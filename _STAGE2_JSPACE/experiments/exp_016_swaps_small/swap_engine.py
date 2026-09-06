"""The intervention engine for EXP_016: patching in lens coordinates.

For a source token s and a target token t at layer l, the two lens vectors
v_s and v_t are stacked into a 768-by-2 matrix V. The residual stream h at a
chosen position is decomposed as c = pinv(V) h, a pair of numbers saying how
much of each lens direction h carries. The swap exchanges those two numbers
and writes back h + alpha * V (sigma(c) - c), where sigma exchanges the two
entries. Everything in the residual stream orthogonal to the plane spanned by
v_s and v_t is left exactly as it was.

Arms:
  lens      the swap above, with the real lens vectors.
  randdir   the same operation with two random directions whose lengths equal
            those of v_s and v_t (the registered norm-matched control).
  randnorm  the randdir directions, but the resulting change to the residual
            stream is rescaled, position by position, to the length of the
            change the lens arm would make on the same residual stream. This
            is the stronger control that asks whether a perturbation of the
            same size in any direction would do the same thing.
"""
from __future__ import annotations
import torch

torch.set_num_threads(1)


def pinv2(V):
    """Pseudo-inverse of a batch of 768-by-2 matrices, shape [B,768,2] to
    [B,2,768], computed from the 2-by-2 Gram matrix."""
    G = V.transpose(1, 2) @ V                       # [B,2,2]
    eye = torch.eye(2, dtype=V.dtype).expand_as(G)
    scale = G.diagonal(dim1=1, dim2=2).abs().amax(dim=1)[:, None, None]
    G = G + 1e-6 * eye * scale.clamp_min(1e-12)
    return torch.linalg.solve(G, V.transpose(1, 2))  # [B,2,768]


def random_pair(v_s, v_t, seed):
    """Two random 768-number directions with the same lengths as v_s and v_t,
    drawn from a Gaussian with the given seed."""
    g = torch.Generator().manual_seed(int(seed) % (2**31))
    r = torch.randn(768, 2, generator=g, dtype=torch.float32)
    r = r / r.norm(dim=0, keepdim=True)
    return r * torch.stack([v_s.norm(), v_t.norm()])


class SwapPlan:
    """One batch of conditions over a single prompt.

    Every batch element carries its own layer set, strength alpha, set of
    token positions to patch, and arm. All elements run in one forward pass.
    """

    def __init__(self, n_tokens, layers):
        self.T = n_tokens
        self.layers = list(layers)
        self.rows = []

    def add(self, per_layer_V_used, per_layer_V_lens, alpha, positions, rescale):
        """per_layer_V_*: {layer: Tensor[768,2]} for the layers this element
        patches at. `positions` is a list of token positions."""
        mask = torch.zeros(self.T)
        mask[list(positions)] = 1.0
        self.rows.append(dict(Vu=per_layer_V_used, Vl=per_layer_V_lens,
                              alpha=float(alpha), mask=mask,
                              rescale=bool(rescale)))
        return len(self.rows) - 1

    def build(self):
        B = len(self.rows)
        self.B = B
        self.alpha = torch.tensor([r["alpha"] for r in self.rows])
        self.mask = torch.stack([r["mask"] for r in self.rows])          # [B,T]
        self.rescale = torch.tensor([r["rescale"] for r in self.rows])
        self.Vu, self.Vl, self.Pu, self.Pl, self.act = {}, {}, {}, {}, {}
        # Squared size of the total change each row makes to the residual
        # stream, summed over patched positions and layers; filled by the
        # hooks during run_plan and read back as patch_norm.
        self.change_sq = torch.zeros(B)
        self.change_sq_layer = {l: torch.zeros(B) for l in self.layers}
        zero = torch.zeros(768, 2)
        for l in self.layers:
            Vu = torch.stack([r["Vu"].get(l, zero) for r in self.rows])
            Vl = torch.stack([r["Vl"].get(l, zero) for r in self.rows])
            act = torch.tensor([1.0 if l in r["Vu"] else 0.0 for r in self.rows])
            # Inactive rows carry a well-conditioned placeholder so that the
            # pseudo-inverse is finite; their change is zeroed by `act` anyway.
            ph = torch.eye(768, 2).expand_as(Vu)
            keep = (act > 0)[:, None, None]
            self.Vu[l] = Vu = torch.where(keep, Vu, ph)
            self.Vl[l] = Vl = torch.where(keep, Vl, ph)
            self.act[l] = act
            self.Pu[l] = pinv2(Vu)
            self.Pl[l] = pinv2(Vl)
        return self

    def hooks(self):
        out = []
        for l in self.layers:
            out.append((f"blocks.{l}.hook_resid_post", self._make(l)))
        return out

    def _make(self, l):
        Vu, Vl, Pu, Pl = self.Vu[l], self.Vl[l], self.Pu[l], self.Pl[l]
        act, alpha, mask, resc = self.act[l], self.alpha, self.mask, self.rescale
        any_resc = bool(resc.any())

        def hook(resid, hook):
            h = resid.float()
            c = torch.einsum("bij,btj->bti", Pu, h)
            patch = torch.einsum("bdi,bti->btd", Vu, c.flip(-1) - c)
            if any_resc:
                cl = torch.einsum("bij,btj->bti", Pl, h)
                pl = torch.einsum("bdi,bti->btd", Vl, cl.flip(-1) - cl)
                nu = patch.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                sc = torch.where(resc[:, None, None],
                                 pl.norm(dim=-1, keepdim=True) / nu,
                                 torch.ones_like(nu))
                patch = patch * sc
            delta = patch * (alpha * act)[:, None, None] * mask[:, :, None]
            sq = (delta * delta).sum(dim=(1, 2))
            self.change_sq += sq
            self.change_sq_layer[l] += sq
            return resid + delta.to(resid.dtype)
        return hook


@torch.no_grad()
def run_plan(model, toks, plan):
    """Run one batched forward pass under the plan and return the
    next-token log-probabilities at the last position, shape [B, vocab].
    Afterwards plan.change_sq[b].sqrt() is the size (Euclidean norm over all
    patched positions and layers) of the change row b made to the residual
    stream, the disturbance size the specification asks to be recorded."""
    plan.change_sq.zero_()
    for v in plan.change_sq_layer.values():
        v.zero_()
    tk = toks.repeat(plan.B, 1)
    resid = model.run_with_hooks(tk, return_type=None,
                                 stop_at_layer=model.cfg.n_layers,
                                 fwd_hooks=plan.hooks())
    logits = model.unembed(model.ln_final(resid[:, -1:, :]))[:, 0, :].float()
    return torch.log_softmax(logits, dim=-1)
