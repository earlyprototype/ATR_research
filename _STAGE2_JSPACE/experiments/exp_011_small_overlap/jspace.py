"""J-space decomposition for EXP_011 (GPT-2 Small, Neuronpedia pre-fitted Jacobian lens).

The J-space at layer l is the union of the cones spanned by at most k = 25 lens
vectors with non-negative coefficients, where a lens vector is one row of
W_U J_l (W_U is the 50257 by 768 unembedding matrix and J_l is the lens's
768 by 768 Jacobian for source layer l). The J-space component of a state is the
nearest point of that set, found by non-negative orthogonal matching pursuit,
which the EXP_011 brief calls gradient pursuit: repeatedly add the atom with the
largest positive correlation with the current residual, re-fit non-negative least
squares on the selected atoms, and stop at 25 atoms with a non-zero coefficient or
when no atom in the whole dictionary has a positive correlation with the residual.
The J-space share of a state is the squared norm of the component divided by the
squared norm of the state.

Everything here is batched over states so that one pass over the 154 MB
dictionary serves every state at a layer.

**Change of 2026-09-05, after the committed EXP_011 run.** The first version kept
an atom in the selected set even when the non-negative least-squares re-fit gave
it a coefficient of exactly zero. Such an atom contributed nothing to the
reconstruction, could never be selected again, and still consumed one of the 25
allowed slots, so a state could stop at 25 selections while using fewer than 25
atoms. `decompose` now drops zero-coefficient atoms from the active set and keeps
going until 25 atoms carry a non-zero coefficient or no atom anywhere has a
positive correlation with the residual, and `n_atoms` now counts atoms that
actually carry weight. Two consequences to know. First, every share this file has
ever returned is a lower bound on the true J-space share, because it is the share
of an actual feasible point of the J-space; the change makes the bound tighter for
the affected states and can only move a share upward. Second, the committed
outputs under `output/` were produced by the earlier version and are unchanged;
the limitation is recorded in RESULTS_EXP011.md rather than repaired by a re-run.

Why the loop terminates. After a non-negative least-squares fit on the active set
the optimality conditions give a_j . r = 0 for every atom with a positive
coefficient and a_j . r <= 0 for every atom held at zero, so the atom chosen next,
which must have a strictly positive correlation with the residual, is never one of
those. Adding it strictly lowers the residual, because a small positive step along
it reduces the error, so the newly added atom always ends with a strictly positive
coefficient and the objective strictly decreases at every iteration. No active set
can therefore repeat, and the number of active sets is finite. `max_iter` is a
safety bound only.
"""
import numpy as np
import torch
from scipy.optimize import nnls

K_ATOMS = 25


def unit_rows(D, eps=1e-12):
    """Row-normalised copy of a dictionary D of shape [n_atoms, d_model]."""
    return D / D.norm(dim=1, keepdim=True).clamp_min(eps)


def decompose(D, H, k=K_ATOMS, Dn=None, record_atoms=False, rel_tol=1e-6,
              max_iter=None):
    """Non-negative sparse decomposition of every state in H against dictionary D.

    Args:
        D: dictionary, torch float32 tensor [n_atoms, d_model].
        H: states, torch float32 tensor [n_states, d_model].
        k: maximum number of atoms per state (25 in the paper's construction).
            The limit is on atoms carrying a non-zero coefficient.
        Dn: optional precomputed row-normalised D.
        record_atoms: also return the selected atom indices and coefficients.
        rel_tol: stop a state early once its residual norm falls below this
            fraction of its own norm (an exactly representable state stops at
            the atoms it is built from instead of padding out to k).
        max_iter: safety bound on the number of selection rounds. Defaults to
            8k. The termination argument in this module's docstring shows the
            loop ends on its own; a state that ever reaches this bound is
            reported in 'hit_max_iter' so it cannot pass unnoticed.

    Returns:
        dict with 'share' (numpy [n_states]), 'n_atoms' (numpy [n_states],
        counting only atoms with a non-zero coefficient), 'resid_norm',
        'state_norm', 'hit_max_iter', and, when record_atoms is set, the lists
        'atoms' and 'coeffs'.
    """
    if Dn is None:
        Dn = unit_rows(D)
    if max_iter is None:
        max_iter = 8 * k
    n = H.shape[0]
    R = H.clone()
    stop_at = H.norm(dim=1) * rel_tol
    sel = [[] for _ in range(n)]          # the ACTIVE set: non-zero coefficients only
    coef = [None] * n
    alive = torch.ones(n, dtype=torch.bool)
    for _ in range(max_iter):
        if not bool(alive.any()):
            break
        C = Dn @ R.T                      # [n_atoms, n_states]
        for i in range(n):
            # Mask only the current active set. An atom dropped for a zero
            # coefficient may be chosen again later, once the residual has moved.
            if sel[i]:
                C[torch.tensor(sel[i], dtype=torch.long), i] = -float("inf")
            if not alive[i]:
                C[:, i] = -float("inf")
        best_val, best_idx = C.max(dim=0)
        for i in range(n):
            if not alive[i]:
                continue
            if float(best_val[i]) <= 0.0:
                # No atom anywhere has a positive correlation with the residual:
                # atoms outside the active set are covered by this test and atoms
                # inside it by the non-negative least-squares optimality
                # conditions. This point is the exact nearest point of the whole
                # positive cone, not an approximation.
                alive[i] = False
                continue
            trial = sel[i] + [int(best_idx[i])]
            A = D[torch.tensor(trial, dtype=torch.long)].T.numpy()   # [d, |S|+1]
            x, _ = nnls(A, H[i].numpy())
            R[i] = H[i] - torch.from_numpy(A @ x)
            keep = [j for j, v in enumerate(x) if v > 0.0]
            sel[i] = [trial[j] for j in keep]
            coef[i] = x[keep]
            if len(sel[i]) >= k:
                alive[i] = False
            elif float(R[i].norm()) <= float(stop_at[i]):
                alive[i] = False
    hit_max = np.array([bool(a) for a in alive.tolist()], dtype=bool)
    state_sq = (H * H).sum(dim=1)
    resid_sq = (R * R).sum(dim=1)
    share = (1.0 - resid_sq / state_sq.clamp_min(1e-30)).clamp(0.0, 1.0)
    out = {
        "share": share.numpy().astype(np.float64),
        "n_atoms": np.array([len(s) for s in sel], dtype=np.int32),
        "resid_norm": resid_sq.sqrt().numpy().astype(np.float64),
        "state_norm": state_sq.sqrt().numpy().astype(np.float64),
        "hit_max_iter": hit_max,
    }
    if record_atoms:
        out["atoms"] = [list(map(int, s)) for s in sel]
        out["coeffs"] = [[] if c is None else [float(v) for v in c] for c in coef]
    return out


def random_rotation(d, seed):
    """A uniformly random orthogonal d by d matrix (QR of a Gaussian, sign-fixed)."""
    g = torch.Generator().manual_seed(int(seed))
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    return Q.float()


def gaussian_dictionary_like(D, seed, chunk=8192):
    """Independent Gaussian atoms with the same shape and per-row norms as D."""
    g = torch.Generator().manual_seed(int(seed))
    out = torch.empty_like(D)
    norms = D.norm(dim=1, keepdim=True)
    for a in range(0, D.shape[0], chunk):
        b = min(a + chunk, D.shape[0])
        block = torch.randn(b - a, D.shape[1], generator=g)
        block = block / block.norm(dim=1, keepdim=True).clamp_min(1e-12)
        out[a:b] = block * norms[a:b]
    return out


def _self_test():
    """A state built from five known atoms must be recovered with share near 1."""
    torch.manual_seed(0)
    d, V = 768, 4000
    D = torch.randn(V, d)
    D = D * (0.5 + torch.rand(V, 1))          # unequal atom norms, as the lens has
    idx = [11, 402, 1500, 2222, 3999]
    coef = torch.tensor([1.0, 0.4, 2.5, 0.9, 0.2])
    h = (D[idx] * coef.unsqueeze(1)).sum(dim=0)
    res = decompose(D, h.unsqueeze(0), record_atoms=True)
    print(f"  synthetic 5-atom state: share={res['share'][0]:.6f} "
          f"n_atoms={res['n_atoms'][0]} recovered={sorted(res['atoms'][0])[:8]}")
    assert res["share"][0] > 0.999, "5-atom recovery failed"
    assert set(idx).issubset(set(res["atoms"][0])), "true atoms not all selected"

    # A generic random state must NOT be fully explained by 25 atoms.
    hr = torch.randn(1, d)
    rr = decompose(D, hr, record_atoms=True)
    print(f"  random state: share={rr['share'][0]:.6f} n_atoms={rr['n_atoms'][0]}")
    assert not rr["hit_max_iter"].any(), "a state ran to the max_iter safety bound"

    # Every reported atom must carry a strictly positive coefficient, and a state
    # that stops at the limit must be using the full allowance of k atoms rather
    # than fewer with some held at zero. This is the 2026-09-05 solver fix.
    for res_ in (res, rr):
        for atoms_, coeffs_ in zip(res_["atoms"], res_["coeffs"]):
            assert len(atoms_) == len(coeffs_), "atom and coefficient lists disagree"
            assert all(c > 0.0 for c in coeffs_), "a zero coefficient was kept"
            assert len(set(atoms_)) == len(atoms_), "an atom was selected twice"
            assert len(atoms_) <= K_ATOMS, "more atoms than the limit allows"
    print(f"  every reported atom carries a positive coefficient "
          f"(synthetic {res['n_atoms'][0]} atoms, random {rr['n_atoms'][0]} atoms)")

    # The stopping condition is the cone optimality condition: when a state stops
    # before the limit, no atom in the whole dictionary may have a positive
    # correlation with its residual.
    Dn_ = unit_rows(D)
    for label_, h_, out_ in (("synthetic", h, res), ("random", hr[0], rr)):
        if int(out_["n_atoms"][0]) >= K_ATOMS:
            continue
        A_ = D[torch.tensor(out_["atoms"][0], dtype=torch.long)].T
        r_ = h_ - A_ @ torch.tensor(out_["coeffs"][0], dtype=torch.float32)
        worst_ = float((Dn_ @ r_).max())
        print(f"  {label_} state stopped below the limit: largest correlation of any "
              f"atom with the residual = {worst_:.3e} (must not be positive)")
        assert worst_ <= 1e-4 * float(h_.norm()), "stopped with a usable atom left"

    # Rotating the dictionary must leave the share unchanged (control (a) identity).
    Q = random_rotation(d, 7)
    Drot = D @ Q.T
    a = decompose(Drot, h.unsqueeze(0))["share"][0]
    b = decompose(D, (h @ Q).unsqueeze(0))["share"][0]
    print(f"  rotation identity: rotated-dictionary share={a:.8f} "
          f"inverse-rotated-state share={b:.8f} (must agree)")
    assert abs(a - b) < 1e-5, "rotation identity broken"

    # Batched and single-state results must agree.
    Hb = torch.stack([h, hr[0], h * 3.0])
    sb = decompose(D, Hb)["share"]
    ss = [decompose(D, x.unsqueeze(0))["share"][0] for x in Hb]
    print(f"  batch vs single: {np.round(sb, 8)} vs {np.round(ss, 8)}")
    assert np.allclose(sb, ss, atol=1e-6), "batching changes the answer"

    # Scale invariance of the share.
    s1 = decompose(D, hr)["share"][0]
    s2 = decompose(D, hr * 137.0)["share"][0]
    print(f"  scale invariance: {s1:.8f} vs {s2:.8f}")
    assert abs(s1 - s2) < 1e-5, "share is not scale invariant"
    print("  jspace self-test PASSED")


if __name__ == "__main__":
    torch.set_num_threads(1)
    _self_test()
