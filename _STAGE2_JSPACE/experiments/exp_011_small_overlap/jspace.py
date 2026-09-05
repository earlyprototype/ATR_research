"""J-space decomposition for EXP_011 (GPT-2 Small, Neuronpedia pre-fitted Jacobian lens).

The J-space at layer l is the union of the cones spanned by at most k = 25 lens
vectors with non-negative coefficients, where a lens vector is one row of
W_U J_l (W_U is the 50257 by 768 unembedding matrix and J_l is the lens's
768 by 768 Jacobian for source layer l). The J-space component of a state is the
nearest point of that set, found by non-negative orthogonal matching pursuit,
which the EXP_011 brief calls gradient pursuit: repeatedly add the atom with the
largest positive correlation with the current residual, re-fit non-negative least
squares on the selected atoms, and stop at 25 atoms or when no atom has a
positive correlation. The J-space share of a state is the squared norm of the
component divided by the squared norm of the state.

Everything here is batched over states so that one pass over the 154 MB
dictionary serves every state at a layer.
"""
import numpy as np
import torch
from scipy.optimize import nnls

K_ATOMS = 25


def unit_rows(D, eps=1e-12):
    """Row-normalised copy of a dictionary D of shape [n_atoms, d_model]."""
    return D / D.norm(dim=1, keepdim=True).clamp_min(eps)


def decompose(D, H, k=K_ATOMS, Dn=None, record_atoms=False, rel_tol=1e-6):
    """Non-negative sparse decomposition of every state in H against dictionary D.

    Args:
        D: dictionary, torch float32 tensor [n_atoms, d_model].
        H: states, torch float32 tensor [n_states, d_model].
        k: maximum number of atoms per state (25 in the paper's construction).
        Dn: optional precomputed row-normalised D.
        record_atoms: also return the selected atom indices and coefficients.
        rel_tol: stop a state early once its residual norm falls below this
            fraction of its own norm (an exactly representable state stops at
            the atoms it is built from instead of padding out to k).

    Returns:
        dict with 'share' (numpy [n_states]), 'n_atoms' (numpy [n_states]),
        'resid_norm', 'state_norm', and, when record_atoms is set, the lists
        'atoms' and 'coeffs'.
    """
    if Dn is None:
        Dn = unit_rows(D)
    n = H.shape[0]
    R = H.clone()
    stop_at = H.norm(dim=1) * rel_tol
    sel = [[] for _ in range(n)]
    coef = [None] * n
    alive = torch.ones(n, dtype=torch.bool)
    for _ in range(k):
        if not bool(alive.any()):
            break
        C = Dn @ R.T                      # [n_atoms, n_states]
        for i in range(n):
            if sel[i]:
                C[torch.tensor(sel[i], dtype=torch.long), i] = -float("inf")
            if not alive[i]:
                C[:, i] = -float("inf")
        best_val, best_idx = C.max(dim=0)
        for i in range(n):
            if not alive[i]:
                continue
            if float(best_val[i]) <= 0.0:
                alive[i] = False
                continue
            sel[i].append(int(best_idx[i]))
            A = D[torch.tensor(sel[i], dtype=torch.long)].T.numpy()   # [d, |S|]
            x, _ = nnls(A, H[i].numpy())
            coef[i] = x
            R[i] = H[i] - torch.from_numpy(A @ x)
            if float(R[i].norm()) <= float(stop_at[i]):
                alive[i] = False
    state_sq = (H * H).sum(dim=1)
    resid_sq = (R * R).sum(dim=1)
    share = (1.0 - resid_sq / state_sq.clamp_min(1e-30)).clamp(0.0, 1.0)
    out = {
        "share": share.numpy().astype(np.float64),
        "n_atoms": np.array([len(s) for s in sel], dtype=np.int32),
        "resid_norm": resid_sq.sqrt().numpy().astype(np.float64),
        "state_norm": state_sq.sqrt().numpy().astype(np.float64),
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
    rr = decompose(D, hr)
    print(f"  random state: share={rr['share'][0]:.6f} n_atoms={rr['n_atoms'][0]}")

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
