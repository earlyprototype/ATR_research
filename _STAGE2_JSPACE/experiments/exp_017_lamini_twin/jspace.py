"""EXP_017 Part 2 — the J-space share, and its synthetic tests.

Spec: ../../EXP_017_SPEC.md section 6.4.

The J-space share of a state h at layer l is the squared length of the point
closest to h in the union of cones spanned by at most 25 lens vectors, divided
by the squared length of h. A lens vector, here called an atom, is a row of
W_U^T J_l: the vector whose inner product with a state gives one vocabulary
token's lens score. Coefficients must be zero or positive, so the reachable
set is a union of cones, one per choice of 25 atoms.

The nearest point is found by gradient pursuit: greedily add the atom with the
largest positive correlation with the current residual, refit every selected
atom jointly by non-negative least squares, and stop at 25 atoms or when no
atom correlates positively.

Two implementation facts, stated because the paper's wording does not pin them
down. First, correlations are taken against unit-length atoms, which is
standard matching-pursuit practice; scaling an atom by a positive number does
not change the cone it spans, so unit-length atoms describe the same reachable
set. Second, greedy selection returns the projection onto the cone of the
atoms it happened to select, which can only be at or below the true nearest
point over all choices of 25 atoms, so the reported share is a lower bound on
the defined quantity.

Self-test: python3 jspace.py --selftest
"""
import argparse

import numpy as np
from scipy.optimize import nnls

K_ATOMS = 25


def unit_atoms(A, eps=1e-12):
    """Row-normalise an atom matrix [n_atoms, d] to unit length.

    Returns the normalised matrix and a boolean mask of atoms whose original
    length was too small to normalise, which are excluded from selection.
    """
    norms = np.linalg.norm(A, axis=1)
    keep = norms > eps
    U = np.zeros_like(A)
    U[keep] = A[keep] / norms[keep, None]
    return U, keep


def pursue_batch(U, H, k=K_ATOMS, keep=None, rtol=1e-6):
    """Gradient pursuit for many states at once.

    Args:
        U: unit-length atoms, [n_atoms, d], float32.
        H: states as columns, [d, n_states], float32.
        k: at most this many atoms per state.
        keep: optional boolean mask over atoms, [n_atoms]; atoms outside it are
            never selected.
        rtol: numerical stopping guard. A state whose residual has fallen to
            rtol of its own length is already explained, and any further
            correlation is floating-point noise, so selection stops. This is
            a guard against noise, not a change to the definition: the fitted
            point is unchanged by atoms added at that point.

    Returns:
        shares: [n_states] squared length of the fitted point over squared
            length of the state.
        n_selected: [n_states] how many atoms each state used.
        supports: list of index arrays, the atoms each state selected.

    Every state runs the same number of outer rounds; a state that has no
    positively correlating atom left simply stops being updated.
    """
    d, n = H.shape
    R = H.copy()
    fits = np.zeros_like(H)
    supports = [[] for _ in range(n)]
    active = np.ones(n, dtype=bool)
    blocked = None if keep is None else ~keep
    floor = rtol * np.linalg.norm(H, axis=0)

    for _ in range(k):
        if not active.any():
            break
        C = U @ R[:, active]                       # [n_atoms, n_active]
        if blocked is not None:
            C[blocked] = -np.inf
        cols = np.flatnonzero(active)
        for local, col in enumerate(cols):
            c = C[:, local]
            for idx in supports[col]:              # never re-select an atom
                c[idx] = -np.inf
            best = int(np.argmax(c))
            if not np.isfinite(c[best]) or c[best] <= 0.0:
                active[col] = False                # no atom points the right way
                continue
            supports[col].append(best)
            S = U[supports[col]].T                 # [d, n_selected]
            x, _ = nnls(S.astype(np.float64), H[:, col].astype(np.float64))
            fit = (S @ x.astype(np.float32))
            fits[:, col] = fit
            R[:, col] = H[:, col] - fit
            if np.linalg.norm(R[:, col]) <= floor[col]:
                active[col] = False                # already explained

    hn2 = np.einsum("ij,ij->j", H, H)
    fn2 = np.einsum("ij,ij->j", fits, fits)
    shares = fn2 / np.maximum(hn2, 1e-30)
    return shares, np.array([len(s) for s in supports]), supports


def random_rotation(d, seed):
    """A Haar-distributed orthogonal matrix [d, d] from a QR decomposition."""
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return (Q * np.sign(np.diag(R))).astype(np.float32)


def rotate_states(H, Rot):
    """Rotating the whole dictionary by Rot gives the same shares as rotating
    the states by Rot transposed and leaving the dictionary alone.

    For coefficients c >= 0, ||sum_i c_i (Rot a_i) - h||^2
    = ||Rot (sum_i c_i a_i - Rot^T h)||^2 = ||sum_i c_i a_i - Rot^T h||^2,
    because a rotation preserves lengths, and the fitted point's length is
    likewise unchanged. The cheaper side is computed.
    """
    return Rot.T @ H


# --------------------------------------------------------------------------
# Synthetic tests, spec section 6.4.
# --------------------------------------------------------------------------

def selftest():
    """Four properties, each fixed in the spec before the probe was run."""
    rng = np.random.default_rng(0)
    d, n_atoms = 64, 800
    A = rng.standard_normal((n_atoms, d)).astype(np.float32)
    U, keep = unit_atoms(A)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}"
              .replace(" — ", ": "))

    # 1. A positive combination of 3 atoms must be recovered exactly.
    idx = [7, 101, 500]
    coef = np.array([1.3, 0.4, 2.2], dtype=np.float32)
    h = (U[idx].T @ coef).reshape(d, 1)
    shares, nsel, sup = pursue_batch(U, h, keep=keep)
    check("exact recovery of a 3-atom positive combination",
          abs(shares[0] - 1.0) < 1e-4, f"share={shares[0]:.6f}")
    check("the three planted atoms are selected",
          set(idx).issubset(set(sup[0])), f"selected={sorted(sup[0])}")

    # 2. A state orthogonal to every atom must give a share near zero.
    #    800 atoms span the whole 64-dimensional space, so an exactly
    #    orthogonal state does not exist; the test uses a dictionary of 20
    #    atoms confined to a subspace and a state outside it.
    A2 = np.zeros((20, d), dtype=np.float32)
    A2[:, :20] = rng.standard_normal((20, 20))
    U2, keep2 = unit_atoms(A2)
    h2 = np.zeros((d, 1), dtype=np.float32)
    h2[20:, 0] = rng.standard_normal(d - 20)
    shares2, _, _ = pursue_batch(U2, h2, keep=keep2)
    check("a state orthogonal to the dictionary gives a share near zero",
          shares2[0] < 1e-8, f"share={shares2[0]:.3e}")

    # 3. A state built from 40 atoms cannot be fully explained by 25.
    idx40 = rng.choice(n_atoms, size=40, replace=False)
    coef40 = rng.random(40).astype(np.float32) + 0.1
    h3 = (U[idx40].T @ coef40).reshape(d, 1)
    shares3, nsel3, _ = pursue_batch(U, h3, keep=keep)
    check("at most 25 atoms are ever selected", nsel3[0] <= K_ATOMS,
          f"n_selected={nsel3[0]}")

    # 4. Shares stay inside [0, 1] on random states, and the fitted point is
    #    orthogonal to the residual, which is the defining property of a
    #    projection onto a convex cone.
    H = rng.standard_normal((d, 40)).astype(np.float32)
    shares4, _, sup4 = pursue_batch(U, H, keep=keep)
    check("shares lie in [0, 1] on 40 random states",
          shares4.min() > -1e-9 and shares4.max() < 1 + 1e-6,
          f"min={shares4.min():.4f} max={shares4.max():.4f}")
    resid_ok = True
    for j in range(H.shape[1]):
        S = U[sup4[j]].T
        x, _ = nnls(S.astype(np.float64), H[:, j].astype(np.float64))
        fit = S @ x.astype(np.float32)
        r = H[:, j] - fit
        denom = max(float(np.linalg.norm(fit) * np.linalg.norm(H[:, j])), 1e-30)
        resid_ok = resid_ok and abs(float(r @ fit)) / denom < 1e-4
    check("the fitted point is orthogonal to the residual", resid_ok)

    # 5. The rotation identity the control relies on.
    Rot = random_rotation(d, 2026)
    Arot = A @ Rot.T
    Urot, keeprot = unit_atoms(Arot)
    hr = rng.standard_normal((d, 1)).astype(np.float32)
    s_dict_rotated, _, _ = pursue_batch(Urot, hr, keep=keeprot)
    s_state_rotated, _, _ = pursue_batch(U, rotate_states(hr, Rot), keep=keep)
    check("rotating the dictionary equals rotating the state the other way",
          abs(s_dict_rotated[0] - s_state_rotated[0]) < 1e-4,
          f"{s_dict_rotated[0]:.6f} vs {s_state_rotated[0]:.6f}")

    print(f"\nselftest: {'ALL PASS' if ok else 'FAILURE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    ap.error("nothing to do; pass --selftest")
