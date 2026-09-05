"""EXP_017 — build a usable lens from a partial fit checkpoint.

Spec section 6.2 sets a wall-clock budget for the twin's lens fit. The fit
checkpoints its running sum of per-prompt Jacobians after every prompt, so if
the budget runs out before the chosen prompt count is reached, the prompts
already done still make a lens: the fitted matrix for each layer is the running
sum divided by the number of prompts completed, which is exactly what the
instrument itself would write on a clean finish.

Using this script is a recorded deviation, because the lens is then fitted on
fewer prompts than the budget rule chose. The actual prompt count is written
into the lens file and reported in the results.

Usage:
    python3 lens_from_checkpoint.py CHECKPOINT.pt OUTPUT.pt
"""
import hashlib
import sys

import torch

torch.set_num_threads(1)

import jlens


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    n = int(state["n_done"])
    if n < 1:
        raise SystemExit(f"checkpoint has {n} completed prompts; nothing to build")
    jac = {int(l): (J / n) for l, J in state["jacobian_sum"].items()}
    d_model = next(iter(jac.values())).shape[0]
    lens = jlens.JacobianLens(jacobians=jac, n_prompts=n, d_model=int(d_model))
    lens.save(out_path)
    h = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    print(f"built from {n} completed prompts (source layers "
          f"{state['source_layers']}, target layer {state['target_layer']}, "
          f"skip_first {state['skip_first']})")
    print(f"{lens}\nsaved -> {out_path}\nsha256 {h}")


if __name__ == "__main__":
    main()
