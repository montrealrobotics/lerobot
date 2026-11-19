"""Save the decoded action chunks behind the offline steerability metrics, for plotting.

For each checkpoint, decodes the same M DSRL-style noises (w ~ U[-1,1]^32, tiled over the chunk)
on the first K cached demo observations, exactly as eval_offline_steerability.py does, and saves
the executed chunk (K, M, 16, 22) in model space (quantile-normalized), plus the demo chunk.

  python scripts/dump_decoded_actions.py --task lamp --k 4 --out results_fp32/decoded \
      --ckpt base=/path/to/pretrained_model --ckpt bf16_EXPERT-ONLY_2000=/path ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_offline_steerability as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(E.TASKS))
    ap.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--k", type=int, default=4, help="number of cached observations to decode")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=str(E.DEFAULT_OUT / "offline_steerability" / "cache"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    # same noise-set arguments as the metrics run (its defaults), so the noises are identical
    args = argparse.Namespace(n_obs=32, m=64, n_lip=16, sigma_delta=0.05, n_fm=16, noise_kind="box",
                              noise_seed=1234, obs_seed=0, device=a.device)

    import torch
    obs = E.load_observations(a.task, args.n_obs, args.obs_seed, Path(a.cache))[: a.k]
    noises = E.noise_sets(a.task, args)[: a.k]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.ckpt:
        label, path = spec.split("=", 1)
        f = out / f"{a.task}_{label}.npz"
        if f.exists():
            print(f"{label}: exists, skipped", flush=True)
            continue
        policy = E.load_policy(path, a.device)
        pre = E.make_preprocessor(path, a.device)
        acts, demos = [], []
        with torch.no_grad():
            for ob, nz in zip(obs, noises):
                batch = E.prepare(pre, ob)
                emb = policy.prepare_embodiment_ids(batch)
                kv, pad, _ = E.prefix_kv(policy, batch)
                W = torch.as_tensor(nz["W"], device=a.device)
                noise = W[:, None, :].expand(-1, E.H, -1).contiguous()
                A = E.decode(policy, kv, pad, noise, emb).float().cpu().numpy()
                acts.append(A[:, : E.H_EXEC, : E.D_ACT])
                demos.append(batch["action"].float().cpu().numpy()[0, : E.H_EXEC, : E.D_ACT])
        np.savez_compressed(f, actions=np.stack(acts), demo=np.stack(demos), checkpoint=path,
                            episodes=np.array([o["meta"]["episode"] for o in obs]))
        print(f"{label}: saved {f} {np.stack(acts).shape}", flush=True)
        del policy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
