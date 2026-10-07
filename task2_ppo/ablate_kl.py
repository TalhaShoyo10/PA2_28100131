"""Task 2 Step 3: reference-KL pressure study.

Matched short continuations (fork_updates) from the identical supplied midpoint for each beta_KL,
epsilon fixed at the config value, then the common held-out evaluation. The beta_KL = 0.10 fork has
exactly the configuration of the epsilon = 0.20 clipping fork, so it is one shared run
(fork_eps020_kl010) and is skipped if already done.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml
from task2_ppo.analyze_clipping import fork_name
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--only", type=float, help="run a single beta_KL")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    kl_values = [float(k) for k in cfg["kl_values"]]
    if args.only is not None:
        if not any(abs(args.only - k) < 1e-9 for k in kl_values):
            raise SystemExit(f"--only {args.only} not in {kl_values}")
        kl_values = [args.only]
    eps = float(cfg["clip_epsilon"])
    print("KL beta conditions:", kl_values, "| epsilon:", eps, "| fork updates:", cfg["fork_updates"])
    for kl in kl_values:
        name = fork_name(eps, kl)
        adapter = run_ppo(args.config, updates=int(cfg["fork_updates"]), clip_epsilon=eps, kl_beta=kl, run_name=name)
        evaluate_policy(args.config, adapter=str(adapter), name=name)


if __name__ == "__main__":
    main()
