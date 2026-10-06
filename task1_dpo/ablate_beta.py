"""Task 1 Step 2: matched short-run DPO forks over beta.

Only beta changes. Every fork: original Qwen2.5-1.5B-Instruct + fresh LoRA, the first
`short_ablation_examples` filtered pairs of the standard training file, same seed, same
optimizer and LoRA config, then the common evaluation protocol (task1_dpo.evaluate).
Finished conditions are skipped, so this can be re-run or split across Colab accounts.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml
from task1_dpo.evaluate import evaluate_policy
from task1_dpo.train import run_training


def beta_tag(beta: float) -> str:
    return f"beta{int(round(beta * 100)):03d}"  # 0.03 -> beta003, 0.10 -> beta010, 0.30 -> beta030


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--only", type=float, help="run a single beta from the configured list")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = [float(b) for b in cfg["betas"]]
    if args.only is not None:
        if not any(abs(args.only - b) < 1e-9 for b in betas):
            raise SystemExit(f"--only {args.only} is not one of the required betas {betas}")
        betas = [args.only]
    print("Beta conditions:", betas, "| examples per fork:", cfg["short_ablation_examples"])
    for beta in betas:
        tag = beta_tag(beta)
        adapter = run_training(args.config, run_name=tag, beta=beta, max_examples=int(cfg["short_ablation_examples"]))
        evaluate_policy(args.config, adapter=str(adapter), name=tag, beta=beta)


if __name__ == "__main__":
    main()
