"""Task 3 Step 3: canonical GRPO vs Dr-GRPO sequence normalization.

Matched short continuations (fork_updates) from the identical supplied midpoint. Only loss_type
changes; reward, beta, epsilon, prompts (same seeded schedule), K, generation settings, and the
completion cap are fixed. Each fork then gets the common held-out evaluation. The length-conditioned
statistic comes from each fork's completions.jsonl (length, advantage, analytic per-token and
per-sequence weights) and is summarized by task3_grpo.summarize.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--only", choices=["grpo", "dr_grpo"], help="run a single normalization")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    for loss_type in ([args.only] if args.only else ["grpo", "dr_grpo"]):
        name = f"fork_{loss_type}"
        adapter = run_grpo(args.config, updates=int(cfg["fork_updates"]), loss_type=loss_type, run_name=name)
        evaluate_policy(args.config, adapter=str(adapter), name=name)


if __name__ == "__main__":
    main()
