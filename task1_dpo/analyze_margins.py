"""Task 1 diagnostic (CPU only, no model calls): what a positive DPO margin is made of.

The margin m = d(y+) - d(y-), with d(y) = log pi(y) - log ref(y), is a gap between two changes.
m > 0 does not imply that the chosen response became more likely: both responses can lose
log-probability, the rejected one by more. This script decomposes the saved per-pair records
(pairs_heldout.jsonl / pairs_length_stratified.jsonl written by task1_dpo.evaluate) into:

- among pairs with m > 0: the share where the chosen log-prob fell (d(y+) < 0); in every such
  pair the rejected log-prob fell by more, since m > 0 and d(y+) < 0 imply d(y-) < d(y+) < 0;
- over all pairs: the share where the chosen log-prob fell, and the mean d(y+) and d(y-).

It reads existing evaluation outputs only and changes none of them.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

CONDITIONS = ["sft", "standard", "beta003", "beta010", "beta030", "length_balanced"]


def decompose(records) -> dict:
    d_c = np.array([r["policy_chosen_logp"] - r["ref_chosen_logp"] for r in records])
    d_r = np.array([r["policy_rejected_logp"] - r["ref_rejected_logp"] for r in records])
    m = np.array([r["margin"] for r in records])
    pos = m > 0
    n_pos = int(pos.sum())
    return {
        "n_pairs": int(m.size),
        "n_margin_positive": n_pos,
        "preference_accuracy": float(pos.mean()),
        "positive_with_chosen_down": int((pos & (d_c < 0)).sum()),
        "frac_positive_with_chosen_down": float((pos & (d_c < 0)).sum() / n_pos) if n_pos else None,
        "frac_all_chosen_down": float((d_c < 0).mean()),
        "mean_delta_chosen": float(d_c.mean()),
        "mean_delta_rejected": float(d_r.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = repo_path(cfg["results_dir"])
    seed = int(cfg["seed"])

    out = {}
    for cond in CONDITIONS:
        run_dir = out_dir / f"task1_dpo_eval_{cond}_seed{seed}"
        heldout = read_jsonl(run_dir / "pairs_heldout.jsonl")
        stratified = read_jsonl(run_dir / "pairs_length_stratified.jsonl")
        strata = sorted({r["length_stratum"] for r in stratified})
        out[cond] = {
            "source": str(run_dir.relative_to(repo_path("."))).replace("\\", "/"),
            "heldout": decompose(heldout),
            "length_stratified": {s: decompose([r for r in stratified if r["length_stratum"] == s]) for s in strata},
        }
    save_json(out_dir / "margin_decomposition.json", out)

    rows = []
    for cond, v in out.items():
        for split, stats in [("heldout", v["heldout"])] + [(f"stratum:{s}", st) for s, st in v["length_stratified"].items()]:
            rows.append({"condition": cond, "split": split, **stats})
    keys = list(rows[0].keys())
    with open(out_dir / "margin_decomposition.csv", "w", encoding="utf-8", newline="") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")
    print(json.dumps({c: out[c]["heldout"] for c in out}, indent=2))


if __name__ == "__main__":
    main()
