"""Task 1 Step 3: length-confounding study.

1. Dataset property (CPU only): how response length relates to the preference label in the
   standard and the length-balanced training files.
2. Train one DPO model from the original initialization on the supplied length-balanced subset,
   with the same one-epoch settings as standard DPO.
3. Evaluate it with the common protocol (per-stratum accuracy, generated length, word limits).
4. Write the standard-vs-balanced comparison table.

The standard DPO model must already be trained and evaluated (Step 1).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_tokenizer
from task1_dpo.evaluate import evaluate_policy
from task1_dpo.summarize import build_summary
from task1_dpo.train import filter_prompts_that_fit, run_training


def dataset_length_profile(rows, tokenizer) -> dict:
    """Length is measured on the raw responses (before the max_sequence_length cut)."""
    diffs = []
    score_diffs = []
    for row in rows:
        yc, yr = preference_responses(row)
        lc = len(tokenizer(yc, add_special_tokens=False)["input_ids"])
        lr = len(tokenizer(yr, add_special_tokens=False)["input_ids"])
        diffs.append(lc - lr)
        if "score_chosen" in row and "score_rejected" in row:
            score_diffs.append(float(row["score_chosen"]) - float(row["score_rejected"]))
    d = np.asarray(diffs, dtype=float)
    out = {
        "n_pairs": int(d.size),
        "fraction_chosen_longer": float((d > 0).mean()),
        "fraction_rejected_longer": float((d < 0).mean()),
        "fraction_equal_length": float((d == 0).mean()),
        "mean_chosen_minus_rejected_tokens": float(d.mean()),
        "median_chosen_minus_rejected_tokens": float(np.median(d)),
    }
    if len(score_diffs) == len(diffs) and np.std(score_diffs) > 0:
        out["corr_length_diff_vs_score_diff"] = float(np.corrcoef(d, score_diffs)[0, 1])
    strata = [row.get("length_stratum") for row in rows]
    if all(strata):
        out["strata_counts"] = {s: strata.count(s) for s in sorted(set(strata))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--dataset-only", action="store_true", help="only compute the CPU dataset length profile")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = repo_path(cfg["results_dir"])

    tokenizer = load_tokenizer(cfg["base_model"])
    max_length = int(cfg["max_sequence_length"])
    profile = {}
    for key in ("dpo_standard_train", "dpo_length_train"):
        rows, _ = filter_prompts_that_fit(read_jsonl(cfg["paths"][key]), tokenizer, max_length)
        profile[key] = dataset_length_profile(rows, tokenizer)
    save_json(out_dir / "dataset_length_profile.json", profile)
    print(json.dumps(profile, indent=2))
    if args.dataset_only:
        return

    adapter = run_training(args.config, run_name="length_balanced", dataset_path=cfg["paths"]["dpo_length_train"])
    evaluate_policy(args.config, adapter=str(adapter), name="length_balanced")

    summary = build_summary(cfg)
    wanted = summary[summary["condition"].isin(["standard", "length_balanced"])]
    if len(wanted) < 2:
        raise SystemExit("Standard DPO evaluation missing: run task1_dpo.train + task1_dpo.evaluate --name standard first.")
    cols = ["condition", "train_pairs", "optimizer_updates", "heldout_preference_accuracy",
            "acc_preferred_longer", "acc_length_matched", "acc_rejected_longer",
            "gen_len_mean", "gen_len_std", "gen_len_iqr", "word_limit_compliance", "word_limit_words_mean"]
    table = wanted[[c for c in cols if c in wanted.columns]]
    table.to_csv(out_dir / "length_comparison.csv", index=False)
    save_json(out_dir / "length_comparison.json", table.to_dict(orient="records"))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
