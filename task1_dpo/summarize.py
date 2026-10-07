"""Collect every finished Task 1 train/eval run into one machine-readable table.

Writes results/task1_dpo/summary.csv and summary.json. Reads only status=done runs, never
archived __attempt directories. Budget columns make the standard one-epoch run visibly
different from the short beta forks (manual Task 1 Required Evidence).
"""
from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def _done_runs(results_dir, prefix):
    root = repo_path(results_dir)
    for d in sorted(root.glob(f"{prefix}*")):
        if "__attempt" in d.name or "smoke" in d.name or not (d / "status.json").exists():
            continue
        if json.loads((d / "status.json").read_text(encoding="utf-8"))["state"] != "done":
            continue
        yield d


def _finite_mean(values):
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else float("nan")


def build_summary(cfg) -> pd.DataFrame:
    seed = int(cfg["seed"])
    train = {}
    for d in _done_runs(cfg["results_dir"], "task1_dpo_train_"):
        cond = d.name[len("task1_dpo_train_"):-len(f"_seed{seed}")]
        manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        log = read_jsonl(d / "train_log.jsonl")
        train[cond] = {
            "train_pairs": manifest["budget"]["train_pairs"],
            "optimizer_updates": manifest["budget"]["optimizer_updates"],
            "train_dataset": json.loads((d / "config.json").read_text(encoding="utf-8"))["train_dataset"],
            "train_beta": metrics["beta"],
            "train_loss_first": metrics["first_update_loss"],
            "train_loss_last5_mean": metrics["mean_train_loss_last_5_updates"],
            "train_wall_clock_s": manifest.get("wall_clock_seconds"),
            "train_peak_vram_gib": manifest.get("peak_vram_allocated_gib"),
            # Overflow updates (skipped by GradScaler) have a non-finite norm; average the rest.
            "train_grad_norm_mean_finite": _finite_mean([r["grad_norm_preclip"] for r in log]),
            "train_updates_skipped_overflow": sum(bool(r["step_skipped_overflow"]) for r in log),
            "train_skipped_update_indices": [r["update"] for r in log if r["step_skipped_overflow"]],
        }

    rows = []
    for d in _done_runs(cfg["results_dir"], "task1_dpo_eval_"):
        cond = d.name[len("task1_dpo_eval_"):-len(f"_seed{seed}")]
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        g = m["heldout_generation"]
        row = {
            "condition": cond,
            **train.get(cond, {"train_pairs": 0, "optimizer_updates": 0} if cond == "sft" else {}),
            "heldout_dpo_loss": m["heldout_pairs"]["dpo_loss"],
            "heldout_loss_beta": m["beta_for_heldout_loss"],
            "heldout_preference_accuracy": m["heldout_pairs"]["preference_accuracy"],
            "heldout_mean_margin": m["heldout_pairs"]["mean_margin"],
            "kl_token_mean": g["kl_token_mean"],
            "kl_sequence_mean": g["kl_sequence_mean"],
            "reward_mean": g.get("reward_mean"),
            "reward_std": g.get("reward_std"),
            "gen_len_mean": g["length_tokens"]["mean"],
            "gen_len_std": g["length_tokens"]["std"],
            "gen_len_iqr": g["length_tokens"]["iqr"],
            "gen_truncation_rate": g["truncation_rate"],
            "word_limit_compliance": m["word_limit"]["compliance_rate"],
            "word_limit_words_mean": m["word_limit"]["words"]["mean"],
        }
        for stratum, agg in m["length_stratified_pairs"]["by_stratum"].items():
            row[f"acc_{stratum}"] = agg["preference_accuracy"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    df = build_summary(cfg)
    out = repo_path(cfg["results_dir"])
    df.to_csv(out / "summary.csv", index=False)
    save_json(out / "summary.json", df.to_dict(orient="records"))
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
