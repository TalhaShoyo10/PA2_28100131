"""Collect finished Task 3 runs into machine-readable tables.

results/task3_grpo/summary.csv               one row per condition: training diagnostics + held-out eval
results/task3_grpo/standard_trajectory.csv   per-update trajectory of the standard continuation
results/task3_grpo/group_size.csv            K-study per K and difficulty bin
results/task3_grpo/normalization_length.csv  length-conditioned gradient allocation per normalization fork
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def _done(results_dir, prefix):
    for d in sorted(repo_path(results_dir).glob(f"{prefix}*")):
        if "__attempt" in d.name or "smoke" in d.name or not (d / "status.json").exists():
            continue
        if json.loads((d / "status.json").read_text(encoding="utf-8"))["state"] == "done":
            yield d


def _finite(xs):
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


def length_conditioned(comps, max_len: int) -> dict:
    """Split a run's completions at the median length (rule fixed here, same for every fork) and
    report how the policy-gradient weight is allocated between short and long completions."""
    used = [c for c in comps if not c["masked_from_loss"]]
    if not used:
        return {}
    med = float(np.median([c["response_tokens"] for c in used]))
    short = [c for c in used if c["response_tokens"] <= med]
    long_ = [c for c in used if c["response_tokens"] > med]
    total = sum(c["sequence_weight"] for c in used) or float("nan")
    def mean(xs, key):
        return float(np.mean([x[key] for x in xs])) if xs else float("nan")
    return {
        "median_length_split": med,
        "n_short": len(short), "n_long": len(long_),
        "share_of_sequence_weight_long": sum(c["sequence_weight"] for c in long_) / total,
        "per_token_weight_short_mean": mean(short, "per_token_weight"),
        "per_token_weight_long_mean": mean(long_, "per_token_weight"),
        "per_token_weight_long_over_short": mean(long_, "per_token_weight") / mean(short, "per_token_weight") if short and long_ else float("nan"),
        "mean_advantage_short": mean(short, "advantage"),
        "mean_advantage_long": mean(long_, "advantage"),
        "corr_length_advantage": float(np.corrcoef([c["response_tokens"] for c in used], [c["advantage"] for c in used])[0, 1]),
        "completions_masked_truncated": sum(c["masked_from_loss"] for c in comps),
    }


def build(cfg):
    seed = int(cfg["seed"])
    max_len = int(cfg["max_completion_length"])
    rows, length_rows = {}, []
    for d in _done(cfg["results_dir"], "task3_grpo_train_"):
        cond = d.name[len("task3_grpo_train_"):-len(f"_seed{seed}")]
        man = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
        log = read_jsonl(d / "train_log.jsonl")
        comps = read_jsonl(d / "completions.jsonl")
        rows[cond] = {
            "condition": cond,
            "loss_type": man["loss_type"],
            "updates": len(log),
            "generations": len(comps),
            "generated_tokens": log[-1]["generated_tokens_cumulative"],
            "train_reward_mean": float(np.mean([r["reward_mean"] for r in log])),
            "train_kl_ref_mean": float(np.mean([r["kl_ref"] for r in log])),
            "train_kl_ref_last": log[-1]["kl_ref"],
            "train_mean_group_reward_std": float(np.mean([r["reward_group_std"] for r in log])),
            "train_uninformative_group_fraction": float(np.mean([r["uninformative_group"] for r in log])),
            "train_policy_loss_mean": float(np.mean([r["policy_loss"] for r in log])),
            "train_grad_norm_mean": float(np.mean(_finite(r["grad_norm"] for r in log))),
            "train_entropy_mean": float(np.mean([r["entropy"] for r in log])),
            "train_response_tokens_mean": float(np.mean([r["response_tokens_mean"] for r in log])),
            "train_truncated_completions": sum(r["truncated_completions"] for r in log),
            "skipped_overflow_steps": sum(r["skipped_overflow_steps"] for r in log),
            "wall_clock_s": man.get("wall_clock_seconds"),
            "peak_vram_gib": man.get("peak_vram_allocated_gib"),
        }
        length_rows.append({"condition": cond, "loss_type": man["loss_type"], **length_conditioned(comps, max_len)})
    for d in _done(cfg["results_dir"], "task3_grpo_eval_"):
        cond = d.name[len("task3_grpo_eval_"):-len(f"_seed{seed}")]
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        row = rows.setdefault(cond, {"condition": cond})
        row.update({
            "heldout_n": m["n_prompts"],
            "heldout_reward_mean": m["reward_raw_mean"],
            "heldout_reward_std": m["reward_raw_std"],
            "heldout_kl_token_mean": m["kl_token_mean"],
            "heldout_kl_sequence_mean": m["kl_sequence_mean"],
            "heldout_entropy_token_mean": m["entropy_token_mean"],
            "heldout_len_mean": m["length_tokens"]["mean"],
            "heldout_len_std": m["length_tokens"]["std"],
            "heldout_len_iqr": m["length_tokens"]["iqr"],
            "heldout_truncation_rate": m["truncation_rate"],
        })
    summary = pd.DataFrame(list(rows.values()))

    traj = None
    std_dir = repo_path(cfg["results_dir"]) / f"task3_grpo_train_standard_seed{seed}"
    if (std_dir / "train_log.jsonl").exists():
        traj = pd.DataFrame(read_jsonl(std_dir / "train_log.jsonl"))

    ksize = None
    kdir = repo_path(cfg["results_dir"]) / f"task3_grpo_group_size_seed{seed}"
    if (kdir / "metrics.json").exists():
        m = json.loads((kdir / "metrics.json").read_text(encoding="utf-8"))
        ksize = pd.DataFrame(
            [{"K": int(k), "bin": "all", **v["overall"]} for k, v in m["by_K"].items()]
            + [{"K": int(k), "bin": b, **s} for k, v in m["by_K"].items() for b, s in v["by_difficulty"].items()]
        )
    return summary, traj, ksize, pd.DataFrame(length_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["results_dir"])
    summary, traj, ksize, lengths = build(cfg)
    summary.to_csv(out / "summary.csv", index=False)
    save_json(out / "summary.json", summary.to_dict(orient="records"))
    if traj is not None:
        traj.to_csv(out / "standard_trajectory.csv", index=False)
    if ksize is not None:
        ksize.to_csv(out / "group_size.csv", index=False)
    lengths.to_csv(out / "normalization_length.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        for name, df in (("summary", summary), ("group size", ksize), ("length-conditioned", lengths)):
            if df is not None:
                print(f"\n== {name}\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
