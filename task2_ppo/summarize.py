"""Collect finished Task 2 runs into machine-readable tables.

results/task2_ppo/summary.csv              one row per condition: training diagnostics + held-out eval
results/task2_ppo/standard_trajectory.csv  per-update trajectory of the standard continuation
results/task2_ppo/cached_clipping.csv      cached-batch clipping geometry per epsilon
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


def train_row(d, seed):
    cond = d.name[len("task2_ppo_train_"):-len(f"_seed{seed}")]
    man = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    log = read_jsonl(d / "train_log.jsonl")
    approx_kl = _finite(r["approx_kl_old_new_last_epoch"] for r in log)
    return cond, {
        "condition": cond,
        "clip_epsilon": man["clip_epsilon"],
        "kl_beta": man["kl_beta"],
        "updates": len(log),
        "generated_tokens": log[-1]["generated_tokens_cumulative"],
        "train_reward_mean": float(np.mean([r["reward_effective"] for r in log])),
        "train_reward_std": float(np.std([r["reward_effective"] for r in log])),
        "train_kl_ref_mean": float(np.mean([r["kl_ref"] for r in log])),
        "train_kl_ref_last": log[-1]["kl_ref"],
        "train_entropy_mean": float(np.mean([r["entropy"] for r in log])),
        "train_policy_loss_mean": float(np.mean([r["policy_loss"] for r in log])),
        "train_value_loss_mean": float(np.mean([r["value_loss"] for r in log])),
        "train_value_explained_variance_mean": float(np.mean(_finite(r["value_explained_variance"] for r in log))) if _finite(r["value_explained_variance"] for r in log) else float("nan"),
        "train_clip_fraction_mean": float(np.mean([r["clip_fraction"] for r in log])),
        "train_clip_fraction_last_epoch_mean": float(np.mean([r["clip_fraction_last_epoch"] for r in log])),
        # stability statistics (pick one and define it in the report)
        "stability_mean_approx_kl_old_new": float(np.mean(approx_kl)) if approx_kl else float("nan"),
        "stability_max_ratio": max(max(e["ratio_max"] for e in r["epochs"]) for r in log),
        "stability_max_policy_grad_norm": max(_finite(r["policy_grad_norm"] for r in log)),
        "train_response_tokens_mean": float(np.mean([r["response_tokens"] for r in log])),
        "train_truncation_rate": float(np.mean([r["truncated"] for r in log])),
        "skipped_overflow_steps": sum(r["skipped_overflow_steps"] for r in log),
        "wall_clock_s": man.get("wall_clock_seconds"),
        "peak_vram_gib": man.get("peak_vram_allocated_gib"),
    }


def build(cfg):
    seed = int(cfg["seed"])
    rows = {}
    for d in _done(cfg["results_dir"], "task2_ppo_train_"):
        cond, row = train_row(d, seed)
        rows[cond] = row
    for d in _done(cfg["results_dir"], "task2_ppo_eval_"):
        cond = d.name[len("task2_ppo_eval_"):-len(f"_seed{seed}")]
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        row = rows.setdefault(cond, {"condition": cond})
        row.update({
            "heldout_n": m["n_prompts"],
            "heldout_reward_raw_mean": m["reward_raw_mean"],
            "heldout_reward_raw_std": m["reward_raw_std"],
            "heldout_reward_effective_mean": m["reward_effective_mean"],
            "heldout_kl_token_mean": m["kl_token_mean"],
            "heldout_kl_sequence_mean": m["kl_sequence_mean"],
            "heldout_entropy_token_mean": m["entropy_token_mean"],
            "heldout_len_mean": m["length_tokens"]["mean"],
            "heldout_len_std": m["length_tokens"]["std"],
            "heldout_len_iqr": m["length_tokens"]["iqr"],
            "heldout_truncation_rate": m["truncation_rate"],
            "heldout_missing_eos_rate": m["missing_eos_rate"],
        })
    summary = pd.DataFrame(list(rows.values()))

    traj = None
    std_dir = repo_path(cfg["results_dir"]) / f"task2_ppo_train_standard_seed{seed}"
    if (std_dir / "train_log.jsonl").exists():
        traj = pd.DataFrame([{k: v for k, v in r.items() if k != "epochs"} for r in read_jsonl(std_dir / "train_log.jsonl")])

    cached = None
    cdir = repo_path(cfg["results_dir"]) / f"task2_ppo_cached_clipping_valid_seed{seed}"
    if (cdir / "metrics.json").exists():
        m = json.loads((cdir / "metrics.json").read_text(encoding="utf-8"))
        cached = pd.DataFrame([
            {"epsilon": float(eps), "phase": phase, **vals}
            for eps, by in m["by_epsilon"].items()
            for phase, vals in by.items() if isinstance(vals, dict)
        ])
    return summary, traj, cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["results_dir"])
    summary, traj, cached = build(cfg)
    summary.to_csv(out / "summary.csv", index=False)
    save_json(out / "summary.json", summary.to_dict(orient="records"))
    if traj is not None:
        traj.to_csv(out / "standard_trajectory.csv", index=False)
    if cached is not None:
        cached.to_csv(out / "cached_clipping.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(summary.to_string(index=False))
        if cached is not None:
            print(cached.to_string(index=False))


if __name__ == "__main__":
    main()
