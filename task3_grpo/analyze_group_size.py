"""Task 3 Step 2: group size at equal total generations (no training; CPU only).

Regrouping rule (decided before looking at any K result): the supplied cache has 8 completions per
prompt. For K in {2, 4, 8}, each prompt's completions, sorted by generation_index, are split into
8/K consecutive disjoint groups. Every K therefore uses exactly the same 192 generations and the same
prompts; only how they are grouped changes.

Per K: informative-group rate (within-group reward std > 1e-6, the release helper's eps), mean
within-group reward std, variance of the group-relative signal (normalized advantages A, and the
unnormalized centred reward r - mean_group), fraction of completions with |A| > 0.

Difficulty bins (fixed rule, defined once): a prompt's difficulty is the mean reward of all its 8
cached completions; prompts are split into tertiles of that mean (low-reward = "hard",
middle, high-reward = "easy"). The same quantities are reported per bin.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, write_jsonl
from common.run_record import RunRecord, experiment_id
from task3_grpo.grpo import group_relative_advantages

ZERO_STD_TOL = 1e-6


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed (see module docstring)."""
    groups = []
    for pid, comps in by_prompt.items():
        comps = comps[:8]
        if 8 % k:
            raise ValueError(f"K={k} does not divide 8")
        for g in range(8 // k):
            groups.append({"prompt": pid, "group": g, "completions": comps[g * k:(g + 1) * k]})
    return groups


def difficulty_bins(by_prompt):
    means = {pid: float(np.mean([c["reward"] for c in comps[:8]])) for pid, comps in by_prompt.items()}
    lo, hi = np.percentile(list(means.values()), [100 / 3, 200 / 3])
    def label(m):
        return "hard" if m < lo else ("easy" if m >= hi else "medium")
    return {pid: label(m) for pid, m in means.items()}, means, {"tertile_cut_low": float(lo), "tertile_cut_high": float(hi)}


def group_stats(groups):
    stds, adv_all, centred_all, rows = [], [], [], []
    for g in groups:
        r = torch.tensor([float(c["reward"]) for c in g["completions"]])
        std = float(r.std(unbiased=False))
        adv = group_relative_advantages(r, torch.zeros(len(r), dtype=torch.long))
        stds.append(std)
        adv_all.extend(adv.tolist())
        centred_all.extend((r - r.mean()).tolist())
        rows.append({"prompt": g["prompt"], "group": g["group"], "std": std, "informative": std > ZERO_STD_TOL})
    adv_all, centred_all = np.array(adv_all), np.array(centred_all)
    return {
        "n_groups": len(groups),
        "n_generations": int(adv_all.size),
        "informative_group_rate": float(np.mean([s > ZERO_STD_TOL for s in stds])),
        "mean_within_group_reward_std": float(np.mean(stds)),
        "median_within_group_reward_std": float(np.median(stds)),
        "variance_normalized_advantage": float(adv_all.var()),
        "variance_centred_reward": float(centred_all.var()),
        "mean_abs_normalized_advantage": float(np.abs(adv_all).mean()),
        "fraction_nonzero_advantage": float((np.abs(adv_all) > 0).mean()),
    }, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    exp_id = experiment_id("task3_grpo", "group_size", int(cfg["seed"]))
    record = RunRecord(cfg["results_dir"], exp_id, cfg, extra={
        "task": "task3_grpo", "condition": "group_size", "cache": cfg["group_cache"],
        "group_sizes": cfg["group_sizes"], "regrouping": "consecutive disjoint groups by generation_index",
        "difficulty_rule": "tertiles of per-prompt mean reward over the 8 cached completions",
    })
    if record.is_done():
        print(f"[skip] {exp_id} already done")
        return
    with record:
        by_prompt = load_k8_cache(cfg["group_cache"])
        bins, prompt_means, cuts = difficulty_bins(by_prompt)
        out, group_rows = {}, []
        for k in [int(x) for x in cfg["group_sizes"]]:
            groups = regroup_equal_generation_budget(by_prompt, k)
            overall, rows = group_stats(groups)
            per_bin = {b: group_stats([g for g in groups if bins[g["prompt"]] == b])[0] for b in ("hard", "medium", "easy")}
            out[str(k)] = {"overall": overall, "by_difficulty": per_bin}
            group_rows.extend({"K": k, "difficulty": bins[r["prompt"]], **r} for r in rows)
            print(f"K={k}: {overall}", flush=True)
        write_jsonl(record.dir / "groups.jsonl", group_rows)
        n_clipped = sum(c.get("clipped_at_max", False) for comps in by_prompt.values() for c in comps)
        record.finish({
            "by_K": out,
            "difficulty_cuts": cuts,
            "prompt_difficulty": {pid: {"mean_reward": prompt_means[pid], "bin": bins[pid]} for pid in by_prompt},
            "n_prompts": len(by_prompt),
            "cached_completions_clipped_at_cap": n_clipped,
            "cache_generation_cap": cfg.get("cache_generation_cap"),
        }, artifacts={"groups": "groups.jsonl"})


if __name__ == "__main__":
    main()
