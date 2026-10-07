"""Validate returned result folders before they are committed (EXECUTION.md §9).

Checks every results/<task>/<experiment_id>/ directory:
  - status.json is "done"; archived __attempt dirs are reported, not validated
  - run_manifest.json has a git commit that exists in this repository
  - metrics.json parses and contains no NaN/inf (outside explicitly allowed keys)
  - Task 1: evaluation counts match the fixed sets after the recorded long-prompt filter,
            training log length matches the planned update budget
  - no file looks like it contains a secret, and no file is unexpectedly large

Usage:  python -m scripts.validate_results [--task task1_dpo]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from common.data import REPO_ROOT, read_jsonl

RESULTS = REPO_ROOT / "results"
MAX_FILE_BYTES = 5 * 2**20
SECRET_PATTERNS = [
    re.compile(p) for p in (
        r"hf_[A-Za-z0-9]{30,}",            # Hugging Face token
        r"ghp_[A-Za-z0-9]{30,}",           # GitHub token
        r"sk-[A-Za-z0-9]{20,}",            # OpenAI-style key
        r"(?i)(api[_-]?key|secret|password|token)\s*[=:]\s*['\"][^'\"]{8,}",
    )
]
# Model-generated example code uses placeholders such as password = 'your_password'.
PLACEHOLDER = re.compile(r"(?i)your[_ -]|<[^>]+>|x{4,}|\*{4,}|example|placeholder|changeme")
# Task 1 fixed-set sizes after the recorded prompt-length filter (raw 300 / 246 / 10 prompts)
T1_EXPECTED = {"heldout": 290, "length_stratified": 237, "word_limit": 50}


def _non_finite_paths(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _non_finite_paths(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _non_finite_paths(v, f"{prefix}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        yield prefix


def _commit_exists(sha: str) -> bool:
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=REPO_ROOT, capture_output=True)
    return r.returncode == 0


def validate_run(d: Path, problems: list[str], info: list[str]) -> None:
    name = d.relative_to(RESULTS)
    status = json.loads((d / "status.json").read_text(encoding="utf-8"))
    if status["state"] != "done":
        problems.append(f"{name}: status is {status['state']}")
        return
    manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    sha = (manifest.get("git") or {}).get("commit")
    if not sha:
        problems.append(f"{name}: no git commit recorded")
    elif not _commit_exists(sha):
        problems.append(f"{name}: recorded commit {sha[:8]} not in this repository")
    if (manifest.get("git") or {}).get("dirty_tracked_files"):
        problems.append(f"{name}: run used modified tracked files (dirty working tree)")
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    bad = list(_non_finite_paths(metrics))
    if bad:
        problems.append(f"{name}: non-finite metrics at {bad}")
    for f in ("config.json", "command.txt"):
        if not (d / f).exists():
            problems.append(f"{name}: missing {f}")

    if d.parent.name == "task1_dpo" and "smoke" not in d.name:
        if d.name.startswith("task1_dpo_eval_"):
            got = {
                "heldout": metrics["heldout_pairs"]["n"],
                "length_stratified": metrics["length_stratified_pairs"]["overall"]["n"],
                "word_limit": metrics["word_limit"]["samples_per_prompt"] * metrics["word_limit"]["n_prompts"],
            }
            if got != T1_EXPECTED:
                problems.append(f"{name}: counts {got} != expected {T1_EXPECTED}")
            n_gen = len(read_jsonl(d / "generations_heldout.jsonl"))
            if n_gen != T1_EXPECTED["heldout"]:
                problems.append(f"{name}: {n_gen} held-out generations")
        if d.name.startswith("task1_dpo_train_"):
            log = read_jsonl(d / "train_log.jsonl")
            planned = manifest["budget"]["optimizer_updates"]
            if len(log) != planned:
                problems.append(f"{name}: {len(log)} logged updates, planned {planned}")
            if abs(log[0]["loss"] - math.log(2)) > 1e-3:
                problems.append(f"{name}: first loss {log[0]['loss']:.4f} is not log 2")
            skipped = [r["update"] for r in log if r["step_skipped_overflow"]]
            if skipped:
                info.append(f"{name}: fp16-overflow skipped updates {skipped} of {planned}")
            if not (REPO_ROOT / manifest["adapter_output"]).exists() and "/content/" not in manifest["adapter_output"]:
                pass  # adapters live on Drive (outputs/), not in the repo; nothing to check locally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="limit to one results/<task> directory")
    args = ap.parse_args()
    roots = [RESULTS / args.task] if args.task else [p for p in RESULTS.iterdir() if p.is_dir() and p.name != "logs"]
    problems, info = [], []
    n_runs = 0
    for root in roots:
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if "__attempt" in d.name:
                info.append(f"{d.relative_to(RESULTS)}: archived attempt (kept as evidence, not validated)")
                continue
            n_runs += 1
            validate_run(d, problems, info)

    for f in RESULTS.rglob("*"):
        if not f.is_file():
            continue
        if f.stat().st_size > MAX_FILE_BYTES:
            problems.append(f"{f.relative_to(RESULTS)}: {f.stat().st_size / 2**20:.1f} MB (too large for Git)")
        if f.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
            problems.append(f"{f.relative_to(RESULTS)}: model weight file inside results/")
        text = f.read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            hits = [m.group(0) for m in pat.finditer(text) if not PLACEHOLDER.search(m.group(0))]
            if hits:
                problems.append(f"{f.relative_to(RESULTS)}: possible secret {hits[0][:40]!r}")

    print(f"Validated {n_runs} runs.")
    for line in info:
        print("INFO   ", line)
    for line in problems:
        print("PROBLEM", line)
    if problems:
        sys.exit(1)
    print("All result folders pass validation.")


if __name__ == "__main__":
    main()
