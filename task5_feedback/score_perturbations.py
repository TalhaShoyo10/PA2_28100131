"""Task 5 Step 2: reward sensitivity on the supplied 100-response controlled diagnostic set
(20 GSM8K problems x 5 variants). Both feedback mechanisms score the same fixed pairs.

Controlled pairs (fixed before scoring): every perturbed variant is paired with the same problem's
clean_correct response, which is the diagnostically better response in each pair:

  reasoning  : clean_correct vs corrupt_reasoning_correct_final   (final held correct, reasoning degraded)
  outcome    : clean_correct vs good_reasoning_wrong_final        (reasoning ~fixed, designated final changed)
  filler     : clean_correct vs persuasive_filler_correct         (same answer + irrelevant persuasive text)
  distractor : clean_correct vs gold_distractor_wrong_final       (gold number mentioned, wrong designated final)

For each mechanism and pair type: better-response rate (prefers clean), tie rate, wrong-preference rate.
  S_reason  = better-response rate on the reasoning pairs
  S_outcome = better-response rate on the outcome pairs (primary; manual: reasoning held ~fixed while the
              final answer changes); also reported pooled with the distractor pairs (both outcome-changing)
Verifier: exact final-answer reward of each response (higher wins, equal = tie). A verifier tie on the
reasoning and filler pairs is the expected invariance, not an error (manual p.13).
AI judge: the fixed PairwiseAIJudge (same orientation balancing and cache as Step 1).
Per-variant table: verifier reward vs the supplied expected_exact_reward (a check of the verifier).
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from common.run_record import RunRecord, experiment_id
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
PAIR_TYPES = {
    "reasoning": "corrupt_reasoning_correct_final",
    "outcome": "good_reasoning_wrong_final",
    "filler": "persuasive_filler_correct",
    "distractor": "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def rates(prefs) -> dict:
    """prefs: list of 'better' / 'tie' / 'wrong' outcomes for one mechanism and pair type."""
    n = len(prefs)
    return {"n": n,
            "better_rate": sum(p == "better" for p in prefs) / n,
            "tie_rate": sum(p == "tie" for p in prefs) / n,
            "wrong_rate": sum(p == "wrong" for p in prefs) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    out_dir = repo_path(cfg["results_dir"]) / "task5_feedback"
    exp_id = experiment_id("task5_feedback", "diagnostics", int(cfg["seed"]))
    record = RunRecord(out_dir, exp_id, cfg, extra={
        "task": "task5_feedback", "condition": "controlled_diagnostics", "judge_model": cfg["ai_judge_model"],
        "pairs": {k: f"clean_correct vs {v}" for k, v in PAIR_TYPES.items()},
    })
    if record.is_done():
        print(f"[skip] {exp_id} already done")
        return
    with record:
        # Per-variant verifier check
        variant_rows = []
        for pid, vs in sorted(groups.items()):
            for vt, r in vs.items():
                rew = exact_reward(r["response"], str(r["gold_final"]))
                variant_rows.append({"problem_id": pid, "variant_type": vt, "verifier_reward": rew,
                                     "expected_exact_reward": float(r["expected_exact_reward"]),
                                     "matches_expected": rew == float(r["expected_exact_reward"])})
        judge = PairwiseAIJudge(cfg, out_dir / "pairwise_judge_cache.json")
        pair_rows = []
        for pid, vs in sorted(groups.items()):
            clean = vs["clean_correct"]
            for ptype, vt in PAIR_TYPES.items():
                other = vs[vt]
                rv_c = exact_reward(clean["response"], str(clean["gold_final"]))
                rv_o = exact_reward(other["response"], str(other["gold_final"]))
                verifier = "better" if rv_c > rv_o else ("wrong" if rv_c < rv_o else "tie")
                pref = judge.compare(clean["question"], clean["response"], other["response"])  # A = clean
                judge_out = {"A": "better", "B": "wrong", "TIE": "tie"}[pref]
                pair_rows.append({"problem_id": pid, "pair_type": ptype, "perturbed_variant": vt,
                                  "verifier_reward_clean": rv_c, "verifier_reward_perturbed": rv_o,
                                  "verifier": verifier, "judge_pref_raw": pref, "judge": judge_out})
            record.heartbeat(problem=pid)
        write_jsonl(record.dir / "variant_verifier_check.jsonl", variant_rows)
        write_jsonl(record.dir / "pair_scores.jsonl", pair_rows)

        by_type = {}
        for ptype in PAIR_TYPES:
            sub = [r for r in pair_rows if r["pair_type"] == ptype]
            by_type[ptype] = {"verifier": rates([r["verifier"] for r in sub]), "judge": rates([r["judge"] for r in sub])}
        outcome_pooled = [r for r in pair_rows if r["pair_type"] in ("outcome", "distractor")]
        metrics = {
            "by_pair_type": by_type,
            "S_reason": {m: by_type["reasoning"][m]["better_rate"] for m in ("verifier", "judge")},
            "S_outcome": {m: by_type["outcome"][m]["better_rate"] for m in ("verifier", "judge")},
            "S_outcome_pooled_with_distractor": {m: rates([r[m] for r in outcome_pooled])["better_rate"] for m in ("verifier", "judge")},
            "verifier_matches_expected_reward": float(np.mean([r["matches_expected"] for r in variant_rows])),
            "verifier_reward_by_variant": {vt: float(np.mean([r["verifier_reward"] for r in variant_rows if r["variant_type"] == vt]))
                                           for vt in sorted(EXPECTED_VARIANTS)},
        }
        save_json(out_dir / "diagnostics_summary.json", metrics)
        record.finish(metrics, artifacts={"pair_scores": "pair_scores.jsonl", "variant_check": "variant_verifier_check.jsonl"})
        for ptype, d in by_type.items():
            print(f"{ptype:10s} verifier {d['verifier']}  judge {d['judge']}")
        print("S_reason", metrics["S_reason"], "| S_outcome", metrics["S_outcome"])


if __name__ == "__main__":
    main()
