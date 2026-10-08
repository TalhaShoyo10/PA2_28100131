"""Task 5: combine in-domain (GSM8K), controlled-diagnostic, and transfer (SVAMP) results.

Writes to results/task5_feedback/:
  feedback_comparison.csv      one row per policy: in-domain vs transfer accuracy, format, length,
                               AI pairwise result vs SFT, and the in-domain -> transfer drop
  diagnostics_table.csv        better / tie / wrong rates per pair type and mechanism
  qualitative_candidates.csv   diagnostic pairs where verifier and judge disagree, for the required
                               reasoning-only / persuasive-filler / wrong-final-gold-distractor examples,
                               in fixed problem_id order (pick by a rule stated before reading them)
Binary verifier rewards and judge win rates are reported side by side but are not calibrated utilities;
do not compare their numerical scales (manual p.13).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["results_dir"]) / "task5_feedback"
    summaries = {d: json.loads((out / f"summary_{d}.json").read_text(encoding="utf-8")) for d in ("gsm", "transfer")
                 if (out / f"summary_{d}.json").exists()}
    if set(summaries) != {"gsm", "transfer"}:
        raise SystemExit(f"Need both summary_gsm.json and summary_transfer.json; found {sorted(summaries)}")

    rows = []
    for name in ("sft", "rlvr", "rlaif"):
        g, t = summaries["gsm"]["policies"][name], summaries["transfer"]["policies"][name]
        row = {"policy": name,
               "gsm_exact_accuracy": g["exact_accuracy"], "svamp_exact_accuracy": t["exact_accuracy"],
               "accuracy_drop_gsm_to_svamp": g["exact_accuracy"] - t["exact_accuracy"],
               "gsm_format_compliance": g["format_compliance"], "svamp_format_compliance": t["format_compliance"],
               "gsm_len_mean": g["response_tokens"]["mean"], "gsm_len_std": g["response_tokens"]["std"],
               "svamp_len_mean": t["response_tokens"]["mean"], "svamp_len_std": t["response_tokens"]["std"],
               "gsm_truncation_rate": g["truncation_rate"], "svamp_truncation_rate": t["truncation_rate"]}
        if name != "sft":
            pg, pt = summaries["gsm"]["pairwise"][f"{name}_vs_sft"], summaries["transfer"]["pairwise"][f"{name}_vs_sft"]
            row.update({"gsm_win_rate_vs_sft": pg["win_rate"], "gsm_ties_vs_sft": pg["ties"],
                        "svamp_win_rate_vs_sft": pt["win_rate"], "svamp_ties_vs_sft": pt["ties"],
                        "win_rate_drop_gsm_to_svamp": pg["win_rate"] - pt["win_rate"],
                        "gsm_verifier_judge_agreement": pg["verifier_judge_agreement"],
                        "gsm_judge_agrees_when_verifier_decisive": pg["judge_agrees_when_verifier_decisive"],
                        "svamp_verifier_judge_agreement": pt["verifier_judge_agreement"]})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "feedback_comparison.csv", index=False)

    diag_path = out / "diagnostics_summary.json"
    if diag_path.exists():
        d = json.loads(diag_path.read_text(encoding="utf-8"))
        drows = [{"pair_type": pt, "mechanism": m, **v[m]} for pt, v in d["by_pair_type"].items() for m in ("verifier", "judge")]
        pd.DataFrame(drows).to_csv(out / "diagnostics_table.csv", index=False)
        diag_dir = next(p for p in sorted(out.glob("task5_feedback_diagnostics_seed*")) if "__attempt" not in p.name)
        pairs = pd.DataFrame(read_jsonl(diag_dir / "pair_scores.jsonl"))
        responses = {(str(r["problem_id"]), r["variant_type"]): r["response"] for r in read_jsonl(cfg["paths"]["task5_diagnostics"])}
        cand = pairs[(pairs["verifier"] != pairs["judge"]) & pairs["pair_type"].isin(["reasoning", "filler", "distractor"])].copy()
        cand["clean_response"] = [responses[(p, "clean_correct")] for p in cand["problem_id"]]
        cand["perturbed_response"] = [responses[(p, v)] for p, v in zip(cand["problem_id"], cand["perturbed_variant"])]
        cand.sort_values(["pair_type", "problem_id"]).to_csv(out / "qualitative_candidates.csv", index=False)
        save_json(out / "feedback_summary.json", {"comparison": rows, "diagnostics": d})
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
