"""Task 5 diagnostic: why does the pairwise judge return TIE for ~97% of GSM8K comparisons?

Read-only inspection; the fixed judge, verifier, and all evaluation results are unchanged.
  1. Identical-response rate: how often an RLVR/RLAIF response is word-for-word the SFT response
     (greedy decoding + small policy drift), where TIE is the correct verdict.
  2. Raw judge output: for a fixed sample of comparisons, the exact text the judge generates under the
     release's prompt, orientation rule, and 4-token budget, and whether the release parser found an
     explicit A / B / TIE or fell back to its default TIE.
     Sample (fixed rule, in problem order): every verifier-decisive pair (exactly one response correct)
     plus the first 30 pairs whose responses differ.
  3. Final-answer formats used by each policy (which explains format compliance).
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import torch

from common.data import load_yaml, read_jsonl, write_jsonl
from common.logging_utils import save_json
from common.run_record import RunRecord, experiment_id
from task5_feedback.evaluate_math import problem_text, task5_dir
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge

N_DIFFERENT = 30
FORMATS = {
    "hash_final (#### n)": re.compile(r"####\s*[-+]?\d"),
    "boxed (\\boxed{...})": re.compile(r"\\boxed\{"),
    "'answer is'": re.compile(r"answer is", re.I),
    "'Final Answer'": re.compile(r"final answer", re.I),
}


@torch.no_grad()
def raw_compare(judge: PairwiseAIJudge, problem: str, a: str, b: str) -> dict:
    """Same prompt, orientation rule, and decoding as PairwiseAIJudge.compare, but returns the raw text.
    The judge cache is neither read nor written."""
    key = judge._key(problem, a, b)
    swap = int(key[:8], 16) % 2 == 1
    aa, bb = (b, a) if swap else (a, b)
    text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
    ids = judge.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors="pt",
                                              add_generation_prompt=True).to(next(judge.model.parameters()).device)
    out = judge.model.generate(ids, max_new_tokens=4, do_sample=False,
                               pad_token_id=judge.tokenizer.eos_token_id, eos_token_id=judge.tokenizer.eos_token_id)
    decoded = judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"\b(A|B|TIE)\b", decoded.strip().upper())
    return {"raw_output": decoded, "swapped": swap, "explicit_parse": m.group(1) if m else None,
            "release_result": ({"A": "B", "B": "A", "TIE": "TIE"}[m.group(1)] if (m and swap) else (m.group(1) if m else "TIE"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = task5_dir(cfg)
    exp_id = experiment_id("task5_feedback", f"judge_inspection_{args.dataset}", int(cfg["seed"]))
    record = RunRecord(out, exp_id, cfg, extra={"task": "task5_feedback", "condition": "judge_inspection",
                                                "dataset": args.dataset, "sample_rule": f"all verifier-decisive pairs + first {N_DIFFERENT} differing pairs"})
    if record.is_done():
        print(f"[skip] {exp_id} already done")
        return
    with record:
        from task5_feedback.evaluate_math import dataset_path
        rows = {str(r.get("prompt_id", r.get("source_index"))): r for r in read_jsonl(dataset_path(cfg, args.dataset))}
        gens = {p: {r["problem_id"]: r for r in read_jsonl(out / f"generated_{args.dataset}_{p}.jsonl")} for p in ("sft", "rlvr", "rlaif")}

        identical = {p: sum(gens[p][k]["response"] == gens["sft"][k]["response"] for k in gens["sft"]) / len(gens["sft"]) for p in ("rlvr", "rlaif")}
        formats = {p: {name: sum(bool(rx.search(g["response"])) for g in gens[p].values()) / len(gens[p]) for name, rx in FORMATS.items()}
                   for p in gens}
        print("identical-to-SFT rate:", identical)
        print("answer formats:", formats)

        judge = PairwiseAIJudge(cfg, out / "judge_inspection_unused_cache.json")
        recs = []
        for p in ("rlvr", "rlaif"):
            differing = [k for k in gens["sft"] if gens[p][k]["response"] != gens["sft"][k]["response"]]
            decisive = [k for k in differing if gens[p][k]["exact_correct"] != gens["sft"][k]["exact_correct"]]
            sample = list(dict.fromkeys(decisive + differing[:N_DIFFERENT]))
            for k in sample:
                r = raw_compare(judge, problem_text(rows[k]), gens[p][k]["response"], gens["sft"][k]["response"])
                recs.append({"policy": p, "problem_id": k, "verifier_decisive": k in decisive,
                             "policy_correct": gens[p][k]["exact_correct"], "sft_correct": gens["sft"][k]["exact_correct"], **r})
            record.heartbeat(policy=p, sampled=len(sample))
        write_jsonl(record.dir / "raw_judge_outputs.jsonl", recs)
        n = len(recs)
        summary = {
            "identical_to_sft_rate": identical,
            "answer_format_rates": formats,
            "n_inspected": n,
            "fraction_explicit_A_or_B": sum(r["explicit_parse"] in ("A", "B") for r in recs) / max(n, 1),
            "fraction_explicit_TIE": sum(r["explicit_parse"] == "TIE" for r in recs) / max(n, 1),
            "fraction_unparsed_default_TIE": sum(r["explicit_parse"] is None for r in recs) / max(n, 1),
            "most_common_raw_outputs": Counter(r["raw_output"].strip() for r in recs).most_common(10),
        }
        save_json(record.dir / "summary.json", summary)
        record.finish(summary, artifacts={"raw_judge_outputs": "raw_judge_outputs.jsonl"})
        for k, v in summary.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
