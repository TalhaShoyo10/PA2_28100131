"""Task 5 Steps 1 and 3: SFT vs RLVR vs RLAIF on the fixed GSM8K subset (in-domain) and the fixed SVAMP
transfer set (out-of-domain). The RLVR/RLAIF adapters are supplied frozen policies; nothing is trained.

Per dataset:
  1. generate one response per problem for each policy with identical greedy decoding
     (math_max_new_tokens) -> exact final-answer accuracy, format compliance, response length
  2. the fixed pairwise AI judge compares RLVR vs SFT and RLAIF vs SFT on every problem
     (win = 1, tie = 0.5, loss = 0; ties counted separately)
  3. verifier-judge agreement: on each comparison the verifier "prefers" whichever response has the
     higher exact reward (tie if equal); agreement is reported overall and on the pairs where the
     verifier is decisive (exactly one of the two responses is correct)
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json
from common.metrics import length_stats
from common.models import clear_gpu, load_policy, load_tokenizer
from common.run_record import RunRecord, experiment_id
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

GEN_BATCH_SIZE = 16
MAX_PROMPT_TOKENS = 512  # GSM8K/SVAMP prompts are short; generation refuses any prompt that would be truncated
POLICIES = ["sft", "rlvr", "rlaif"]


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def task5_dir(cfg):
    return repo_path(cfg["results_dir"]) / "task5_feedback"


def problem_text(row) -> str:
    return str(row.get("question") or prompt_messages(row)[-1]["content"])


def generate_policy(cfg, rows, tok, dataset: str, name: str):
    exp_id = experiment_id("task5_feedback", f"generate_{dataset}_{name}", int(cfg["seed"]))
    out_file = task5_dir(cfg) / f"generated_{dataset}_{name}.jsonl"
    record = RunRecord(task5_dir(cfg), exp_id, cfg, extra={
        "task": "task5_feedback", "dataset": dataset, "condition": name, "model": cfg["base_model"],
        "checkpoint": policy_specs(cfg)[name] or "SFT (no adapter)",
        "decoding": {"do_sample": False, "max_new_tokens": int(cfg["math_max_new_tokens"]), "batch_size": GEN_BATCH_SIZE},
    })
    if record.is_done():
        print(f"[skip] {exp_id} already done")
        return read_jsonl(out_file)
    with record:
        too_long = [r.get("prompt_id", r.get("source_index")) for r in rows
                    if len(tok.apply_chat_template(prompt_messages(r), tokenize=True, add_generation_prompt=True)) > MAX_PROMPT_TOKENS]
        if too_long:
            raise SystemExit(f"{len(too_long)} prompts exceed {MAX_PROMPT_TOKENS} tokens: {too_long[:5]}")
        policy = load_frozen_policy(cfg, name)
        records = []
        for start in range(0, len(rows), GEN_BATCH_SIZE):
            chunk = rows[start:start + GEN_BATCH_SIZE]
            gen = batch_generate(policy, tok, [prompt_messages(r) for r in chunk], max_prompt_length=MAX_PROMPT_TOKENS,
                                 max_new_tokens=int(cfg["math_max_new_tokens"]), temperature=0.0, top_p=1.0, do_sample=False)
            for j, r in enumerate(chunk):
                resp = gen["responses"][j]
                pred = extract_designated_final(resp)
                records.append({
                    "problem_id": str(r.get("prompt_id", r.get("source_index"))),
                    "source_index": r.get("source_index"),
                    "policy": name,
                    "gold_final": str(r["gold_final"]),
                    "predicted_final": pred,
                    "format_ok": pred is not None,
                    "exact_correct": exact_reward(resp, str(r["gold_final"])),
                    "response": resp,
                    "response_tokens": int(gen["response_lengths"][j]),
                    "truncated": bool(gen["truncated"][j]),
                })
            record.heartbeat(update=min(start + GEN_BATCH_SIZE, len(rows)), of=len(rows))
            print(f"  {name}: generated {min(start + GEN_BATCH_SIZE, len(rows))}/{len(rows)}", flush=True)
        write_jsonl(out_file, records)
        write_jsonl(record.dir / "generated.jsonl", records)
        del policy
        clear_gpu()
        record.finish(policy_metrics(records), artifacts={"generated": out_file.name})
    return records


def policy_metrics(records) -> dict:
    return {
        "n": len(records),
        "exact_accuracy": float(np.mean([r["exact_correct"] for r in records])),
        "format_compliance": float(np.mean([r["format_ok"] for r in records])),
        "response_tokens": length_stats([r["response_tokens"] for r in records]),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
    }


def pairwise_vs_sft(cfg, rows, gens: dict, dataset: str):
    exp_id = experiment_id("task5_feedback", f"pairwise_{dataset}", int(cfg["seed"]))
    out_file = task5_dir(cfg) / f"pairwise_{dataset}.jsonl"
    record = RunRecord(task5_dir(cfg), exp_id, cfg, extra={
        "task": "task5_feedback", "dataset": dataset, "judge_model": cfg["ai_judge_model"],
        "comparisons": ["rlvr_vs_sft", "rlaif_vs_sft"], "orientation": "release PairwiseAIJudge (hash-based A/B swap)",
    })
    if record.is_done():
        print(f"[skip] {exp_id} already done")
        return read_jsonl(out_file)
    with record:
        judge = PairwiseAIJudge(cfg, task5_dir(cfg) / "pairwise_judge_cache.json")
        sft = {r["problem_id"]: r for r in gens["sft"]}
        results = []
        for name in ("rlvr", "rlaif"):
            for i, r in enumerate(gens[name], start=1):
                base = sft[r["problem_id"]]
                row = next(x for x in rows if str(x.get("prompt_id", x.get("source_index"))) == r["problem_id"])
                pref = judge.compare(problem_text(row), r["response"], base["response"])  # A = policy, B = SFT
                score = {"A": 1.0, "TIE": 0.5, "B": 0.0}[pref]
                v = r["exact_correct"] - base["exact_correct"]
                verifier_pref = "A" if v > 0 else ("B" if v < 0 else "TIE")
                results.append({"problem_id": r["problem_id"], "policy": name, "judge_pref": pref, "judge_score": score,
                                "verifier_pref": verifier_pref, "policy_correct": r["exact_correct"], "sft_correct": base["exact_correct"]})
                if i % 25 == 0:
                    record.heartbeat(update=i, of=len(gens[name]), policy=name)
        write_jsonl(out_file, results)
        record.finish(pairwise_metrics(results), artifacts={"pairwise": out_file.name})
    return results


def pairwise_metrics(results) -> dict:
    out = {}
    for name in ("rlvr", "rlaif"):
        rs = [r for r in results if r["policy"] == name]
        decisive = [r for r in rs if r["verifier_pref"] != "TIE"]
        out[f"{name}_vs_sft"] = {
            "n": len(rs),
            "win_rate": float(np.mean([r["judge_score"] for r in rs])),
            "wins": sum(r["judge_pref"] == "A" for r in rs),
            "ties": sum(r["judge_pref"] == "TIE" for r in rs),
            "losses": sum(r["judge_pref"] == "B" for r in rs),
            "verifier_judge_agreement": float(np.mean([r["judge_pref"] == r["verifier_pref"] for r in rs])),
            "n_verifier_decisive": len(decisive),
            "judge_agrees_when_verifier_decisive": float(np.mean([r["judge_pref"] == r["verifier_pref"] for r in decisive])) if decisive else float("nan"),
            "judge_tie_when_verifier_decisive": float(np.mean([r["judge_pref"] == "TIE" for r in decisive])) if decisive else float("nan"),
        }
    return out


def evaluate_dataset(config_path: str, dataset: str, stage: str = "all"):
    cfg, rows, tok = load_math_evaluation(config_path, dataset)
    print("Rows:", len(rows), "| policies:", POLICIES)
    gens = {}
    for name in POLICIES:
        gens[name] = generate_policy(cfg, rows, tok, dataset, name)
    if stage == "generate":
        return
    pairs = pairwise_vs_sft(cfg, rows, gens, dataset)
    summary = {"dataset": dataset, "n_problems": len(rows),
               "policies": {n: policy_metrics(g) for n, g in gens.items()},
               "pairwise": pairwise_metrics(pairs)}
    save_json(task5_dir(cfg) / f"summary_{dataset}.json", summary)
    for n, m in summary["policies"].items():
        print(f"{n:6s} acc {m['exact_accuracy']:.3f} format {m['format_compliance']:.3f} len {m['response_tokens']['mean']:.0f}")
    for k, m in summary["pairwise"].items():
        print(f"{k}: win {m['win_rate']:.3f} (W/T/L {m['wins']}/{m['ties']}/{m['losses']}) agreement {m['verifier_judge_agreement']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--stage", choices=["generate", "all"], default="all", help="generate only, or generate + judge + summary")
    args = ap.parse_args()
    evaluate_dataset(args.config, args.dataset, args.stage)


if __name__ == "__main__":
    main()
