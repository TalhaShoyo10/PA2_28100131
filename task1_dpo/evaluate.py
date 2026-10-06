"""Common Task 1 evaluation protocol, applied identically to every DPO condition (and SFT).

1. Held-out pairs (dpo_standard_eval), teacher forcing: DPO loss at the run's beta and
   preference accuracy, m = [log pi(y+) - log ref(y+)] - [log pi(y-) - log ref(y-)] > 0.
2. Length-stratified pairs (dpo_length_stratified_eval): same, broken down per stratum.
3. Generation on the held-out prompts: sampled-response KL from the reference, reward-model
   score, response-length statistics.
4. Generation on the fixed word-limit prompts: explicit word-limit compliance.

Pairs whose prompt does not fit max_sequence_length are excluded by the same rule as training
(task1_dpo.train.filter_prompts_that_fit); excluded IDs are saved with the results.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    write_jsonl,
)
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import word_count, word_limit_compliance
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from common.run_record import RunRecord, experiment_id
from task1_dpo.train import filter_prompts_that_fit

WORD_LIMIT_SAMPLES_PER_PROMPT = 5


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def length_stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "max": float(a.max()),
    }


def _device(model):
    return next(model.parameters()).device


@torch.no_grad()
def pair_records(model, tokenizer, rows, max_length, beta, has_adapter, batch_size=4):
    device = _device(model)
    out = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        enc_c, enc_r = [], []
        for row in chunk:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            enc_c.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            enc_r.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        bc = {k: v.to(device) for k, v in pad_batch(tokenizer, enc_c).items()}
        br = {k: v.to(device) for k, v in pad_batch(tokenizer, enc_r).items()}
        pc = response_sequence_logprobs(model, bc)[0]
        pr = response_sequence_logprobs(model, br)[0]
        if has_adapter:
            with reference_mode(model):
                rc = response_sequence_logprobs(model, bc)[0]
                rr = response_sequence_logprobs(model, br)[0]
        else:  # SFT: the evaluated policy IS the reference, so m = 0 by definition
            rc, rr = pc, pr
        margin = (pc - rc) - (pr - rr)
        loss = -F.logsigmoid(beta * margin)
        for j, row in enumerate(chunk):
            out.append({
                "prompt_id": row.get("prompt_id"),
                "source_index": row.get("source_index"),
                "length_stratum": row.get("length_stratum"),
                "chosen_response_tokens": int(sum(enc_c[j][1])),
                "rejected_response_tokens": int(sum(enc_r[j][1])),
                "policy_chosen_logp": float(pc[j]),
                "policy_rejected_logp": float(pr[j]),
                "ref_chosen_logp": float(rc[j]),
                "ref_rejected_logp": float(rr[j]),
                "margin": float(margin[j]),
                "dpo_loss": float(loss[j]),
            })
    return out


def aggregate_pairs(records) -> dict:
    m = np.array([r["margin"] for r in records])
    return {
        "n": len(records),
        "dpo_loss": float(np.mean([r["dpo_loss"] for r in records])),
        "preference_accuracy": float((m > 0).mean()),
        "mean_margin": float(m.mean()),
        "median_margin": float(np.median(m)),
    }


@torch.no_grad()
def generate_records(model, tokenizer, items, cfg, has_adapter, reward=None, gen_batch=16, score_batch=4):
    """items: list of dicts with prompt_id and messages. Returns one record per generated response."""
    gcfg = cfg["generation"]
    max_new = int(cfg["max_generation_tokens"])
    max_prompt = int(cfg["max_sequence_length"])
    # Sort by prompt length so batches pad less; order is deterministic, so sampling is too.
    lens = [len(tokenizer.apply_chat_template(it["messages"], tokenize=True, add_generation_prompt=True)) for it in items]
    order = sorted(range(len(items)), key=lambda i: (lens[i], str(items[i]["prompt_id"])))
    set_seed(int(cfg["seed"]))
    out = []
    for start in range(0, len(order), gen_batch):
        batch_items = [items[i] for i in order[start:start + gen_batch]]
        gen = batch_generate(
            model, tokenizer, [it["messages"] for it in batch_items],
            max_prompt_length=max_prompt, max_new_tokens=max_new,
            temperature=float(gcfg["temperature"]), top_p=float(gcfg["top_p"]), do_sample=bool(gcfg["do_sample"]),
        )
        kl_sum, n_tok = [], []
        for s in range(0, len(batch_items), score_batch):
            sl = slice(s, s + score_batch)
            args = (gen["sequences"][sl], gen["attention_mask"][sl], gen["prompt_width"], gen["response_ids"][sl])
            plp = response_token_logprobs(model, *args)[0]
            if has_adapter:
                with reference_mode(model):
                    rlp = response_token_logprobs(model, *args)[0]
            else:
                rlp = plp
            mask = gen["response_mask"][sl]
            kl_sum.extend(((plp - rlp) * mask).sum(-1).tolist())
            n_tok.extend(mask.sum(-1).tolist())
        rewards = [None] * len(batch_items)
        if reward is not None:
            rm, rm_tok = reward
            rewards = score_reward_pairs(
                rm, rm_tok, [it["messages"] for it in batch_items], gen["responses"],
                max_length=int(cfg.get("reward_max_length", 1280)),
            ).tolist()
        for j, it in enumerate(batch_items):
            out.append({
                "prompt_id": it["prompt_id"],
                "sample_index": it.get("sample_index", 0),
                "response": gen["responses"][j],
                "response_tokens": int(gen["response_lengths"][j]),
                "terminated_with_eos": bool(gen["terminated_with_eos"][j]),
                "truncated": bool(gen["truncated"][j]),
                "kl_sum": float(kl_sum[j]),
                "kl_tokens": float(n_tok[j]),
                "reward": rewards[j],
            })
        print(f"  generated {min(start + gen_batch, len(order))}/{len(order)}", flush=True)
    return out


def summarize_generation(records) -> dict:
    kl_total = sum(r["kl_sum"] for r in records)
    tok_total = sum(r["kl_tokens"] for r in records)
    out = {
        "n_responses": len(records),
        # Primary KL convention: token-level mean over all valid response tokens, i.e. the course
        # helper common.metrics.sampled_kl applied to the whole evaluation set.
        "kl_token_mean": kl_total / max(tok_total, 1.0),
        "kl_sequence_mean": float(np.mean([r["kl_sum"] for r in records])),
        "length_tokens": length_stats([r["response_tokens"] for r in records]),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
    }
    rewards = [r["reward"] for r in records if r["reward"] is not None]
    if rewards:
        out["reward_mean"] = float(np.mean(rewards))
        out["reward_std"] = float(np.std(rewards))
    return out


def evaluate_policy(config_path: str, adapter: str | None, name: str, beta: float | None = None, limit: int | None = None):
    """`limit` truncates every evaluation set; only for smoke tests (name must contain 'smoke')."""
    if limit is not None and "smoke" not in name:
        raise ValueError("--limit is only allowed for smoke runs; real evaluations use the full fixed sets")
    cfg = load_yaml(config_path)
    beta = float(cfg["beta"] if beta is None else beta)
    exp_id = experiment_id("task1_dpo", f"eval_{name}", int(cfg["seed"]))
    record = RunRecord(cfg["results_dir"], exp_id, dict(cfg, eval_policy=name, eval_adapter=adapter, beta_for_loss=beta), extra={
        "task": "task1_dpo",
        "condition": name,
        "model": cfg["base_model"],
        "checkpoint": adapter or "SFT (no adapter)",
        "decoding": dict(cfg["generation"], max_new_tokens=int(cfg["max_generation_tokens"])),
        "word_limit_samples_per_prompt": WORD_LIMIT_SAMPLES_PER_PROMPT,
    })
    if record.is_done():
        print(f"[skip] {exp_id} already done")
        return record.dir

    with record:
        tokenizer = load_tokenizer(cfg["base_model"])
        max_length = int(cfg["max_sequence_length"])
        std_rows, std_dropped = filter_prompts_that_fit(read_jsonl(cfg["paths"]["dpo_standard_eval"]), tokenizer, max_length)
        len_rows, len_dropped = filter_prompts_that_fit(read_jsonl(cfg["paths"]["dpo_length_eval"]), tokenizer, max_length)
        if limit is not None:
            std_rows, len_rows = std_rows[:limit], len_rows[:limit]
        save_json(record.dir / "dropped_long_prompts.json", {"dpo_standard_eval": std_dropped, "dpo_length_stratified_eval": len_dropped})

        policy = load_policy(cfg, adapter_path=adapter, trainable=False)
        has_adapter = adapter is not None

        print("[1/4] held-out pairs", flush=True)
        std_pairs = pair_records(policy, tokenizer, std_rows, max_length, beta, has_adapter)
        write_jsonl(record.dir / "pairs_heldout.jsonl", std_pairs)

        print("[2/4] length-stratified pairs", flush=True)
        len_pairs = pair_records(policy, tokenizer, len_rows, max_length, beta, has_adapter)
        write_jsonl(record.dir / "pairs_length_stratified.jsonl", len_pairs)
        strata = sorted({r["length_stratum"] for r in len_pairs})

        print("[3/4] held-out generation", flush=True)
        reward = load_reward_model(cfg)
        items = [{"prompt_id": r["prompt_id"], "messages": prompt_messages_from_preference(r)} for r in std_rows]
        gens = generate_records(policy, tokenizer, items, cfg, has_adapter, reward=reward)
        write_jsonl(record.dir / "generations_heldout.jsonl", gens)

        print("[4/4] word-limit prompts", flush=True)
        wl_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])[: limit or None]
        wl_items = [
            {"prompt_id": r["prompt_id"], "sample_index": s, "messages": prompt_messages(r)}
            for r in wl_rows for s in range(WORD_LIMIT_SAMPLES_PER_PROMPT)
        ]
        wl = generate_records(policy, tokenizer, wl_items, cfg, has_adapter)
        prompt_text = {r["prompt_id"]: prompt_messages(r)[-1]["content"] for r in wl_rows}
        for r in wl:
            r["words"] = word_count(r["response"])
            r["compliant"] = word_limit_compliance(prompt_text[r["prompt_id"]], r["response"])
        write_jsonl(record.dir / "generations_word_limit.jsonl", wl)
        if any(r["compliant"] is None for r in wl):
            raise ValueError("a word-limit prompt has no parseable limit")

        metrics = {
            "policy": name,
            "adapter": adapter,
            "beta_for_heldout_loss": beta,
            "heldout_pairs": aggregate_pairs(std_pairs),
            "length_stratified_pairs": {
                "overall": aggregate_pairs(len_pairs),
                "by_stratum": {s: aggregate_pairs([r for r in len_pairs if r["length_stratum"] == s]) for s in strata},
            },
            "heldout_generation": summarize_generation(gens),
            "word_limit": {
                "n_prompts": len(wl_rows),
                "samples_per_prompt": WORD_LIMIT_SAMPLES_PER_PROMPT,
                "compliance_rate": float(np.mean([r["compliant"] for r in wl])),
                "words": length_stats([r["words"] for r in wl]),
                "length_tokens": length_stats([r["response_tokens"] for r in wl]),
                "per_prompt_compliance": {
                    pid: float(np.mean([r["compliant"] for r in wl if r["prompt_id"] == pid])) for pid in prompt_text
                },
            },
            "excluded_long_prompts": {"heldout": len(std_dropped), "length_stratified": len(len_dropped)},
        }
        record.finish(metrics, artifacts={
            "pairs_heldout": "pairs_heldout.jsonl",
            "pairs_length_stratified": "pairs_length_stratified.jsonl",
            "generations_heldout": "generations_heldout.jsonl",
            "generations_word_limit": "generations_word_limit.jsonl",
        })
    return record.dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", help="LoRA adapter directory; omit for the SFT baseline")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the held-out DPO loss (default: config beta)")
    ap.add_argument("--limit", type=int, help="smoke tests only: evaluate the first N items of each set")
    args = ap.parse_args()
    evaluate_policy(args.config, args.adapter, args.name, args.beta, args.limit)


if __name__ == "__main__":
    main()
