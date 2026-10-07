"""Common held-out PPO evaluation: one sampled response per filtered held-out prompt, generated with
the frozen-evaluation cap (eval_max_response_length), learned reward, sampled KL to the reference,
sampled-token entropy, and length statistics. Identical for every PPO condition."""
from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, write_jsonl
from common.logging_utils import save_json
from common.models import load_policy, load_reward_model, load_tokenizer
from common.rl import evaluate_rl_policy, load_prompt_rows, summarize_rl_eval
from common.run_record import RunRecord, experiment_id


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_policy(config_path: str, adapter: str | None, name: str, limit: int | None = None):
    if limit is not None and "smoke" not in name:
        raise ValueError("--limit is only allowed for smoke runs")
    cfg = load_yaml(config_path)
    max_new = int(cfg["eval_max_response_length"])
    exp_id = experiment_id("task2_ppo", f"eval_{name}", int(cfg["seed"]))
    record = RunRecord(cfg["results_dir"], exp_id, dict(cfg, eval_policy=name, eval_adapter=adapter), extra={
        "task": "task2_ppo",
        "condition": name,
        "model": cfg["base_model"],
        "checkpoint": adapter or "SFT (no adapter)",
        "decoding": dict(cfg["generation"], max_new_tokens=max_new),
    })
    if record.is_done():
        print(f"[skip] {exp_id} already done")
        return record.dir
    with record:
        tok = load_tokenizer(cfg["base_model"])
        rows, dropped = load_prompt_rows(cfg, "rl_prompt_eval", tok)
        if limit is not None:
            rows = rows[:limit]
        save_json(record.dir / "dropped_long_prompts.json", dropped)
        policy = load_policy(cfg, adapter_path=adapter, trainable=False)
        reward = load_reward_model(cfg)
        recs = evaluate_rl_policy(policy, tok, reward, rows, cfg, has_adapter=adapter is not None, max_new_tokens=max_new,
                                  progress=lambda i, n: record.heartbeat(update=i, of=n))
        write_jsonl(record.dir / "generations_heldout.jsonl", recs)
        metrics = dict(summarize_rl_eval(recs), policy=name, adapter=adapter, excluded_long_prompts=len(dropped))
        record.finish(metrics, artifacts={"generations_heldout": "generations_heldout.jsonl"})
    return record.dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", help="policy LoRA adapter; omit for SFT")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--limit", type=int, help="smoke tests only")
    args = ap.parse_args()
    evaluate_policy(args.config, args.adapter, args.name, args.limit)


if __name__ == "__main__":
    main()
