from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.metrics import length_stats
from common.models import load_policy, load_tokenizer
from common.run_record import RunRecord, experiment_id


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    return records


# The standard (Step 1) training run that produced each fixed Task 4 adapter. Generation refuses to
# start unless that run finished, so an ablation fork or a half-trained adapter can never be used.
STANDARD_TRAIN_RUNS = {
    "dpo": ("task1_dpo", "task1_dpo_train_standard"),
    "ppo": ("task2_ppo", "task2_ppo_train_standard"),
    "grpo": ("task3_grpo", "task3_grpo_train_standard"),
}
GEN_BATCH_SIZE = 16  # greedy decoding; the same batching for every policy


def safety_dir(cfg):
    return repo_path(cfg["results_dir"]) / "task4_safety"


def check_standard_policy(cfg, name: str) -> None:
    if name == "sft":
        return
    task, prefix = STANDARD_TRAIN_RUNS[name]
    status = repo_path(cfg["results_dir"]) / task / f"{prefix}_seed{int(cfg['seed'])}" / "status.json"
    if not status.exists() or json.loads(status.read_text(encoding="utf-8"))["state"] != "done":
        raise SystemExit(f"{name}: standard training run {status.parent.name} is not done; Task 4 needs the finished standard policy")
    if not (repo_path(cfg["policies"][name]) / "adapter_config.json").exists():
        raise SystemExit(f"{name}: adapter not found at {cfg['policies'][name]}")


def run_policy(cfg, name: str) -> None:
    exp_id = experiment_id("task4_safety", f"generate_{name}", int(cfg["seed"]))
    record = RunRecord(safety_dir(cfg), exp_id, cfg, extra={
        "task": "task4_safety", "condition": name, "model": cfg["base_model"],
        "checkpoint": cfg["policies"][name] or "SFT (no adapter)",
        "decoding": {"do_sample": False, "max_new_tokens": int(cfg["safety_max_new_tokens"]), "max_prompt_length": 256,
                     "batch_size": GEN_BATCH_SIZE},
    })
    if record.is_done():
        print(f"[skip] {exp_id} already done")
        return
    check_standard_policy(cfg, name)
    with record:
        records = generate_for_policy(cfg, name, batch_size=GEN_BATCH_SIZE)
        write_jsonl(record.dir / "generated.jsonl", records)
        # Canonical copy at the fixed path the release's make_audit_sheet reads.
        write_jsonl(safety_dir(cfg) / f"generated_{name}.jsonl", records)
        lengths = [r["response_tokens"] for r in records]
        record.finish({
            "policy": name,
            "n_prompts": len(records),
            "n_safe": sum(r["benchmark_class"] == "SAFE" for r in records),
            "n_unsafe": sum(r["benchmark_class"] == "UNSAFE" for r in records),
            "response_tokens": length_stats(lengths),
            "hit_max_new_tokens": sum(n >= int(cfg["safety_max_new_tokens"]) for n in lengths),
        }, artifacts={"generated": f"generated_{name}.jsonl"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policy", choices=["sft", "dpo", "ppo", "grpo"], help="one policy (default: all four)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Policies:", list(policy_specs(cfg)))
    print("XSTest rows:", len(load_xstest(cfg)))
    for name in ([args.policy] if args.policy else list(policy_specs(cfg))):
        run_policy(cfg, name)


if __name__ == "__main__":
    main()
