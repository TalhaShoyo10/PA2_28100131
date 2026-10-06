from __future__ import annotations

import argparse
import math
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from common.run_record import RunRecord, experiment_id
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prompt_token_length(tokenizer, row) -> int:
    ids = tokenizer.apply_chat_template(
        prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True
    )
    return len(ids)


def filter_prompts_that_fit(rows, tokenizer, max_length):
    """Drop pairs whose prompt alone does not fit in max_length.

    Decision (student-approved 2026-10-07): `encode_prompt_response` refuses these rows rather
    than truncating the prompt, and the released max_sequence_length is kept unchanged. The
    same rule is applied to every Task 1 dataset and condition; dropped IDs are recorded.
    """
    kept, dropped = [], []
    for row in rows:
        n = prompt_token_length(tokenizer, row)
        if n < max_length:
            kept.append(row)
        else:
            dropped.append({"prompt_id": row.get("prompt_id"), "source_index": row.get("source_index"), "prompt_tokens": n})
    return kept, dropped


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    tokenizer = load_tokenizer(cfg["base_model"])
    max_length = int(cfg["max_sequence_length"])

    # Filter first, then take the first N in file order, so a short fork still sees exactly N pairs.
    rows, dropped = filter_prompts_that_fit(read_jsonl(path), tokenizer, max_length)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    # A dedicated generator makes the shuffle order depend only on the seed, not on how much
    # global RNG model construction consumed, so every condition sees the same example order.
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=make_collate(tokenizer, max_length),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dropped": dropped,
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def dpo_forward(model, chosen, rejected, beta):
    """Policy (with grad) and reference (adapter disabled, no grad) sequence log-probs."""
    pc, _, _ = response_sequence_logprobs(model, chosen)
    pr, _, _ = response_sequence_logprobs(model, rejected)
    with torch.no_grad(), reference_mode(model):
        rc, _, _ = response_sequence_logprobs(model, chosen)
        rr, _, _ = response_sequence_logprobs(model, rejected)
    loss, diag = dpo_loss(pc, pr, rc, rr, beta)
    with torch.no_grad():
        diag["chosen_reward"] = (beta * (pc - rc)).mean()
        diag["rejected_reward"] = (beta * (pr - rr)).mean()
        diag["policy_chosen_logp"] = pc.mean()
        diag["policy_rejected_logp"] = pr.mean()
    return loss, diag


def default_output(cfg: dict, run_name: str) -> str:
    if run_name == "standard":
        return cfg["standard_output"]
    if run_name == "length_balanced":
        return cfg["length_output"]
    return f"outputs/task1_dpo/{run_name}"


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg0 = load_yaml(config_path)
    output = repo_path(output_path or default_output(cfg0, run_name))
    exp_id = experiment_id("task1_dpo", f"train_{run_name}", int(cfg0["seed"]))
    if RunRecord(cfg0["results_dir"], exp_id, cfg0).is_done():
        print(f"[skip] {exp_id} already done; adapter at {output}")
        return output

    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    beta = bundle["beta"]
    model, loader, optimizer = bundle["model"], bundle["loader"], bundle["optimizer"]
    output.parent.mkdir(parents=True, exist_ok=True)

    params = trainable_parameters(model)
    # PEFT keeps LoRA weights in fp32 over an fp16 base; GradScaler protects fp16 activation
    # gradients from underflow on the T4 (no bf16 support).
    assert all(p.dtype == torch.float32 for p in params), "expected fp32 LoRA parameters"
    use_scaler = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    device = next(model.parameters()).device

    accum = int(cfg["grad_accum_steps"])
    max_grad_norm = float(cfg["max_grad_norm"])
    n_micro = len(loader)
    n_updates = math.ceil(n_micro / accum)
    rec_cfg = dict(cfg, beta=beta, run_name=run_name, train_dataset=bundle["dataset_path"], max_examples=max_examples)
    record = RunRecord(cfg["results_dir"], exp_id, rec_cfg, extra={
        "task": "task1_dpo",
        "condition": run_name,
        "model": cfg["base_model"],
        "initialization": "original Qwen2.5-1.5B-Instruct + fresh LoRA",
        "adapter_output": str(output),
        "budget": {
            "train_pairs": len(bundle["rows"]),
            "epochs": int(cfg["epochs"]),
            "micro_batches": n_micro,
            "optimizer_updates": n_updates,
            "pairs_per_update": int(cfg["batch_size"]) * accum,
        },
        "dropped_long_prompts": len(bundle["dropped"]),
    })

    with record:
        save_json(record.dir / "dropped_long_prompts.json", bundle["dropped"])
        save_json(record.dir / "train_prompt_ids.json", [r["prompt_id"] for r in bundle["rows"]])
        log_path = record.dir / "train_log.jsonl"
        t0 = time.perf_counter()
        update, micro = 0, 0
        for epoch in range(int(cfg["epochs"])):
            window, window_diags, window_loss = [], [], 0.0
            for i, (chosen, rejected) in enumerate(loader):
                # Divide by the real window size so a short final window is not under-weighted.
                window_start = (micro // accum) * accum
                window_size = min(accum, n_micro - window_start)
                loss, diag = dpo_forward(model, _to_device(chosen, device), _to_device(rejected, device), beta)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite DPO loss at micro-batch {micro}")
                scaler.scale(loss / window_size).backward()
                window_loss += loss.item() / window_size
                window_diags.append({k: float(v) for k, v in diag.items()})
                micro += 1

                if micro % accum == 0 or micro == n_micro:
                    scale_before = scaler.get_scale()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    skipped = use_scaler and scaler.get_scale() < scale_before
                    update += 1
                    row = {
                        "update": update,
                        "epoch": epoch,
                        "pairs_seen": min(micro * int(cfg["batch_size"]), len(bundle["rows"])),
                        "loss": window_loss,
                        "grad_norm_preclip": float(grad_norm),
                        "step_skipped_overflow": bool(skipped),
                        "loss_scale": float(scale_before) if use_scaler else None,
                        "lr": optimizer.param_groups[0]["lr"],
                        "elapsed_s": round(time.perf_counter() - t0, 1),
                        "peak_vram_gib": record.peak_vram_gib(),
                    }
                    for key in window_diags[0]:
                        row[key] = sum(d[key] for d in window_diags) / len(window_diags)
                    append_jsonl(log_path, row)
                    record.heartbeat(update=update, of=n_updates)
                    print(f"[{run_name}] update {update}/{n_updates} loss {row['loss']:.4f} "
                          f"acc {row['preference_accuracy']:.3f} gnorm {row['grad_norm_preclip']:.3f} "
                          f"t {row['elapsed_s']:.0f}s", flush=True)
                    window_diags, window_loss = [], 0.0

        model.save_pretrained(str(output))
        log = read_jsonl(log_path)
        record.finish(
            metrics={
                "beta": beta,
                "optimizer_updates": update,
                "train_pairs": len(bundle["rows"]),
                "first_update_loss": log[0]["loss"],
                "final_update_loss": log[-1]["loss"],
                "final_update_train_preference_accuracy": log[-1]["preference_accuracy"],
                "mean_train_loss_last_5_updates": sum(r["loss"] for r in log[-5:]) / len(log[-5:]),
                "skipped_overflow_updates": sum(r["step_skipped_overflow"] for r in log),
            },
            artifacts={"adapter": str(output), "train_log": str(log_path)},
        )
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
