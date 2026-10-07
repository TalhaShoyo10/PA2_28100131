"""Task 3: GRPO continuation from the supplied midpoint adapter.

One update (config: 1 prompt per update, K = num_generations completions, policy_epochs = 1):
  1. sample K completions for the prompt (<= max_completion_length tokens)
  2. old log-probs (policy) and reference log-probs (adapter off); learned reward per completion
  3. group-relative advantages A_k = (r_k - mean_group) / (std_group + eps)
  4. completions that hit the length cap are masked out of the loss (mask_truncated_completions)
  5. grpo_policy_loss: clipped ratio objective with canonical (1/T_k) or Dr-GRPO (1/max_len)
     sequence normalization, plus beta * KL(pi || ref) (k3 estimator inside the release loss)
Every condition starts from the identical midpoint and uses the same seeded prompt schedule.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import sampled_kl, sample_entropy
from common.models import load_policy, load_reward_model, load_tokenizer, trainable_parameters
from common.rl import disable_dropout, generate, learned_rewards, load_prompt_rows, policy_and_reference_logprobs, prompt_schedule
from common.run_record import RunRecord, experiment_id
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

# A group is uninformative when its reward std is at most the release helper's eps (1e-6).
ZERO_STD_TOL = 1e-6
# Loss-scaler start value. Task 2 lost its first PPO steps while the default (65536) calibrated down;
# a lower start spends fewer of GRPO's 20 single-step updates on calibration. Same for every condition.
GRAD_SCALER_INIT = 2.0 ** 12


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    disable_dropout(policy)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts, dropped = load_prompt_rows(cfg, "rl_prompt_train", tokenizer)
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "dropped_prompts": dropped,
        "optimizer": optimizer,
    }


def sequence_weights(lengths, adv, loss_type, max_len):
    """Analytic gradient allocation of the policy term at ratio = 1: per-token weight |A_k| / N_k and
    per-sequence total |A_k| * T_k / N_k, with N_k = T_k (canonical) or max_len (Dr-GRPO)."""
    out = []
    for T, a in zip(lengths, adv):
        denom = max(T, 1) if loss_type == "grpo" else max_len
        out.append({"per_token_weight": abs(a) / denom, "sequence_weight": abs(a) * T / denom})
    return out


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    cfg0 = load_yaml(config_path)
    out = repo_path(output or (cfg0["output"] if run_name == "standard" else f"outputs/task3_grpo/{run_name}"))
    exp_id = experiment_id("task3_grpo", f"train_{run_name}", int(cfg0["seed"]))
    if RunRecord(cfg0["results_dir"], exp_id, cfg0).is_done():
        print(f"[skip] {exp_id} already done; adapter at {out}")
        return out

    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out.parent.mkdir(parents=True, exist_ok=True)
    policy, opt, tok = bundle["policy"], bundle["optimizer"], bundle["tokenizer"]
    params = trainable_parameters(policy)
    assert all(p.dtype == torch.float32 for p in params), "expected fp32 LoRA parameters"
    use_scaler = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", init_scale=GRAD_SCALER_INIT, enabled=use_scaler)
    K = int(cfg["num_generations"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_len = int(cfg["max_completion_length"])
    n_updates = int(cfg["updates"])
    schedule = prompt_schedule(bundle["prompt_rows"], int(cfg["seed"]), n_updates)

    record = RunRecord(cfg["results_dir"], exp_id, dict(cfg, run_name=run_name, loss_type=loss_type), extra={
        "task": "task3_grpo",
        "condition": run_name,
        "loss_type": loss_type,
        "model": cfg["base_model"],
        "checkpoint": cfg["paths"]["grpo_midpoint_policy"],
        "decoding": dict(cfg["generation"], max_new_tokens=max_len),
        "budget": {"updates": n_updates, "prompts_per_update": int(cfg["prompts_per_update"]),
                   "generations_per_prompt": K, "policy_epochs": int(cfg["policy_epochs"])},
        "grad_scaler_init": GRAD_SCALER_INIT,
        "adapter_output": str(out),
        "dropped_long_prompts": len(bundle["dropped_prompts"]),
    })

    with record:
        save_json(record.dir / "dropped_long_prompts.json", bundle["dropped_prompts"])
        save_json(record.dir / "prompt_schedule.json", [r["prompt_id"] for r in schedule])
        log_path = record.dir / "train_log.jsonl"
        comp_path = record.dir / "completions.jsonl"
        t0 = time.perf_counter()
        total_tokens = 0
        for u, prompt_row in enumerate(schedule, start=1):
            msgs = [prompt_messages(prompt_row)] * K
            gen = generate(policy, tok, msgs, cfg, max_len)
            old_logp, ref_logp = policy_and_reference_logprobs(policy, gen, has_adapter=True)
            full_mask = gen["response_mask"].float()
            raw, _ = learned_rewards((bundle["reward_model"], bundle["reward_tokenizer"]), msgs, gen["responses"], gen["terminated_with_eos"], cfg)
            rewards = raw.to(old_logp.device)
            adv = group_relative_advantages(rewards, torch.zeros(K, dtype=torch.long, device=rewards.device))
            mask = mask_truncated_sequences(full_mask, gen["truncated"]) if cfg.get("mask_truncated_completions") else full_mask
            total_tokens += int(full_mask.sum())
            group_std = float(rewards.std(unbiased=False))

            stats = []
            for _ in range(int(cfg["policy_epochs"])):
                new_logp = response_token_logprobs(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])[0].float()
                loss, diag = grpo_policy_loss(new_logp, old_logp, adv, mask, ref_logp, eps, beta, loss_type=loss_type, max_completion_length=max_len)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite GRPO loss at update {u}")
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                stats.append({
                    "loss": float(loss),
                    "policy_term": float(diag["policy_term"]),
                    "kl_term_k3": float(diag["sampled_kl"]),
                    "clip_fraction": float(diag["clip_fraction"]),
                    "grad_norm": float(gn),
                    "skipped_overflow": bool(use_scaler and scaler.get_scale() < scale_before),
                })

            lengths = [int(x) for x in gen["response_lengths"]]
            weights = sequence_weights(lengths, adv.tolist(), loss_type, max_len)
            for k in range(K):
                append_jsonl(comp_path, {
                    "update": u, "prompt_id": prompt_row["prompt_id"], "k": k,
                    "response_tokens": lengths[k], "truncated": bool(gen["truncated"][k]),
                    "masked_from_loss": bool(mask[k].sum() == 0 and full_mask[k].sum() > 0),
                    "reward": float(rewards[k]), "advantage": float(adv[k]), **weights[k],
                    "response": gen["responses"][k],
                })
            row = {
                "update": u,
                "prompt_id": prompt_row["prompt_id"],
                "reward_mean": float(rewards.mean()),
                "reward_group_std": group_std,
                "uninformative_group": group_std <= ZERO_STD_TOL,
                # KL to the reference over all generated tokens (course helper; same as Task 2)
                "kl_ref": float(sampled_kl(old_logp, ref_logp, full_mask)),
                "entropy": float(sample_entropy(old_logp, full_mask)),
                "response_tokens_mean": sum(lengths) / K,
                "truncated_completions": int(sum(bool(t) for t in gen["truncated"])),
                "loss": stats[-1]["loss"],
                "policy_loss": stats[-1]["policy_term"],
                "kl_term_k3": stats[-1]["kl_term_k3"],
                "clip_fraction": stats[-1]["clip_fraction"],
                "grad_norm": stats[-1]["grad_norm"],
                "skipped_overflow_steps": sum(s["skipped_overflow"] for s in stats),
                "generated_tokens_cumulative": total_tokens,
                "elapsed_s": round(time.perf_counter() - t0, 1),
                "peak_vram_gib": record.peak_vram_gib(),
            }
            append_jsonl(log_path, row)
            record.heartbeat(update=u, of=n_updates)
            print(f"[{run_name}] update {u}/{n_updates} reward {row['reward_mean']:+.3f} std {group_std:.3f} "
                  f"kl {row['kl_ref']:+.5f} loss {row['loss']:+.4f} gnorm {row['grad_norm']:.3f} "
                  f"len {row['response_tokens_mean']:.0f} trunc {row['truncated_completions']} t {row['elapsed_s']:.0f}s", flush=True)

        policy.save_pretrained(str(out))
        log = read_jsonl(log_path)
        record.finish(
            metrics={
                "updates": n_updates,
                "loss_type": loss_type,
                "generated_tokens": total_tokens,
                "reward_mean": sum(r["reward_mean"] for r in log) / len(log),
                "uninformative_group_fraction": sum(r["uninformative_group"] for r in log) / len(log),
                "mean_group_reward_std": sum(r["reward_group_std"] for r in log) / len(log),
                "skipped_overflow_steps": sum(r["skipped_overflow_steps"] for r in log),
            },
            artifacts={"adapter": str(out), "train_log": str(log_path), "completions": str(comp_path)},
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
