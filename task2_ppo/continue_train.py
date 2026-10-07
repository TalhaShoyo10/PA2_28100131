"""Task 2: PPO continuation from the supplied midpoint policy + critic.

One update (config: prompts_per_update = 1, ppo_epochs = 2):
  1. rollout: the current policy samples a response (<= max_response_length tokens)
  2. frozen statistics: old log-probs (policy), reference log-probs (adapter off), critic values,
     learned reward minus the missing-EOS penalty
  3. shaped rewards  r_t = -beta_KL (log pi_old - log ref) + reward on the last token
  4. GAE(gamma, lambda) -> advantages (whitened over valid tokens) and returns
  5. ppo_epochs gradient steps on this rollout: clipped policy loss + value_coef * value MSE
Every condition (standard and forks) starts from the identical supplied policy/critic state and
uses the same seeded prompt schedule.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean, sampled_kl, sample_entropy
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from common.rl import disable_dropout, generate, learned_rewards, load_prompt_rows, policy_and_reference_logprobs, prompt_schedule
from common.run_record import RunRecord, experiment_id
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    # Trainable critic weights in fp32 (LoRA already is; the copied score head is fp16).
    for p in value_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    head = value_model.score if hasattr(value_model, "score") else value_model.classifier
    for p in head.parameters():
        p.data = p.data.float()
    disable_dropout(policy)
    disable_dropout(value_model)

    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts, dropped = load_prompt_rows(cfg, "rl_prompt_train", tokenizer)

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "dropped_prompts": dropped,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def response_values(value_model, sequences, attention_mask, prompt_width, n_response):
    """V(s_t) for each response position t: value-head output at the last token before token t."""
    v = token_values(value_model, sequences, attention_mask)
    return v[:, prompt_width - 1: prompt_width - 1 + n_response].float()


def explained_variance(pred, target, mask):
    m = mask.bool()
    y, p = target[m], pred[m]
    var = y.var(unbiased=False)
    return float(1.0 - (y - p).var(unbiased=False) / var) if var > 0 else float("nan")


def collect_rollout(bundle, prompt_row):
    cfg, policy, tok = bundle["cfg"], bundle["policy"], bundle["tokenizer"]
    msgs = [prompt_messages(prompt_row)]
    gen = generate(policy, tok, msgs, cfg, int(cfg["max_response_length"]))
    old_logp, ref_logp = policy_and_reference_logprobs(policy, gen, has_adapter=True)
    mask = gen["response_mask"].float()
    T = gen["response_ids"].shape[1]
    with torch.no_grad():
        values = response_values(bundle["value_model"], gen["sequences"], gen["attention_mask"], gen["prompt_width"], T)
    raw, eff = learned_rewards((bundle["reward_model"], bundle["reward_tokenizer"]), msgs, gen["responses"], gen["terminated_with_eos"], cfg)
    return gen, old_logp, ref_logp, values * mask, mask, raw, eff


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    cfg0 = load_yaml(config_path)
    out = repo_path(output or (cfg0["output"] if run_name == "standard" else f"outputs/task2_ppo/{run_name}"))
    exp_id = experiment_id("task2_ppo", f"train_{run_name}", int(cfg0["seed"]))
    if RunRecord(cfg0["results_dir"], exp_id, cfg0).is_done():
        print(f"[skip] {exp_id} already done; adapter at {out}")
        return out

    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, value_model = bundle["policy"], bundle["value_model"]
    p_opt, v_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    p_params, v_params = trainable_parameters(policy), trainable_parameters(value_model)
    assert all(p.dtype == torch.float32 for p in p_params + v_params), "expected fp32 trainable parameters"
    use_scaler = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    eps, beta_kl = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    n_updates = int(cfg["updates"])
    schedule = prompt_schedule(bundle["prompt_rows"], int(cfg["seed"]), n_updates)

    record = RunRecord(cfg["results_dir"], exp_id, dict(cfg, run_name=run_name), extra={
        "task": "task2_ppo",
        "condition": run_name,
        "model": cfg["base_model"],
        "checkpoint": {"policy": cfg["paths"]["ppo_midpoint_policy"], "value": cfg["paths"]["ppo_midpoint_value"]},
        "clip_epsilon": eps,
        "kl_beta": beta_kl,
        "decoding": dict(cfg["generation"], max_new_tokens=int(cfg["max_response_length"])),
        "budget": {"updates": n_updates, "prompts_per_update": int(cfg["prompts_per_update"]), "ppo_epochs": int(cfg["ppo_epochs"])},
        "adapter_output": str(out),
        "dropped_long_prompts": len(bundle["dropped_prompts"]),
    })

    with record:
        save_json(record.dir / "dropped_long_prompts.json", bundle["dropped_prompts"])
        save_json(record.dir / "prompt_schedule.json", [r["prompt_id"] for r in schedule])
        log_path = record.dir / "train_log.jsonl"
        t0 = time.perf_counter()
        total_tokens = 0
        for u, prompt_row in enumerate(schedule, start=1):
            gen, old_logp, ref_logp, values, mask, raw, eff = collect_rollout(bundle, prompt_row)
            dev = old_logp.device
            rewards = shaped_rewards(eff.to(dev), old_logp, ref_logp, mask, beta_kl)
            adv_raw, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
            adv = normalize_advantages(adv_raw, mask)
            total_tokens += int(mask.sum())

            epoch_stats = []
            for epoch in range(int(cfg["ppo_epochs"])):
                new_logp = response_token_logprobs(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"])[0].float()
                p_loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, adv, mask, eps=eps)
                v_new = response_values(value_model, gen["sequences"], gen["attention_mask"], gen["prompt_width"], mask.shape[1])
                v_loss = value_mse_loss(v_new, returns, mask)
                loss = p_loss + float(cfg["value_coef"]) * v_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite PPO loss at update {u} epoch {epoch}")
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(p_opt)
                scaler.unscale_(v_opt)
                p_gn = torch.nn.utils.clip_grad_norm_(p_params, float(cfg["max_grad_norm"]))
                v_gn = torch.nn.utils.clip_grad_norm_(v_params, float(cfg["max_grad_norm"]))
                scaler.step(p_opt)
                scaler.step(v_opt)
                scaler.update()
                p_opt.zero_grad(set_to_none=True)
                v_opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    log_ratio = (new_logp - old_logp) * mask
                    epoch_stats.append({
                        "policy_loss": float(p_loss),
                        "value_loss": float(v_loss),
                        "clip_fraction": float(clip_frac),
                        "approx_kl_old_new": float(masked_mean(-log_ratio, mask)),
                        "ratio_max": float(ratio[mask.bool()].max()),
                        "ratio_min": float(ratio[mask.bool()].min()),
                        "policy_grad_norm": float(p_gn),
                        "value_grad_norm": float(v_gn),
                        "skipped_overflow": bool(use_scaler and scaler.get_scale() < scale_before),
                    })

            row = {
                "update": u,
                "prompt_id": prompt_row["prompt_id"],
                "reward_raw": float(raw[0]),
                "reward_effective": float(eff[0]),
                "kl_ref": float(sampled_kl(old_logp, ref_logp, mask)),
                "entropy": float(sample_entropy(old_logp, mask)),
                "response_tokens": int(gen["response_lengths"][0]),
                "terminated_with_eos": bool(gen["terminated_with_eos"][0]),
                "truncated": bool(gen["truncated"][0]),
                "value_mean": float(masked_mean(values, mask)),
                "return_mean": float(masked_mean(returns, mask)),
                "value_explained_variance": explained_variance(values, returns, mask),
                "policy_loss": sum(e["policy_loss"] for e in epoch_stats) / len(epoch_stats),
                "value_loss": sum(e["value_loss"] for e in epoch_stats) / len(epoch_stats),
                "clip_fraction": sum(e["clip_fraction"] for e in epoch_stats) / len(epoch_stats),
                "clip_fraction_last_epoch": epoch_stats[-1]["clip_fraction"],
                "approx_kl_old_new_last_epoch": epoch_stats[-1]["approx_kl_old_new"],
                "policy_grad_norm": max(e["policy_grad_norm"] for e in epoch_stats),
                "value_grad_norm": max(e["value_grad_norm"] for e in epoch_stats),
                "skipped_overflow_steps": sum(e["skipped_overflow"] for e in epoch_stats),
                "epochs": epoch_stats,
                "generated_tokens_cumulative": total_tokens,
                "elapsed_s": round(time.perf_counter() - t0, 1),
                "peak_vram_gib": record.peak_vram_gib(),
            }
            append_jsonl(log_path, row)
            record.heartbeat(update=u, of=n_updates)
            print(f"[{run_name}] update {u}/{n_updates} reward {row['reward_effective']:+.3f} kl {row['kl_ref']:+.4f} "
                  f"ploss {row['policy_loss']:+.4f} vloss {row['value_loss']:.3f} clip {row['clip_fraction']:.3f} "
                  f"len {row['response_tokens']} t {row['elapsed_s']:.0f}s", flush=True)

        policy.save_pretrained(str(out))
        value_model.save_pretrained(str(out) + "_value")
        log = read_jsonl(log_path)
        record.finish(
            metrics={
                "updates": n_updates,
                "clip_epsilon": eps,
                "kl_beta": beta_kl,
                "generated_tokens": total_tokens,
                "skipped_overflow_steps": sum(r["skipped_overflow_steps"] for r in log),
                "reward_effective_mean": sum(r["reward_effective"] for r in log) / len(log),
                "kl_ref_last": log[-1]["kl_ref"],
                "clip_fraction_mean": sum(r["clip_fraction"] for r in log) / len(log),
            },
            artifacts={"policy_adapter": str(out), "value_adapter": str(out) + "_value", "train_log": str(log_path)},
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
