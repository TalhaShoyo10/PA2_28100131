"""Task 2 Step 2: clipping study.

Part A - cached rollout diagnostic (one fixed batch, only epsilon changes):
  1. Rebuild the 32 cached rollouts as token sequences (prompt from the RL prompt pools + re-tokenized
     response). A rollout is kept only if re-tokenization reproduces the cached token count.
  2. Step-0 check: the supplied midpoint's log-probs on the rebuilt tokens vs the cached old log-probs.
  3. Advantages/returns exactly as in training: KL-shaped rewards (config beta_KL), GAE with the
     cached critic values, per-rollout advantage whitening.
  4. For each epsilon, from a fresh copy of the identical midpoint policy, run the configured
     ppo_epochs passes over the batch (one optimizer step per rollout, fixed order) and record, before
     each step: clip fraction (rho outside [1-eps, 1+eps]), affected-token fraction (tokens whose
     clipped term is the active minimum, i.e. A>0 & rho>1+eps or A<0 & rho<1-eps -> zero gradient),
     unclipped and clipped surrogate. A final no-grad pass reports the same quantities for the
     end-of-batch policy.
  Since ratios are exactly 1 before any update, epsilon can only matter once the policy has moved
  within the batch; this protocol measures precisely that.

Part B - matched short forks: run_ppo for fork_updates from the same midpoint for each epsilon
(beta_KL fixed at the config value), then the common held-out evaluation.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import response_token_logprobs
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer, trainable_parameters
from common.rl import disable_dropout
from common.run_record import RunRecord, experiment_id
from task2_ppo.evaluate import evaluate_policy
from task2_ppo.continue_train import run_ppo
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def fork_name(eps: float, kl: float) -> str:
    return f"fork_eps{int(round(eps * 100)):03d}_kl{int(round(kl * 100)):03d}"


def rebuild_batch(cfg, tokenizer, rows):
    lookup = {}
    for key in ("rl_prompt_train", "rl_prompt_eval"):
        for r in read_jsonl(cfg["paths"][key]):
            lookup[r["prompt_id"]] = r
    items, rejected = [], []
    for i, r in enumerate(rows):
        pr = lookup.get(r.get("prompt_id"))
        if pr is None:
            rejected.append({"cache_index": i, "reason": "prompt_id not in prompt pools"})
            continue
        # No generation happens here, so the rollout-time prompt-length filter does not apply:
        # every supplied rollout is used with its full prompt (step0_check.json shows the match).
        prompt_ids = tokenizer.apply_chat_template(prompt_messages(pr), tokenize=True, add_generation_prompt=True)
        resp_ids = tokenizer(r["response"], add_special_tokens=False)["input_ids"]
        if r.get("terminated_with_eos"):
            resp_ids = resp_ids + [tokenizer.eos_token_id]
        T = len(r["old_logprobs"])
        if len(resp_ids) != T:
            rejected.append({"cache_index": i, "reason": f"re-tokenized {len(resp_ids)} tokens != cached {T}"})
            continue
        items.append({
            "cache_index": i,
            "prompt_id": r["prompt_id"],
            "input_ids": torch.tensor([prompt_ids + resp_ids]),
            "prompt_width": len(prompt_ids),
            "response_ids": torch.tensor([resp_ids]),
            "old": torch.as_tensor(r["old_logprobs"]).float().unsqueeze(0),
            "ref": torch.as_tensor(r["ref_logprobs"]).float().unsqueeze(0),
            "values": torch.as_tensor(r["values"]).float().unsqueeze(0),
            "reward": float(r.get("effective_terminal_reward", r.get("raw_terminal_reward"))),
        })
    return items, rejected


def attach_advantages(items, cfg):
    for it in items:
        mask = torch.ones_like(it["old"])
        rewards = shaped_rewards(torch.tensor([it["reward"]]), it["old"], it["ref"], mask, float(cfg["kl_beta"]))
        adv, _ = compute_gae(rewards, it["values"], mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
        it["adv"] = normalize_advantages(adv, mask)


def _new_logp(policy, it, device):
    ids = it["input_ids"].to(device)
    return response_token_logprobs(policy, ids, torch.ones_like(ids), it["prompt_width"], it["response_ids"].to(device))[0].float()


def _geometry(new_logp, old, adv, eps):
    ratio = torch.exp(new_logp - old)
    outside = (ratio < 1 - eps) | (ratio > 1 + eps)
    affected = ((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps))
    surr_unclipped = ratio * adv
    surr_clipped = torch.minimum(surr_unclipped, ratio.clamp(1 - eps, 1 + eps) * adv)
    return {
        "clip_fraction": float(outside.float().mean()),
        "affected_fraction": float(affected.float().mean()),
        "surrogate_unclipped": float(surr_unclipped.mean()),
        "surrogate_clipped": float(surr_clipped.mean()),
        "ratio_max": float(ratio.max()),
        "ratio_min": float(ratio.min()),
        "approx_kl_old_new": float((old - new_logp).mean()),
        "tokens": int(ratio.numel()),
    }


def _pool(stats):
    """Token-weighted aggregate over rollouts."""
    n = sum(s["tokens"] for s in stats)
    out = {k: sum(s[k] * s["tokens"] for s in stats) / n for k in ("clip_fraction", "affected_fraction", "surrogate_unclipped", "surrogate_clipped", "approx_kl_old_new")}
    out["ratio_max"] = max(s["ratio_max"] for s in stats)
    out["ratio_min"] = min(s["ratio_min"] for s in stats)
    out["tokens"] = n
    return out


def cached_study(config_path: str):
    cfg = load_yaml(config_path)
    exp_id = experiment_id("task2_ppo", "cached_clipping", int(cfg["seed"]))
    record = RunRecord(cfg["results_dir"], exp_id, cfg, extra={
        "task": "task2_ppo", "condition": "cached_clipping", "checkpoint": cfg["paths"]["ppo_midpoint_policy"],
        "cache": cfg["cached_rollouts"], "epsilons": cfg["clip_values"], "ppo_epochs": int(cfg["ppo_epochs"]),
    })
    if record.is_done():
        print(f"[skip] {exp_id} already done")
        return
    with record:
        tok = load_tokenizer(cfg["base_model"])
        items, rejected = rebuild_batch(cfg, tok, load_cached_rollouts(cfg["cached_rollouts"]))
        save_json(record.dir / "rebuild_report.json", {"kept": [it["cache_index"] for it in items], "rejected": rejected})
        print(f"Rebuilt {len(items)}/{len(items) + len(rejected)} cached rollouts", flush=True)
        attach_advantages(items, cfg)

        results, steps = {}, []
        for eps in [float(e) for e in cfg["clip_values"]]:
            set_seed(int(cfg["seed"]))
            policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
            disable_dropout(policy)
            device = next(policy.parameters()).device
            params = trainable_parameters(policy)
            opt = AdamW(params, lr=float(cfg["policy_learning_rate"]))
            scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

            if not results:  # step-0 reconstruction check, once (same start policy for every epsilon)
                with torch.no_grad():
                    diffs = [(_new_logp(policy, it, device).cpu() - it["old"]).abs() for it in items]
                allp = torch.cat([d.flatten() for d in diffs])
                save_json(record.dir / "step0_check.json", {
                    "per_rollout": [
                        {"cache_index": it["cache_index"], "prompt_tokens": it["prompt_width"],
                         "mean_abs_logp_diff": float(d.mean()), "max_abs_logp_diff": float(d.max())}
                        for it, d in zip(items, diffs)
                    ],
                    "mean_abs_logp_diff": float(allp.mean()), "max_abs_logp_diff": float(allp.max()),
                    "fraction_tokens_abs_logratio_gt_0.01": float((allp > 0.01).float().mean()),
                })

            step_stats = []
            for epoch in range(int(cfg["ppo_epochs"])):
                for it in items:
                    new_logp = _new_logp(policy, it, device)
                    old, adv = it["old"].to(device), it["adv"].to(device)
                    with torch.no_grad():
                        g = _geometry(new_logp.detach(), old, adv, eps)
                    loss, _, _ = ppo_policy_loss(new_logp, old, adv, torch.ones_like(old), eps=eps)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    g.update({"epsilon": eps, "epoch": epoch, "cache_index": it["cache_index"], "grad_norm": float(gn)})
                    step_stats.append(g)
            with torch.no_grad():
                final = [_geometry(_new_logp(policy, it, device), it["old"].to(device), it["adv"].to(device), eps) for it in items]
            results[f"{eps:.2f}"] = {
                "during_updates_epoch1": _pool([s for s in step_stats if s["epoch"] == 0]),
                "during_updates_epoch2": _pool([s for s in step_stats if s["epoch"] == 1]),
                "end_of_batch_policy": _pool(final),
                "grad_norm_mean": float(np.mean([s["grad_norm"] for s in step_stats if np.isfinite(s["grad_norm"])])),
            }
            steps.extend(step_stats)
            print(f"eps {eps:.2f}: {results[f'{eps:.2f}']['end_of_batch_policy']}", flush=True)
            del policy, opt, params
            clear_gpu()
        write_jsonl(record.dir / "cached_steps.jsonl", steps)
        record.finish({"n_rollouts": len(items), "by_epsilon": results}, artifacts={"steps": "cached_steps.jsonl"})


def forks(config_path: str, only: float | None = None):
    cfg = load_yaml(config_path)
    eps_values = [float(e) for e in cfg["clip_values"]]
    if only is not None:
        if not any(abs(only - e) < 1e-9 for e in eps_values):
            raise SystemExit(f"--only {only} not in {eps_values}")
        eps_values = [only]
    kl = float(cfg["kl_beta"])
    for eps in eps_values:
        name = fork_name(eps, kl)
        adapter = run_ppo(config_path, updates=int(cfg["fork_updates"]), clip_epsilon=eps, kl_beta=kl, run_name=name)
        evaluate_policy(config_path, adapter=str(adapter), name=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--part", choices=["cached", "forks", "all"], default="all")
    ap.add_argument("--only", type=float, help="forks: run a single epsilon")
    args = ap.parse_args()
    if args.part in ("cached", "all"):
        cached_study(args.config)
    if args.part in ("forks", "all"):
        forks(args.config, args.only)


if __name__ == "__main__":
    main()
