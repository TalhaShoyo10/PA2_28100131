"""Shared online-RL plumbing for Task 2 (PPO) and Task 3 (GRPO).

- prompt filtering: prompts longer than max_prompt_length are skipped and recorded, because
  common.generation.batch_generate truncates from the right and would cut the assistant header
  (same rule as Task 1, approved by the student 2026-10-07)
- a fixed, seeded prompt schedule shared by every condition of a task
- token-level log-probs for the policy and the frozen reference (adapter disabled)
- learned-reward scoring with the configured missing-EOS penalty
- the common held-out evaluation protocol for PPO/GRPO policies
"""
from __future__ import annotations

import numpy as np
import torch

from common.data import prompt_messages, read_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import length_stats
from common.models import reference_mode


def prompt_length(tokenizer, row) -> int:
    return len(tokenizer.apply_chat_template(prompt_messages(row), tokenize=True, add_generation_prompt=True))


def filter_prompt_rows(rows, tokenizer, max_prompt_length: int):
    kept, dropped = [], []
    for row in rows:
        n = prompt_length(tokenizer, row)
        if n <= max_prompt_length:
            kept.append(row)
        else:
            dropped.append({"prompt_id": row.get("prompt_id"), "source_index": row.get("source_index"), "prompt_tokens": n})
    return kept, dropped


def prompt_schedule(rows, seed: int, n: int):
    """First n prompts of one seeded permutation. Every condition (standard run and all forks)
    uses this same list, so a fork with fewer updates sees a prefix of the standard prompts."""
    if n > len(rows):
        raise ValueError(f"need {n} prompts, only {len(rows)} available")
    perm = np.random.default_rng(seed).permutation(len(rows))
    return [rows[int(i)] for i in perm[:n]]


def disable_dropout(model) -> None:
    """Make the old-policy and new-policy log-probs come from the same deterministic function,
    so the PPO/GRPO ratio is exactly exp(0) = 1 before the first gradient step."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


@torch.no_grad()
def policy_and_reference_logprobs(policy, gen: dict, has_adapter: bool = True, score_batch: int = 2):
    """Token log-probs of the generated responses under the policy and the reference. [B, T] fp32."""
    pol, ref = [], []
    for s in range(0, gen["sequences"].shape[0], score_batch):
        sl = slice(s, s + score_batch)
        args = (gen["sequences"][sl], gen["attention_mask"][sl], gen["prompt_width"], gen["response_ids"][sl])
        p = response_token_logprobs(policy, *args)[0]
        if has_adapter:
            with reference_mode(policy):
                r = response_token_logprobs(policy, *args)[0]
        else:
            r = p
        pol.append(p.float())
        ref.append(r.float())
    return torch.cat(pol), torch.cat(ref)


@torch.no_grad()
def learned_rewards(reward_bundle, prompts, responses, terminated, cfg):
    rm, rm_tok = reward_bundle
    raw = score_reward_pairs(rm, rm_tok, prompts, responses, max_length=int(cfg.get("reward_max_length", 1280))).float().cpu()
    penalty = float(cfg.get("missing_eos_penalty", 0.0))
    eff = raw - penalty * (~torch.as_tensor(terminated, dtype=torch.bool)).float()
    return raw, eff


def generate(policy, tokenizer, messages, cfg, max_new_tokens: int):
    g = cfg["generation"]
    return batch_generate(
        policy, tokenizer, messages,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(max_new_tokens),
        temperature=float(g["temperature"]), top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]),
    )


@torch.no_grad()
def evaluate_rl_policy(policy, tokenizer, reward_bundle, rows, cfg, has_adapter: bool, max_new_tokens: int, gen_batch: int = 16):
    """Common held-out protocol: one sampled response per prompt (seed reset), learned reward,
    sampled KL to the reference, sampled-token entropy, and length statistics."""
    lens = [prompt_length(tokenizer, r) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: (lens[i], str(rows[i]["prompt_id"])))
    set_seed(int(cfg["seed"]))
    records = []
    for start in range(0, len(order), gen_batch):
        batch = [rows[i] for i in order[start:start + gen_batch]]
        msgs = [prompt_messages(r) for r in batch]
        gen = generate(policy, tokenizer, msgs, cfg, max_new_tokens)
        pol, ref = policy_and_reference_logprobs(policy, gen, has_adapter)
        mask = gen["response_mask"].float()
        raw, eff = learned_rewards(reward_bundle, msgs, gen["responses"], gen["terminated_with_eos"], cfg)
        for j, r in enumerate(batch):
            m = mask[j]
            records.append({
                "prompt_id": r["prompt_id"],
                "response": gen["responses"][j],
                "response_tokens": int(gen["response_lengths"][j]),
                "terminated_with_eos": bool(gen["terminated_with_eos"][j]),
                "truncated": bool(gen["truncated"][j]),
                "reward_raw": float(raw[j]),
                "reward_effective": float(eff[j]),
                "kl_sum": float(((pol[j] - ref[j]) * m).sum()),
                "neg_logp_sum": float((-pol[j] * m).sum()),
                "tokens": float(m.sum()),
            })
        print(f"  evaluated {min(start + gen_batch, len(order))}/{len(order)}", flush=True)
    return records


def summarize_rl_eval(records) -> dict:
    tokens = sum(r["tokens"] for r in records)
    raw = [r["reward_raw"] for r in records]
    eff = [r["reward_effective"] for r in records]
    return {
        "n_prompts": len(records),
        "reward_raw_mean": float(np.mean(raw)),
        "reward_raw_std": float(np.std(raw)),
        "reward_effective_mean": float(np.mean(eff)),
        # token-level means over all valid response tokens (course helper convention)
        "kl_token_mean": sum(r["kl_sum"] for r in records) / max(tokens, 1.0),
        "kl_sequence_mean": float(np.mean([r["kl_sum"] for r in records])),
        "entropy_token_mean": sum(r["neg_logp_sum"] for r in records) / max(tokens, 1.0),
        "length_tokens": length_stats([r["response_tokens"] for r in records]),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "missing_eos_rate": float(np.mean([not r["terminated_with_eos"] for r in records])),
    }


def load_prompt_rows(cfg, key: str, tokenizer):
    rows = read_jsonl(cfg["paths"][key])
    return filter_prompt_rows(rows, tokenizer, int(cfg["max_prompt_length"]))
