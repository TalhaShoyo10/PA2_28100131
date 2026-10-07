"""Mathematical validation of the Task 1-3 core objectives against ATML PA2 manual equations.

Each test uses tiny hand-computable tensors on CPU. Every test here fails on the
released starter code and passes once the deliberate defect is corrected.

Run from the repository root:  python -m pytest tests/ -q
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards
from task3_grpo.grpo import group_relative_advantages


def t(*xs):
    return torch.tensor(xs, dtype=torch.float32)


# --------------------------------------------------------------------------------------
# Task 1 - DPO (manual p.4)
#   L = -log sigma( beta * ( [log pi(y+) - log ref(y+)] - [log pi(y-) - log ref(y-)] ) )
# --------------------------------------------------------------------------------------

def test_dpo_matches_manual_equation():
    pc, pr, rc, rr = t(-1.0), t(-3.0), t(-1.0), t(-4.0)
    beta = 0.1
    # Implicit-reward margin: (-1 - -1) - (-3 - -4) = 0 - 1 = -1
    expected = -F.logsigmoid(torch.tensor(beta * -1.0))
    loss, _ = dpo_loss(pc, pr, rc, rr, beta)
    assert torch.allclose(loss, expected, atol=1e-6), (loss, expected)


def test_dpo_equals_log2_when_policy_equals_reference():
    # At initialization pi_theta == pi_ref, so every log-ratio is 0 and loss = log 2,
    # regardless of how different the chosen/rejected likelihoods are.
    pc, pr = t(-5.0, -2.0, -40.0), t(-1.0, -9.0, -3.0)
    loss, diag = dpo_loss(pc, pr, pc.clone(), pr.clone(), beta=0.3)
    assert math.isclose(loss.item(), math.log(2.0), rel_tol=1e-6), loss.item()
    assert diag["preference_accuracy"].item() == 0.0  # m = 0 is not > 0


def test_dpo_gradient_raises_chosen_and_lowers_rejected():
    pc = t(-2.0).requires_grad_()
    pr = t(-2.0).requires_grad_()
    loss, _ = dpo_loss(pc, pr, t(-1.0), t(-3.0), beta=0.1)
    loss.backward()
    assert pc.grad.item() < 0  # gradient descent increases log pi(y+)
    assert pr.grad.item() > 0  # gradient descent decreases log pi(y-)


# --------------------------------------------------------------------------------------
# Task 2 - PPO clipped surrogate (manual p.5)
#   L_clip = E_t[ min( rho_t A_t, clip(rho_t, 1-eps, 1+eps) A_t ) ];  loss = -L_clip
# --------------------------------------------------------------------------------------

def _ppo(ratio, adv, eps=0.2):
    new_logp = torch.log(t(*ratio)).unsqueeze(0)
    old_logp = torch.zeros_like(new_logp)
    a = t(*adv).unsqueeze(0)
    mask = torch.ones_like(a)
    return ppo_policy_loss(new_logp, old_logp, a, mask, eps=eps)


def test_ppo_positive_advantage_is_capped_at_one_plus_eps():
    loss, _, _ = _ppo([1.5], [2.0])
    # min(1.5*2, 1.2*2) = 2.4
    assert torch.allclose(loss, torch.tensor(-2.4), atol=1e-5), loss


def test_ppo_negative_advantage_takes_pessimistic_bound():
    loss, _, _ = _ppo([0.5], [-2.0])
    # min(0.5*-2, 0.8*-2) = min(-1.0, -1.6) = -1.6
    assert torch.allclose(loss, torch.tensor(1.6), atol=1e-5), loss


def test_ppo_no_gradient_once_clipped_in_the_improving_direction():
    new_logp = torch.log(t(1.5)).unsqueeze(0).requires_grad_()
    loss, _, _ = ppo_policy_loss(new_logp, torch.zeros(1, 1), t(1.0).unsqueeze(0), torch.ones(1, 1), eps=0.2)
    loss.backward()
    assert new_logp.grad.abs().item() < 1e-8, new_logp.grad


def test_ppo_clip_fraction_counts_tokens_outside_band_under_mask():
    new_logp = torch.log(t(1.0, 1.5, 0.5, 3.0)).unsqueeze(0)
    mask = t(1.0, 1.0, 1.0, 0.0).unsqueeze(0)  # last token is padding
    _, _, frac = ppo_policy_loss(new_logp, torch.zeros_like(new_logp), torch.ones_like(new_logp), mask, eps=0.2)
    assert math.isclose(frac.item(), 2.0 / 3.0, rel_tol=1e-6), frac


def test_shaped_rewards_kl_cost_per_token_plus_terminal_reward():
    # learning_guide.md 3.3: log pi - log ref = [0.5, 0.1, -0.2], beta_kl = 0.1, RM = 1.0
    policy = t(0.5, 0.1, -0.2, 9.0).unsqueeze(0)
    ref = torch.zeros_like(policy)
    mask = t(1.0, 1.0, 1.0, 0.0).unsqueeze(0)  # 4th position is padding
    r = shaped_rewards(t(1.0), policy, ref, mask, beta_kl=0.1)
    assert torch.allclose(r, t(-0.05, -0.01, 1.02, 0.0).unsqueeze(0), atol=1e-6), r


def test_gae_matches_hand_computation_and_ignores_padding():
    # learning_guide.md 3.3: gamma = 1, lambda = 0.95
    rewards = t(-0.05, -0.01, 1.02, 0.0).unsqueeze(0)
    values = t(0.6, 0.8, 0.9, 5.0).unsqueeze(0)  # value at the padded position must be ignored
    mask = t(1.0, 1.0, 1.0, 0.0).unsqueeze(0)
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    assert torch.allclose(adv[0, :3], t(0.3438, 0.204, 0.12), atol=1e-4), adv
    assert adv[0, 3].item() == 0.0
    assert torch.allclose(ret[0, :3], t(0.9438, 1.004, 1.02), atol=1e-4), ret


def test_gae_with_lambda_one_is_return_minus_value():
    rewards = t(-0.05, -0.01, 1.02).unsqueeze(0)
    values = t(0.6, 0.8, 0.9).unsqueeze(0)
    adv, _ = compute_gae(rewards, values, torch.ones_like(rewards), gamma=1.0, lam=1.0)
    # Monte-Carlo: A_t = sum_{k>=t} r_k - V_t
    assert torch.allclose(adv, t(0.96 - 0.6, 1.01 - 0.8, 1.02 - 0.9).unsqueeze(0), atol=1e-5), adv


# --------------------------------------------------------------------------------------
# Task 3 - GRPO group-relative advantage (manual p.7)
#   A_k = (r_k - mu_r) / (sigma_r + eps), statistics computed WITHIN each prompt group
# --------------------------------------------------------------------------------------

def test_grpo_advantages_are_computed_within_each_group():
    rewards = t(1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0)
    gids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    adv = group_relative_advantages(rewards, gids)
    g0 = rewards[:4]
    expected0 = (g0 - g0.mean()) / (g0.std(unbiased=False) + 1e-6)
    assert torch.allclose(adv[:4], expected0, atol=1e-5), adv
    # Constant-reward group is uninformative: zero advantage, not a large offset.
    assert torch.allclose(adv[4:], torch.zeros(4), atol=1e-6), adv


def test_grpo_advantages_sum_to_zero_per_group_with_interleaved_ids():
    rewards = t(0.3, 5.0, -1.0, 7.0, 0.9, 6.5)
    gids = torch.tensor([2, 9, 2, 9, 2, 9])  # non-contiguous, non-sorted ids
    adv = group_relative_advantages(rewards, gids)
    for g in (2, 9):
        assert abs(adv[gids == g].sum().item()) < 1e-5
    # A reward shift applied to one group must not change any advantage.
    shifted = rewards.clone()
    shifted[gids == 9] += 100.0
    assert torch.allclose(group_relative_advantages(shifted, gids), adv, atol=1e-4)
