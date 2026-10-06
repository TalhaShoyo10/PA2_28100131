# Understanding — implementation notes for the student

Learning and traceability notes. **Not report prose** — the report must be written by the student.

Validation for everything below: `python -m pytest tests/ -q` (CPU, tiny tensors, ~10 s).

---

## 1. DPO objective — `task1_dpo/dpo.py::dpo_loss`

**Concept.** DPO learns from fixed (prompt, preferred, rejected) triples with no reward model and no sampling. It raises the likelihood of y+ relative to y− — but *measured relative to the frozen reference policy*, so the model is credited only for what it changed, not for what the reference already believed.

**Math (manual p.4).**
```
m(x) = [log πθ(y+|x) − log πref(y+|x)] − [log πθ(y−|x) − log πref(y−|x)]
     = (log πθ(y+) − log πθ(y−)) − (log πref(y+) − log πref(y−))
     =        policy_margin       −          ref_margin
L    = −mean( log σ(β · m) )
```
Inputs are shape `[B]`: one *sequence* log-prob per example, the sum of response-token log-probs (prompt tokens excluded via `response_mask`, see `common/generation.py::response_sequence_logprobs`). β scales how sharply the sigmoid saturates — small β needs a big margin change to reduce loss (weak preference pressure, policy can drift further before the loss stops pushing); large β saturates quickly.

**Defect.** The starter used `β·(policy_margin + ref_margin)`.
- Original behavior: at initialization (πθ = πref) the logit is `2β·ref_margin`, not 0, so the loss is not log 2 and pairs the reference already ranks correctly look "already learned".
- Requirement: the logit must be zero when πθ = πref.
- Correction: `β·(policy_margin − ref_margin)`.
- Tests: `test_dpo_matches_manual_equation`, `test_dpo_equals_log2_when_policy_equals_reference` (both fail on the starter), `test_dpo_gradient_raises_chosen_and_lowers_rejected` (sign guard).
- Affected experiments: every Task 1 run (standard, β forks, length-balanced) and therefore the Task 4 DPO policy.

**How to sanity-check a real run.** First-step loss ≈ log 2 ≈ 0.693 and preference accuracy (m > 0) starts near 0 (ties at m = 0 count as not > 0). If the first-step loss is far from 0.693, the reference log-probs are wrong (e.g. LoRA not disabled when computing them).

---

## 2. PPO clipped surrogate — `task2_ppo/ppo.py::ppo_policy_loss`

**Concept.** PPO reuses a batch sampled from an *old* policy snapshot for several gradient steps. The ratio ρt = πθ/πold re-weights old samples; clipping stops the policy from exploiting one batch by moving too far. This is a *per-batch trust region*, separate from the reference-KL penalty (`shaped_rewards`), which is a *long-horizon* pull toward πref.

**Math (manual p.5).**
```
ρt = exp(log πθ(at|st) − log πold(at|st))
L_clip = E_t[ min( ρt·At , clip(ρt, 1−ε, 1+ε)·At ) ]       loss = −L_clip (masked token mean)
```
Shapes: `new_logp, old_logp, advantage, mask` are `[B, T_response]`.
- A > 0: objective = min(ρA, (1+ε)A) → gain capped once ρ > 1+ε, and the gradient becomes 0.
- A < 0: objective = min(ρA, clip(ρ)A) → when ρ < 1−ε the *more negative* (clipped) value is taken, so the policy is not rewarded for pushing the probability down further. (When ρ grows on a bad action the unclipped, more-negative term is kept: penalties are never capped.)

**Defect.** The starter used `torch.maximum`.
- Original behavior: optimistic bound — for A > 0 the unclipped ρA always wins, so there is no trust region at all.
- Correction: `torch.minimum`.
- Tests: `test_ppo_positive_advantage_is_capped_at_one_plus_eps`, `test_ppo_negative_advantage_takes_pessimistic_bound`, `test_ppo_no_gradient_once_clipped_in_the_improving_direction` (fail on starter); `test_ppo_clip_fraction_counts_tokens_outside_band_under_mask` (guard: clip fraction = tokens with ρ outside [1−ε,1+ε] *before* clipping, padding excluded — manual p.3 definition).
- Affected experiments: all Task 2 runs; the cached-rollout ε study's surrogate values.

**Still to validate when writing the loop:** GAE (`compute_gae`) and KL shaping (`shaped_rewards`) are not flagged as defective but will be tested against hand-computed trajectories before the PPO loop is trusted.

---

## 3. GRPO group-relative advantage — `task3_grpo/grpo.py::group_relative_advantages`

**Concept.** GRPO drops the critic. For each prompt it samples K completions and asks "was this completion better or worse *than its siblings for the same prompt*?" The baseline is the group mean, so prompt difficulty cancels out.

**Math (manual p.7).**
```
A_k = (r_k − μ_g) / (σ_g + ε),    μ_g, σ_g over the K completions of prompt g
```
Inputs: `rewards [N]`, `group_ids [N]` (N = prompts × K). Output: one scalar advantage per completion, broadcast over its tokens in `grpo_policy_loss`.
- If all K rewards in a group are equal, σ_g = 0 → every A_k = 0 → that prompt contributes **no policy gradient**. This is an *uninformative group*; its frequency is a required Task 3 metric.

**Defect.** The starter computed μ, σ over the whole batch, ignoring `group_ids`.
- Original behavior: an uninformative group (e.g. all rewards 10) got large non-zero advantages just because other prompts had lower rewards — prompt difficulty leaked into the update, and the critic-free baseline was destroyed.
- Correction: compute μ, σ separately per unique group id.
- Tests: `test_grpo_advantages_are_computed_within_each_group`, `test_grpo_advantages_sum_to_zero_per_group_with_interleaved_ids` (fail on starter; second one also checks a reward shift in one group leaves all advantages unchanged, and that non-contiguous ids work).
- Affected experiments: all Task 3 runs; any K-study code that reuses this helper.

**Decision → Reason → Constraint → Effect → Validation**
- *Decision:* denominator `σ_g + ε` (manual form) instead of the starter's `max(σ_g, ε)`; keep population std (`unbiased=False`) from the release.
- *Reason:* manual is the source of truth and writes σ+ε explicitly.
- *Constraint:* manual p.7 equation; release eps = 1e-6.
- *Effect:* numerically negligible (relative difference ≤ 1e-6 for σ ≫ ε); both give exactly 0 for zero-std groups.
- *Validation:* tests compare against `(r−μ)/(σ+1e-6)`.

**Normalization (not a defect, Task 3 Step 3).** `loss_type="grpo"` divides each sequence's token sum by its own length T_k (each completion gets equal total weight → per-token gradient ∝ 1/T_k, short completions' tokens weigh more). `loss_type="dr_grpo"` divides by the constant `max_completion_length` (each token gets equal weight → long completions carry more total gradient). This is what the normalization study measures.

---

## 4. DPO training loop — `task1_dpo/train.py::run_training`

**Data flow per micro-batch** (batch_size = 2 pairs):
1. `make_collate` → `encode_prompt_response` builds `prompt + response + EOS` token ids and a
   `response_mask` that is 1 only on response tokens. Chosen and rejected are padded separately
   (left padding), giving two dicts of `[2, L]` tensors.
2. `dpo_forward`:
   - policy pass (LoRA active, dropout on, gradients on) → `pc, pr` = summed response-token log-probs `[2]`;
   - reference pass inside `reference_mode` (LoRA **disabled**, eval mode, `no_grad`) → `rc, rr`.
     The reference is the same base weights without the adapter, so no second 1.5B model is needed.
   - `dpo_loss(pc, pr, rc, rr, β)`.
3. Gradient accumulation: 8 micro-batches = 16 pairs per optimizer update. Each micro-loss is divided
   by the real window size, so the final shorter window is not under-weighted.
4. Update: unscale → clip grad-norm to 1.0 (config) → AdamW step (lr 2e-5, constant — the config
   defines no schedule).

**Why the reference must have the adapter disabled.** If the reference pass used the LoRA weights,
rc = pc and m ≡ 0: the loss would be log 2 forever and gradients would vanish.

**Budget.** Standard: 1 epoch over the filtered 1,500-pair file (~1,446 pairs → ~91 updates).
β forks: first 600 filtered pairs (38 updates). Length-balanced: 1 epoch over the filtered
balanced file, with the same settings as standard. So standard and β forks have **different budgets**
(manual requires saying so); standard and length-balanced are matched.

**Logged per update** (`train_log.jsonl`): loss, train preference accuracy, chosen/rejected implicit
rewards β(log πθ − log πref), pre-clip grad norm, skipped-overflow flag, lr, elapsed time, peak VRAM.

**Decision record — long prompts**
- *Decision:* skip pairs whose prompt alone is ≥ 768 tokens (standard train 54/1500, balanced train
  58/1500, held-out 10/300, stratified 9/246); save the excluded IDs.
- *Reason:* released `max_sequence_length: 768` plus the released `encode_prompt_response`, which
  refuses rather than truncates such prompts; approved by the student 2026-10-07.
- *Constraint:* config constants unchanged; the same rule applies to every condition and eval set.
- *Expected effect:* slightly smaller fixed sets (~4%), biased toward excluding very long prompts.
- *Validation:* `dropped_long_prompts.json` in every run; counts appear in each manifest.

**Decision record — fp16 on T4**
- *Decision:* GradScaler (dynamic loss scaling) for training; the LoRA weights are fp32.
- *Reason:* T4 has no bf16; fp16 activation gradients can underflow. PEFT keeps LoRA weights in
  fp32 over the fp16 base (asserted at start).
- *Effect:* mathematically the same objective; overflow steps are skipped and counted
  (`step_skipped_overflow`).

**Decision record — shuffle generator**
- A `torch.Generator` seeded with the config seed drives the DataLoader shuffle, so every condition
  sees the same example order regardless of other RNG use.

---

## 5. Task 1 evaluation protocol — `task1_dpo/evaluate.py`

Same protocol for every condition (SFT, standard, β forks, length-balanced):

| Quantity | How | File |
|---|---|---|
| Held-out DPO loss / preference accuracy | teacher forcing on the filtered `dpo_standard_eval`; m > 0 with reference adjustment; loss at the run's own β | `pairs_heldout.jsonl` |
| Length-stratified accuracy | same, on `dpo_length_stratified_eval`, split by `length_stratum` | `pairs_length_stratified.jsonl` |
| KL from the reference | sample one response per held-out prompt (T=0.7, top-p 0.9, max 256 new tokens, seed reset), then Σ(log πθ − log πref) over response tokens; primary = token mean over all tokens (course helper convention), also the per-sequence mean | `generations_heldout.jsonl` |
| Reward-model score | course RM on the same generated responses | same |
| Length | response tokens: mean, std, median, IQR; truncation rate | same |
| Word-limit compliance | 10 fixed prompts × 5 samples, same decoding; `word_count ≤ parsed limit` | `generations_word_limit.jsonl` |

**SFT note:** with no adapter the policy *is* the reference, so m = 0 (accuracy 0 by the strict
m > 0 rule, loss = log 2) and KL = 0 by definition. SFT is useful for reward, length, and
word-limit baselines only.

**Decision record — 5 samples per word-limit prompt:** the prompt set is fixed at 10 prompts; one
sample each would make compliance move in steps of 10%. Five samples under the same decoding
lower the variance without changing the prompt set or decoding settings.
