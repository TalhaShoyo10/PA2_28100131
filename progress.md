# Progress

Last updated: 2026-10-10. Due: 11 October 2026 (LMS).

## Setup (CLAUDE.md §8)
- [x] Starter repo cloned — commit `1d64ac65acd5e45d1e4e1f415edc80455e21274e`
- [x] Complete manual read (15 pages)
- [x] Repo tree, configs, task scripts inspected
- [x] Pinned environment installed **on Colab** (transformers 4.57.1, tokenizers 0.22.1, PEFT 0.17.1, TRL 0.27.2; Python 3.13, torch 2.11 cu130, Tesla T4). All execution runs on Colab (student decision); the local machine has no GPU.
- [x] `python -m scripts.download_assets` (revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`, ~1.07 GB)
- [x] `python -m scripts.validate_assets` passes (19/19 assets, schemas valid)
- [x] `python -m scripts.check_environment` recorded in the Colab logs
- [x] `.gitignore` extended (`*.pt`, `*.pth`, `*.safetensors`, `*.bin`, `.env`)
- [x] Student's public GitHub remote: `origin` = TalhaShoyo10/PA2_28100131, `upstream` = course repo
- [x] Colab runner `colab/PA2_runner.ipynb`; shared Drive state `MyDrive/ATML_PA2` (`outputs/` symlink, results via `PA2_RESULTS_ROOT`); per-run `status.json`; `scripts/release_run.py` for lost sessions; `scripts/validate_results.py` before every commit

## Deliberate defects — corrected and validated (`tests/test_objectives.py`, see `understanding.md`)

| Task | Location | Starter behavior | Manual requirement |
|---|---|---|---|
| 1 DPO | `task1_dpo/dpo.py` | `logits = β·(policy_margin + ref_margin)` | `β·(policy_margin − ref_margin)` |
| 2 PPO | `task2_ppo/ppo.py` | `torch.maximum(surr1, surr2)` (optimistic bound) | `min(ρA, clip(ρ)A)` |
| 3 GRPO | `task3_grpo/grpo.py` | mean/std over the whole batch, ignores `group_ids` | per-prompt-group `(r_k − μ_group)/(σ_group + ε)` |

12 tests: 7 fail on the starter code, all pass after correction (also on Colab). GAE and KL-shaping helpers (not defective) have hand-computed tests.

Other release issues found and fixed (no computational change to the objectives):
- `common.models.token_values` resolved the PEFT critic to the full classifier; now unwraps to the transformer body (needed for the fp32 value head).
- `batch_generate` returns inference-mode tensors; `common.rl.generate` clones them before PPO/GRPO updates.

## Task 1 — DPO ✅ complete, committed
- [x] standard DPO (1 epoch, 1446 pairs, 91 updates; 1 fp16-overflow skip at #78)
- [x] beta study (0.03 / 0.10 / 0.30, 600 pairs, 38 updates each; beta030 skips #2, #25)
- [x] length-balanced DPO (1442 pairs, 91 updates; skips #20, #90) + stratified eval + word-limit compliance
- [x] SFT evaluated under the same protocol (additional reference point)
- [x] results: `results/task1_dpo/summary.csv`, `length_comparison.csv`, `dataset_length_profile.json`; all 13 runs on commit c526ffb
- Note: grad-norm clipping (max 1.0) active on 0% of beta003 updates and 100% of all other runs' updates
- Audit (no rerun needed). Reporting caveats:
  - training responses truncated to fit 768 tokens: standard chosen 11.9% / rejected 9.2%, balanced ~2.2% (confounder for standard vs length-balanced);
  - 8/290 held-out pairs have a response cut to ≤2 tokens by a near-limit prompt;
  - ~42% of generations hit the 256-token cap in every condition;
  - long-prompt filter: pairs with prompt ≥768 tokens excluded (train 54/1500, balanced 58/1500, held-out 10/300, stratified 9/246).
- [ ] qualitative evidence (student selects from `pairs_*.jsonl` / `generations_*.jsonl` by a stated rule)

## Task 2 — PPO ✅ complete, committed
- [x] standard 20-update continuation (+ VRAM 7.19 GiB, wall clock 289 s) and midpoint eval
- [x] cached clipping study: `task2_ppo_cached_clipping_valid` (29/32 rollouts; step-0 clip fraction 9.9% / 0.11% / 0% for ε 0.05 / 0.20 / 0.50)
- [x] matched 8-update clipping forks: bit-identical across ε (max ratio 1.03 — clipping never binds at lr 3e-6)
- [x] KL forks (βKL 0 / 0.10 / 0.20); βKL 0.10 fork = ε 0.20 fork (shared run)
- [x] code: runs at 0c32299 or 7a6b99a (no computational difference); cached study at cd5c9a7
- Archived evidence: first cached run (3 rollouts beyond the rollout prompt cap), eval_fork_eps005 attempt1 (lost session), train_smoke attempt1
- Caveats: fp16 overflow skips 4/40 steps (standard), 3/16 (each fork, same pattern); critic weak by design (negative explained variance)
- [ ] qualitative evidence (reward/quality agree + disagree)

## Task 3 — GRPO ✅ complete, committed (commit c2fdf51)
- [x] standard 20-update continuation (+ VRAM 8.28 GiB, wall clock 390 s), midpoint eval; 0 overflow skips (scaler init 2^12)
- [x] K-study from cache (2/4/8, 192 generations each, difficulty tertiles fixed once); use `variance_centred_reward` (normalized-advantage variance equals the informative rate by construction)
- [x] canonical vs Dr.-GRPO forks; length-conditioned weights match theory (long-completion weight share 0.46 vs 0.94)
- Caveats: Dr-GRPO grad norm 0.04 vs 0.63 (step-size confounder); ~25% of completions truncated and masked (some updates fully masked → zero gradient)
- [ ] qualitative evidence

## Task 4 — Safety calibration (run; results on Drive, not yet committed)
- [x] greedy generation for the 4 fixed policies (standard runs only, enforced in code)
- [x] AI judge on 450 prompts × 4 policies; 0 parse failures
- [x] blind audit sheet: 60 fixed prompts × 4 policies = 240 items (student decision), labelled via `task4_safety/audit_labeler.html`
- [x] student manual labels → `manual_audit_labels.csv` (240/240 valid; 3 Trail of Tears items corrected by the student from OVER_REFUSAL to JUSTIFIED_REFUSAL after learning the historical context, before seeing any AI label)
- [x] evaluate_safety: agreement 0.775, κ 0.524 (SAFE prompts 0.658, UNSAFE 0.892); manual AMBIGUOUS 3.3%, judge 0%
- Finding to report: the judge never assigns OVER_REFUSAL to safe prompts (over-refusal rate 0.0 for all policies); ~29% of labels belong to the other class (`class_inconsistent_label_rate`) → judge failure mode, characterized by the audit
- [ ] student classifies `audit_disagreements.csv` (`disagreement_type`: judge_error / policy_difference / both)
- [ ] download results from Drive → validate → commit
- [ ] qualitative examples: harmful compliance (none found by the judge), justified refusal, exaggerated refusal

## Task 5 — RLVR vs RLAIF (code pushed, commit 0c424da; not yet confirmed run)
- [ ] GSM8K in-domain eval (`task5_gsm`)
- [ ] diagnostic set: S_reason, S_outcome, per pair type (`task5_diagnostics`)
- [ ] SVAMP transfer eval (`task5_transfer`)
- [ ] combine (`task5_compare`) → `feedback_comparison.csv`, `diagnostics_table.csv`, `qualitative_candidates.csv`
- [ ] download → validate → commit

## Task 6 — Synthesis (student)
- [ ] cross-task evidence assembled
- [ ] report evidence map complete

## Submission
- [ ] README attribution and final commands check
- [ ] final freeze commit; GitHub link at the end of the abstract
