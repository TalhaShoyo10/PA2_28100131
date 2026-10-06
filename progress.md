# Progress

## Setup (CLAUDE.md §8)
- [x] Starter repo cloned — commit `1d64ac65acd5e45d1e4e1f415edc80455e21274e`
- [x] Complete manual read (15 pages)
- [x] Repo tree, configs, task scripts inspected
- [ ] Pinned environment installed (local machine has no NVIDIA GPU; `peft`/`trl` missing locally) — run on Colab
- [ ] `python -m scripts.download_assets` (~1.07 GB, revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`)
- [ ] `python -m scripts.validate_assets` passes
- [ ] `python -m scripts.check_environment` recorded
- [x] `.gitignore` extended (`*.pt`, `*.pth`, `*.safetensors`, `.env` per manual appendix)
- [x] Student's own public GitHub remote configured (`origin` = TalhaShoyo10/PA2_28100131, `upstream` = course repo)

## Deliberate defects — corrected and validated (`tests/test_objectives.py`, see `understanding.md`)

| Task | Location | Starter behavior | Manual requirement |
|---|---|---|---|
| 1 DPO | `task1_dpo/dpo.py:32-34` | `logits = β·(policy_margin + ref_margin)` | `β·(policy_margin − ref_margin)` |
| 2 PPO | `task2_ppo/ppo.py:56` | `torch.maximum(surr1, surr2)` (pessimistic bound inverted) | `min(ρA, clip(ρ)A)` |
| 3 GRPO | `task3_grpo/grpo.py:15-17` | mean/std over the whole batch, ignores `group_ids` | per-prompt-group mean/std: `(r_k − μ_group)/(σ_group + ε)` |

All 9 objective tests fail 7/9 on the starter code and pass 9/9 after correction (verified locally on CPU, 2026-10-07).

Other items to verify (not confirmed defects):
- `common.metrics.preference_accuracy(a, b)` has no reference term; it must be called with reference-adjusted log-ratios.
- `task1_dpo/dpo.py` diagnostic `preference_accuracy` uses `policy_margin − ref_margin` (correct), unlike its loss line.
- GRPO loss uses the k3 KL estimator internally; reported KL must use the common `sampled_kl` helper consistently across conditions.
- `scripts/download_assets.py` does not call `prepare_transfer_eval`; the HF bundle already ships `data/math_transfer_eval.jsonl` — confirm via `validate_assets`.

## Task 1
- [x] objective validated
- [ ] standard DPO (1 epoch)
- [ ] beta study (0.03 / 0.10 / 0.30, 600 examples each)
- [ ] length-balanced DPO + stratified eval + word-limit compliance
- [ ] required metrics
- [ ] qualitative evidence

## Task 2
- [x] objective validated
- [ ] standard 20-update continuation (+ VRAM, wall time)
- [ ] cached clipping diagnostic + matched 8-update forks (ε 0.05/0.20/0.50)
- [ ] KL forks (βKL 0/0.10/0.20)
- [ ] qualitative evidence

## Task 3
- [x] objective validated
- [ ] standard 20-update continuation (+ VRAM, wall time)
- [ ] K-study from cache (2/4/8, equal generations, difficulty bins fixed once)
- [ ] canonical vs Dr.-GRPO forks
- [ ] qualitative evidence

## Task 4 (only after Task 1–3 standard policies exist)
- [ ] deterministic generation, 4 fixed policies
- [ ] AI judge labels + aggregate rates
- [ ] blind audit sheet generated
- [ ] student manual labels (60) — student only
- [ ] agreement / confusion analysis

## Task 5
- [ ] GSM8K in-domain eval
- [ ] diagnostic set (Sreason, Soutcome, per-category)
- [ ] SVAMP transfer eval

## Task 6
- [ ] cross-task evidence assembled
- [ ] report evidence map complete
