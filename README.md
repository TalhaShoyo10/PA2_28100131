# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

```bash
python -m pytest tests/ -q                                                    # objective validation
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard      # Step 1: one epoch
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --name sft             # reference point (SFT)
python -m task1_dpo.ablate_beta --config configs/dpo.yaml                     # Step 2: beta 0.03/0.10/0.30 (or --only <beta>)
python -m task1_dpo.analyze_length --config configs/dpo.yaml                  # Step 3: length-balanced + comparison
python -m task1_dpo.summarize --config configs/dpo.yaml                       # results/task1_dpo/summary.csv
```

Pairs whose prompt alone is at least `max_sequence_length` (768) tokens are excluded from every Task 1 training and evaluation set by one rule; excluded IDs are saved in each run's `dropped_long_prompts.json`.

### Running on Colab

`colab/PA2_runner.ipynb` clones this repository, installs the pinned environment, downloads/validates assets, runs the tests, and executes the commands above. `outputs/` and `results/` are linked to a shared Google Drive folder so runs can continue across Colab accounts; every experiment records its status, git commit, command, configuration, runtime, and metrics under `results/<task>/<experiment_id>/`.

### Task 2 - PPO

```bash
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --part cached       # Step 2a: cached-batch geometry
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard     # Step 1: 20 updates
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --part forks        # Step 2b: eps forks (or --only <eps>)
python -m task2_ppo.ablate_kl --config configs/ppo.yaml                            # Step 3: beta_KL forks (or --only <beta>)
python -m task2_ppo.summarize --config configs/ppo.yaml                            # summary.csv, standard_trajectory.csv, cached_clipping.csv
```

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard     # Step 1: 20 updates, K = 4
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml                     # Step 2: K = 2/4/8 from the cache
python -m task3_grpo.compare_normalization --config configs/grpo.yaml                  # Step 3: grpo vs dr_grpo forks (or --only)
python -m task3_grpo.summarize --config configs/grpo.yaml
```

### Task 4 - Safety calibration

The judge loader/parser are supplied. You must implement the requested generation aggregation and evaluation.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml   # greedy, all 4 fixed policies (or --policy <name>)
python -m task4_safety.judge_responses --config configs/feedback.yaml      # fixed AI judge (or --policy <name>)
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml     # blind sheet: 60 fixed prompts x 4 policies
# the student labels audit_sheet_blind.csv by hand (task4_safety/audit_labeler.html, offline) -> manual_audit_labels.csv
python -m task4_safety.evaluate_safety --config configs/feedback.yaml      # rates, categories, agreement
```

Generation refuses to start unless the policy's standard (Step 1) training run is finished, so ablation checkpoints cannot be used.

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

RLVR and RLAIF are the supplied frozen adapters (evaluation only). All three policies use identical greedy decoding (`math_max_new_tokens`); outputs are in `results/task5_feedback/` (`feedback_comparison.csv`, `diagnostics_table.csv`, `qualitative_candidates.csv`).

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.

## 7. Running on Colab and validating results

All experiments were run on Google Colab (T4) through `colab/PA2_runner.ipynb`, which clones this repository at a
commit, installs `requirements.txt`, downloads and validates the course assets, runs `tests/`, and then executes the
commands in Section 5. Shared state lives on Google Drive:

- `outputs/` (trained adapters) is a symlink to Drive;
- results are written to Drive via `PA2_RESULTS_ROOT` and later copied into `results/` in this repository.

Every experiment directory `results/<task>/<experiment_id>/` contains `status.json`, `run_manifest.json` (git commit,
command, seed, runtime, timestamps, wall clock, peak VRAM), `config.json`, `command.txt`, `metrics.json`, and its
generations/logs. A run that was interrupted is archived as `<experiment_id>__attemptN`, never overwritten.

```bash
python -m pytest tests/ -q                 # objective validation (12 tests)
python -m scripts.validate_results         # checks every committed result folder before commit
python -m scripts.release_run <exp_id>     # marks a run from a lost Colab session as failed so it can restart
```

Implementation notes and decision records: `understanding.md`. Status of every required experiment: `progress.md`.

## 8. Attribution

- Starter code, configurations, fixed data, cached diagnostics, and checkpoints: course release
  (`AbDu11aHHH/ATML-PA2-LLM-PostTraining`, assets `AbDu11aHHH/ATML-PA2-assets` at revision `0b35048`).
- Libraries: PyTorch, Hugging Face Transformers, PEFT, TRL (pinned in `requirements.txt`), pandas, NumPy.
- Models: `Qwen/Qwen2.5-1.5B-Instruct` (policy), `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback` (reward model),
  `Qwen/Qwen2.5-3B-Instruct` (AI judge), supplied PPO/GRPO/RLVR/RLAIF adapters.
- No external code was materially copied. Implementations follow the equations in the assignment manual. Coding
  assistance from an LLM (Claude) was used for code, as permitted by the course policy; the student is responsible for
  every submitted line, and `understanding.md` documents what each component does and why.
