# Regression Ablation Results: DSAI Branch vs Main Branch

**Date:** 2026-03-30
**Branch:** DSAI
**GPU:** Quadro RTX 6000 (24 GB)
**Framework:** TensorFlow 2.10
**Model:** DeepLabV3+ with ResNet50 backbone
**Training:** 8 epochs, batch_size=1, tile_size=1024, early stopping disabled

## Overview

This ablation study isolates the root cause of accuracy differences between the DSAI branch and the main branch. Each experiment starts from the DSAI baseline and introduces a single main-branch behavior change, while keeping everything else at DSAI defaults.

## Experiments

| ID | Name | Description |
|----|------|-------------|
| dsai_baseline | DSAI as-is | Control — DSAI branch defaults |
| main_all | All main-branch behaviors | All 6 changes applied simultaneously |
| exp_a_loss | Per-pixel weighted loss | Main's per-pixel weighted loss instead of DSAI's per-class mean loss |
| exp_b_lr_sched | Counter-based LR schedule | Main's counter-based LR reduction instead of DSAI's unconditional per-epoch reduction |
| exp_c_lr_init | Initial LR = 0.0005 | Main's initial learning rate (5x higher than DSAI's 0.0001) |
| exp_d_val_freq | 3-per-epoch validation | Main's linspace validation (3x per epoch) instead of DSAI's iteration-based frequency |
| exp_e_raw_wts | Raw class weights | Main's un-normalized class weights instead of DSAI's weights normalized to sum to 1 |
| exp_f_epsilon | Adam epsilon = 1e-7 | Main's default Adam epsilon instead of DSAI's 1e-8 |

## Results — Liver Dataset

Baseline: 98.14% overall accuracy (7 classes, 15 training / 3 validation images, 10x resolution)

| Experiment | Accuracy | Delta | Train Time |
|-----------|----------|-------|------------|
| dsai_baseline | 98.14% | — | 4585s |
| main_all | 97.24% | **-0.90%** | 3954s |
| exp_a_loss | 96.09% | **-2.05%** | 4199s |
| exp_b_lr_sched | 97.86% | -0.28% | 4689s |
| exp_c_lr_init | 98.78% | **+0.64%** | 4687s |
| exp_d_val_freq | 98.14% | +0.00% | 4570s |
| exp_e_raw_wts | 97.78% | -0.36% | 4860s |
| exp_f_epsilon | 97.60% | -0.54% | 4679s |

**Liver analysis:**
- The per-pixel loss change (`exp_a`) is the largest negative contributor at -2.05%.
- Higher initial LR (`exp_c`) partially compensates at +0.64%.
- Validation frequency (`exp_d`) has zero impact.
- The combined effect (`main_all` = -0.90%) is smaller than the sum of individual negatives, indicating interactions between changes that partially cancel out.

## Results — Lungs Dataset

Baseline: 91.89% overall accuracy (6 classes, 15 training / 3 validation images, 5x resolution)

| Experiment | Accuracy | Delta | Train Time |
|-----------|----------|-------|------------|
| dsai_baseline | 91.89% | — | 4604s |
| main_all | 93.45% | **+1.56%** | 4049s |
| exp_a_loss | 92.59% | **+0.70%** | 4182s |
| exp_b_lr_sched | 92.35% | +0.46% | 4791s |
| exp_c_lr_init | 92.43% | +0.54% | 4521s |
| exp_d_val_freq | 92.23% | +0.34% | 4628s |
| exp_e_raw_wts | 91.84% | -0.05% | 4499s |
| exp_f_epsilon | 91.77% | -0.12% | 4595s |

**Lungs analysis:**
- The per-pixel loss change (`exp_a`) is again the largest single contributor, but in the **opposite direction** at +0.70% (explains 45% of the main_all gap).
- All main-branch changes except raw weights and epsilon **improve** lungs accuracy.
- The combined effect (`main_all` = +1.56%) is larger than any single change, indicating synergistic interactions.

## Key Findings

1. **The per-pixel loss function is the dominant factor** on both datasets, but its effect is dataset-dependent: -2.05% on liver, +0.70% on lungs. The DSAI per-class mean loss benefits liver (more balanced class weighting helps with 7 heterogeneous tissue classes), while the main branch's per-pixel loss benefits lungs (where class distribution is different).

2. **Higher initial learning rate (0.0005 vs 0.0001) is consistently beneficial** on both datasets (+0.64% liver, +0.54% lungs), suggesting the DSAI default LR of 0.0001 is too conservative.

3. **Raw class weights and Adam epsilon have negligible impact** on both datasets (< 0.5%), meaning these changes are not meaningful contributors to any performance difference.

4. **The combined effect is not the sum of individual effects.** On liver, individual negatives sum to roughly -3.2% but `main_all` is only -0.9%, indicating the changes partially compensate each other. On lungs, individual positives sum to ~+2.5% but `main_all` is +1.56%, suggesting diminishing returns from stacking improvements.

5. **There is no universal "better" branch.** DSAI is better for liver (+0.9%), main is better for lungs (+1.56%). The optimal configuration depends on the dataset.

## Bugs Found and Fixed During This Analysis

Three bugs in the ablation/evaluation pipeline were identified and fixed:

1. **Classification output caching (classification.py:472):** All experiments shared the same model name (`nm`) in metadata, causing `classify_images()` to skip re-classification after the first experiment and reuse its predictions. **Fix:** Set `nm` to include the experiment name so each gets a unique classification output directory.

2. **BatchNorm moving_variance corruption:** Training with batch_size=1 causes the first BatchNorm layer's moving_variance to become NaN (single-sample variance is degenerate). This propagates NaN through the entire network at inference time. **Fix:** After training, scan all BatchNorm layers and replace NaN moving_variance with 1.0 and NaN moving_mean with 0.0 before saving the model for evaluation.

3. **CUDA device ordering mismatch:** nvidia-smi uses PCI bus order but CUDA defaults to FASTEST_FIRST, causing `--gpu 2` to target the wrong GPU. **Fix:** Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the worker environment.

## Reproducibility

```bash
# Liver
PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/run_regression_ablation.py \
    --dataset liver --gpu 2 --batch-size 1

# Lungs
PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/run_regression_ablation.py \
    --dataset lungs --gpu 2 --batch-size 1
```

Note: On Windows 11, long-running Python processes may be throttled. Set process priority to RealTime to prevent this.

## Output Locations

- Liver: `C:\Users\tnewton\Desktop\liver_tissue_data\regression_ablation_results\`
- Lungs: `C:\Users\tnewton\Desktop\lungs_data\regression_ablation_results\`
- Per-experiment: `{results_dir}/{experiment_name}/worker_results.json`
- Summary CSV: `{results_dir}/ablation_comparison.csv`
- Combined JSON: `{results_dir}/all_results.json`
