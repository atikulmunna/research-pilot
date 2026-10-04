# Does label smoothing improve calibration of logistic regression and small MLPs beyond tuned L2 and post-hoc scaling? A protocol with preliminary, inconclusive pilot results

> **Status:** Draft with open issues (failing checks: central_claims_supported, claims_within_evidence, required_experiments_complete, reviewer_objections_addressed). 0 of 3 central claims are SUPPORTED by the evidence graph (3 red-team review round(s)).

**Plain-language summary.** This pilot study tested whether label smoothing helps logistic regression or small neural networks on tabular data after tuning regularization and temperature scaling. In synthetic experiments, tuned label smoothing showed no detectable log-loss advantage over tuned training without smoothing, and the study did not find evidence of smoothing-induced probability error; one deliberately constrained model was overconfident. These inconclusive results do not test the hypothesis about overconfident baselines and are limited to synthetic data and a few small datasets.

## Abstract

Label smoothing (LS) is often proposed as a calibration aid, but the evidence comes mainly from deep networks [P001, P042]. We ask whether LS helps logistic regression (LR) or small MLPs on tabular data once L2 strength and post-hoc temperature scaling are tuned. We specify three comparative and mechanistic questions with directional hypotheses (H0-H4) and report five completed runs. All hypotheses remain at evidence level HYPOTHESIS and every run is labelled inconclusive. The study is a protocol with pilot results, not a confirmatory test, and it has no frozen, timestamped pre-registration. The primary comparison (R001, synthetic LR and MLP cells that we defined ourselves, 20 seeds) found no detectable log-loss difference between best-tuned LS and best-tuned eps=0: +0.0009, 95% CI [-0.0006, 0.0025], Holm p=1.0. Both arms were better than raw eps=0 (LS minus eps=0 raw: -0.0120, CI [-0.0149, -0.0090]). A gate run (R002) found no positive signed error for tuned-L2 eps=0 LR in the pooled mean (-0.0027, CI [-0.0101, 0.0046]). A deliberately constrained-L2 arm was overconfident (0.0249, CI [0.0194, 0.0303]). The mechanism test (R003) did not show LS-induced probability error in any of 8 cells. Accuracy differences were small (R004). H1, which concerns regimes where the baseline is overconfident, is untested. Results are limited to synthetic data and a few small datasets, and several protocol deviations are disclosed.

## 1. Introduction

Label smoothing replaces one-hot targets with q = (1-eps)*onehot + eps/K. In deep networks it is reported to affect calibration, with a mixed picture of when it helps [P001]. Modern networks are often overconfident, and temperature scaling is a strong post-hoc remedy [P042]. Whether these findings carry over to small tabular classifiers is unclear. LR is a strong baseline in tabular and clinical prediction, where performance depends on sample size and data quality [P052, P053, P059]. L2 and LS are both regularisers [P017, P023], so LS should be compared against tuned L2 and post-hoc scaling, not against an unregularised model.

We frame three questions. Q-a asks whether LS ever beats tuned L2 plus post-hoc scaling, and in which regimes. Q-b asks whether LS causes underconfidence in well-specified LR, measured against ground-truth probabilities. Q-c asks whether the accuracy-calibration trade-off traced by eps is dominated by the one traced by L2 strength. The overconfidence premise is treated as an empirical question (H0). Practitioners would use the answers to decide whether to add an eps hyperparameter to an LR or small-MLP pipeline that already tunes L2 and temperature scaling. The v1 hypotheses H1 and H3 were superseded by v2 and no conclusions are drawn from them.

Contribution: a comparative, mechanistic framing with decision rules that make null results informative (C23). It comes with pilot runs, none of which tests the central regime-resolved claim.

## 2. Related Work

Calibration assessment. Surveys and metric papers motivate our metrics (log loss, Brier, ECE, signed confidence-minus-accuracy) [P018]. P012 introduces the Integrated Calibration Index for LR, a binning-free measure we have not yet added. P037 discusses how calibration should be compared and improved. Label smoothing and post-hoc scaling. P001 studies when LS helps. P042 documents overconfidence in modern networks and compares post-hoc remedies such as temperature scaling, which we adopt as the strong comparator. Corrections to LR. P051 reports that corrections applied to LR risk models can harm calibration, which motivates asking whether LS can cause underconfidence. Small tabular models. P052, P053 and P059 describe LR as a strong baseline whose performance depends on sample size and data quality. Surveys of regularisation and loss functions [P017, P023] frame L2 and LS as related; they concern deep networks and are loosely relevant here.

## 3. Method

Models. L2-regularised LR trained on the smoothed cross-entropy, with the smoothed target q above. The MLP is a custom numpy implementation, because sklearn's MLPClassifier has no LS. In R001 the MLP is trained for a fixed 150 epochs with L2 only and without early stopping.

Arms in R001. eps0_raw and ls_raw (no recalibration), eps0_cvts and ls_cvts (cross-fitted temperature scaling fitted on inner out-of-fold predictions, then the base model refit), ls_desmooth (inverting the smoothing, with clipping), and eps0_best_variant and ls_best_variant (raw or cvts chosen on inner validation). The primary contrast is ls_best_variant minus eps0_best_variant in test log loss on 20k fresh test points per replicate. Holm adjustment is used, but the family composition is not fully specified.

Hypotheses and rules. H0: tuned eps=0 LR is not systematically overconfident (signed mean(confidence - accuracy)). H1: LS helps where the tuned baseline is overconfident. H2: LS does not beat eps=0 plus temperature scaling. H3: in well-specified synthetic LR, LS increases error against the true probabilities, with harm growing with eps. H4: accuracy non-inferiority at -1 pp, with 'inconclusive' for CI width above 2 pp or ceiling accuracy.

Disclosed deviations from protocol (see also Limitations). R001: E1 cells replaced by self-defined synthetic cells; MLP without early stopping (fixed 150 epochs, L2 only); real-data analysis limited to breast_cancer, iris and wine with LR only; digits and real-data MLPs omitted; Nadeau-Bengio intervals not computed. R002: each invocation ran one pilot replicate per cell, so the 10-replicate gate statistics (one-sided t, bootstrap, 80/20 aggregation, 70/30 and 90/10 sensitivity) and cell labels had to be computed downstream; reserve cells were not run; a reviewer changed the solver before execution. R003: robustness tests (d=30 with n/d in {3,30}), sensitivity analyses (joint grid, oracle and Brier-based tuning) and the misspecified generator were not run; the G9/G9' grids are engineer-assumed. R004: bootstrap CIs, Holm adjustment and TOST/ceiling rules had to be computed downstream across seeds. R005: regimes R1-R5 were defined by the engineer; the model was refit; the bootstrap was over the 5 outer folds of one seed; fixed-eps and clip-floor robustness runs were deferred. R006 and R007: stand-in cells replaced the E1 cells; no R002 artefacts were supplied, so reproduction audits were not run (NaN entries); R007 used a 200,000-point reference sample and a simplified time guard.

## 4. Experimental Setup

All code is CPU-only Python (numpy, scikit-learn). Each experiment runs in under 10 minutes; R002 total elapsed time was about 56 s. Runs: R001 (E2, primary, seeds 0-19), R002 (E1, gate, seeds 1000-1009), R003 (E4, mechanism, seeds 0-19), R004 (E5, accuracy and Pareto, seeds 0-19), R005 (E3, de-smoothing ablation, seeds 0-2), R006 (E8) and R007 (E9) (validation and diagnostic, seeds 2000-2019). Metrics are log loss, Brier, 15-bin equal-mass top-label ECE, signed error versus true probability (synthetic) or confidence minus accuracy, and accuracy. R002 tables report n=8 per cell although 10 seeds were run; the reason for the difference is not documented in the supplied outputs. The full specification of generators, cells, grids, folds and regime thresholds (RT1-3, RT2-3) is not available in the evidence supplied and is therefore not reproduced here. A frozen configuration hash has not been created.

## 5. Results

All results are preliminary. Every run is labelled inconclusive.

Primary comparison (R001, Table 1). Test log loss was 0.4991 for eps0_best_variant and 0.5001 for ls_best_variant. The paired difference was +0.0009 (CI [-0.0006, 0.0025], Holm p=1.0). Against eps0_cvts the difference was -0.0007 (CI [-0.0026, 0.0011]). Against eps0_raw it was -0.0120 (CI [-0.0149, -0.0090], Holm p<0.0001). Brier and ECE contrasts against eps0_best_variant were also null (Brier -0.0004, CI [-0.0041, 0.0033]; ECE +0.0005, CI [-0.0039, 0.0048]). Raw ECE was 0.0460 for eps0_raw and 0.0332 for ls_raw, and about 0.028 for both best variants. Within LS, ls_cvts had log loss 0.4993 against 0.5001 for ls_best_variant, so inner-validation variant selection did not help LS here. De-smoothing was worse than raw LS (log loss 0.5176 against 0.5041), with 2.88% of entries clipped for ls_desmooth. This anomaly is not explained, and a clipping or implementation problem has not been excluded. The tuned eps=0 best variant had a signed error against the true probability of -0.0069 (CI [-0.0116, -0.0021]), so it was slightly underconfident. The ablation rows in Table 1 repeat identical values across arms and are treated as a reporting defect; we do not interpret them.

Gate (R002, Table 2). Pooled over cells, tuned-L2 eps=0 signed error was -0.0027 (CI [-0.0101, 0.0046]). The constrained-L2 arm was +0.0249 (CI [0.0194, 0.0303]), and the paired tuned-minus-constrained difference was -0.0276 (CI [-0.0360, -0.0192]). By cell, constrained-L2 CIs lay entirely above zero in C1, C5, C7 and D1 under accuracy-based error (C1 0.0612; C5 0.1376; C7 0.0544; D1 0.0262, with CI [0.0062, 0.0462]). Under the signed error versus truth, C1, C5, C7 and D1 also lay above zero. The earlier summary named three cells (C1, C5, C7), so D1 should be reconciled. Under tuned L2, no cell had a CI entirely above zero in the SO set. The fires indicators nevertheless equalled 1.0 for C1, C5 and C7 and for most MLP256 cells; their definition is not given here and they conflict with the signed-error summary. Implementation checks: fam1_check_pass had mean 0.8, so some replicates failed it; how those replicates were handled is not documented, and the CI [0.4984, 1.1016] exceeds 1 because a normal approximation was used.

Mechanism (R003, Table 3). The primary paired MSE difference (ls_best_lr minus eps0_best_lr) was -0.0010 (CI [-0.0024, 0.0005], Holm p=0.9731). No n x signal cell had a CI entirely below 0. Point estimates were negative at n30 and n100 and positive at larger n; all of those intervals spanned 0 except those for fixed-eps arms. At n3000_b4, best-tuned LS had higher MSE (+0.0013, CI [0.0011, 0.0014]) and signed error -0.0306, which is consistent with LS underconfidence there but is a single cell and not a pre-specified test. In the fixed-eps arms, the Spearman correlation between eps and MSE was 0.99 at n3000_b4 and -0.60 at n30_b1. Pooled signed error for ls_best_lr was +0.0023 (CI [-0.0043, 0.0089]), against +0.0159 for eps0_best_lr (CI [0.0064, 0.0254]). Both implementation checks passed.

Accuracy and Pareto (R004, Table 4). The paired accuracy difference (ls_best_raw minus eps0_best_raw) was -0.0003 (CI [-0.0008, 0.0001]), and test accuracy was 0.8033 against 0.8036. The argmax changed in 1.14% of test predictions. LS-epsilon curve versus L2 curve in log loss: ls_best_raw was better than the eps curve at tuned L2 (-0.0182) and than the L2 curve at eps=0 (-0.1264); these compare different curve points and do not settle Q-c. Real-data descriptive differences were small: breast_cancer -0.0083, digits -0.0051, iris +0.0010, wine +0.0028; their relation to the run deviations (digits and real-data MLPs are listed as omitted) is unresolved. Wide seed-level intervals for S5_lr and S4_mlp would be labelled inconclusive. Full TOST rules were not computed.

De-smoothing ablation (R005, Table 5). De-smoothing minus raw LS log loss was -0.0106 (CI [-0.0170, -0.0041], Holm p=0.0583), with 21.9% of entries clipped. This is the opposite sign to the H2 falsification criterion and does not refute H2. It is based on 3 seeds with a single-seed bootstrap.

Validation runs (R006, R007, Tables 6-7). Both are diagnostic and do not test H1. R002 reproduction audits could not run and most R007 metrics are NaN. In R006, inner-CV selection of C was below the oracle by about 0.5 in log10 C (cv5: -0.5250), with tuned-minus-oracle test log loss of 0.0052. The constrained-L2 arm had a gap of 0.1613. The log-loss MDE at 20 replicates was 0.0294 for best-tuned LS and 0.0701 for fixed-eps 0.1 LS, far above the 0.01 margin. These come from a single stand-in configuration.

### Result tables

**Table 1: R001, E2@v2 (primary) testing H2@v1, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
| ablation_cvts_minus_raw_log_loss | eps0_best_variant | 20 | -0.0048 | 0.0053 | [-0.0073, -0.0023] |
| ablation_cvts_minus_raw_log_loss | eps0_cvts | 20 | -0.0048 | 0.0053 | [-0.0073, -0.0023] |
| ablation_cvts_minus_raw_log_loss | eps0_raw | 20 | -0.0048 | 0.0053 | [-0.0073, -0.0023] |
| ablation_cvts_minus_raw_log_loss | ls_best_variant | 20 | -0.0048 | 0.0053 | [-0.0073, -0.0023] |
| ablation_cvts_minus_raw_log_loss | ls_cvts | 20 | -0.0048 | 0.0053 | [-0.0073, -0.0023] |
| ablation_cvts_minus_raw_log_loss | ls_desmooth | 20 | -0.0048 | 0.0053 | [-0.0073, -0.0023] |
| ablation_cvts_minus_raw_log_loss | ls_raw | 20 | -0.0048 | 0.0053 | [-0.0073, -0.0023] |
| ablation_desmooth_minus_raw_log_loss | eps0_best_variant | 20 | 0.0135 | 0.0073 | [0.0100, 0.0169] |
| ablation_desmooth_minus_raw_log_loss | eps0_cvts | 20 | 0.0135 | 0.0073 | [0.0100, 0.0169] |
| ablation_desmooth_minus_raw_log_loss | eps0_raw | 20 | 0.0135 | 0.0073 | [0.0100, 0.0169] |
| ablation_desmooth_minus_raw_log_loss | ls_best_variant | 20 | 0.0135 | 0.0073 | [0.0100, 0.0169] |
| ablation_desmooth_minus_raw_log_loss | ls_cvts | 20 | 0.0135 | 0.0073 | [0.0100, 0.0169] |
| ablation_desmooth_minus_raw_log_loss | ls_desmooth | 20 | 0.0135 | 0.0073 | [0.0100, 0.0169] |
| ablation_desmooth_minus_raw_log_loss | ls_raw | 20 | 0.0135 | 0.0073 | [0.0100, 0.0169] |
| accuracy | eps0_best_variant | 20 | 0.7456 | 0.0053 | [0.7431, 0.7481] |
| accuracy | eps0_cvts | 20 | 0.7439 | 0.0056 | [0.7413, 0.7465] |
| accuracy | eps0_raw | 20 | 0.7380 | 0.0101 | [0.7333, 0.7427] |
| accuracy | ls_best_variant | 20 | 0.7460 | 0.0050 | [0.7436, 0.7483] |
| accuracy | ls_cvts | 20 | 0.7449 | 0.0044 | [0.7428, 0.7470] |
| accuracy | ls_desmooth | 20 | 0.7372 | 0.0090 | [0.7330, 0.7414] |
| accuracy | ls_raw | 20 | 0.7451 | 0.0050 | [0.7427, 0.7474] |
| brier_score | eps0_best_variant | 20 | 0.3356 | 0.0058 | [0.3328, 0.3383] |
| brier_score | eps0_cvts | 20 | 0.3371 | 0.0054 | [0.3345, 0.3396] |
| brier_score | eps0_raw | 20 | 0.3435 | 0.0072 | [0.3401, 0.3468] |
| brier_score | ls_best_variant | 20 | 0.3352 | 0.0057 | [0.3325, 0.3378] |
| brier_score | ls_cvts | 20 | 0.3360 | 0.0049 | [0.3337, 0.3383] |
| brier_score | ls_desmooth | 20 | 0.3446 | 0.0065 | [0.3416, 0.3476] |
| brier_score | ls_raw | 20 | 0.3383 | 0.0055 | [0.3357, 0.3409] |
| equal_mass_ECE_15_bins (top-label) | eps0_best_variant | 20 | 0.0276 | 0.0065 | [0.0246, 0.0307] |
| equal_mass_ECE_15_bins (top-label) | eps0_cvts | 20 | 0.0288 | 0.0061 | [0.0259, 0.0317] |
| equal_mass_ECE_15_bins (top-label) | eps0_raw | 20 | 0.0460 | 0.0070 | [0.0427, 0.0493] |
| equal_mass_ECE_15_bins (top-label) | ls_best_variant | 20 | 0.0281 | 0.0070 | [0.0248, 0.0314] |
| equal_mass_ECE_15_bins (top-label) | ls_cvts | 20 | 0.0272 | 0.0062 | [0.0243, 0.0300] |
| equal_mass_ECE_15_bins (top-label) | ls_desmooth | 20 | 0.0443 | 0.0084 | [0.0403, 0.0482] |
| equal_mass_ECE_15_bins (top-label) | ls_raw | 20 | 0.0332 | 0.0068 | [0.0301, 0.0364] |
| fixed_variant_log_loss_differences | eps0_best_variant | 20 | -0.0017 | 0.0023 | [-0.0028, -0.0006] |
| fixed_variant_log_loss_differences | eps0_cvts | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fixed_variant_log_loss_differences | eps0_raw | 20 | 0.0113 | 0.0053 | [0.0088, 0.0137] |
| fixed_variant_log_loss_differences | ls_best_variant | 20 | -0.0007 | 0.0038 | [-0.0025, 0.0010] |
| fixed_variant_log_loss_differences | ls_cvts | 20 | -0.0015 | 0.0031 | [-0.0029, -0.0001] |
| fixed_variant_log_loss_differences | ls_desmooth | 20 | 0.0167 | 0.0067 | [0.0136, 0.0199] |
| fixed_variant_log_loss_differences | ls_raw | 20 | 0.0033 | 0.0039 | [0.0015, 0.0051] |
| log_loss | eps0_best_variant | 20 | 0.4991 | 0.0086 | [0.4951, 0.5031] |
| log_loss | eps0_cvts | 20 | 0.5008 | 0.0078 | [0.4971, 0.5045] |
| log_loss | eps0_raw | 20 | 0.5121 | 0.0087 | [0.5080, 0.5161] |
| log_loss | ls_best_variant | 20 | 0.5001 | 0.0085 | [0.4961, 0.5040] |
| log_loss | ls_cvts | 20 | 0.4993 | 0.0072 | [0.4959, 0.5027] |
| log_loss | ls_desmooth | 20 | 0.5176 | 0.0083 | [0.5136, 0.5215] |
| log_loss | ls_raw | 20 | 0.5041 | 0.0074 | [0.5006, 0.5076] |
| paired_test_log_loss_difference | eps0_best_variant | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_test_log_loss_difference | eps0_cvts | 20 | 0.0017 | 0.0023 | [0.0006, 0.0028] |
| paired_test_log_loss_difference | eps0_raw | 20 | 0.0129 | 0.0055 | [0.0103, 0.0155] |
| paired_test_log_loss_difference | ls_best_variant | 20 | 0.0009 | 0.0034 | [-0.0006, 0.0025] |
| paired_test_log_loss_difference | ls_cvts | 20 | 0.0002 | 0.0034 | [-0.0014, 0.0018] |
| paired_test_log_loss_difference | ls_desmooth | 20 | 0.0184 | 0.0068 | [0.0152, 0.0216] |
| paired_test_log_loss_difference | ls_raw | 20 | 0.0050 | 0.0044 | [0.0029, 0.0070] |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | eps0_best_variant | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | eps0_cvts | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | eps0_raw | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | ls_best_variant | 20 | 0.1132 | 0.0125 | [0.1074, 0.1190] |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | ls_cvts | 20 | 0.1286 | 0.0144 | [0.1218, 0.1353] |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | ls_desmooth | 20 | 0.0870 | 0.0162 | [0.0794, 0.0946] |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | ls_raw | 20 | 0.0975 | 0.0128 | [0.0915, 0.1035] |
| selected_frac_clipped_desmooth | eps0_best_variant | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_frac_clipped_desmooth | eps0_cvts | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_frac_clipped_desmooth | eps0_raw | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_frac_clipped_desmooth | ls_best_variant | 20 | 0.0112 | 0.0079 | [0.0075, 0.0149] |
| selected_frac_clipped_desmooth | ls_cvts | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_frac_clipped_desmooth | ls_desmooth | 20 | 0.0288 | 0.0113 | [0.0235, 0.0340] |
| selected_frac_clipped_desmooth | ls_raw | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_frac_variant_cvts | eps0_best_variant | 20 | 0.6286 | 0.0751 | [0.5934, 0.6637] |
| selected_frac_variant_cvts | eps0_cvts | 20 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| selected_frac_variant_cvts | eps0_raw | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_frac_variant_cvts | ls_best_variant | 20 | 0.4810 | 0.1058 | [0.4314, 0.5305] |
| selected_frac_variant_cvts | ls_cvts | 20 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| selected_frac_variant_cvts | ls_desmooth | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_frac_variant_cvts | ls_raw | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_log10_C_or_wd | eps0_best_variant | 20 | -1.2524 | 0.1442 | [-1.3199, -1.1849] |
| selected_log10_C_or_wd | eps0_cvts | 20 | -1.3357 | 0.2098 | [-1.4339, -1.2375] |
| selected_log10_C_or_wd | eps0_raw | 20 | -1.0452 | 0.1042 | [-1.0940, -0.9965] |
| selected_log10_C_or_wd | ls_best_variant | 20 | -1.0024 | 0.0894 | [-1.0442, -0.9605] |
| selected_log10_C_or_wd | ls_cvts | 20 | -1.1524 | 0.1110 | [-1.2043, -1.1004] |
| selected_log10_C_or_wd | ls_desmooth | 20 | -1.0214 | 0.0894 | [-1.0633, -0.9796] |
| selected_log10_C_or_wd | ls_raw | 20 | -1.0262 | 0.0732 | [-1.0605, -0.9919] |
| signed_error_vs_true_prob | eps0_best_variant | 20 | -0.0069 | 0.0102 | [-0.0116, -0.0021] |
| signed_error_vs_true_prob | eps0_cvts | 20 | -0.0068 | 0.0105 | [-0.0117, -0.0019] |
| signed_error_vs_true_prob | eps0_raw | 20 | -0.0077 | 0.0147 | [-0.0146, -0.0008] |
| signed_error_vs_true_prob | ls_best_variant | 20 | -0.0115 | 0.0104 | [-0.0164, -0.0067] |
| signed_error_vs_true_prob | ls_cvts | 20 | -0.0077 | 0.0115 | [-0.0130, -0.0023] |
| signed_error_vs_true_prob | ls_desmooth | 20 | -0.0180 | 0.0127 | [-0.0239, -0.0121] |
| signed_error_vs_true_prob | ls_raw | 20 | -0.0200 | 0.0107 | [-0.0250, -0.0150] |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | eps0_best_variant | 20 | -0.0067 | 0.0102 | [-0.0115, -0.0019] |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | eps0_cvts | 20 | -0.0066 | 0.0103 | [-0.0115, -0.0018] |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | eps0_raw | 20 | -0.0076 | 0.0146 | [-0.0145, -0.0008] |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | ls_best_variant | 20 | -0.0114 | 0.0104 | [-0.0163, -0.0066] |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | ls_cvts | 20 | -0.0076 | 0.0113 | [-0.0128, -0.0023] |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | ls_desmooth | 20 | -0.0179 | 0.0124 | [-0.0237, -0.0121] |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | ls_raw | 20 | -0.0199 | 0.0107 | [-0.0250, -0.0149] |
| true_prob_log_loss_gap_KL | eps0_best_variant | 20 | 0.0727 | 0.0086 | [0.0686, 0.0767] |
| true_prob_log_loss_gap_KL | eps0_cvts | 20 | 0.0744 | 0.0079 | [0.0707, 0.0781] |
| true_prob_log_loss_gap_KL | eps0_raw | 20 | 0.0856 | 0.0087 | [0.0816, 0.0897] |
| true_prob_log_loss_gap_KL | ls_best_variant | 20 | 0.0736 | 0.0086 | [0.0696, 0.0776] |
| true_prob_log_loss_gap_KL | ls_cvts | 20 | 0.0728 | 0.0074 | [0.0694, 0.0763] |
| true_prob_log_loss_gap_KL | ls_desmooth | 20 | 0.0911 | 0.0085 | [0.0872, 0.0951] |
| true_prob_log_loss_gap_KL | ls_raw | 20 | 0.0776 | 0.0074 | [0.0742, 0.0811] |

| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |
|---|---|---|---|---|---|---|---|
| paired_test_log_loss_difference (ls_best_variant minus eps0_best_variant, 20k fresh test, per replicate) (primary) | ls_best_variant vs eps0_best_variant | 0.0009 | [-0.0006, 0.0025] | 1.0000 | 0.39 | no | no |
| paired_test_log_loss_difference (ls_best_variant minus eps0_best_variant, 20k fresh test, per replicate) (primary) | ls_best_variant vs eps0_cvts | -0.0007 | [-0.0026, 0.0011] | 1.0000 | -0.24 | no | no |
| paired_test_log_loss_difference (ls_best_variant minus eps0_best_variant, 20k fresh test, per replicate) (primary) | ls_best_variant vs eps0_raw | -0.0120 | [-0.0149, -0.0090] | 0.0000 | -2.56 | yes | yes |
| paired_test_log_loss_difference (ls_best_variant minus eps0_best_variant, 20k fresh test, per replicate) (primary) | ls_best_variant vs ls_raw | -0.0040 | [-0.0065, -0.0015] | 0.0567 | -1.00 | no | no |
| paired_test_log_loss_difference (ls_best_variant minus eps0_best_variant, 20k fresh test, per replicate) (primary) | ls_best_variant vs ls_cvts | 0.0008 | [-0.0014, 0.0030] | 1.0000 | 0.23 | no | no |
| paired_test_log_loss_difference (ls_best_variant minus eps0_best_variant, 20k fresh test, per replicate) (primary) | ls_best_variant vs ls_desmooth | -0.0175 | [-0.0210, -0.0140] | 0.0000 | -3.19 | yes | yes |
| fixed_variant_log_loss_differences (ls_cvts minus eps0_cvts; ls_raw minus eps0_cvts; best variant minus eps0_cvts) | ls_best_variant vs eps0_best_variant | 0.0009 | [-0.0011, 0.0030] | 1.0000 | 0.30 | no | no |
| fixed_variant_log_loss_differences (ls_cvts minus eps0_cvts; ls_raw minus eps0_cvts; best variant minus eps0_cvts) | ls_best_variant vs eps0_cvts | -0.0007 | [-0.0025, 0.0010] | 1.0000 | -0.27 | no | no |
| fixed_variant_log_loss_differences (ls_cvts minus eps0_cvts; ls_raw minus eps0_cvts; best variant minus eps0_cvts) | ls_best_variant vs eps0_raw | -0.0120 | [-0.0149, -0.0090] | 0.0000 | -2.54 | yes | yes |
| fixed_variant_log_loss_differences (ls_cvts minus eps0_cvts; ls_raw minus eps0_cvts; best variant minus eps0_cvts) | ls_best_variant vs ls_raw | -0.0040 | [-0.0065, -0.0016] | 0.0429 | -1.03 | yes | no |
| fixed_variant_log_loss_differences (ls_cvts minus eps0_cvts; ls_raw minus eps0_cvts; best variant minus eps0_cvts) | ls_best_variant vs ls_cvts | 0.0008 | [-0.0014, 0.0030] | 1.0000 | 0.22 | no | no |
| fixed_variant_log_loss_differences (ls_cvts minus eps0_cvts; ls_raw minus eps0_cvts; best variant minus eps0_cvts) | ls_best_variant vs ls_desmooth | -0.0175 | [-0.0210, -0.0140] | 0.0000 | -3.17 | yes | yes |
| brier_score | ls_best_variant vs eps0_best_variant | -0.0004 | [-0.0041, 0.0033] | 1.0000 | -0.07 | no | no |
| brier_score | ls_best_variant vs eps0_cvts | -0.0019 | [-0.0055, 0.0016] | 1.0000 | -0.34 | no | no |
| brier_score | ls_best_variant vs eps0_raw | -0.0083 | [-0.0125, -0.0042] | 0.0063 | -1.26 | yes | yes |
| brier_score | ls_best_variant vs ls_raw | -0.0031 | [-0.0067, 0.0004] | 1.0000 | -0.55 | no | no |
| brier_score | ls_best_variant vs ls_cvts | -0.0008 | [-0.0042, 0.0026] | 1.0000 | -0.15 | no | no |
| brier_score | ls_best_variant vs ls_desmooth | -0.0094 | [-0.0134, -0.0055] | 0.0005 | -1.52 | yes | yes |
| equal_mass_ECE_15_bins (top-label) | ls_best_variant vs eps0_best_variant | 0.0005 | [-0.0039, 0.0048] | 1.0000 | 0.07 | no | no |
| equal_mass_ECE_15_bins (top-label) | ls_best_variant vs eps0_cvts | -0.0007 | [-0.0049, 0.0035] | 1.0000 | -0.11 | no | no |
| equal_mass_ECE_15_bins (top-label) | ls_best_variant vs eps0_raw | -0.0179 | [-0.0224, -0.0134] | 0.0000 | -2.50 | yes | yes |
| equal_mass_ECE_15_bins (top-label) | ls_best_variant vs ls_raw | -0.0052 | [-0.0096, -0.0007] | 0.4449 | -0.73 | no | no |
| equal_mass_ECE_15_bins (top-label) | ls_best_variant vs ls_cvts | 0.0009 | [-0.0033, 0.0052] | 1.0000 | 0.14 | no | no |
| equal_mass_ECE_15_bins (top-label) | ls_best_variant vs ls_desmooth | -0.0162 | [-0.0211, -0.0112] | 0.0000 | -2.05 | yes | yes |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | ls_best_variant vs eps0_best_variant | -0.0047 | [-0.0113, 0.0019] | 1.0000 | -0.45 | no | no |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | ls_best_variant vs eps0_cvts | -0.0048 | [-0.0114, 0.0019] | 1.0000 | -0.45 | no | no |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | ls_best_variant vs eps0_raw | -0.0038 | [-0.0119, 0.0043] | 1.0000 | -0.29 | no | no |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | ls_best_variant vs ls_raw | 0.0085 | [0.0018, 0.0153] | 0.2943 | 0.79 | no | yes |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | ls_best_variant vs ls_cvts | -0.0038 | [-0.0108, 0.0031] | 1.0000 | -0.35 | no | no |
| signed_error_vs_true_prob (synthetic) and signed_calibration_error mean(confidence - accuracy) | ls_best_variant vs ls_desmooth | 0.0065 | [-0.0008, 0.0138] | 1.0000 | 0.56 | no | yes |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | ls_best_variant vs eps0_best_variant | 0.1132 | [0.1074, 0.1190] | 0.0000 | 12.59 | yes | yes |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | ls_best_variant vs eps0_cvts | 0.1132 | [0.1074, 0.1190] | 0.0000 | 12.59 | yes | yes |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | ls_best_variant vs eps0_raw | 0.1132 | [0.1074, 0.1190] | 0.0000 | 12.59 | yes | yes |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | ls_best_variant vs ls_raw | 0.0157 | [0.0076, 0.0238] | 0.0081 | 1.22 | yes | yes |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | ls_best_variant vs ls_cvts | -0.0154 | [-0.0240, -0.0067] | 0.0210 | -1.12 | yes | yes |
| selected eps, C/wd and variant frequencies; fraction of clipped entries under de-smoothing | ls_best_variant vs ls_desmooth | 0.0262 | [0.0169, 0.0355] | 0.0000 | 1.78 | yes | yes |

**Table 2: R002, E1@v2 (validation) testing H1@v2, seeds [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
| acc_signed_error__C1 | eps0_constrained_l2 | 8 | 0.0612 | 0.0637 | [0.0079, 0.1144] |
| acc_signed_error__C1 | eps0_tuned_l2 | 8 | -0.0101 | 0.0249 | [-0.0309, 0.0107] |
| acc_signed_error__C1 | eps0_tuned_l2_altgrid | 8 | -0.0039 | 0.0263 | [-0.0259, 0.0181] |
| acc_signed_error__C2 | eps0_constrained_l2 | 8 | -0.0087 | 0.0135 | [-0.0200, 0.0026] |
| acc_signed_error__C2 | eps0_tuned_l2 | 8 | -0.0105 | 0.0134 | [-0.0217, 0.0007] |
| acc_signed_error__C2 | eps0_tuned_l2_altgrid | 8 | -0.0121 | 0.0120 | [-0.0221, -0.0020] |
| acc_signed_error__C3 | eps0_constrained_l2 | 8 | 0.0010 | 0.0049 | [-0.0031, 0.0051] |
| acc_signed_error__C3 | eps0_tuned_l2 | 8 | 0.0004 | 0.0051 | [-0.0038, 0.0047] |
| acc_signed_error__C3 | eps0_tuned_l2_altgrid | 8 | 0.0000 | 0.0052 | [-0.0043, 0.0043] |
| acc_signed_error__C4 | eps0_constrained_l2 | 8 | 0.0007 | 0.0050 | [-0.0035, 0.0048] |
| acc_signed_error__C4 | eps0_tuned_l2 | 8 | 0.0002 | 0.0054 | [-0.0044, 0.0047] |
| acc_signed_error__C4 | eps0_tuned_l2_altgrid | 8 | 0.0004 | 0.0052 | [-0.0039, 0.0047] |
| acc_signed_error__C5 | eps0_constrained_l2 | 8 | 0.1376 | 0.0069 | [0.1318, 0.1434] |
| acc_signed_error__C5 | eps0_tuned_l2 | 8 | -0.0245 | 0.0620 | [-0.0764, 0.0274] |
| acc_signed_error__C5 | eps0_tuned_l2_altgrid | 8 | -0.0197 | 0.0088 | [-0.0271, -0.0123] |
| acc_signed_error__C6 | eps0_constrained_l2 | 8 | 0.0018 | 0.0054 | [-0.0027, 0.0063] |
| acc_signed_error__C6 | eps0_tuned_l2 | 8 | 0.0012 | 0.0054 | [-0.0033, 0.0057] |
| acc_signed_error__C6 | eps0_tuned_l2_altgrid | 8 | -0.0001 | 0.0053 | [-0.0046, 0.0044] |
| acc_signed_error__C7 | eps0_constrained_l2 | 8 | 0.0544 | 0.0174 | [0.0398, 0.0689] |
| acc_signed_error__C7 | eps0_tuned_l2 | 8 | 0.0029 | 0.0407 | [-0.0312, 0.0369] |
| acc_signed_error__C7 | eps0_tuned_l2_altgrid | 8 | -0.0091 | 0.0102 | [-0.0176, -0.0006] |
| acc_signed_error__C8 | eps0_constrained_l2 | 8 | 0.0011 | 0.0062 | [-0.0041, 0.0063] |
| acc_signed_error__C8 | eps0_tuned_l2 | 8 | -0.0011 | 0.0056 | [-0.0058, 0.0036] |
| acc_signed_error__C8 | eps0_tuned_l2_altgrid | 8 | -0.0005 | 0.0062 | [-0.0057, 0.0046] |
| acc_signed_error__D1 | eps0_constrained_l2 | 8 | 0.0262 | 0.0239 | [0.0062, 0.0462] |
| acc_signed_error__D1 | eps0_tuned_l2 | 8 | 0.0139 | 0.0374 | [-0.0174, 0.0451] |
| acc_signed_error__D1 | eps0_tuned_l2_altgrid | 8 | 0.0145 | 0.0294 | [-0.0101, 0.0391] |
| acc_signed_error__D2 | eps0_constrained_l2 | 8 | -0.0007 | 0.0020 | [-0.0023, 0.0010] |
| acc_signed_error__D2 | eps0_tuned_l2 | 8 | -0.0007 | 0.0020 | [-0.0023, 0.0010] |
| acc_signed_error__D2 | eps0_tuned_l2_altgrid | 8 | 0.0005 | 0.0011 | [-0.0004, 0.0013] |
| acc_signed_error__D3 | eps0_constrained_l2 | 8 | -0.0011 | 0.0026 | [-0.0033, 0.0011] |
| acc_signed_error__D3 | eps0_tuned_l2 | 8 | -0.0019 | 0.0029 | [-0.0043, 0.0005] |
| acc_signed_error__D3 | eps0_tuned_l2_altgrid | 8 | -0.0014 | 0.0027 | [-0.0036, 0.0009] |
| accuracy_based_signed_error (mean confidence minus accuracy) | eps0_constrained_l2 | 8 | 0.0249 | 0.0065 | [0.0194, 0.0303] |
| accuracy_based_signed_error (mean confidence minus accuracy) | eps0_tuned_l2 | 8 | -0.0027 | 0.0088 | [-0.0101, 0.0046] |
| accuracy_based_signed_error (mean confidence minus accuracy) | eps0_tuned_l2_altgrid | 8 | -0.0029 | 0.0039 | [-0.0062, 0.0004] |
| altgrid_cells_skipped_by_time_guard | eps0_tuned_l2_altgrid | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| check_i_pass | direct_soft_ce_minimiser | 10 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| check_ii_pass | direct_soft_ce_minimiser | 10 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| check_iii_pass | direct_soft_ce_minimiser | 10 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| constrained_eps0_signed_error_vs_true_prob | eps0_constrained_l2 | 8 | 0.0250 | 0.0065 | [0.0195, 0.0304] |
| constrained_eps0_signed_error_vs_true_prob__also_primary_name | eps0_constrained_l2 | 8 | 0.0250 | 0.0065 | [0.0195, 0.0304] |
| elapsed_seconds_total | direct_soft_ce_minimiser | 10 | 56.2988 | 1.4978 | [55.2273, 57.3703] |
| elapsed_seconds_total | eps0_constrained_l2 | 10 | 56.2988 | 1.4978 | [55.2273, 57.3703] |
| elapsed_seconds_total | eps0_tuned_l2 | 10 | 56.2988 | 1.4978 | [55.2273, 57.3703] |
| elapsed_seconds_total | eps0_tuned_l2_altgrid | 10 | 56.2988 | 1.4978 | [55.2273, 57.3703] |
| elapsed_seconds_total | sklearn_logreg_reference | 10 | 56.2988 | 1.4978 | [55.2273, 57.3703] |
| eps_gt0_max_abs_coef_diff_vs_direct_minimiser (K=2 and K=3; pass < 1e-4) | direct_soft_ce_minimiser | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| eps_gt0_stationarity_grad_inf_norm (pass < 1e-6) | direct_soft_ce_minimiser | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fam1_check_pass | direct_soft_ce_minimiser | 10 | 0.8000 | 0.4216 | [0.4984, 1.1016] |
| fam1_true_prob_bin_check_max_abs_z | direct_soft_ce_minimiser | 10 | 2.4241 | 0.6675 | [1.9466, 2.9016] |
| fires_SO_LR__C1 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_LR__C2 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR__C3 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR__C4 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR__C5 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_LR__C6 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR__C7 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_LR__C8 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR__D1 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR__D2 | eps0_tuned_l2 | 8 | 0.6250 | 0.5175 | [0.1923, 1.0577] |
| fires_SO_LR__D3 | eps0_tuned_l2 | 8 | 0.6250 | 0.5175 | [0.1923, 1.0577] |
| fires_SO_LR_alt_a__C1 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_LR_alt_a__C2 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_a__C3 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_a__C4 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_a__C5 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_LR_alt_a__C6 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_a__C7 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_LR_alt_a__C8 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_a__D1 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_a__D2 | eps0_tuned_l2 | 8 | 0.7500 | 0.4629 | [0.3630, 1.1370] |
| fires_SO_LR_alt_a__D3 | eps0_tuned_l2 | 8 | 0.6250 | 0.5175 | [0.1923, 1.0577] |
| fires_SO_LR_alt_c__C1 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_LR_alt_c__C2 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_c__C3 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_c__C4 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_c__C5 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_LR_alt_c__C6 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_c__C7 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_LR_alt_c__C8 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_c__D1 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_c__D2 | eps0_tuned_l2 | 8 | 0.2500 | 0.4629 | [-0.1370, 0.6370] |
| fires_SO_LR_alt_c__D3 | eps0_tuned_l2 | 8 | 0.5000 | 0.5345 | [0.0531, 0.9469] |
| fires_SO_LR_alt_d__C1 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_LR_alt_d__C2 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_d__C3 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_d__C4 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_d__C5 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_LR_alt_d__C6 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_d__C7 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_LR_alt_d__C8 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_d__D1 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_d__D2 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_LR_alt_d__D3 | eps0_tuned_l2 | 8 | 0.5000 | 0.5345 | [0.0531, 0.9469] |
| fires_SO_MLP16__C1 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_MLP16__C2 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_MLP16__C3 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_MLP16__C4 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_MLP16__C5 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_MLP16__C6 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_MLP16__C7 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_MLP16__C8 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_MLP16__D1 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_MLP16__D2 | eps0_tuned_l2 | 8 | 0.6250 | 0.5175 | [0.1923, 1.0577] |
| fires_SO_MLP16__D3 | eps0_tuned_l2 | 8 | 0.6250 | 0.5175 | [0.1923, 1.0577] |
| fires_SO_MLP256__C1 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_MLP256__C2 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_MLP256__C3 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_MLP256__C4 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_MLP256__C5 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_MLP256__C6 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_MLP256__C7 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_MLP256__C8 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fires_SO_MLP256__D1 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_MLP256__D2 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| fires_SO_MLP256__D3 | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| large_n_mean_abs_prob_dev_max | direct_soft_ce_minimiser | 10 | 0.0018 | 0.0003 | [0.0015, 0.0020] |
| large_n_orth_norm_max | direct_soft_ce_minimiser | 10 | 0.0150 | 0.0037 | [0.0123, 0.0176] |
| large_n_relative_deviation_from_population_LR_LS_solution (pass < 0.02 on a_eps, < 0.005 mean abs prob deviation) | direct_soft_ce_minimiser | 10 | 0.0015 | 0.0011 | [0.0007, 0.0023] |
| log10C_selected__C1 | eps0_constrained_l2 | 8 | 2.0000 | 0.0000 | [2.0000, 2.0000] |
| log10C_selected__C1 | eps0_tuned_l2 | 8 | 0.1250 | 0.6409 | [-0.4108, 0.6608] |
| log10C_selected__C2 | eps0_constrained_l2 | 8 | 2.0000 | 0.0000 | [2.0000, 2.0000] |
| log10C_selected__C2 | eps0_tuned_l2 | 8 | -0.2500 | 0.4629 | [-0.6370, 0.1370] |
| log10C_selected__C3 | eps0_constrained_l2 | 8 | 2.0000 | 0.0000 | [2.0000, 2.0000] |
| log10C_selected__C3 | eps0_tuned_l2 | 8 | -0.5000 | 1.5119 | [-1.7639, 0.7639] |
| log10C_selected__C4 | eps0_constrained_l2 | 8 | 2.0000 | 0.0000 | [2.0000, 2.0000] |
| log10C_selected__C4 | eps0_tuned_l2 | 8 | -0.1250 | 0.8345 | [-0.8227, 0.5727] |
| log10C_selected__C5 | eps0_constrained_l2 | 8 | 2.0000 | 0.0000 | [2.0000, 2.0000] |
| log10C_selected__C5 | eps0_tuned_l2 | 8 | -0.5000 | 0.5345 | [-0.9469, -0.0531] |
| log10C_selected__C6 | eps0_constrained_l2 | 8 | 2.0000 | 0.0000 | [2.0000, 2.0000] |
| log10C_selected__C6 | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| log10C_selected__C7 | eps0_constrained_l2 | 8 | 2.0000 | 0.0000 | [2.0000, 2.0000] |
| log10C_selected__C7 | eps0_tuned_l2 | 8 | -0.2500 | 0.4629 | [-0.6370, 0.1370] |
| log10C_selected__C8 | eps0_constrained_l2 | 8 | 2.0000 | 0.0000 | [2.0000, 2.0000] |
| log10C_selected__C8 | eps0_tuned_l2 | 8 | -0.3750 | 0.5175 | [-0.8077, 0.0577] |
| log10C_selected__D1 | eps0_constrained_l2 | 8 | 2.0000 | 0.0000 | [2.0000, 2.0000] |
| log10C_selected__D1 | eps0_tuned_l2 | 8 | -2.0000 | 1.0690 | [-2.8937, -1.1063] |
| log10C_selected__D2 | eps0_constrained_l2 | 8 | 2.2500 | 0.4629 | [1.8630, 2.6370] |
| log10C_selected__D2 | eps0_tuned_l2 | 8 | 2.2500 | 0.4629 | [1.8630, 2.6370] |
| log10C_selected__D3 | eps0_constrained_l2 | 8 | 3.0000 | 1.0690 | [2.1063, 3.8937] |
| log10C_selected__D3 | eps0_tuned_l2 | 8 | 2.5000 | 1.6036 | [1.1594, 3.8406] |
| mlp_grad_pass | direct_soft_ce_minimiser | 10 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| mlp_gradient_check_max_relative_error (pass < 1e-5) | direct_soft_ce_minimiser | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| pilot_eps0_signed_error_vs_true_prob (mean over 20k test points of top-label confidence minus true P(top label|x), tuned eps0 LR; gate uses its one-sided 95% lower bound) | eps0_tuned_l2 | 8 | -0.0027 | 0.0087 | [-0.0101, 0.0046] |
| pilot_eps0_signed_error_vs_true_prob (mean over 20k test points of top-label confidence minus true P(top label|x), tuned eps0 LR; gate uses its one-sided 95% lower bound) | eps0_tuned_l2_altgrid | 8 | -0.0028 | 0.0041 | [-0.0063, 0.0006] |
| signed_error__C1 | eps0_constrained_l2 | 8 | 0.0610 | 0.0637 | [0.0078, 0.1142] |
| signed_error__C1 | eps0_tuned_l2 | 8 | -0.0103 | 0.0249 | [-0.0311, 0.0105] |
| signed_error__C1 | eps0_tuned_l2_altgrid | 8 | -0.0041 | 0.0264 | [-0.0262, 0.0180] |
| signed_error__C2 | eps0_constrained_l2 | 8 | -0.0093 | 0.0143 | [-0.0213, 0.0026] |
| signed_error__C2 | eps0_tuned_l2 | 8 | -0.0111 | 0.0142 | [-0.0230, 0.0008] |
| signed_error__C2 | eps0_tuned_l2_altgrid | 8 | -0.0127 | 0.0128 | [-0.0234, -0.0020] |
| signed_error__C3 | eps0_constrained_l2 | 8 | 0.0009 | 0.0051 | [-0.0034, 0.0051] |
| signed_error__C3 | eps0_tuned_l2 | 8 | 0.0003 | 0.0053 | [-0.0041, 0.0048] |
| signed_error__C3 | eps0_tuned_l2_altgrid | 8 | -0.0001 | 0.0054 | [-0.0046, 0.0044] |
| signed_error__C4 | eps0_constrained_l2 | 8 | 0.0005 | 0.0047 | [-0.0034, 0.0044] |
| signed_error__C4 | eps0_tuned_l2 | 8 | 0.0000 | 0.0051 | [-0.0042, 0.0043] |
| signed_error__C4 | eps0_tuned_l2_altgrid | 8 | 0.0003 | 0.0049 | [-0.0038, 0.0043] |
| signed_error__C5 | eps0_constrained_l2 | 8 | 0.1380 | 0.0087 | [0.1307, 0.1453] |
| signed_error__C5 | eps0_tuned_l2 | 8 | -0.0247 | 0.0625 | [-0.0769, 0.0275] |
| signed_error__C5 | eps0_tuned_l2_altgrid | 8 | -0.0196 | 0.0093 | [-0.0274, -0.0118] |
| signed_error__C6 | eps0_constrained_l2 | 8 | 0.0017 | 0.0048 | [-0.0023, 0.0057] |
| signed_error__C6 | eps0_tuned_l2 | 8 | 0.0011 | 0.0047 | [-0.0028, 0.0051] |
| signed_error__C6 | eps0_tuned_l2_altgrid | 8 | -0.0002 | 0.0047 | [-0.0041, 0.0037] |
| signed_error__C7 | eps0_constrained_l2 | 8 | 0.0548 | 0.0166 | [0.0409, 0.0687] |
| signed_error__C7 | eps0_tuned_l2 | 8 | 0.0032 | 0.0409 | [-0.0310, 0.0374] |
| signed_error__C7 | eps0_tuned_l2_altgrid | 8 | -0.0085 | 0.0088 | [-0.0158, -0.0011] |
| signed_error__C8 | eps0_constrained_l2 | 8 | 0.0008 | 0.0041 | [-0.0027, 0.0042] |
| signed_error__C8 | eps0_tuned_l2 | 8 | -0.0015 | 0.0042 | [-0.0050, 0.0020] |
| signed_error__C8 | eps0_tuned_l2_altgrid | 8 | -0.0009 | 0.0041 | [-0.0043, 0.0026] |
| signed_error__D1 | eps0_constrained_l2 | 8 | 0.0277 | 0.0229 | [0.0086, 0.0468] |
| signed_error__D1 | eps0_tuned_l2 | 8 | 0.0149 | 0.0368 | [-0.0158, 0.0457] |
| signed_error__D1 | eps0_tuned_l2_altgrid | 8 | 0.0150 | 0.0291 | [-0.0093, 0.0393] |
| signed_error__D2 | eps0_constrained_l2 | 8 | -0.0004 | 0.0023 | [-0.0022, 0.0015] |
| signed_error__D2 | eps0_tuned_l2 | 8 | -0.0004 | 0.0023 | [-0.0022, 0.0015] |
| signed_error__D2 | eps0_tuned_l2_altgrid | 8 | 0.0008 | 0.0011 | [-0.0002, 0.0017] |
| signed_error__D3 | eps0_constrained_l2 | 8 | -0.0011 | 0.0026 | [-0.0033, 0.0011] |
| signed_error__D3 | eps0_tuned_l2 | 8 | -0.0019 | 0.0029 | [-0.0043, 0.0005] |
| signed_error__D3 | eps0_tuned_l2_altgrid | 8 | -0.0014 | 0.0027 | [-0.0036, 0.0009] |
| signed_error_mean_over_replicate_SO_cells | eps0_constrained_l2 | 8 | 0.0612 | 0.0149 | [0.0487, 0.0736] |
| signed_error_mean_over_replicate_SO_cells | eps0_tuned_l2 | 8 | -0.0097 | 0.0253 | [-0.0308, 0.0115] |
| signed_error_mean_over_replicate_SO_cells | eps0_tuned_l2_altgrid | 8 | -0.0078 | 0.0073 | [-0.0140, -0.0017] |
| smoke_dup_vs_plain_eps0_max_abs_coef_diff_NONINFORMATIVE | sklearn_logreg_reference | 10 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition | eps0_tuned_l2 | 8 | 0.2841 | 0.1024 | [0.1985, 0.3697] |
| structural_proxy_firing_rate per cell x proxy definition__C1__a | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C1__b | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C1__c | eps0_tuned_l2 | 8 | 0.8750 | 0.3536 | [0.5794, 1.1706] |
| structural_proxy_firing_rate per cell x proxy definition__C1__d | eps0_tuned_l2 | 8 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C2__a | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C2__b | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C2__c | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C2__d | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C3__a | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C3__b | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C3__c | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C3__d | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C4__a | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C4__b | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C4__c | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C4__d | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C5__a | eps0_tuned_l2 | 8 | 0.7500 | 0.4629 | [0.3630, 1.1370] |
| structural_proxy_firing_rate per cell x proxy definition__C5__b | eps0_tuned_l2 | 8 | 0.7500 | 0.4629 | [0.3630, 1.1370] |
| structural_proxy_firing_rate per cell x proxy definition__C5__c | eps0_tuned_l2 | 8 | 0.6250 | 0.5175 | [0.1923, 1.0577] |
| structural_proxy_firing_rate per cell x proxy definition__C5__d | eps0_tuned_l2 | 8 | 0.8750 | 0.3536 | [0.5794, 1.1706] |
| structural_proxy_firing_rate per cell x proxy definition__C6__a | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C6__b | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C6__c | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C6__d | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C7__a | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C7__b | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C7__c | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C7__d | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C8__a | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C8__b | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C8__c | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__C8__d | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__D1__a | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__D1__b | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__D1__c | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__D1__d | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__D2__a | eps0_tuned_l2 | 8 | 0.7500 | 0.4629 | [0.3630, 1.1370] |
| structural_proxy_firing_rate per cell x proxy definition__D2__b | eps0_tuned_l2 | 8 | 0.6250 | 0.5175 | [0.1923, 1.0577] |
| structural_proxy_firing_rate per cell x proxy definition__D2__c | eps0_tuned_l2 | 8 | 0.2500 | 0.4629 | [-0.1370, 0.6370] |
| structural_proxy_firing_rate per cell x proxy definition__D2__d | eps0_tuned_l2 | 8 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| structural_proxy_firing_rate per cell x proxy definition__D3__a | eps0_tuned_l2 | 8 | 0.6250 | 0.5175 | [0.1923, 1.0577] |
| structural_proxy_firing_rate per cell x proxy definition__D3__b | eps0_tuned_l2 | 8 | 0.6250 | 0.5175 | [0.1923, 1.0577] |
| structural_proxy_firing_rate per cell x proxy definition__D3__c | eps0_tuned_l2 | 8 | 0.5000 | 0.5345 | [0.0531, 0.9469] |
| structural_proxy_firing_rate per cell x proxy definition__D3__d | eps0_tuned_l2 | 8 | 0.5000 | 0.5345 | [0.0531, 0.9469] |
| wall_time_seconds for the largest LR fit (K=10, d=64, n=1797 duplicated rows) and the largest MLP fit (n=4000, d=40, width=256, 300 epochs) | eps0_tuned_l2 | 10 | 15.8818 | 0.5614 | [15.4802, 16.2833] |
| wall_time_seconds__lr_largest | eps0_tuned_l2 | 10 | 1.3085 | 0.1208 | [1.2221, 1.3950] |
| wall_time_seconds__mlp_largest | eps0_tuned_l2 | 10 | 14.5732 | 0.6278 | [14.1241, 15.0223] |

| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |
|---|---|---|---|---|---|---|---|
| pilot_eps0_signed_error_vs_true_prob (mean over 20k test points of top-label confidence minus true P(top label|x), tuned eps0 LR; gate uses its one-sided 95% lower bound) (primary) | eps0_tuned_l2 vs eps0_tuned_l2_altgrid | 0.0001 | [-0.0075, 0.0077] | 1.0000 | 0.01 | no | no |
| accuracy_based_signed_error (mean confidence minus accuracy) | eps0_tuned_l2 vs eps0_constrained_l2 | -0.0276 | [-0.0360, -0.0192] | 0.0000 | -3.37 | yes | yes |
| accuracy_based_signed_error (mean confidence minus accuracy) | eps0_tuned_l2 vs eps0_tuned_l2_altgrid | 0.0001 | [-0.0075, 0.0077] | 1.0000 | 0.02 | no | no |

**Table 3: R003, E4@v2 (primary) testing H3@v1, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
| KL(true || predicted) averaged over test points | eps0_best_lr | 20 | 0.0711 | 0.0311 | [0.0566, 0.0856] |
| KL(true || predicted) averaged over test points | eps0_best_lr_altgrid | 20 | 0.0742 | 0.0342 | [0.0583, 0.0902] |
| KL(true || predicted) averaged over test points | ls_best_lr | 20 | 0.0599 | 0.0181 | [0.0515, 0.0684] |
| KL(true || predicted) averaged over test points | ls_fixed_eps_at_eps0_C | 20 | 0.0635 | 0.0153 | [0.0563, 0.0706] |
| KL(true || predicted) averaged over test points | ls_fixed_eps_tuned_C | 20 | 0.0626 | 0.0167 | [0.0548, 0.0704] |
| KL(true || predicted) averaged over test points@n100_b1 | eps0_best_lr | 20 | 0.0496 | 0.0228 | [0.0389, 0.0602] |
| KL(true || predicted) averaged over test points@n100_b1 | eps0_best_lr_altgrid | 20 | 0.0514 | 0.0136 | [0.0450, 0.0577] |
| KL(true || predicted) averaged over test points@n100_b1 | ls_best_lr | 20 | 0.0476 | 0.0202 | [0.0382, 0.0571] |
| KL(true || predicted) averaged over test points@n100_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0482 | 0.0219 | [0.0380, 0.0585] |
| KL(true || predicted) averaged over test points@n100_b1 | ls_fixed_eps_tuned_C | 20 | 0.0471 | 0.0208 | [0.0373, 0.0568] |
| KL(true || predicted) averaged over test points@n100_b4 | eps0_best_lr | 20 | 0.0694 | 0.0444 | [0.0487, 0.0902] |
| KL(true || predicted) averaged over test points@n100_b4 | eps0_best_lr_altgrid | 20 | 0.0644 | 0.0306 | [0.0501, 0.0787] |
| KL(true || predicted) averaged over test points@n100_b4 | ls_best_lr | 20 | 0.0594 | 0.0267 | [0.0469, 0.0718] |
| KL(true || predicted) averaged over test points@n100_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0686 | 0.0283 | [0.0553, 0.0818] |
| KL(true || predicted) averaged over test points@n100_b4 | ls_fixed_eps_tuned_C | 20 | 0.0566 | 0.0213 | [0.0466, 0.0666] |
| KL(true || predicted) averaged over test points@n3000_b1 | eps0_best_lr | 20 | 0.0020 | 0.0010 | [0.0015, 0.0025] |
| KL(true || predicted) averaged over test points@n3000_b1 | eps0_best_lr_altgrid | 20 | 0.0020 | 0.0011 | [0.0015, 0.0025] |
| KL(true || predicted) averaged over test points@n3000_b1 | ls_best_lr | 20 | 0.0021 | 0.0012 | [0.0016, 0.0027] |
| KL(true || predicted) averaged over test points@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0032 | 0.0016 | [0.0025, 0.0040] |
| KL(true || predicted) averaged over test points@n3000_b1 | ls_fixed_eps_tuned_C | 20 | 0.0029 | 0.0015 | [0.0022, 0.0036] |
| KL(true || predicted) averaged over test points@n3000_b4 | eps0_best_lr | 20 | 0.0019 | 0.0010 | [0.0015, 0.0024] |
| KL(true || predicted) averaged over test points@n3000_b4 | eps0_best_lr_altgrid | 20 | 0.0020 | 0.0010 | [0.0015, 0.0024] |
| KL(true || predicted) averaged over test points@n3000_b4 | ls_best_lr | 20 | 0.0088 | 0.0017 | [0.0080, 0.0096] |
| KL(true || predicted) averaged over test points@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0239 | 0.0022 | [0.0229, 0.0250] |
| KL(true || predicted) averaged over test points@n3000_b4 | ls_fixed_eps_tuned_C | 20 | 0.0233 | 0.0022 | [0.0223, 0.0244] |
| KL(true || predicted) averaged over test points@n300_b1 | eps0_best_lr | 20 | 0.0165 | 0.0062 | [0.0136, 0.0194] |
| KL(true || predicted) averaged over test points@n300_b1 | eps0_best_lr_altgrid | 20 | 0.0206 | 0.0065 | [0.0175, 0.0236] |
| KL(true || predicted) averaged over test points@n300_b1 | ls_best_lr | 20 | 0.0175 | 0.0070 | [0.0142, 0.0208] |
| KL(true || predicted) averaged over test points@n300_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0172 | 0.0066 | [0.0141, 0.0203] |
| KL(true || predicted) averaged over test points@n300_b1 | ls_fixed_eps_tuned_C | 20 | 0.0173 | 0.0065 | [0.0142, 0.0204] |
| KL(true || predicted) averaged over test points@n300_b4 | eps0_best_lr | 20 | 0.0191 | 0.0057 | [0.0164, 0.0217] |
| KL(true || predicted) averaged over test points@n300_b4 | eps0_best_lr_altgrid | 20 | 0.0221 | 0.0056 | [0.0194, 0.0247] |
| KL(true || predicted) averaged over test points@n300_b4 | ls_best_lr | 20 | 0.0200 | 0.0066 | [0.0169, 0.0231] |
| KL(true || predicted) averaged over test points@n300_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0355 | 0.0085 | [0.0316, 0.0395] |
| KL(true || predicted) averaged over test points@n300_b4 | ls_fixed_eps_tuned_C | 20 | 0.0298 | 0.0077 | [0.0262, 0.0334] |
| KL(true || predicted) averaged over test points@n30_b1 | eps0_best_lr | 20 | 0.1812 | 0.2040 | [0.0857, 0.2766] |
| KL(true || predicted) averaged over test points@n30_b1 | eps0_best_lr_altgrid | 20 | 0.1844 | 0.1987 | [0.0913, 0.2774] |
| KL(true || predicted) averaged over test points@n30_b1 | ls_best_lr | 20 | 0.1431 | 0.1183 | [0.0877, 0.1985] |
| KL(true || predicted) averaged over test points@n30_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.1324 | 0.0898 | [0.0904, 0.1744] |
| KL(true || predicted) averaged over test points@n30_b1 | ls_fixed_eps_tuned_C | 20 | 0.1378 | 0.1064 | [0.0880, 0.1876] |
| KL(true || predicted) averaged over test points@n30_b4 | eps0_best_lr | 20 | 0.2290 | 0.1533 | [0.1573, 0.3008] |
| KL(true || predicted) averaged over test points@n30_b4 | eps0_best_lr_altgrid | 20 | 0.2472 | 0.1944 | [0.1563, 0.3382] |
| KL(true || predicted) averaged over test points@n30_b4 | ls_best_lr | 20 | 0.1808 | 0.0808 | [0.1430, 0.2186] |
| KL(true || predicted) averaged over test points@n30_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.1787 | 0.0670 | [0.1474, 0.2101] |
| KL(true || predicted) averaged over test points@n30_b4 | ls_fixed_eps_tuned_C | 20 | 0.1858 | 0.0659 | [0.1550, 0.2167] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms | ls_fixed_eps_at_eps0_C | 20 | 0.4706 | 0.2059 | [0.3742, 0.5670] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms | ls_fixed_eps_tuned_C | 20 | 0.3856 | 0.2133 | [0.2858, 0.4855] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n100_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.3400 | 0.8580 | [-0.0616, 0.7416] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n100_b1 | ls_fixed_eps_tuned_C | 20 | 0.3550 | 0.7722 | [-0.0064, 0.7164] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n100_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.6400 | 0.6344 | [0.3431, 0.9369] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n100_b4 | ls_fixed_eps_tuned_C | 20 | 0.4100 | 0.6382 | [0.1113, 0.7087] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.8550 | 0.2502 | [0.7379, 0.9721] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n3000_b1 | ls_fixed_eps_tuned_C | 20 | 0.8150 | 0.2796 | [0.6841, 0.9459] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.9900 | 0.0308 | [0.9756, 1.0044] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n3000_b4 | ls_fixed_eps_tuned_C | 20 | 0.9900 | 0.0308 | [0.9756, 1.0044] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n300_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.6150 | 0.5509 | [0.3572, 0.8728] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n300_b1 | ls_fixed_eps_tuned_C | 20 | 0.2750 | 0.7383 | [-0.0705, 0.6205] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n300_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.9100 | 0.1714 | [0.8298, 0.9902] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n300_b4 | ls_fixed_eps_tuned_C | 20 | 0.8050 | 0.1849 | [0.7185, 0.8915] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n30_b1 | ls_fixed_eps_at_eps0_C | 20 | -0.6000 | 0.7539 | [-0.9529, -0.2471] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n30_b1 | ls_fixed_eps_tuned_C | 20 | -0.5450 | 0.7060 | [-0.8754, -0.2146] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n30_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0150 | 0.8647 | [-0.3897, 0.4197] |
| Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n30_b4 | ls_fixed_eps_tuned_C | 20 | -0.0200 | 0.7764 | [-0.3833, 0.3433] |
| crosscheck_a_1e6 | ls_best_lr | 20 | 1.8306 | 0.0012 | [1.8300, 1.8312] |
| crosscheck_a_quadrature | ls_best_lr | 20 | 1.8300 | 0.0000 | [1.8300, 1.8300] |
| impl_check_a_fail | ls_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| impl_check_a_fail@n3000_b1 | ls_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| impl_check_a_fail@n3000_b4 | ls_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| impl_check_b_fail | ls_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| mean_abs_deviation_from_population_LR_LS_probabilities (n=3000, unpenalised C=1e4 fit) | eps0_best_lr | 20 | 0.0185 | 0.0033 | [0.0170, 0.0201] |
| mean_abs_deviation_from_population_LR_LS_probabilities (n=3000, unpenalised C=1e4 fit) | ls_best_lr | 20 | 0.0148 | 0.0027 | [0.0136, 0.0161] |
| mean_abs_deviation_from_population_LR_LS_probabilities (n=3000, unpenalised C=1e4 fit)@n3000_b1 | eps0_best_lr | 20 | 0.0225 | 0.0058 | [0.0198, 0.0252] |
| mean_abs_deviation_from_population_LR_LS_probabilities (n=3000, unpenalised C=1e4 fit)@n3000_b1 | ls_best_lr | 20 | 0.0179 | 0.0046 | [0.0157, 0.0200] |
| mean_abs_deviation_from_population_LR_LS_probabilities (n=3000, unpenalised C=1e4 fit)@n3000_b4 | eps0_best_lr | 20 | 0.0145 | 0.0038 | [0.0128, 0.0163] |
| mean_abs_deviation_from_population_LR_LS_probabilities (n=3000, unpenalised C=1e4 fit)@n3000_b4 | ls_best_lr | 20 | 0.0118 | 0.0027 | [0.0106, 0.0131] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | eps0_best_lr | 20 | 0.0159 | 0.0203 | [0.0064, 0.0254] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | eps0_best_lr_altgrid | 20 | 0.0157 | 0.0204 | [0.0062, 0.0253] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | ls_best_lr | 20 | 0.0023 | 0.0141 | [-0.0043, 0.0089] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | ls_fixed_eps_at_eps0_C | 20 | -0.0175 | 0.0169 | [-0.0254, -0.0096] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | ls_fixed_eps_tuned_C | 20 | -0.0083 | 0.0127 | [-0.0143, -0.0024] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n100_b1 | eps0_best_lr | 20 | 0.0038 | 0.0366 | [-0.0133, 0.0209] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n100_b1 | eps0_best_lr_altgrid | 20 | -0.0034 | 0.0530 | [-0.0282, 0.0214] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n100_b1 | ls_best_lr | 20 | -0.0051 | 0.0352 | [-0.0216, 0.0114] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n100_b1 | ls_fixed_eps_at_eps0_C | 20 | -0.0088 | 0.0316 | [-0.0236, 0.0060] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n100_b1 | ls_fixed_eps_tuned_C | 20 | -0.0079 | 0.0318 | [-0.0228, 0.0069] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n100_b4 | eps0_best_lr | 20 | 0.0000 | 0.0495 | [-0.0231, 0.0232] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n100_b4 | eps0_best_lr_altgrid | 20 | 0.0065 | 0.0487 | [-0.0163, 0.0292] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n100_b4 | ls_best_lr | 20 | -0.0132 | 0.0405 | [-0.0321, 0.0058] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n100_b4 | ls_fixed_eps_at_eps0_C | 20 | -0.0508 | 0.0321 | [-0.0659, -0.0358] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n100_b4 | ls_fixed_eps_tuned_C | 20 | -0.0262 | 0.0251 | [-0.0379, -0.0144] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n3000_b1 | eps0_best_lr | 20 | 0.0002 | 0.0078 | [-0.0034, 0.0039] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n3000_b1 | eps0_best_lr_altgrid | 20 | 0.0011 | 0.0083 | [-0.0028, 0.0050] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n3000_b1 | ls_best_lr | 20 | -0.0068 | 0.0074 | [-0.0102, -0.0033] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | -0.0188 | 0.0067 | [-0.0219, -0.0156] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n3000_b1 | ls_fixed_eps_tuned_C | 20 | -0.0162 | 0.0069 | [-0.0194, -0.0129] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n3000_b4 | eps0_best_lr | 20 | -0.0002 | 0.0049 | [-0.0025, 0.0021] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n3000_b4 | eps0_best_lr_altgrid | 20 | 0.0003 | 0.0055 | [-0.0023, 0.0029] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n3000_b4 | ls_best_lr | 20 | -0.0306 | 0.0040 | [-0.0325, -0.0288] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | -0.0590 | 0.0035 | [-0.0606, -0.0573] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n3000_b4 | ls_fixed_eps_tuned_C | 20 | -0.0581 | 0.0036 | [-0.0597, -0.0564] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n300_b1 | eps0_best_lr | 20 | 0.0001 | 0.0158 | [-0.0073, 0.0074] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n300_b1 | eps0_best_lr_altgrid | 20 | -0.0067 | 0.0359 | [-0.0235, 0.0101] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n300_b1 | ls_best_lr | 20 | -0.0119 | 0.0209 | [-0.0217, -0.0022] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n300_b1 | ls_fixed_eps_at_eps0_C | 20 | -0.0167 | 0.0139 | [-0.0232, -0.0102] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n300_b1 | ls_fixed_eps_tuned_C | 20 | -0.0110 | 0.0197 | [-0.0202, -0.0017] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n300_b4 | eps0_best_lr | 20 | 0.0007 | 0.0166 | [-0.0071, 0.0085] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n300_b4 | eps0_best_lr_altgrid | 20 | 0.0040 | 0.0265 | [-0.0084, 0.0165] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n300_b4 | ls_best_lr | 20 | -0.0169 | 0.0142 | [-0.0235, -0.0102] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n300_b4 | ls_fixed_eps_at_eps0_C | 20 | -0.0530 | 0.0116 | [-0.0584, -0.0476] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n300_b4 | ls_fixed_eps_tuned_C | 20 | -0.0431 | 0.0109 | [-0.0482, -0.0380] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n30_b1 | eps0_best_lr | 20 | 0.1064 | 0.0902 | [0.0642, 0.1486] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n30_b1 | eps0_best_lr_altgrid | 20 | 0.0975 | 0.1080 | [0.0470, 0.1481] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n30_b1 | ls_best_lr | 20 | 0.0980 | 0.0830 | [0.0591, 0.1368] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n30_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0885 | 0.0793 | [0.0513, 0.1256] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n30_b1 | ls_fixed_eps_tuned_C | 20 | 0.0950 | 0.0778 | [0.0586, 0.1314] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n30_b4 | eps0_best_lr | 20 | 0.0160 | 0.0865 | [-0.0245, 0.0565] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n30_b4 | eps0_best_lr_altgrid | 20 | 0.0265 | 0.0858 | [-0.0137, 0.0666] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n30_b4 | ls_best_lr | 20 | 0.0047 | 0.0649 | [-0.0257, 0.0351] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n30_b4 | ls_fixed_eps_at_eps0_C | 20 | -0.0212 | 0.0723 | [-0.0550, 0.0127] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n30_b4 | ls_fixed_eps_tuned_C | 20 | 0.0006 | 0.0770 | [-0.0354, 0.0367] |
| mse_eps0.0 | ls_fixed_eps_at_eps0_C | 20 | 0.0219 | 0.0043 | [0.0199, 0.0239] |
| mse_eps0.0 | ls_fixed_eps_tuned_C | 20 | 0.0219 | 0.0043 | [0.0199, 0.0239] |
| mse_eps0.02 | ls_fixed_eps_at_eps0_C | 20 | 0.0216 | 0.0044 | [0.0196, 0.0237] |
| mse_eps0.02 | ls_fixed_eps_tuned_C | 20 | 0.0221 | 0.0052 | [0.0196, 0.0245] |
| mse_eps0.02@n100_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0216 | 0.0102 | [0.0168, 0.0264] |
| mse_eps0.02@n100_b1 | ls_fixed_eps_tuned_C | 20 | 0.0216 | 0.0102 | [0.0168, 0.0264] |
| mse_eps0.02@n100_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0197 | 0.0103 | [0.0149, 0.0245] |
| mse_eps0.02@n100_b4 | ls_fixed_eps_tuned_C | 20 | 0.0200 | 0.0107 | [0.0150, 0.0251] |
| mse_eps0.02@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0009 | 0.0005 | [0.0006, 0.0011] |
| mse_eps0.02@n3000_b1 | ls_fixed_eps_tuned_C | 20 | 0.0008 | 0.0005 | [0.0006, 0.0011] |
| mse_eps0.02@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0009 | 0.0003 | [0.0007, 0.0010] |
| mse_eps0.02@n3000_b4 | ls_fixed_eps_tuned_C | 20 | 0.0008 | 0.0003 | [0.0007, 0.0010] |
| mse_eps0.02@n300_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0072 | 0.0028 | [0.0059, 0.0085] |
| mse_eps0.02@n300_b1 | ls_fixed_eps_tuned_C | 20 | 0.0072 | 0.0028 | [0.0059, 0.0085] |
| mse_eps0.02@n300_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0062 | 0.0019 | [0.0054, 0.0071] |
| mse_eps0.02@n300_b4 | ls_fixed_eps_tuned_C | 20 | 0.0062 | 0.0017 | [0.0054, 0.0070] |
| mse_eps0.02@n30_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0531 | 0.0246 | [0.0416, 0.0647] |
| mse_eps0.02@n30_b1 | ls_fixed_eps_tuned_C | 20 | 0.0553 | 0.0294 | [0.0415, 0.0690] |
| mse_eps0.02@n30_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0634 | 0.0249 | [0.0517, 0.0750] |
| mse_eps0.02@n30_b4 | ls_fixed_eps_tuned_C | 20 | 0.0646 | 0.0290 | [0.0510, 0.0781] |
| mse_eps0.05 | ls_fixed_eps_at_eps0_C | 20 | 0.0215 | 0.0042 | [0.0195, 0.0235] |
| mse_eps0.05 | ls_fixed_eps_tuned_C | 20 | 0.0211 | 0.0046 | [0.0189, 0.0233] |
| mse_eps0.05@n100_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0215 | 0.0102 | [0.0167, 0.0263] |
| mse_eps0.05@n100_b1 | ls_fixed_eps_tuned_C | 20 | 0.0215 | 0.0102 | [0.0167, 0.0263] |
| mse_eps0.05@n100_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0197 | 0.0097 | [0.0152, 0.0243] |
| mse_eps0.05@n100_b4 | ls_fixed_eps_tuned_C | 20 | 0.0183 | 0.0084 | [0.0144, 0.0222] |
| mse_eps0.05@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0009 | 0.0005 | [0.0007, 0.0012] |
| mse_eps0.05@n3000_b1 | ls_fixed_eps_tuned_C | 20 | 0.0009 | 0.0005 | [0.0007, 0.0011] |
| mse_eps0.05@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0020 | 0.0004 | [0.0018, 0.0022] |
| mse_eps0.05@n3000_b4 | ls_fixed_eps_tuned_C | 20 | 0.0019 | 0.0004 | [0.0017, 0.0021] |
| mse_eps0.05@n300_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0072 | 0.0028 | [0.0059, 0.0085] |
| mse_eps0.05@n300_b1 | ls_fixed_eps_tuned_C | 20 | 0.0073 | 0.0027 | [0.0060, 0.0086] |
| mse_eps0.05@n300_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0071 | 0.0021 | [0.0061, 0.0081] |
| mse_eps0.05@n300_b4 | ls_fixed_eps_tuned_C | 20 | 0.0064 | 0.0020 | [0.0054, 0.0073] |
| mse_eps0.05@n30_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0514 | 0.0227 | [0.0408, 0.0620] |
| mse_eps0.05@n30_b1 | ls_fixed_eps_tuned_C | 20 | 0.0509 | 0.0252 | [0.0391, 0.0627] |
| mse_eps0.05@n30_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0622 | 0.0244 | [0.0508, 0.0736] |
| mse_eps0.05@n30_b4 | ls_fixed_eps_tuned_C | 20 | 0.0617 | 0.0255 | [0.0498, 0.0736] |
| mse_eps0.0@n100_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0217 | 0.0102 | [0.0169, 0.0264] |
| mse_eps0.0@n100_b1 | ls_fixed_eps_tuned_C | 20 | 0.0217 | 0.0102 | [0.0169, 0.0264] |
| mse_eps0.0@n100_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0205 | 0.0112 | [0.0153, 0.0258] |
| mse_eps0.0@n100_b4 | ls_fixed_eps_tuned_C | 20 | 0.0205 | 0.0112 | [0.0153, 0.0258] |
| mse_eps0.0@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0008 | 0.0005 | [0.0006, 0.0011] |
| mse_eps0.0@n3000_b1 | ls_fixed_eps_tuned_C | 20 | 0.0008 | 0.0005 | [0.0006, 0.0011] |
| mse_eps0.0@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0006 | 0.0003 | [0.0005, 0.0008] |
| mse_eps0.0@n3000_b4 | ls_fixed_eps_tuned_C | 20 | 0.0006 | 0.0003 | [0.0005, 0.0008] |
| mse_eps0.0@n300_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0072 | 0.0028 | [0.0059, 0.0085] |
| mse_eps0.0@n300_b1 | ls_fixed_eps_tuned_C | 20 | 0.0072 | 0.0028 | [0.0059, 0.0085] |
| mse_eps0.0@n300_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0062 | 0.0018 | [0.0053, 0.0070] |
| mse_eps0.0@n300_b4 | ls_fixed_eps_tuned_C | 20 | 0.0062 | 0.0018 | [0.0053, 0.0070] |
| mse_eps0.0@n30_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0543 | 0.0260 | [0.0421, 0.0665] |
| mse_eps0.0@n30_b1 | ls_fixed_eps_tuned_C | 20 | 0.0543 | 0.0260 | [0.0421, 0.0665] |
| mse_eps0.0@n30_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0638 | 0.0252 | [0.0520, 0.0756] |
| mse_eps0.0@n30_b4 | ls_fixed_eps_tuned_C | 20 | 0.0638 | 0.0252 | [0.0520, 0.0756] |
| mse_eps0.1 | ls_fixed_eps_at_eps0_C | 20 | 0.0220 | 0.0040 | [0.0202, 0.0239] |
| mse_eps0.1 | ls_fixed_eps_tuned_C | 20 | 0.0216 | 0.0044 | [0.0196, 0.0237] |
| mse_eps0.1@n100_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0215 | 0.0103 | [0.0166, 0.0263] |
| mse_eps0.1@n100_b1 | ls_fixed_eps_tuned_C | 20 | 0.0209 | 0.0097 | [0.0163, 0.0255] |
| mse_eps0.1@n100_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0215 | 0.0094 | [0.0171, 0.0259] |
| mse_eps0.1@n100_b4 | ls_fixed_eps_tuned_C | 20 | 0.0186 | 0.0075 | [0.0151, 0.0221] |
| mse_eps0.1@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0013 | 0.0006 | [0.0010, 0.0016] |
| mse_eps0.1@n3000_b1 | ls_fixed_eps_tuned_C | 20 | 0.0012 | 0.0006 | [0.0009, 0.0014] |
| mse_eps0.1@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0052 | 0.0005 | [0.0049, 0.0054] |
| mse_eps0.1@n3000_b4 | ls_fixed_eps_tuned_C | 20 | 0.0050 | 0.0006 | [0.0048, 0.0053] |
| mse_eps0.1@n300_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0075 | 0.0029 | [0.0061, 0.0088] |
| mse_eps0.1@n300_b1 | ls_fixed_eps_tuned_C | 20 | 0.0075 | 0.0029 | [0.0061, 0.0088] |
| mse_eps0.1@n300_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0097 | 0.0024 | [0.0086, 0.0108] |
| mse_eps0.1@n300_b4 | ls_fixed_eps_tuned_C | 20 | 0.0084 | 0.0022 | [0.0073, 0.0094] |
| mse_eps0.1@n30_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0488 | 0.0199 | [0.0395, 0.0581] |
| mse_eps0.1@n30_b1 | ls_fixed_eps_tuned_C | 20 | 0.0497 | 0.0214 | [0.0397, 0.0597] |
| mse_eps0.1@n30_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0610 | 0.0236 | [0.0499, 0.0720] |
| mse_eps0.1@n30_b4 | ls_fixed_eps_tuned_C | 20 | 0.0620 | 0.0226 | [0.0514, 0.0725] |
| mse_eps0.2 | ls_fixed_eps_at_eps0_C | 20 | 0.0250 | 0.0037 | [0.0232, 0.0267] |
| mse_eps0.2 | ls_fixed_eps_tuned_C | 20 | 0.0239 | 0.0042 | [0.0220, 0.0259] |
| mse_eps0.2@n100_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0218 | 0.0104 | [0.0169, 0.0266] |
| mse_eps0.2@n100_b1 | ls_fixed_eps_tuned_C | 20 | 0.0215 | 0.0097 | [0.0169, 0.0260] |
| mse_eps0.2@n100_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0284 | 0.0090 | [0.0242, 0.0326] |
| mse_eps0.2@n100_b4 | ls_fixed_eps_tuned_C | 20 | 0.0241 | 0.0064 | [0.0211, 0.0271] |
| mse_eps0.2@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0026 | 0.0008 | [0.0022, 0.0030] |
| mse_eps0.2@n3000_b1 | ls_fixed_eps_tuned_C | 20 | 0.0024 | 0.0008 | [0.0020, 0.0028] |
| mse_eps0.2@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0144 | 0.0008 | [0.0140, 0.0147] |
| mse_eps0.2@n3000_b4 | ls_fixed_eps_tuned_C | 20 | 0.0142 | 0.0008 | [0.0139, 0.0146] |
| mse_eps0.2@n300_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0085 | 0.0030 | [0.0071, 0.0099] |
| mse_eps0.2@n300_b1 | ls_fixed_eps_tuned_C | 20 | 0.0078 | 0.0031 | [0.0064, 0.0093] |
| mse_eps0.2@n300_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0179 | 0.0027 | [0.0166, 0.0191] |
| mse_eps0.2@n300_b4 | ls_fixed_eps_tuned_C | 20 | 0.0164 | 0.0026 | [0.0152, 0.0176] |
| mse_eps0.2@n30_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0443 | 0.0157 | [0.0369, 0.0517] |
| mse_eps0.2@n30_b1 | ls_fixed_eps_tuned_C | 20 | 0.0469 | 0.0239 | [0.0357, 0.0580] |
| mse_eps0.2@n30_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0618 | 0.0231 | [0.0510, 0.0726] |
| mse_eps0.2@n30_b4 | ls_fixed_eps_tuned_C | 20 | 0.0582 | 0.0200 | [0.0488, 0.0676] |
| mse_vs_true_probability | eps0_best_lr | 20 | 0.0219 | 0.0043 | [0.0199, 0.0239] |
| mse_vs_true_probability | eps0_best_lr_altgrid | 20 | 0.0224 | 0.0052 | [0.0200, 0.0249] |
| mse_vs_true_probability | ls_best_lr | 20 | 0.0209 | 0.0045 | [0.0188, 0.0230] |
| mse_vs_true_probability | ls_fixed_eps_at_eps0_C | 20 | 0.0220 | 0.0040 | [0.0202, 0.0239] |
| mse_vs_true_probability | ls_fixed_eps_tuned_C | 20 | 0.0216 | 0.0044 | [0.0196, 0.0237] |
| mse_vs_true_probability@n100_b1 | eps0_best_lr | 20 | 0.0217 | 0.0102 | [0.0169, 0.0264] |
| mse_vs_true_probability@n100_b1 | eps0_best_lr_altgrid | 20 | 0.0221 | 0.0067 | [0.0190, 0.0252] |
| mse_vs_true_probability@n100_b1 | ls_best_lr | 20 | 0.0211 | 0.0096 | [0.0166, 0.0255] |
| mse_vs_true_probability@n100_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0215 | 0.0103 | [0.0166, 0.0263] |
| mse_vs_true_probability@n100_b1 | ls_fixed_eps_tuned_C | 20 | 0.0209 | 0.0097 | [0.0163, 0.0255] |
| mse_vs_true_probability@n100_b4 | eps0_best_lr | 20 | 0.0205 | 0.0112 | [0.0153, 0.0258] |
| mse_vs_true_probability@n100_b4 | eps0_best_lr_altgrid | 20 | 0.0197 | 0.0091 | [0.0154, 0.0240] |
| mse_vs_true_probability@n100_b4 | ls_best_lr | 20 | 0.0192 | 0.0087 | [0.0151, 0.0233] |
| mse_vs_true_probability@n100_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0215 | 0.0094 | [0.0171, 0.0259] |
| mse_vs_true_probability@n100_b4 | ls_fixed_eps_tuned_C | 20 | 0.0186 | 0.0075 | [0.0151, 0.0221] |
| mse_vs_true_probability@n3000_b1 | eps0_best_lr | 20 | 0.0008 | 0.0005 | [0.0006, 0.0011] |
| mse_vs_true_probability@n3000_b1 | eps0_best_lr_altgrid | 20 | 0.0009 | 0.0005 | [0.0006, 0.0011] |
| mse_vs_true_probability@n3000_b1 | ls_best_lr | 20 | 0.0009 | 0.0005 | [0.0007, 0.0011] |
| mse_vs_true_probability@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0013 | 0.0006 | [0.0010, 0.0016] |
| mse_vs_true_probability@n3000_b1 | ls_fixed_eps_tuned_C | 20 | 0.0012 | 0.0006 | [0.0009, 0.0014] |
| mse_vs_true_probability@n3000_b4 | eps0_best_lr | 20 | 0.0006 | 0.0003 | [0.0005, 0.0008] |
| mse_vs_true_probability@n3000_b4 | eps0_best_lr_altgrid | 20 | 0.0006 | 0.0003 | [0.0005, 0.0008] |
| mse_vs_true_probability@n3000_b4 | ls_best_lr | 20 | 0.0019 | 0.0004 | [0.0017, 0.0021] |
| mse_vs_true_probability@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0052 | 0.0005 | [0.0049, 0.0054] |
| mse_vs_true_probability@n3000_b4 | ls_fixed_eps_tuned_C | 20 | 0.0050 | 0.0006 | [0.0048, 0.0053] |
| mse_vs_true_probability@n300_b1 | eps0_best_lr | 20 | 0.0072 | 0.0028 | [0.0059, 0.0085] |
| mse_vs_true_probability@n300_b1 | eps0_best_lr_altgrid | 20 | 0.0087 | 0.0029 | [0.0073, 0.0101] |
| mse_vs_true_probability@n300_b1 | ls_best_lr | 20 | 0.0076 | 0.0031 | [0.0062, 0.0090] |
| mse_vs_true_probability@n300_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0075 | 0.0029 | [0.0061, 0.0088] |
| mse_vs_true_probability@n300_b1 | ls_fixed_eps_tuned_C | 20 | 0.0075 | 0.0029 | [0.0061, 0.0088] |
| mse_vs_true_probability@n300_b4 | eps0_best_lr | 20 | 0.0062 | 0.0018 | [0.0053, 0.0070] |
| mse_vs_true_probability@n300_b4 | eps0_best_lr_altgrid | 20 | 0.0067 | 0.0017 | [0.0059, 0.0075] |
| mse_vs_true_probability@n300_b4 | ls_best_lr | 20 | 0.0064 | 0.0020 | [0.0055, 0.0073] |
| mse_vs_true_probability@n300_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0097 | 0.0024 | [0.0086, 0.0108] |
| mse_vs_true_probability@n300_b4 | ls_fixed_eps_tuned_C | 20 | 0.0084 | 0.0022 | [0.0073, 0.0094] |
| mse_vs_true_probability@n30_b1 | eps0_best_lr | 20 | 0.0543 | 0.0260 | [0.0421, 0.0665] |
| mse_vs_true_probability@n30_b1 | eps0_best_lr_altgrid | 20 | 0.0557 | 0.0312 | [0.0411, 0.0703] |
| mse_vs_true_probability@n30_b1 | ls_best_lr | 20 | 0.0507 | 0.0268 | [0.0382, 0.0633] |
| mse_vs_true_probability@n30_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0488 | 0.0199 | [0.0395, 0.0581] |
| mse_vs_true_probability@n30_b1 | ls_fixed_eps_tuned_C | 20 | 0.0497 | 0.0214 | [0.0397, 0.0597] |
| mse_vs_true_probability@n30_b4 | eps0_best_lr | 20 | 0.0638 | 0.0252 | [0.0520, 0.0756] |
| mse_vs_true_probability@n30_b4 | eps0_best_lr_altgrid | 20 | 0.0651 | 0.0226 | [0.0545, 0.0757] |
| mse_vs_true_probability@n30_b4 | ls_best_lr | 20 | 0.0594 | 0.0227 | [0.0488, 0.0701] |
| mse_vs_true_probability@n30_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0610 | 0.0236 | [0.0499, 0.0720] |
| mse_vs_true_probability@n30_b4 | ls_fixed_eps_tuned_C | 20 | 0.0620 | 0.0226 | [0.0514, 0.0725] |
| n_convergence_warnings_total | eps0_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| n_convergence_warnings_total | eps0_best_lr_altgrid | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| n_convergence_warnings_total | ls_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| n_convergence_warnings_total | ls_fixed_eps_at_eps0_C | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| n_convergence_warnings_total | ls_fixed_eps_tuned_C | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) | eps0_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) | eps0_best_lr_altgrid | 20 | 0.0006 | 0.0031 | [-0.0009, 0.0020] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) | ls_best_lr | 20 | -0.0010 | 0.0031 | [-0.0024, 0.0005] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) | ls_fixed_eps_at_eps0_C | 20 | 0.0002 | 0.0017 | [-0.0007, 0.0010] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) | ls_fixed_eps_tuned_C | 20 | -0.0002 | 0.0034 | [-0.0019, 0.0014] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n100_b1 | eps0_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n100_b1 | eps0_best_lr_altgrid | 20 | 0.0004 | 0.0062 | [-0.0025, 0.0034] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n100_b1 | ls_best_lr | 20 | -0.0006 | 0.0030 | [-0.0020, 0.0008] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n100_b1 | ls_fixed_eps_at_eps0_C | 20 | -0.0002 | 0.0016 | [-0.0010, 0.0005] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n100_b1 | ls_fixed_eps_tuned_C | 20 | -0.0008 | 0.0029 | [-0.0021, 0.0006] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n100_b4 | eps0_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n100_b4 | eps0_best_lr_altgrid | 20 | -0.0008 | 0.0061 | [-0.0037, 0.0020] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n100_b4 | ls_best_lr | 20 | -0.0013 | 0.0054 | [-0.0039, 0.0012] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n100_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0010 | 0.0070 | [-0.0023, 0.0043] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n100_b4 | ls_fixed_eps_tuned_C | 20 | -0.0020 | 0.0072 | [-0.0054, 0.0014] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n3000_b1 | eps0_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n3000_b1 | eps0_best_lr_altgrid | 20 | 0.0000 | 0.0001 | [-0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n3000_b1 | ls_best_lr | 20 | 0.0001 | 0.0001 | [-0.0000, 0.0001] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0004 | 0.0003 | [0.0003, 0.0006] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n3000_b1 | ls_fixed_eps_tuned_C | 20 | 0.0003 | 0.0003 | [0.0002, 0.0005] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n3000_b4 | eps0_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n3000_b4 | eps0_best_lr_altgrid | 20 | 0.0000 | 0.0000 | [-0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n3000_b4 | ls_best_lr | 20 | 0.0013 | 0.0003 | [0.0011, 0.0014] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0045 | 0.0005 | [0.0043, 0.0048] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n3000_b4 | ls_fixed_eps_tuned_C | 20 | 0.0044 | 0.0005 | [0.0042, 0.0046] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n300_b1 | eps0_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n300_b1 | eps0_best_lr_altgrid | 20 | 0.0015 | 0.0012 | [0.0010, 0.0021] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n300_b1 | ls_best_lr | 20 | 0.0004 | 0.0010 | [-0.0001, 0.0009] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n300_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0003 | 0.0006 | [0.0000, 0.0006] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n300_b1 | ls_fixed_eps_tuned_C | 20 | 0.0003 | 0.0005 | [0.0001, 0.0006] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n300_b4 | eps0_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n300_b4 | eps0_best_lr_altgrid | 20 | 0.0005 | 0.0010 | [0.0001, 0.0010] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n300_b4 | ls_best_lr | 20 | 0.0002 | 0.0010 | [-0.0002, 0.0007] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n300_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0035 | 0.0017 | [0.0027, 0.0043] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n300_b4 | ls_fixed_eps_tuned_C | 20 | 0.0022 | 0.0013 | [0.0016, 0.0028] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n30_b1 | eps0_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n30_b1 | eps0_best_lr_altgrid | 20 | 0.0014 | 0.0176 | [-0.0068, 0.0096] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n30_b1 | ls_best_lr | 20 | -0.0036 | 0.0207 | [-0.0133, 0.0061] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n30_b1 | ls_fixed_eps_at_eps0_C | 20 | -0.0055 | 0.0079 | [-0.0092, -0.0018] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n30_b1 | ls_fixed_eps_tuned_C | 20 | -0.0046 | 0.0183 | [-0.0131, 0.0039] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n30_b4 | eps0_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n30_b4 | eps0_best_lr_altgrid | 20 | 0.0013 | 0.0142 | [-0.0053, 0.0080] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n30_b4 | ls_best_lr | 20 | -0.0044 | 0.0154 | [-0.0116, 0.0028] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n30_b4 | ls_fixed_eps_at_eps0_C | 20 | -0.0028 | 0.0101 | [-0.0075, 0.0019] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr)@n30_b4 | ls_fixed_eps_tuned_C | 20 | -0.0018 | 0.0176 | [-0.0101, 0.0064] |
| pop_a_b1_eps0.0 | ls_best_lr | 20 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| pop_a_b1_eps0.02 | ls_best_lr | 20 | 0.9717 | 0.0000 | [0.9717, 0.9717] |
| pop_a_b1_eps0.05 | ls_best_lr | 20 | 0.9305 | 0.0000 | [0.9305, 0.9305] |
| pop_a_b1_eps0.1 | ls_best_lr | 20 | 0.8647 | 0.0000 | [0.8647, 0.8647] |
| pop_a_b1_eps0.2 | ls_best_lr | 20 | 0.7429 | 0.0000 | [0.7429, 0.7429] |
| pop_a_b4_eps0.0 | ls_best_lr | 20 | 4.0000 | 0.0000 | [4.0000, 4.0000] |
| pop_a_b4_eps0.02 | ls_best_lr | 20 | 3.5603 | 0.0000 | [3.5603, 3.5603] |
| pop_a_b4_eps0.05 | ls_best_lr | 20 | 3.0733 | 0.0000 | [3.0733, 3.0733] |
| pop_a_b4_eps0.1 | ls_best_lr | 20 | 2.5144 | 0.0000 | [2.5144, 2.5144] |
| pop_a_b4_eps0.2 | ls_best_lr | 20 | 1.8300 | 0.0000 | [1.8300, 1.8300] |
| pop_mse_b1_eps0.0 | ls_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| pop_mse_b1_eps0.02 | ls_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| pop_mse_b1_eps0.05 | ls_best_lr | 20 | 0.0001 | 0.0000 | [0.0001, 0.0001] |
| pop_mse_b1_eps0.1 | ls_best_lr | 20 | 0.0005 | 0.0000 | [0.0005, 0.0005] |
| pop_mse_b1_eps0.2 | ls_best_lr | 20 | 0.0019 | 0.0000 | [0.0019, 0.0019] |
| pop_mse_b4_eps0.0 | ls_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| pop_mse_b4_eps0.02 | ls_best_lr | 20 | 0.0003 | 0.0000 | [0.0003, 0.0003] |
| pop_mse_b4_eps0.05 | ls_best_lr | 20 | 0.0014 | 0.0000 | [0.0014, 0.0014] |
| pop_mse_b4_eps0.1 | ls_best_lr | 20 | 0.0046 | 0.0000 | [0.0046, 0.0046] |
| pop_mse_b4_eps0.2 | ls_best_lr | 20 | 0.0139 | 0.0000 | [0.0139, 0.0139] |
| popdev_eps0.0 | ls_fixed_eps_at_eps0_C | 20 | 0.0185 | 0.0033 | [0.0170, 0.0201] |
| popdev_eps0.05 | ls_fixed_eps_at_eps0_C | 20 | 0.0175 | 0.0032 | [0.0160, 0.0190] |
| popdev_eps0.05@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0213 | 0.0055 | [0.0187, 0.0239] |
| popdev_eps0.05@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0137 | 0.0034 | [0.0122, 0.0153] |
| popdev_eps0.0@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0225 | 0.0058 | [0.0198, 0.0252] |
| popdev_eps0.0@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0145 | 0.0038 | [0.0128, 0.0163] |
| popdev_eps0.1 | ls_fixed_eps_at_eps0_C | 20 | 0.0166 | 0.0030 | [0.0152, 0.0181] |
| popdev_eps0.1@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0202 | 0.0052 | [0.0177, 0.0226] |
| popdev_eps0.1@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0131 | 0.0031 | [0.0116, 0.0145] |
| popdev_eps0.2 | ls_fixed_eps_at_eps0_C | 20 | 0.0148 | 0.0027 | [0.0136, 0.0161] |
| popdev_eps0.2@n3000_b1 | ls_fixed_eps_at_eps0_C | 20 | 0.0179 | 0.0046 | [0.0157, 0.0200] |
| popdev_eps0.2@n3000_b4 | ls_fixed_eps_at_eps0_C | 20 | 0.0118 | 0.0027 | [0.0106, 0.0131] |
| sel_C_log10 | eps0_best_lr | 20 | -0.5312 | 0.2921 | [-0.6680, -0.3945] |
| sel_C_log10 | eps0_best_lr_altgrid | 20 | -0.4062 | 0.2290 | [-0.5134, -0.2991] |
| sel_C_log10 | ls_best_lr | 20 | 0.0375 | 0.3038 | [-0.1047, 0.1797] |
| sel_C_log10@n100_b1 | eps0_best_lr | 20 | -1.2500 | 0.5501 | [-1.5075, -0.9925] |
| sel_C_log10@n100_b1 | eps0_best_lr_altgrid | 20 | -1.2000 | 0.5712 | [-1.4673, -0.9327] |
| sel_C_log10@n100_b1 | ls_best_lr | 20 | -1.1000 | 0.6407 | [-1.3999, -0.8001] |
| sel_C_log10@n100_b4 | eps0_best_lr | 20 | 0.1500 | 0.4894 | [-0.0790, 0.3790] |
| sel_C_log10@n100_b4 | eps0_best_lr_altgrid | 20 | 0.2500 | 0.4443 | [0.0421, 0.4579] |
| sel_C_log10@n100_b4 | ls_best_lr | 20 | 0.9500 | 0.6863 | [0.6288, 1.2712] |
| sel_C_log10@n3000_b1 | eps0_best_lr | 20 | -1.0000 | 0.0000 | [-1.0000, -1.0000] |
| sel_C_log10@n3000_b1 | eps0_best_lr_altgrid | 20 | -0.6500 | 0.3663 | [-0.8215, -0.4785] |
| sel_C_log10@n3000_b1 | ls_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| sel_C_log10@n3000_b4 | eps0_best_lr | 20 | 0.2500 | 0.4443 | [0.0421, 0.4579] |
| sel_C_log10@n3000_b4 | eps0_best_lr_altgrid | 20 | 0.4000 | 0.3078 | [0.2559, 0.5441] |
| sel_C_log10@n3000_b4 | ls_best_lr | 20 | 1.2500 | 0.4443 | [1.0421, 1.4579] |
| sel_C_log10@n300_b1 | eps0_best_lr | 20 | -1.0000 | 0.0000 | [-1.0000, -1.0000] |
| sel_C_log10@n300_b1 | eps0_best_lr_altgrid | 20 | -1.0000 | 0.5130 | [-1.2401, -0.7599] |
| sel_C_log10@n300_b1 | ls_best_lr | 20 | -0.7500 | 0.4443 | [-0.9579, -0.5421] |
| sel_C_log10@n300_b4 | eps0_best_lr | 20 | 0.1500 | 0.3663 | [-0.0215, 0.3215] |
| sel_C_log10@n300_b4 | eps0_best_lr_altgrid | 20 | 0.3000 | 0.4104 | [0.1079, 0.4921] |
| sel_C_log10@n300_b4 | ls_best_lr | 20 | 1.0500 | 0.5104 | [0.8111, 1.2889] |
| sel_C_log10@n30_b1 | eps0_best_lr | 20 | -1.7500 | 1.7434 | [-2.5659, -0.9341] |
| sel_C_log10@n30_b1 | eps0_best_lr_altgrid | 20 | -1.6500 | 1.4965 | [-2.3504, -0.9496] |
| sel_C_log10@n30_b1 | ls_best_lr | 20 | -1.6000 | 1.8180 | [-2.4509, -0.7491] |
| sel_C_log10@n30_b4 | eps0_best_lr | 20 | 0.2000 | 0.8944 | [-0.2186, 0.6186] |
| sel_C_log10@n30_b4 | eps0_best_lr_altgrid | 20 | 0.3000 | 0.9515 | [-0.1453, 0.7453] |
| sel_C_log10@n30_b4 | ls_best_lr | 20 | 0.5000 | 0.8885 | [0.0842, 0.9158] |
| sel_eps | ls_best_lr | 20 | 0.0850 | 0.0145 | [0.0782, 0.0918] |
| sel_eps@n100_b1 | ls_best_lr | 20 | 0.1150 | 0.0727 | [0.0810, 0.1490] |
| sel_eps@n100_b4 | ls_best_lr | 20 | 0.0775 | 0.0472 | [0.0554, 0.0996] |
| sel_eps@n3000_b1 | ls_best_lr | 20 | 0.0500 | 0.0000 | [0.0500, 0.0500] |
| sel_eps@n3000_b4 | ls_best_lr | 20 | 0.0500 | 0.0000 | [0.0500, 0.0500] |
| sel_eps@n300_b1 | ls_best_lr | 20 | 0.1050 | 0.0667 | [0.0738, 0.1362] |
| sel_eps@n300_b4 | ls_best_lr | 20 | 0.0500 | 0.0000 | [0.0500, 0.0500] |
| sel_eps@n30_b1 | ls_best_lr | 20 | 0.1250 | 0.0659 | [0.0942, 0.1558] |
| sel_eps@n30_b4 | ls_best_lr | 20 | 0.1075 | 0.0712 | [0.0742, 0.1408] |

| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |
|---|---|---|---|---|---|---|---|
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) (primary) | ls_best_lr vs eps0_best_lr | -0.0010 | [-0.0024, 0.0005] | 0.9731 | -0.44 | no | no |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) (primary) | ls_best_lr vs eps0_best_lr_altgrid | -0.0015 | [-0.0035, 0.0005] | 0.8768 | -0.49 | no | yes |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) (primary) | ls_best_lr vs ls_fixed_eps_at_eps0_C | -0.0011 | [-0.0028, 0.0005] | 0.9731 | -0.44 | no | yes |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) (primary) | ls_best_lr vs ls_fixed_eps_tuned_C | -0.0007 | [-0.0028, 0.0014] | 1.0000 | -0.22 | no | no |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | ls_best_lr vs eps0_best_lr | -0.0136 | [-0.0248, -0.0024] | 0.1917 | -0.76 | no | yes |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | ls_best_lr vs eps0_best_lr_altgrid | -0.0135 | [-0.0247, -0.0022] | 0.1917 | -0.75 | no | yes |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | ls_best_lr vs ls_fixed_eps_at_eps0_C | 0.0198 | [0.0098, 0.0297] | 0.0037 | 1.24 | yes | yes |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | ls_best_lr vs ls_fixed_eps_tuned_C | 0.0106 | [0.0020, 0.0192] | 0.1871 | 0.77 | no | yes |
| KL(true || predicted) averaged over test points | ls_best_lr vs eps0_best_lr | -0.0112 | [-0.0276, 0.0052] | 0.9731 | -0.43 | no | no |
| KL(true || predicted) averaged over test points | ls_best_lr vs eps0_best_lr_altgrid | -0.0143 | [-0.0320, 0.0034] | 0.8663 | -0.51 | no | yes |
| KL(true || predicted) averaged over test points | ls_best_lr vs ls_fixed_eps_at_eps0_C | -0.0036 | [-0.0143, 0.0072] | 1.0000 | -0.21 | no | no |
| KL(true || predicted) averaged over test points | ls_best_lr vs ls_fixed_eps_tuned_C | -0.0027 | [-0.0138, 0.0085] | 1.0000 | -0.15 | no | no |
| mean_abs_deviation_from_population_LR_LS_probabilities (n=3000, unpenalised C=1e4 fit) | ls_best_lr vs eps0_best_lr | -0.0037 | [-0.0056, -0.0017] | 0.0057 | -1.19 | yes | yes |

**Table 4: R004, E5@v2 (primary) testing H4@v1, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
| area_between_pareto_frontiers (eps frontier minus L2 frontier in log loss, integrated over the shared accuracy range) with bootstrap CI | eps_curve_at_tuned_l2 | 20 | -0.0000 | 0.0000 | [-0.0000, 0.0000] |
| diff_S1_lr | ls_best_raw | 20 | 0.0003 | 0.0014 | [-0.0004, 0.0009] |
| diff_S2_lr | ls_best_raw | 20 | 0.0000 | 0.0002 | [-0.0001, 0.0001] |
| diff_S2_mlp | ls_best_raw | 20 | -0.0005 | 0.0011 | [-0.0010, 0.0000] |
| diff_S3_lr | ls_best_raw | 20 | -0.0000 | 0.0007 | [-0.0003, 0.0003] |
| diff_S4_lr | ls_best_raw | 20 | -0.0000 | 0.0004 | [-0.0002, 0.0002] |
| diff_S4_mlp | ls_best_raw | 20 | -0.0006 | 0.0042 | [-0.0026, 0.0013] |
| diff_S5_lr | ls_best_raw | 20 | -0.0014 | 0.0063 | [-0.0043, 0.0016] |
| diff_S6_lr | ls_best_raw | 20 | -0.0004 | 0.0006 | [-0.0007, -0.0001] |
| frac_units_at_ceiling | ls_best_raw | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fraction_of_test_predictions_with_changed_argmax | eps0_best_raw | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fraction_of_test_predictions_with_changed_argmax | ls_best_raw | 20 | 0.0114 | 0.0057 | [0.0088, 0.0141] |
| non_dominance_frequency per eps point: share of replicate-bootstrap resamples in which no L2 point has mean accuracy >= and mean log loss <= that eps point | eps_curve_at_tuned_l2 | 20 | 0.3337 | 0.0915 | [0.2909, 0.3766] |
| paired_test_accuracy_difference (ls_best_raw minus eps0_best_raw), one-sided 95% lower bound | eps0_best_raw | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_test_accuracy_difference (ls_best_raw minus eps0_best_raw), one-sided 95% lower bound | ls_best_raw | 20 | -0.0003 | 0.0010 | [-0.0008, 0.0001] |
| real_breast_cancer_area | ls_best_raw | 20 | 0.0001 | 0.0001 | [0.0000, 0.0001] |
| real_breast_cancer_diff | ls_best_raw | 20 | -0.0083 | 0.0039 | [-0.0102, -0.0065] |
| real_breast_cancer_nondom | ls_best_raw | 20 | 0.0520 | 0.0650 | [0.0216, 0.0824] |
| real_digits_area | ls_best_raw | 20 | 0.0000 | 0.0001 | [0.0000, 0.0001] |
| real_digits_diff | ls_best_raw | 20 | -0.0051 | 0.0024 | [-0.0062, -0.0039] |
| real_digits_nondom | ls_best_raw | 20 | 0.0820 | 0.0826 | [0.0434, 0.1206] |
| real_iris_area | ls_best_raw | 20 | 0.0011 | 0.0006 | [0.0008, 0.0014] |
| real_iris_diff | ls_best_raw | 20 | 0.0010 | 0.0069 | [-0.0022, 0.0042] |
| real_iris_nondom | ls_best_raw | 20 | 0.0280 | 0.0451 | [0.0069, 0.0491] |
| real_wine_area | ls_best_raw | 20 | -0.0000 | 0.0004 | [-0.0002, 0.0002] |
| real_wine_diff | ls_best_raw | 20 | 0.0028 | 0.0085 | [-0.0012, 0.0068] |
| real_wine_nondom | ls_best_raw | 20 | 0.2160 | 0.1000 | [0.1692, 0.2628] |
| test_accuracy | eps0_best_raw | 20 | 0.8036 | 0.0032 | [0.8021, 0.8051] |
| test_accuracy | eps_curve_at_tuned_l2 | 20 | 0.8031 | 0.0031 | [0.8016, 0.8045] |
| test_accuracy | l2_curve_at_eps0 | 20 | 0.7697 | 0.0054 | [0.7671, 0.7722] |
| test_accuracy | ls_best_raw | 20 | 0.8033 | 0.0033 | [0.8017, 0.8048] |
| test_log_loss and accuracy per curve point | eps0_best_raw | 20 | 0.4367 | 0.0073 | [0.4333, 0.4401] |
| test_log_loss and accuracy per curve point | eps_curve_at_tuned_l2 | 20 | 0.4584 | 0.0058 | [0.4557, 0.4611] |
| test_log_loss and accuracy per curve point | l2_curve_at_eps0 | 20 | 0.5667 | 0.0245 | [0.5552, 0.5781] |
| test_log_loss and accuracy per curve point | ls_best_raw | 20 | 0.4403 | 0.0053 | [0.4378, 0.4427] |

| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |
|---|---|---|---|---|---|---|---|
| paired_test_accuracy_difference (ls_best_raw minus eps0_best_raw), one-sided 95% lower bound (primary) | ls_best_raw vs eps0_best_raw | -0.0003 | [-0.0008, 0.0001] | 0.1795 | -0.46 | no | no |
| fraction_of_test_predictions_with_changed_argmax | ls_best_raw vs eps0_best_raw | 0.0114 | [0.0088, 0.0141] | 0.0000 | 2.77 | yes | yes |
| test_log_loss and accuracy per curve point | ls_best_raw vs eps0_best_raw | 0.0035 | [-0.0006, 0.0077] | 0.1795 | 0.54 | no | yes |
| test_log_loss and accuracy per curve point | ls_best_raw vs eps_curve_at_tuned_l2 | -0.0182 | [-0.0217, -0.0146] | 0.0000 | -3.20 | yes | yes |
| test_log_loss and accuracy per curve point | ls_best_raw vs l2_curve_at_eps0 | -0.1264 | [-0.1381, -0.1147] | 0.0000 | -6.98 | yes | yes |

**Table 5: R005, E3@v1 (ablation) testing H2@v1, seeds [0, 1, 2]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
| brier_score | ls_desmooth | 3 | 0.1441 | 0.0094 | [0.1206, 0.1675] |
| brier_score | ls_raw | 3 | 0.1470 | 0.0087 | [0.1253, 0.1687] |
| fraction_of_entries_clipped | ls_desmooth | 3 | 0.2186 | 0.0050 | [0.2063, 0.2310] |
| fraction_of_entries_clipped | ls_raw | 3 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_outer_test_log_loss_difference (ls_desmooth minus ls_raw) | ls_desmooth | 3 | -0.0106 | 0.0026 | [-0.0170, -0.0041] |
| paired_outer_test_log_loss_difference (ls_desmooth minus ls_raw) | ls_raw | 3 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| signed_calibration_error mean(confidence - accuracy) | ls_desmooth | 3 | -0.0302 | 0.0066 | [-0.0466, -0.0138] |
| signed_calibration_error mean(confidence - accuracy) | ls_raw | 3 | -0.0485 | 0.0055 | [-0.0623, -0.0348] |

| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |
|---|---|---|---|---|---|---|---|
| paired_outer_test_log_loss_difference (ls_desmooth minus ls_raw) (primary) | ls_desmooth vs ls_raw | -0.0106 | [-0.0170, -0.0041] | 0.0583 | -4.62 | no | yes |
| fraction_of_entries_clipped | ls_desmooth vs ls_raw | 0.2186 | [0.2063, 0.2310] | 0.0007 | 49.68 | yes | yes |
| signed_calibration_error mean(confidence - accuracy) | ls_desmooth vs ls_raw | 0.0183 | [0.0044, 0.0323] | 0.0583 | 2.41 | no | yes |
| brier_score | ls_desmooth vs ls_raw | -0.0030 | [-0.0237, 0.0177] | 0.7086 | -0.26 | no | no |

**Table 6: R006, E8@v2 (validation) testing H1@v2, seeds [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
| calibration_slope_vs_true_logit_at_fixed_C | eps0_fixed_C_sweep | 1 | 0.7198 | 0.0000 | [n/a, n/a] |
| implied_logloss_cost_of_C_gap | eps0_inner_cv3 | 1 | 0.0209 | 0.0000 | [n/a, n/a] |
| implied_logloss_cost_of_C_gap | eps0_inner_cv5 | 1 | 0.0223 | 0.0000 | [n/a, n/a] |
| logloss_curvature_at_oracle_log10C | eps0_oracle_C | 1 | 0.0830 | 0.0000 | [n/a, n/a] |
| nonconverged_fit_fraction | eps0_constrained_l2 | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| nonconverged_fit_fraction | eps0_fixed_C_sweep | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| nonconverged_fit_fraction | eps0_inner_cv10 | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| nonconverged_fit_fraction | eps0_inner_cv3 | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| nonconverged_fit_fraction | eps0_inner_cv5 | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| nonconverged_fit_fraction | eps0_oracle_C | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| nonconverged_fit_fraction | eps0_reduced_n_oracle_C | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| nonconverged_fit_fraction | ls_best_tuned_cv5 | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| nonconverged_fit_fraction | ls_fixed_eps0p1_cv5 | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| paired_logloss_diff_mde_20reps | ls_best_tuned_cv5 | 1 | 0.0294 | 0.0000 | [n/a, n/a] |
| paired_logloss_diff_mde_20reps | ls_fixed_eps0p1_cv5 | 1 | 0.0701 | 0.0000 | [n/a, n/a] |
| paired_logloss_diff_mde_r002_reps | ls_best_tuned_cv5 | 1 | nan | 0.0000 | [n/a, n/a] |
| paired_logloss_diff_mde_r002_reps | ls_fixed_eps0p1_cv5 | 1 | nan | 0.0000 | [n/a, n/a] |
| paired_logloss_diff_sd | ls_best_tuned_cv5 | 1 | 0.0063 | 0.0000 | [n/a, n/a] |
| paired_logloss_diff_sd | ls_fixed_eps0p1_cv5 | 1 | 0.0119 | 0.0000 | [n/a, n/a] |
| paired_logloss_diff_sd_fixed_eps_minus_tuned_eps | ls_fixed_eps0p1_cv5 | 1 | 0.0090 | 0.0000 | [n/a, n/a] |
| r002_replicates_completed_per_cell_arm | ls_best_tuned_cv5 | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| r002_reproduction_max_abs_logloss_diff | ls_best_tuned_cv5 | 1 | nan | 0.0000 | [n/a, n/a] |
| selected_minus_oracle_log10C | eps0_inner_cv10 | 1 | -0.3950 | 0.0000 | [n/a, n/a] |
| selected_minus_oracle_log10C | eps0_inner_cv3 | 1 | -0.5137 | 0.0000 | [n/a, n/a] |
| selected_minus_oracle_log10C | eps0_inner_cv5 | 1 | -0.5250 | 0.0000 | [n/a, n/a] |
| selected_minus_oracle_log10C | eps0_reduced_n_oracle_C | 1 | -0.0388 | 0.0000 | [n/a, n/a] |
| selected_minus_reduced_n_oracle_log10C | eps0_inner_cv3 | 1 | -0.4750 | 0.0000 | [n/a, n/a] |
| selected_minus_reduced_n_oracle_log10C | eps0_inner_cv5 | 1 | -0.4862 | 0.0000 | [n/a, n/a] |
| signed_error_slope_per_decade_C | eps0_fixed_C_sweep | 1 | 0.0111 | 0.0000 | [n/a, n/a] |
| signed_error_vs_true_prob_at_fixed_C | eps0_fixed_C_sweep | 1 | 0.0498 | 0.0000 | [n/a, n/a] |
| so_minus_sc_contrast_simulated_mde | eps0_inner_cv5 | 1 | 0.0372 | 0.0000 | [n/a, n/a] |
| so_minus_sc_signed_error_at_tuned_C | eps0_inner_cv5 | 1 | -0.0035 | 0.0000 | [n/a, n/a] |
| so_minus_sc_signed_error_constrained | eps0_constrained_l2 | 1 | 0.0692 | 0.0000 | [n/a, n/a] |
| so_minus_sc_signed_error_slope | eps0_fixed_C_sweep | 1 | 0.0240 | 0.0000 | [n/a, n/a] |
| soft_target_newton_max_abs_prob_diff | ls_best_tuned_cv5 | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| tost_power_0p01_at_20reps | ls_best_tuned_cv5 | 1 | 0.0047 | 0.0000 | [n/a, n/a] |
| tost_power_0p01_at_20reps | ls_fixed_eps0p1_cv5 | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| tost_power_0p01_at_r002_reps | ls_best_tuned_cv5 | 1 | nan | 0.0000 | [n/a, n/a] |
| tost_power_0p01_at_r002_reps | ls_fixed_eps0p1_cv5 | 1 | nan | 0.0000 | [n/a, n/a] |
| tuned_minus_oracle_test_logloss | eps0_constrained_l2 | 1 | 0.1613 | 0.0000 | [n/a, n/a] |
| tuned_minus_oracle_test_logloss | eps0_inner_cv10 | 1 | 0.0097 | 0.0000 | [n/a, n/a] |
| tuned_minus_oracle_test_logloss | eps0_inner_cv3 | 1 | 0.0039 | 0.0000 | [n/a, n/a] |
| tuned_minus_oracle_test_logloss | eps0_inner_cv5 | 1 | 0.0052 | 0.0000 | [n/a, n/a] |
| tuned_minus_oracle_test_logloss | eps0_reduced_n_oracle_C | 1 | 0.0005 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_constrained_l2 | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_fixed_C_sweep | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_inner_cv10 | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_inner_cv3 | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_inner_cv5 | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_oracle_C | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_reduced_n_oracle_C | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | ls_best_tuned_cv5 | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | ls_fixed_eps0p1_cv5 | 1 | 51.9614 | 0.0000 | [n/a, n/a] |

| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |
|---|---|---|---|---|---|---|---|
| paired_logloss_diff_mde_20reps (primary) | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | -0.0407 | [n/a, n/a] | n/a | n/a | no | yes |
| paired_logloss_diff_sd | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | -0.0057 | [n/a, n/a] | n/a | n/a | no | no |
| paired_logloss_diff_mde_r002_reps | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [n/a, n/a] | n/a | n/a | no | no |
| tost_power_0p01_at_r002_reps | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [n/a, n/a] | n/a | n/a | no | no |
| tost_power_0p01_at_20reps | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0047 | [n/a, n/a] | n/a | n/a | no | no |
| nonconverged_fit_fraction | ls_best_tuned_cv5 vs eps0_inner_cv5 | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| nonconverged_fit_fraction | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| nonconverged_fit_fraction | ls_best_tuned_cv5 vs eps0_inner_cv3 | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| nonconverged_fit_fraction | ls_best_tuned_cv5 vs eps0_inner_cv10 | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| nonconverged_fit_fraction | ls_best_tuned_cv5 vs eps0_oracle_C | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| nonconverged_fit_fraction | ls_best_tuned_cv5 vs eps0_reduced_n_oracle_C | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| nonconverged_fit_fraction | ls_best_tuned_cv5 vs eps0_fixed_C_sweep | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| nonconverged_fit_fraction | ls_best_tuned_cv5 vs eps0_constrained_l2 | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs eps0_inner_cv5 | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs eps0_inner_cv3 | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs eps0_inner_cv10 | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs eps0_oracle_C | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs eps0_reduced_n_oracle_C | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs eps0_fixed_C_sweep | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs eps0_constrained_l2 | 0.0000 | [n/a, n/a] | n/a | n/a | no | no |

**Table 7: R007, E9@v2 (validation) testing H1@v2, seeds [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
| curvature_cost_unit_test_max_rel_error | eps0_inner_cv3 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| curvature_cost_unit_test_max_rel_error | eps0_inner_cv5 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| curvature_cost_unit_test_max_rel_error | eps0_inner_cv5_one_se | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| curvature_cost_unit_test_max_rel_error | ls_best_tuned_cv5 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| curvature_cost_unit_test_max_rel_error | ls_best_tuned_cv5_one_se | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| curvature_cost_unit_test_max_rel_error | ls_fixed_eps0p1_cv5 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| curvature_cost_unit_test_max_rel_error | ls_oracle_eps_on_B | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| duplicate_replicate_fraction | eps0_inner_cv3 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| duplicate_replicate_fraction | eps0_inner_cv5 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| duplicate_replicate_fraction | eps0_inner_cv5_one_se | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| duplicate_replicate_fraction | ls_best_tuned_cv5 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| duplicate_replicate_fraction | ls_best_tuned_cv5_one_se | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| duplicate_replicate_fraction | ls_fixed_eps0p1_cv5 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| duplicate_replicate_fraction | ls_oracle_eps_on_B | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| eps0_noise_floor_sd_cv3_vs_cv5 | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| eps0_noise_floor_sd_cv3_vs_cv5 | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| eps0_noise_floor_sd_cv3_vs_cv5 | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| eps0_noise_floor_sd_cv3_vs_cv5 | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| eps0_noise_floor_sd_cv3_vs_cv5 | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| eps0_noise_floor_sd_cv3_vs_cv5 | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| eps0_noise_floor_sd_cv3_vs_cv5 | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| mde_without_selection_variance | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| mde_without_selection_variance | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| mde_without_selection_variance | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| mde_without_selection_variance | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| mde_without_selection_variance | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| mde_without_selection_variance | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| mde_without_selection_variance | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| near_identical_prediction_fraction | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| near_identical_prediction_fraction | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| near_identical_prediction_fraction | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| near_identical_prediction_fraction | ls_best_tuned_cv5 | 20 | 0.0045 | 0.0203 | [-0.0050, 0.0141] |
| near_identical_prediction_fraction | ls_best_tuned_cv5_one_se | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| near_identical_prediction_fraction | ls_fixed_eps0p1_cv5 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| near_identical_prediction_fraction | ls_oracle_eps_on_B | 20 | 0.0045 | 0.0203 | [-0.0050, 0.0141] |
| nonconverged_fit_fraction_per_cell | eps0_inner_cv3 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction_per_cell | eps0_inner_cv5 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction_per_cell | eps0_inner_cv5_one_se | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction_per_cell | ls_best_tuned_cv5 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction_per_cell | ls_best_tuned_cv5_one_se | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction_per_cell | ls_fixed_eps0p1_cv5 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction_per_cell | ls_oracle_eps_on_B | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| p_equivalent_verdict_at_mu_grid | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| p_equivalent_verdict_at_mu_grid | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| p_equivalent_verdict_at_mu_grid | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| p_equivalent_verdict_at_mu_grid | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| p_equivalent_verdict_at_mu_grid | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| p_equivalent_verdict_at_mu_grid | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| p_equivalent_verdict_at_mu_grid | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| p_ls_better_verdict_at_mu_grid | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| p_ls_better_verdict_at_mu_grid | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| p_ls_better_verdict_at_mu_grid | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| p_ls_better_verdict_at_mu_grid | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| p_ls_better_verdict_at_mu_grid | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| p_ls_better_verdict_at_mu_grid | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| p_ls_better_verdict_at_mu_grid | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_20reps | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_20reps | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_20reps | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_20reps | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_20reps | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_20reps | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_20reps | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_plain_sd_r002_reps | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_plain_sd_r002_reps | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_plain_sd_r002_reps | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_plain_sd_r002_reps | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_plain_sd_r002_reps | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_plain_sd_r002_reps | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_plain_sd_r002_reps | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_r002_reps_recomputed | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_r002_reps_recomputed | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_r002_reps_recomputed | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_r002_reps_recomputed | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_r002_reps_recomputed | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_r002_reps_recomputed | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_mde_r002_reps_recomputed | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_sd_20reps | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_sd_20reps | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_sd_20reps | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_sd_20reps | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_sd_20reps | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_sd_20reps | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| paired_logloss_diff_sd_20reps | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| pipeline_sim_mde_power_at_mde | eps0_inner_cv3 | 20 | 0.7924 | 0.0000 | [0.7924, 0.7924] |
| pipeline_sim_mde_power_at_mde | eps0_inner_cv5 | 20 | 0.7924 | 0.0000 | [0.7924, 0.7924] |
| pipeline_sim_mde_power_at_mde | eps0_inner_cv5_one_se | 20 | 0.7924 | 0.0000 | [0.7924, 0.7924] |
| pipeline_sim_mde_power_at_mde | ls_best_tuned_cv5 | 20 | 0.7924 | 0.0000 | [0.7924, 0.7924] |
| pipeline_sim_mde_power_at_mde | ls_best_tuned_cv5_one_se | 20 | 0.7924 | 0.0000 | [0.7924, 0.7924] |
| pipeline_sim_mde_power_at_mde | ls_fixed_eps0p1_cv5 | 20 | 0.7924 | 0.0000 | [0.7924, 0.7924] |
| pipeline_sim_mde_power_at_mde | ls_oracle_eps_on_B | 20 | 0.7924 | 0.0000 | [0.7924, 0.7924] |
| pipeline_sim_power_abs_error | eps0_inner_cv3 | 20 | 0.0045 | 0.0000 | [0.0045, 0.0045] |
| pipeline_sim_power_abs_error | eps0_inner_cv5 | 20 | 0.0045 | 0.0000 | [0.0045, 0.0045] |
| pipeline_sim_power_abs_error | eps0_inner_cv5_one_se | 20 | 0.0045 | 0.0000 | [0.0045, 0.0045] |
| pipeline_sim_power_abs_error | ls_best_tuned_cv5 | 20 | 0.0045 | 0.0000 | [0.0045, 0.0045] |
| pipeline_sim_power_abs_error | ls_best_tuned_cv5_one_se | 20 | 0.0045 | 0.0000 | [0.0045, 0.0045] |
| pipeline_sim_power_abs_error | ls_fixed_eps0p1_cv5 | 20 | 0.0045 | 0.0000 | [0.0045, 0.0045] |
| pipeline_sim_power_abs_error | ls_oracle_eps_on_B | 20 | 0.0045 | 0.0000 | [0.0045, 0.0045] |
| r002_holm_family_size_mismatch | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| r002_holm_family_size_mismatch | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| r002_holm_family_size_mismatch | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| r002_holm_family_size_mismatch | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| r002_holm_family_size_mismatch | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| r002_holm_family_size_mismatch | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| r002_holm_family_size_mismatch | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| r002_label_mismatch_count | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| r002_label_mismatch_count | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| r002_label_mismatch_count | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| r002_label_mismatch_count | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| r002_label_mismatch_count | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| r002_label_mismatch_count | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| r002_label_mismatch_count | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| r002_replicates_completed_per_cell_arm | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| r002_replicates_completed_per_cell_arm | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| r002_replicates_completed_per_cell_arm | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| r002_replicates_completed_per_cell_arm | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| r002_replicates_completed_per_cell_arm | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| r002_replicates_completed_per_cell_arm | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| r002_replicates_completed_per_cell_arm | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| r002_verdict_mismatch_count | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| r002_verdict_mismatch_count | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| r002_verdict_mismatch_count | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| r002_verdict_mismatch_count | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| r002_verdict_mismatch_count | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| r002_verdict_mismatch_count | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| r002_verdict_mismatch_count | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| reproduction_mismatch_class_counts | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| reproduction_mismatch_class_counts | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| reproduction_mismatch_class_counts | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| reproduction_mismatch_class_counts | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| reproduction_mismatch_class_counts | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| reproduction_mismatch_class_counts | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| reproduction_mismatch_class_counts | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| required_reps_for_mde_0p01 | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| required_reps_for_mde_0p01 | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| required_reps_for_mde_0p01 | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| required_reps_for_mde_0p01 | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| required_reps_for_mde_0p01 | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| required_reps_for_mde_0p01 | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| required_reps_for_mde_0p01 | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| sd_ci_coverage_normal_chisq | eps0_inner_cv3 | 20 | 0.9495 | 0.0000 | [0.9495, 0.9495] |
| sd_ci_coverage_normal_chisq | eps0_inner_cv5 | 20 | 0.9495 | 0.0000 | [0.9495, 0.9495] |
| sd_ci_coverage_normal_chisq | eps0_inner_cv5_one_se | 20 | 0.9495 | 0.0000 | [0.9495, 0.9495] |
| sd_ci_coverage_normal_chisq | ls_best_tuned_cv5 | 20 | 0.9495 | 0.0000 | [0.9495, 0.9495] |
| sd_ci_coverage_normal_chisq | ls_best_tuned_cv5_one_se | 20 | 0.9495 | 0.0000 | [0.9495, 0.9495] |
| sd_ci_coverage_normal_chisq | ls_fixed_eps0p1_cv5 | 20 | 0.9495 | 0.0000 | [0.9495, 0.9495] |
| sd_ci_coverage_normal_chisq | ls_oracle_eps_on_B | 20 | 0.9495 | 0.0000 | [0.9495, 0.9495] |
| sd_ci_coverage_t5_bootstrap | eps0_inner_cv3 | 20 | 0.7063 | 0.0000 | [0.7063, 0.7063] |
| sd_ci_coverage_t5_bootstrap | eps0_inner_cv5 | 20 | 0.7063 | 0.0000 | [0.7063, 0.7063] |
| sd_ci_coverage_t5_bootstrap | eps0_inner_cv5_one_se | 20 | 0.7063 | 0.0000 | [0.7063, 0.7063] |
| sd_ci_coverage_t5_bootstrap | ls_best_tuned_cv5 | 20 | 0.7063 | 0.0000 | [0.7063, 0.7063] |
| sd_ci_coverage_t5_bootstrap | ls_best_tuned_cv5_one_se | 20 | 0.7063 | 0.0000 | [0.7063, 0.7063] |
| sd_ci_coverage_t5_bootstrap | ls_fixed_eps0p1_cv5 | 20 | 0.7063 | 0.0000 | [0.7063, 0.7063] |
| sd_ci_coverage_t5_bootstrap | ls_oracle_eps_on_B | 20 | 0.7063 | 0.0000 | [0.7063, 0.7063] |
| sd_ci_coverage_t5_chisq | eps0_inner_cv3 | 20 | 0.8586 | 0.0000 | [0.8586, 0.8586] |
| sd_ci_coverage_t5_chisq | eps0_inner_cv5 | 20 | 0.8586 | 0.0000 | [0.8586, 0.8586] |
| sd_ci_coverage_t5_chisq | eps0_inner_cv5_one_se | 20 | 0.8586 | 0.0000 | [0.8586, 0.8586] |
| sd_ci_coverage_t5_chisq | ls_best_tuned_cv5 | 20 | 0.8586 | 0.0000 | [0.8586, 0.8586] |
| sd_ci_coverage_t5_chisq | ls_best_tuned_cv5_one_se | 20 | 0.8586 | 0.0000 | [0.8586, 0.8586] |
| sd_ci_coverage_t5_chisq | ls_fixed_eps0p1_cv5 | 20 | 0.8586 | 0.0000 | [0.8586, 0.8586] |
| sd_ci_coverage_t5_chisq | ls_oracle_eps_on_B | 20 | 0.8586 | 0.0000 | [0.8586, 0.8586] |
| sd_ratio_one_se_over_argmin | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| sd_ratio_one_se_over_argmin | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| sd_ratio_one_se_over_argmin | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| sd_ratio_one_se_over_argmin | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| sd_ratio_one_se_over_argmin | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| sd_ratio_one_se_over_argmin | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| sd_ratio_one_se_over_argmin | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| sealed_mean_magnitude_category | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| sealed_mean_magnitude_category | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| sealed_mean_magnitude_category | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| sealed_mean_magnitude_category | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| sealed_mean_magnitude_category | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| sealed_mean_magnitude_category | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| sealed_mean_magnitude_category | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| selected_c_eps_distribution_one_se_vs_argmin | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| selected_c_eps_distribution_one_se_vs_argmin | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| selected_c_eps_distribution_one_se_vs_argmin | eps0_inner_cv5_one_se | 20 | 1.1909 | 0.1840 | [1.1048, 1.2770] |
| selected_c_eps_distribution_one_se_vs_argmin | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| selected_c_eps_distribution_one_se_vs_argmin | ls_best_tuned_cv5_one_se | 20 | 1.9455 | 0.1851 | [1.8588, 2.0321] |
| selected_c_eps_distribution_one_se_vs_argmin | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| selected_c_eps_distribution_one_se_vs_argmin | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| selected_eps_entropy_per_cell | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| selected_eps_entropy_per_cell | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| selected_eps_entropy_per_cell | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| selected_eps_entropy_per_cell | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| selected_eps_entropy_per_cell | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| selected_eps_entropy_per_cell | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| selected_eps_entropy_per_cell | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| selection_variance_share | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| selection_variance_share | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| selection_variance_share | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| selection_variance_share | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| selection_variance_share | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| selection_variance_share | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| selection_variance_share | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| test_draw_variance_share | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| test_draw_variance_share | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| test_draw_variance_share | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| test_draw_variance_share | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| test_draw_variance_share | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| test_draw_variance_share | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| test_draw_variance_share | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| train_draw_variance_share | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| train_draw_variance_share | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| train_draw_variance_share | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| train_draw_variance_share | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| train_draw_variance_share | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| train_draw_variance_share | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| train_draw_variance_share | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| var_ratio_expected_over_sampled_logloss | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| var_ratio_expected_over_sampled_logloss | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| var_ratio_expected_over_sampled_logloss | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| var_ratio_expected_over_sampled_logloss | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| var_ratio_expected_over_sampled_logloss | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| var_ratio_expected_over_sampled_logloss | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| var_ratio_expected_over_sampled_logloss | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |
| wall_time_seconds | eps0_inner_cv3 | 20 | 53.4209 | 13.3826 | [47.1577, 59.6842] |
| wall_time_seconds | eps0_inner_cv5 | 20 | 53.4209 | 13.3826 | [47.1577, 59.6842] |
| wall_time_seconds | eps0_inner_cv5_one_se | 20 | 53.4209 | 13.3826 | [47.1577, 59.6842] |
| wall_time_seconds | ls_best_tuned_cv5 | 20 | 53.4209 | 13.3826 | [47.1577, 59.6842] |
| wall_time_seconds | ls_best_tuned_cv5_one_se | 20 | 53.4209 | 13.3826 | [47.1577, 59.6842] |
| wall_time_seconds | ls_fixed_eps0p1_cv5 | 20 | 53.4209 | 13.3826 | [47.1577, 59.6842] |
| wall_time_seconds | ls_oracle_eps_on_B | 20 | 53.4209 | 13.3826 | [47.1577, 59.6842] |
| x_power_verdict_flip_count_plain_vs_upper | eps0_inner_cv3 | 20 | nan | nan | [nan, nan] |
| x_power_verdict_flip_count_plain_vs_upper | eps0_inner_cv5 | 20 | nan | nan | [nan, nan] |
| x_power_verdict_flip_count_plain_vs_upper | eps0_inner_cv5_one_se | 20 | nan | nan | [nan, nan] |
| x_power_verdict_flip_count_plain_vs_upper | ls_best_tuned_cv5 | 20 | nan | nan | [nan, nan] |
| x_power_verdict_flip_count_plain_vs_upper | ls_best_tuned_cv5_one_se | 20 | nan | nan | [nan, nan] |
| x_power_verdict_flip_count_plain_vs_upper | ls_fixed_eps0p1_cv5 | 20 | nan | nan | [nan, nan] |
| x_power_verdict_flip_count_plain_vs_upper | ls_oracle_eps_on_B | 20 | nan | nan | [nan, nan] |

| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |
|---|---|---|---|---|---|---|---|
| paired_logloss_diff_mde_r002_reps_recomputed (primary) | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_r002_reps_recomputed (primary) | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_r002_reps_recomputed (primary) | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_r002_reps_recomputed (primary) | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_r002_reps_recomputed (primary) | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_r002_reps_recomputed (primary) | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_plain_sd_r002_reps | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_plain_sd_r002_reps | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_plain_sd_r002_reps | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_plain_sd_r002_reps | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_plain_sd_r002_reps | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_plain_sd_r002_reps | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| x_power_verdict_flip_count_plain_vs_upper | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| x_power_verdict_flip_count_plain_vs_upper | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| x_power_verdict_flip_count_plain_vs_upper | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| x_power_verdict_flip_count_plain_vs_upper | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| x_power_verdict_flip_count_plain_vs_upper | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| x_power_verdict_flip_count_plain_vs_upper | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| p_equivalent_verdict_at_mu_grid | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| p_equivalent_verdict_at_mu_grid | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| p_equivalent_verdict_at_mu_grid | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| p_equivalent_verdict_at_mu_grid | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| p_equivalent_verdict_at_mu_grid | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| p_equivalent_verdict_at_mu_grid | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| p_ls_better_verdict_at_mu_grid | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| p_ls_better_verdict_at_mu_grid | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| p_ls_better_verdict_at_mu_grid | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| p_ls_better_verdict_at_mu_grid | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| p_ls_better_verdict_at_mu_grid | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| p_ls_better_verdict_at_mu_grid | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| sealed_mean_magnitude_category | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| sealed_mean_magnitude_category | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| sealed_mean_magnitude_category | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| sealed_mean_magnitude_category | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| sealed_mean_magnitude_category | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| sealed_mean_magnitude_category | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| pipeline_sim_power_abs_error | ls_best_tuned_cv5 vs eps0_inner_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| pipeline_sim_power_abs_error | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| pipeline_sim_power_abs_error | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| pipeline_sim_power_abs_error | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| pipeline_sim_power_abs_error | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| pipeline_sim_power_abs_error | ls_best_tuned_cv5 vs eps0_inner_cv3 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| pipeline_sim_mde_power_at_mde | ls_best_tuned_cv5 vs eps0_inner_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| pipeline_sim_mde_power_at_mde | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| pipeline_sim_mde_power_at_mde | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| pipeline_sim_mde_power_at_mde | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| pipeline_sim_mde_power_at_mde | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| pipeline_sim_mde_power_at_mde | ls_best_tuned_cv5 vs eps0_inner_cv3 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| sd_ci_coverage_normal_chisq | ls_best_tuned_cv5 vs eps0_inner_cv5 | 0.0000 | [-0.0000, 0.0000] | 1.0000 | 0.00 | no | no |
| sd_ci_coverage_normal_chisq | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0000 | [-0.0000, 0.0000] | 1.0000 | 0.00 | no | no |
| sd_ci_coverage_normal_chisq | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | 0.0000 | [-0.0000, 0.0000] | 1.0000 | 0.00 | no | no |
| sd_ci_coverage_normal_chisq | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | 0.0000 | [-0.0000, 0.0000] | 1.0000 | 0.00 | no | no |
| sd_ci_coverage_normal_chisq | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | 0.0000 | [-0.0000, 0.0000] | 1.0000 | 0.00 | no | no |
| sd_ci_coverage_normal_chisq | ls_best_tuned_cv5 vs eps0_inner_cv3 | 0.0000 | [-0.0000, 0.0000] | 1.0000 | 0.00 | no | no |
| sd_ci_coverage_t5_chisq | ls_best_tuned_cv5 vs eps0_inner_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| sd_ci_coverage_t5_chisq | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| sd_ci_coverage_t5_chisq | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| sd_ci_coverage_t5_chisq | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| sd_ci_coverage_t5_chisq | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| sd_ci_coverage_t5_chisq | ls_best_tuned_cv5 vs eps0_inner_cv3 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| sd_ci_coverage_t5_bootstrap | ls_best_tuned_cv5 vs eps0_inner_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| sd_ci_coverage_t5_bootstrap | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| sd_ci_coverage_t5_bootstrap | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| sd_ci_coverage_t5_bootstrap | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| sd_ci_coverage_t5_bootstrap | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| sd_ci_coverage_t5_bootstrap | ls_best_tuned_cv5 vs eps0_inner_cv3 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| curvature_cost_unit_test_max_rel_error | ls_best_tuned_cv5 vs eps0_inner_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| curvature_cost_unit_test_max_rel_error | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| curvature_cost_unit_test_max_rel_error | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| curvature_cost_unit_test_max_rel_error | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| curvature_cost_unit_test_max_rel_error | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| curvature_cost_unit_test_max_rel_error | ls_best_tuned_cv5 vs eps0_inner_cv3 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| r002_verdict_mismatch_count | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_verdict_mismatch_count | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_verdict_mismatch_count | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_verdict_mismatch_count | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_verdict_mismatch_count | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_verdict_mismatch_count | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_label_mismatch_count | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_label_mismatch_count | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_label_mismatch_count | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_label_mismatch_count | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_label_mismatch_count | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_label_mismatch_count | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_holm_family_size_mismatch | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_holm_family_size_mismatch | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_holm_family_size_mismatch | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_holm_family_size_mismatch | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_holm_family_size_mismatch | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_holm_family_size_mismatch | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_replicates_completed_per_cell_arm | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_replicates_completed_per_cell_arm | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_replicates_completed_per_cell_arm | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_replicates_completed_per_cell_arm | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_replicates_completed_per_cell_arm | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| r002_replicates_completed_per_cell_arm | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| duplicate_replicate_fraction | ls_best_tuned_cv5 vs eps0_inner_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| duplicate_replicate_fraction | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| duplicate_replicate_fraction | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| duplicate_replicate_fraction | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| duplicate_replicate_fraction | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| duplicate_replicate_fraction | ls_best_tuned_cv5 vs eps0_inner_cv3 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| reproduction_mismatch_class_counts | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| reproduction_mismatch_class_counts | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| reproduction_mismatch_class_counts | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| reproduction_mismatch_class_counts | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| reproduction_mismatch_class_counts | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| reproduction_mismatch_class_counts | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_sd_20reps | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_sd_20reps | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_sd_20reps | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_sd_20reps | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_sd_20reps | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_sd_20reps | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_20reps | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_20reps | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_20reps | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_20reps | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_20reps | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| paired_logloss_diff_mde_20reps | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| required_reps_for_mde_0p01 | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| required_reps_for_mde_0p01 | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| required_reps_for_mde_0p01 | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| required_reps_for_mde_0p01 | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| required_reps_for_mde_0p01 | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| required_reps_for_mde_0p01 | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| train_draw_variance_share | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| train_draw_variance_share | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| train_draw_variance_share | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| train_draw_variance_share | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| train_draw_variance_share | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| train_draw_variance_share | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| selection_variance_share | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| selection_variance_share | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| selection_variance_share | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| selection_variance_share | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| selection_variance_share | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| selection_variance_share | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| test_draw_variance_share | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| test_draw_variance_share | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| test_draw_variance_share | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| test_draw_variance_share | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| test_draw_variance_share | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| test_draw_variance_share | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| mde_without_selection_variance | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| mde_without_selection_variance | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| mde_without_selection_variance | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| mde_without_selection_variance | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| mde_without_selection_variance | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| mde_without_selection_variance | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| sd_ratio_one_se_over_argmin | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| sd_ratio_one_se_over_argmin | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| sd_ratio_one_se_over_argmin | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| sd_ratio_one_se_over_argmin | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| sd_ratio_one_se_over_argmin | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| sd_ratio_one_se_over_argmin | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| near_identical_prediction_fraction | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| near_identical_prediction_fraction | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0045 | [-0.0050, 0.0141] | 1.0000 | 0.31 | no | no |
| near_identical_prediction_fraction | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | 0.0000 | [-0.0130, 0.0130] | 1.0000 | 0.00 | no | no |
| near_identical_prediction_fraction | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | 0.0045 | [-0.0050, 0.0141] | 1.0000 | 0.31 | no | no |
| near_identical_prediction_fraction | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| near_identical_prediction_fraction | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| selected_c_eps_distribution_one_se_vs_argmin | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| selected_c_eps_distribution_one_se_vs_argmin | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| selected_c_eps_distribution_one_se_vs_argmin | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| selected_c_eps_distribution_one_se_vs_argmin | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| selected_c_eps_distribution_one_se_vs_argmin | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| selected_c_eps_distribution_one_se_vs_argmin | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| var_ratio_expected_over_sampled_logloss | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| var_ratio_expected_over_sampled_logloss | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| var_ratio_expected_over_sampled_logloss | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| var_ratio_expected_over_sampled_logloss | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| var_ratio_expected_over_sampled_logloss | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| var_ratio_expected_over_sampled_logloss | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| eps0_noise_floor_sd_cv3_vs_cv5 | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| eps0_noise_floor_sd_cv3_vs_cv5 | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| eps0_noise_floor_sd_cv3_vs_cv5 | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| eps0_noise_floor_sd_cv3_vs_cv5 | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| eps0_noise_floor_sd_cv3_vs_cv5 | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| eps0_noise_floor_sd_cv3_vs_cv5 | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| selected_eps_entropy_per_cell | ls_best_tuned_cv5 vs eps0_inner_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| selected_eps_entropy_per_cell | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | nan | [nan, nan] | 1.0000 | nan | no | no |
| selected_eps_entropy_per_cell | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | nan | [nan, nan] | 1.0000 | nan | no | no |
| selected_eps_entropy_per_cell | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| selected_eps_entropy_per_cell | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | nan | [nan, nan] | 1.0000 | nan | no | no |
| selected_eps_entropy_per_cell | ls_best_tuned_cv5 vs eps0_inner_cv3 | nan | [nan, nan] | 1.0000 | nan | no | no |
| nonconverged_fit_fraction_per_cell | ls_best_tuned_cv5 vs eps0_inner_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| nonconverged_fit_fraction_per_cell | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| nonconverged_fit_fraction_per_cell | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| nonconverged_fit_fraction_per_cell | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| nonconverged_fit_fraction_per_cell | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| nonconverged_fit_fraction_per_cell | ls_best_tuned_cv5 vs eps0_inner_cv3 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs eps0_inner_cv5 | 0.0000 | [-8.5671, 8.5671] | 1.0000 | 0.00 | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs ls_fixed_eps0p1_cv5 | 0.0000 | [-8.5671, 8.5671] | 1.0000 | 0.00 | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs ls_oracle_eps_on_B | 0.0000 | [-8.5671, 8.5671] | 1.0000 | 0.00 | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs ls_best_tuned_cv5_one_se | 0.0000 | [-8.5671, 8.5671] | 1.0000 | 0.00 | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs eps0_inner_cv5_one_se | 0.0000 | [-8.5671, 8.5671] | 1.0000 | 0.00 | no | no |
| wall_time_seconds | ls_best_tuned_cv5 vs eps0_inner_cv3 | 0.0000 | [-8.5671, 8.5671] | 1.0000 | 0.00 | no | no |

## 6. Discussion

The evidence is compatible with H2 in the R001 setting. No LS advantage over tuned eps=0, with or without temperature scaling, was detected. However, R001 does not establish H2, because of the pooled replicates, self-defined cells, and the absence of regime-specific analysis. The CI half-width (about 0.002) is smaller than the -0.01 margin, which is itself unjustified and large relative to the between-arm differences, so the non-superiority verdict is easy to reach and sensitivity to smaller margins (e.g. 0.002 or 0.005) is not yet shown.

The H0 premise check suggests that overconfidence of LR depends on how L2 is chosen: tuned L2 gave no systematic overconfidence in the pooled estimate, while constrained L2 did. The latter is expected by construction. Both results are uncertain for the reasons given below; absence of a CI above zero is weak evidence of absence, and no equivalence bound was set.

The mechanism test does not support a claim that LS harms or helps probability accuracy at tuned settings. The n3000_b4 cell shows harm in one setting, and the cross-cell pattern is a hypothesis for follow-up, not a finding. Accuracy cost appears small, but non-inferiority is not established for all cells.

What would change practice: if confirmed, the pilot pattern would mean that adding an eps hyperparameter to an LR pipeline that already tunes L2 and temperature scaling brings no calibration gain, and that the useful regimes would be those with weak or mis-tuned regularisation. That has not been tested in label-noise, near-separable or wide-MLP regimes, where LS could plausibly win. Next steps are to run E2 across the regimes the gate identifies, add Platt, vector scaling and isotonic arms with equalised tuning budgets, implement early stopping for MLPs, add ICI [P012] and confidence-stratified signed error, validate bootstrap CI coverage on eps=0 versus eps=0 nulls, and run a power analysis for H4. The lack of any regime-resolved result means the title question is not answered.

## 7. Limitations

1. No test of H1. R002, R006 and R007 are gate, diagnostic or validation runs and do not test H1 components. R006 and R007 could not audit R002 because no R002/E1 artefacts were supplied. H1@v1 and H3@v1 were superseded by v2, and no conclusions are drawn from them. 2. Protocol deviations (listed in Methods): self-defined and stand-in cells replacing E1 cells, so run results are not directly comparable; custom numpy MLP with fixed 150 epochs, L2 only and no early stopping, so the 'tuned L2 plus early stopping' MLP comparator is not implemented; real-data analysis limited to LR on breast_cancer, iris and wine (plus a digits entry whose status is unclear); Nadeau-Bengio intervals not computed. 3. Aggregation dependence: the 10-replicate gate decisions, one-sided CIs, Holm adjustment and cross-seed pooling were computed downstream, so all CI and decision-rule statements are provisional. The R005 bootstrap covers only 5 outer folds of one seed. Robustness and sensitivity analyses were deferred, including the misspecified generator, d=30, oracle and Brier-based tuning, fixed-eps and clip-floor runs. Grids G9/G9' and regimes R1-R5 are engineer-defined. 4. Statistical issues: outer-fold CIs share training data and may be too narrow; Holm families and the -0.01 margin are unjustified; no multiplicity strategy across metrics, regimes or the joint/any-metric H1 rule is set; no power analysis for H4; no equivalence bound for H0. 5. Internal inconsistencies: R002 reports n=8 per cell though 10 seeds were run (the reason is not documented); fam1_check_pass was 0.8, so some checks failed and the handling of those replicates is unknown; CIs for proportions use a normal approximation; the constrained-L2 cell count differs between the summary text (three cells) and the table (four). The 'fires' proxies are undefined and conflict with the signed-error conclusion. 6. R001 anomalies: de-smoothing is worse than raw LS despite raw LS being underconfident, which could reflect a clipping issue or bug; eps=0 is itself slightly underconfident; Table 1 ablation rows repeat identical values across arms and effect-size values (g) do not match the tabulated differences; the LS-versus-eps0+TS CI differs slightly between the table sections. 7. Baselines are incomplete: no Platt, vector or isotonic arms; tuning budgets are not equalised. 8. Calibration measurement is coarse: top-label signed error, 15-bin ECE only, no ICI and no bin-sensitivity analysis. 9. Reproducibility: the generator, cell, grid and fold specifications are not given, there is no frozen config hash, and the work is not pre-registered in a timestamped sense. 10. Scope: all evidence is synthetic or from a few small bundled scikit-learn datasets with small CPU models; results do not speak to deep networks, large datasets or non-tabular data, where the premise of [P001, P042] was developed. 11. Citations: P017, P023 and similar sources are loosely relevant to tabular LS.

## 8. Conclusion

In the completed pilots, best-tuned label smoothing showed no detectable log-loss, Brier or ECE advantage over best-tuned eps=0 on self-defined synthetic LR and MLP cells (log-loss difference +0.0009, CI [-0.0006, 0.0025]). The mechanism test and the accuracy cost test did not show clear harm, but they also did not establish its absence. Overconfidence of LR in these pilots depended on how L2 was chosen. The key regime-resolved claim (H1) remains untested, all hypotheses remain at evidence level HYPOTHESIS, and every run is inconclusive. The contribution at this stage is the protocol, the decision rules and a documented list of deviations. A confirmatory study needs a frozen specification, equalised baselines including more post-hoc calibrators, early-stopped MLPs, regime-resolved runs, and validated confidence intervals.

## Claim-Evidence Map

| claim | kind | importance | evidence state | traceability |
|---|---|---|---|---|
| C23: Contribution: we reframe the LS-calibration question for small tabular classifiers as three pre-registered comparative and mechanistic questions. Q-a asks wh... | contribution | central | HYPOTHESIS | H1, H2, H3, H4, P001, P042, P018, P037; 0 experimental, 4 literature, untested: literature can motivate this claim but not establish it |
| C24: Finding (preliminary, inconclusive): in the E1 gate run, the overconfidence premise for LR is not supported under the tuned eps=0 arm, since no SO LR cell ha... | finding | central | HYPOTHESIS | H1, R002, P042, P001; 0 experimental, 2 literature, untested: literature can motivate this claim but not establish it |
| C25: Finding (preliminary, inconclusive): in the primary comparison (E2), best-tuned LS shows no detectable log-loss advantage over best-tuned eps=0. The paired d... | finding | central | HYPOTHESIS | H2, R001, P042, P018; 0 experimental, 2 literature, untested: literature can motivate this claim but not establish it |
| C26: Finding (not confirmed): the mechanism test of LS-induced underconfidence in well-specified synthetic LR did not meet its falsification criterion. In none of... | finding | supporting | HYPOTHESIS | H3, R003; 0 experimental, 0 literature |
| C27: Finding (preliminary): the accuracy cost of LS in the synthetic LR cells is small. Paired accuracy differences have |mean| ≤ 0.14 pp with intervals well insi... | finding | supporting | HYPOTHESIS | H4, R004; 0 experimental, 0 literature |
| C28: Finding (ablation, inconclusive): in the de-smoothing ablation (E3), the paired difference was -0.0106 log loss (CI [-0.0170, -0.0041], Holm p=0.0583). Its s... | finding | supporting | HYPOTHESIS | H2, R005; 0 experimental, 0 literature |
| C29: Limitation: H1, the claim that LS helps where the tuned eps=0 baseline is overconfident, is untested. R002 (E1), R006 (E8) and R007 (E9) are gate, diagnostic... | limitation | supporting | HYPOTHESIS | H1, R002, R006, R007; 0 experimental, 0 literature |
| C30: Limitation: protocol deviations limit comparability. E1 cells were replaced by self-defined synthetic cells in E2 and by stand-ins in E8 and E9. The E2 MLP i... | limitation | supporting | HYPOTHESIS | H1, H2, R001, R006, R007; 0 experimental, 0 literature |
| C31: Limitation: several analyses depend on aggregation across replicates that single-run outputs did not perform. This applies to the 10-replicate gate decisions... | limitation | supporting | HYPOTHESIS | H1, H3, H4, R002, R003, R004, R005; 0 experimental, 0 literature |
| C32: Limitation: all evidence is synthetic or from a few small bundled scikit-learn datasets, with CPU-only small models. The results do not speak to deep network... | limitation | supporting | SUPPORTED | H1, H2, H3, H4, R001, R002, R003, R004, R005, P001, P042; 0 experimental, 2 literature |
| C33: Background: the premise that LS affects calibration comes from deep-network work. P001 studies when LS helps, and P042 documents miscalibration (overconfiden... | background | supporting | SUPPORTED | H2, P001, P042; 0 experimental, 2 literature |
| C34: Background: calibration assessment and comparison methodology motivates our metrics and protocol. This covers log loss, Brier, ECE and signed confidence-minu... | background | supporting | SUPPORTED | H3, P012, P018, P037, P051; 0 experimental, 4 literature |
| C35: Background: small tabular classifiers such as LR are strong baselines in clinical prediction, and their performance depends on sample size and data quality (... | background | supporting | SUPPORTED | H1, P052, P053, P059, P017, P023, P011; 0 experimental, 5 literature |

## References

[P001] Rafael Rios Müller et al. (2019). When Does Label Smoothing Help?. *arXiv (Cornell University)*. https://doi.org/10.48550/arxiv.1906.02629
[P012] Peter C. Austin and Ewout Willem Steyerberg (2019). The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models. *Statistics in Medicine*. https://doi.org/10.1002/sim.8281
[P017] Claudio Filipi Gonçalves dos Santos and João Paulo Papa (2022). Avoiding Overfitting: A Survey on Regularization Methods for Convolutional Neural Networks. *ACM Computing Surveys*. https://doi.org/10.1145/3510413
[P018] Telmo M. Silva Filho et al. (2023). Classifier calibration: a survey on how to assess and improve predicted class probabilities. *Machine Learning*. https://doi.org/10.1007/s10994-023-06336-7
[P023] Juan R. Terven et al. (2025). A comprehensive survey of loss functions and metrics in deep learning. *Artificial Intelligence Review*. https://doi.org/10.1007/s10462-025-11198-7
[P037] Muthu Chidambaram and Rong Ge (2024). Reassessing How to Compare and Improve the Calibration of Machine Learning Models. *arXiv*. https://arxiv.org/abs/2406.04068v2
[P042] Chuan Guo et al. (2017). On Calibration of Modern Neural Networks. *arXiv (Cornell University)*. https://doi.org/10.48550/arxiv.1706.04599
[P051] Ruben van den Goorbergh et al. (2022). The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. *Journal of the American Medical Informatics Association*. https://doi.org/10.1093/jamia/ocac093
[P052] Yanan Hu et al. (2025). Beyond Comparing Machine Learning and Logistic Regression in Clinical Prediction Modelling: Shifting from Model Debate to Data Quality. *Journal of Medical Internet Research*. https://doi.org/10.2196/77721
[P053] Scott G. Silvey and Jinze Liu (2024). Sample Size Requirements for Popular Classification Algorithms in Tabular Clinical Data: Empirical Study. *Journal of Medical Internet Research*. https://doi.org/10.2196/60231
[P059] Rishi J. Desai et al. (2020). Comparison of Machine Learning Methods With Traditional Models for Use of Administrative Claims With Electronic Medical Records to Predict Heart Failure Outcomes. *JAMA Network Open*. https://doi.org/10.1001/jamanetworkopen.2019.18962
