# Does label smoothing ever beat tuned L2 and post-hoc scaling for calibrating logistic regression and small MLPs? A protocol with pilot results on synthetic generators

> **Status:** Draft with open issues (failing checks: central_claims_supported, claims_within_evidence, required_experiments_complete, reviewer_objections_addressed). 0 of 3 central claims are SUPPORTED by the evidence graph (4 red-team review round(s)).

**Plain-language summary.** # Plain-Language Summary

Label smoothing is sometimes recommended to improve model confidence calibration, but most evidence comes from large neural networks; this pilot study tests whether it helps simpler models like logistic regression and small neural networks when other tuning methods are also optimized. Across multiple experiments with different data sizes and model setups, label smoothing showed no clear advantage over baseline approaches, with differences in prediction accuracy typically near zero and not statistically meaningful. The researchers emphasize this is preliminary work without pre-registration, and the core question remains unanswered pending a complete, properly frozen experiment.

## Abstract

Label smoothing (LS) is often proposed as a calibration aid, but the supporting evidence comes mainly from deep networks [P001, P042]. We ask whether LS helps logistic regression (LR) or small MLPs once L2 strength and post-hoc scaling are tuned. This is a protocol paper with pilot results, not a confirmatory study. No frozen, timestamped pre-registration exists. Every hypothesis remains at evidence level HYPOTHESIS and every run is labelled inconclusive. Several runs used stand-in or reconstructed generators because the frozen grids were unavailable, so the regime-resolved question is not answered. Findings: (i) pooled over self-defined synthetic cells (R001, 20 seeds), best-tuned LS minus best-tuned eps=0 (both post-hoc corrected) in test log loss was +0.0009, 95% CI [-0.0006, 0.0025], Holm p=1.0, so no LS benefit was detected. The Holm family was not fully defined, and the MLP had no early stopping. (ii) In a later run on a reconstructed grid (R009), nested-tuned LS differed from tuned eps=0 by -0.0008 (CI [-0.0016, -0.0001], Holm p=0.4858) and from eps=0 plus temperature scaling by -0.0002 (CI [-0.0015, 0.0010]). (iii) With known P(y|x) at n=100,000 (R010), LS reduces the calibration slope to about 0.747 at eps=0.1 and 0.596 at eps=0.2, which is the expected shrinkage direction. (iv) No finite-sample regime was found where LS lowers error against the truth (R003). (v) The premise that tuned LR is overconfident was not supported under the tuned eps=0 arm in R002, though it was under constrained L2. The title question remains open until the real regime grid is run.

## 1. Introduction

Label smoothing replaces hard targets with (1-eps)y + eps/K. In deep networks it is reported to improve calibration and generalisation [P001], while modern networks are overconfident and are corrected by simple post-hoc temperature scaling [P042]. Whether this transfers to logistic regression and small MLPs on tabular-style data is the premise we test rather than assume. Practitioners could choose between three levers: tuning L2 (and early stopping for MLPs), post-hoc scaling, or LS. The decision-relevant question is whether LS ever adds anything beyond the first two. Questions: Q-a, is LS ever better than tuned L2 plus post-hoc scaling, and in which regimes? Q-b, where does LS-induced underconfidence against ground-truth probabilities appear, and is there a finite-sample crossover where shrinkage lowers error? Q-c, is the accuracy/log-loss curve traced by eps dominated by the curve traced by L2 strength? H1 (version 4) is the single confirmatory hypothesis. H2-H7 are exploratory under the H1 registry. The hypothesis registry was revised repeatedly: H1 v1-v3, H2 v1, H3 v1-v2, H4 v1, H5 v1, H6 v1 and H7 v1 are superseded. Experiments ran on earlier specifications, so results below are not confirmatory. We make no claim to be the first to study LS on LR. The contribution is a regime-resolved, head-to-head comparison design plus pilot results. The targeted literature search required before any novelty claim was limited to what the swarm tools could retrieve and is not documented in full here.

## 2. Related Work

LS and its effects on calibration in deep networks are studied in [P001]. Temperature scaling as a strong post-hoc baseline is from [P042]. Surveys and comparisons of calibration assessment and improvement [P018, P037] motivate treating post-hoc scaling as the baseline to beat, and the ICI [P012] is a binning-free metric for logistic models (not yet used here). Shrinkage-and-scaling post-hoc methods [P061] are conceptually related to LS as shrinkage toward uniform. Finite-sample behaviour and imbalance corrections in LR risk models are sensitive to sample size and can harm calibration [P051, P062, P053, P052], which motivates regime-resolved analysis and the exploratory test of prior-directed shrinkage (H6). LS combined with distributionally robust optimisation [P036] and loss-based calibration under imbalance [P022] give context for non-uniform smoothing. Most of these sources concern deep or clinical models, so transfer to LR and small MLPs is the premise being tested.

## 3. Method

Models. Custom numpy LR and one-hidden-layer MLPs with soft-target cross-entropy (sklearn's MLPClassifier has no LS). Arms: raw eps=0 with L2; eps=0 with cross-fitted temperature scaling (cvts); LS with tuned eps and L2; LS with TS; and a de-smoothing ablation that inverts the LS shrinkage with clipping. Variant selection (raw, cvts, desmooth) uses inner validation. Hypotheses: H1, benefit of best-tuned eps>0 over best-tuned eps=0 (confirmatory, not tested by any run); H2, LS gives no benefit beyond post-hoc correction; H3, mechanism on LR against known P(y|x); H4, accuracy non-inferiority (-1 pp margin) and Q-c frontier dominance; H5, LS substituting for early stopping (not tested); H6, prior-directed shrinkage (not tested); H7, leakage and consistency checks. Population prediction (written before running): the unrestricted minimiser of smoothed cross-entropy is (1-eps)P(y|x)+eps/K, but a linear-logit LR cannot represent it exactly, so the check is approximate shrinkage. Metrics: log loss (primary), Brier, 15-bin equal-mass top-label ECE, signed error versus true probability and versus accuracy, calibration slope against the true logit, and KL(true||predicted). Statistics: paired differences with 95% CIs over seeds and Holm adjustment; the Holm family is not fully enumerated in the run outputs. Many scripts emit one replicate per invocation, and the cross-seed aggregation (permutation tests, TOST, power) was not completed.

Declared deviations from protocol. R001: self-defined synthetic cells replace the E1 cells; the MLP is a custom numpy network with fixed 150 epochs, L2 only and no early stopping; real data covered only breast_cancer, iris and wine with LR, with no Nadeau-Bengio intervals and digits omitted. R002: one pilot replicate per invocation; the 10-replicate statistics, the published SO/SC labels and the reserve cells were not computed; a solver change was made. R003: robustness (d=30) and sensitivity analyses not run; the misspecified generator only runs via a flag and was not run; the G9/G9' grids are assumed. R004: single runs cannot yield bootstrap, Holm or TOST; curve points are reported as one mean; the LS tuning grid was changed to exclude eps=0. R005: regimes R1-R5 self-defined; models refit; bootstrap over 5 outer folds of one seed, so Holm p-values are provisional; a KeyError fix was applied. R006: stand-in cells replace E1 cells and R002 artefacts were not compared. R007: Part 2 (R002 reconciliation) not run, stand-in cells, a 200,000-point reference sample, a simplified time guard, and a changed hand-test verdict and curvature-check setting before Part 1 passed. R008: power metrics emitted as NaN, structural labels not output, and cells C1/C5/C7 not included. R009: reconstructed grid, with A/B conditioning, TOST and per-cell Holm left to downstream aggregation. R010: pilot, power calculation and permutation tests left to the aggregator; E6 frozen values reconstructed; 30 seeds run against a spec of 20. R011: forward-difference Jacobian, seeds varying rather than frozen, and robustness tests in a single cell. R012: reconstructed generator, eps0_tuned refit rather than loaded, MLP capped at 200 full-batch Adam epochs, and MLP ablations skipped under the time guard.

## 4. Experimental Setup

Runs and seeds: R001 (E2, seeds 0-19, primary, H2); R002 (E1 gate, seeds 1000-1009, H1); R003 (E4, seeds 0-19, H3); R004 (E5, seeds 0-19, H4); R005 (E3 de-smoothing ablation, seeds 0-2); R006 (E8, seeds 2000-2019, validation); R007 (E9, validation); R008 (E11, seeds 1000-1019); R009 (E12, seeds 0-19, H1); R010 (E7, seeds 0-29, H3); R011 (E6 numerical gate, seeds 0-2); R012 (E13 ablation, seeds 0-19). The Stage-0 gate covers matching sklearn coefficients at eps=0, gradient checks, the population-shrinkage sanity check and a leakage permutation test. Of these, only the shrinkage-solution check (R011) is reported here; the others are not shown in the supplied tables, and H7(a) leakage results are absent. Compute: CPU only, numpy and scikit-learn, no downloads, each experiment under 10 minutes. The full grids of eps, C, MLP width and fold counts are in the run artefacts and are not reproduced here, which is a reproducibility gap (see Limitations). Datasets: synthetic generators (self-defined or reconstructed) plus breast_cancer, iris and wine for consistency checks. Digits is image pixels, not tabular.

## 5. Results

**Premise (R002, gate).** Under the tuned eps=0 arm the pooled signed confidence-minus-accuracy error is -0.0027 (CI [-0.0101, 0.0046]; Table 2), so the overconfidence premise is not supported for LR. Under constrained L2 it is +0.0249 (CI [0.0194, 0.0303]); tuned minus constrained is -0.0276 (CI [-0.0360, -0.0192]). Three strongly-overconfident cells (C1, C5, C7) have intervals above 0 under constrained L2. This is absence of evidence, not equivalence, since no equivalence margin was set. On R008, tuned eps=0 has signed top-label error -0.0003 (CI [-0.0018, 0.0013]) versus +0.0532 for constrained L2 (Table 8). The signs of the tuned-baseline error differ across runs: R001 gives -0.0069 (CI [-0.0116, -0.0021]) and R003 gives +0.0159 (CI [0.0064, 0.0254]; Table 3), and I have not reconciled them. In R003, LS reduces this to +0.0023 (CI [-0.0043, 0.0089]).

**Primary comparison (R001).** Best-tuned LS minus best-tuned eps=0: +0.0009 (CI [-0.0006, 0.0025], Holm p=1.0, Table 1). Against eps0+TS (cvts): -0.0007 (CI [-0.0026, 0.0011]). Versus raw eps=0: -0.0120 (CI [-0.0149, -0.0090]). Raw LS minus eps0_cvts is +0.0050 (CI [0.0029, 0.0070]). Thus tuning or post-hoc correction accounts for the gain and LS adds none detectably. In R009 (reconstructed grid), nested LS minus eps=0 is -0.0008 (CI [-0.0016, -0.0001], Holm p=0.4858) and minus eps0+TS is -0.0002 (CI [-0.0015, 0.0010]; Table 9). A matched-grid LS variant selected eps near 0.042 with 59% of selections at the grid edge, against 0.015 for nested LS. The calibration component of the true-probability decomposition was lower for nested LS than eps=0 (-0.0011, Holm p=0.0413) but not versus eps0+TS. Given the pooling, stand-in cells and absent MLP early stopping, none of this is a regime-resolved test of H1.

**Mechanism (R010, R012, R003).** At n=100,000 the calibration slope is 0.7474 (CI [0.7468, 0.7481]) at eps=0.1 and 0.5956 (CI [0.5952, 0.5961]) at eps=0.2, against 0.9996 at eps=0 (Table 10). With L2 fixed, s(eps)-s(0) is -0.0518 (CI [-0.0524, -0.0511]) at eps=0.1 and -0.0948 (CI [-0.0957, -0.0938]) at eps=0.2 (Table 12). Test log loss rises with eps in those cells: 0.2488 at eps=0, 0.2759 at 0.1, 0.3107 at 0.2. This is a descriptive confirmation of the expected shrinkage direction on reconstructed generators. The numerical check of the population solution (R011) gave a relative deviation of 0.0000, below display resolution, with only a subset of metrics reported, so I treat it as favourable but not a clean pass. For finite-sample crossover (R003) the pooled paired MSE difference, LS minus eps=0, is -0.0010 (CI [-0.0024, 0.0005], Holm p=0.9731; Table 3). In none of 8 n x signal cells does the CI lie below 0, and the point estimates are negative only at n30 and n100. A crossover is not demonstrated. In R010, tuned joint LS versus best eps=0 on MSE to the truth is -0.0002 (CI [-0.0007, 0.0003]).

**Accuracy (R004).** Paired accuracy difference LS minus eps=0 is -0.0003 (CI [-0.0008, 0.0001], Holm p=0.1795), with 1.14% of argmax predictions changed (Table 4). In the synthetic LR cells |mean| is at most 0.14 pp. S5_lr and S4_mlp cells have wide intervals (about 5.9 and 3.9 pp), labelled inconclusive. TOST, ceiling and frontier-area analyses were not completed. Mean log loss at the eps-curve point (0.4584) and the L2-curve point at eps=0 (0.5667) compares different curve points, so Q-c is not settled.

**De-smoothing.** In R005 (3 seeds), de-smoothing minus raw LS in log loss is -0.0106 (CI [-0.0170, -0.0041], Holm p=0.0583) with 21.9% of entries clipped. In R001 de-smoothed LS is worse than raw LS (Brier 0.3446 vs 0.3383, ECE 0.0443 vs 0.0332 versus tuned variants). The two runs disagree and the cause (regime, clipping or bug) is unresolved.

**Power and validation.** R006 reports a minimum detectable log-loss difference of 0.0294 for best-tuned LS in a single stand-in configuration with SD 0.0063, and R007's analytic outputs are mostly NaN. These cannot be reconciled with R001's paired CI half-width of about 0.0016 and are not used as evidence.

### Result tables

**Table 1: R001, E2@v2 (primary) testing H2@v1, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
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

8 further reported metrics are summarised in analysis/quantitative/R001.yaml: ablation_cvts_minus_raw_log_loss, ablation_desmooth_minus_raw_log_loss, accuracy, log_loss, selected_frac_clipped_desmooth, selected_frac_variant_cvts, selected_log10_C_or_wd, true_prob_log_loss_gap_KL

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
| accuracy_based_signed_error (mean confidence minus accuracy) | eps0_constrained_l2 | 8 | 0.0249 | 0.0065 | [0.0194, 0.0303] |
| accuracy_based_signed_error (mean confidence minus accuracy) | eps0_tuned_l2 | 8 | -0.0027 | 0.0088 | [-0.0101, 0.0046] |
| accuracy_based_signed_error (mean confidence minus accuracy) | eps0_tuned_l2_altgrid | 8 | -0.0029 | 0.0039 | [-0.0062, 0.0004] |
| pilot_eps0_signed_error_vs_true_prob (mean over 20k test points of top-label confidence minus true P(top label|x), tuned eps0 LR; gate uses its one-sided 95% lower bound) | eps0_tuned_l2 | 8 | -0.0027 | 0.0087 | [-0.0101, 0.0046] |
| pilot_eps0_signed_error_vs_true_prob (mean over 20k test points of top-label confidence minus true P(top label|x), tuned eps0 LR; gate uses its one-sided 95% lower bound) | eps0_tuned_l2_altgrid | 8 | -0.0028 | 0.0041 | [-0.0063, 0.0006] |

165 further reported metrics are summarised in analysis/quantitative/R002.yaml: acc_signed_error__C1, acc_signed_error__C2, acc_signed_error__C3, acc_signed_error__C4, acc_signed_error__C5, acc_signed_error__C6, acc_signed_error__C7, acc_signed_error__C8, acc_signed_error__D1, acc_signed_error__D2, acc_signed_error__D3, altgrid_cells_skipped_by_time_guard, check_i_pass, check_ii_pass, check_iii_pass, constrained_eps0_signed_error_vs_true_prob, constrained_eps0_signed_error_vs_true_prob__also_primary_name, elapsed_seconds_total, eps_gt0_max_abs_coef_diff_vs_direct_minimiser (K=2 and K=3; pass < 1e-4), eps_gt0_stationarity_grad_inf_norm (pass < 1e-6), fam1_check_pass, fam1_true_prob_bin_check_max_abs_z, fires_SO_LR__C1, fires_SO_LR__C2, fires_SO_LR__C3, fires_SO_LR__C4, fires_SO_LR__C5, fires_SO_LR__C6, fires_SO_LR__C7, fires_SO_LR__C8, and 135 more

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
| mean_abs_deviation_from_population_LR_LS_probabilities (n=3000, unpenalised C=1e4 fit) | eps0_best_lr | 20 | 0.0185 | 0.0033 | [0.0170, 0.0201] |
| mean_abs_deviation_from_population_LR_LS_probabilities (n=3000, unpenalised C=1e4 fit) | ls_best_lr | 20 | 0.0148 | 0.0027 | [0.0136, 0.0161] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | eps0_best_lr | 20 | 0.0159 | 0.0203 | [0.0064, 0.0254] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | eps0_best_lr_altgrid | 20 | 0.0157 | 0.0204 | [0.0062, 0.0253] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | ls_best_lr | 20 | 0.0023 | 0.0141 | [-0.0043, 0.0089] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | ls_fixed_eps_at_eps0_C | 20 | -0.0175 | 0.0169 | [-0.0254, -0.0096] |
| mean_signed_error (predicted minus true probability, oriented toward the predicted class) | ls_fixed_eps_tuned_C | 20 | -0.0083 | 0.0127 | [-0.0143, -0.0024] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) | eps0_best_lr | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) | eps0_best_lr_altgrid | 20 | 0.0006 | 0.0031 | [-0.0009, 0.0020] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) | ls_best_lr | 20 | -0.0010 | 0.0031 | [-0.0024, 0.0005] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) | ls_fixed_eps_at_eps0_C | 20 | 0.0002 | 0.0017 | [-0.0007, 0.0010] |
| paired_MSE_vs_true_probability_difference (ls_best_lr minus eps0_best_lr) | ls_fixed_eps_tuned_C | 20 | -0.0002 | 0.0034 | [-0.0019, 0.0014] |

146 further reported metrics are summarised in analysis/quantitative/R003.yaml: KL(true || predicted) averaged over test points@n100_b1, KL(true || predicted) averaged over test points@n100_b4, KL(true || predicted) averaged over test points@n3000_b1, KL(true || predicted) averaged over test points@n3000_b4, KL(true || predicted) averaged over test points@n300_b1, KL(true || predicted) averaged over test points@n300_b4, KL(true || predicted) averaged over test points@n30_b1, KL(true || predicted) averaged over test points@n30_b4, Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms, Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n100_b1, Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n100_b4, Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n3000_b1, Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n3000_b4, Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n300_b1, Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n300_b4, Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n30_b1, Spearman(eps, MSE-vs-truth) per seed, for both fixed-eps arms@n30_b4, crosscheck_a_1e6, crosscheck_a_quadrature, impl_check_a_fail, impl_check_a_fail@n3000_b1, impl_check_a_fail@n3000_b4, impl_check_b_fail, mean_abs_deviation_from_population_LR_LS_probabilities (n=3000, unpenalised C=1e4 fit)@n3000_b1, mean_abs_deviation_from_population_LR_LS_probabilities (n=3000, unpenalised C=1e4 fit)@n3000_b4, mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n100_b1, mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n100_b4, mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n3000_b1, mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n3000_b4, mean_signed_error (predicted minus true probability, oriented toward the predicted class)@n300_b1, and 116 more

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
| fraction_of_test_predictions_with_changed_argmax | eps0_best_raw | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| fraction_of_test_predictions_with_changed_argmax | ls_best_raw | 20 | 0.0114 | 0.0057 | [0.0088, 0.0141] |
| paired_test_accuracy_difference (ls_best_raw minus eps0_best_raw), one-sided 95% lower bound | eps0_best_raw | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_test_accuracy_difference (ls_best_raw minus eps0_best_raw), one-sided 95% lower bound | ls_best_raw | 20 | -0.0003 | 0.0010 | [-0.0008, 0.0001] |
| test_log_loss and accuracy per curve point | eps0_best_raw | 20 | 0.4367 | 0.0073 | [0.4333, 0.4401] |
| test_log_loss and accuracy per curve point | eps_curve_at_tuned_l2 | 20 | 0.4584 | 0.0058 | [0.4557, 0.4611] |
| test_log_loss and accuracy per curve point | l2_curve_at_eps0 | 20 | 0.5667 | 0.0245 | [0.5552, 0.5781] |
| test_log_loss and accuracy per curve point | ls_best_raw | 20 | 0.4403 | 0.0053 | [0.4378, 0.4427] |

24 further reported metrics are summarised in analysis/quantitative/R004.yaml: area_between_pareto_frontiers (eps frontier minus L2 frontier in log loss, integrated over the shared accuracy range) with bootstrap CI, diff_S1_lr, diff_S2_lr, diff_S2_mlp, diff_S3_lr, diff_S4_lr, diff_S4_mlp, diff_S5_lr, diff_S6_lr, frac_units_at_ceiling, non_dominance_frequency per eps point: share of replicate-bootstrap resamples in which no L2 point has mean accuracy >= and mean log loss <= that eps point, real_breast_cancer_area, real_breast_cancer_diff, real_breast_cancer_nondom, real_digits_area, real_digits_diff, real_digits_nondom, real_iris_area, real_iris_diff, real_iris_nondom, real_wine_area, real_wine_diff, real_wine_nondom, test_accuracy

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
| tost_power_0p01_at_20reps | ls_best_tuned_cv5 | 1 | 0.0047 | 0.0000 | [n/a, n/a] |
| tost_power_0p01_at_20reps | ls_fixed_eps0p1_cv5 | 1 | 0.0000 | 0.0000 | [n/a, n/a] |
| tost_power_0p01_at_r002_reps | ls_best_tuned_cv5 | 1 | nan | 0.0000 | [n/a, n/a] |
| tost_power_0p01_at_r002_reps | ls_fixed_eps0p1_cv5 | 1 | nan | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_constrained_l2 | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_fixed_C_sweep | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_inner_cv10 | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_inner_cv3 | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_inner_cv5 | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_oracle_C | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | eps0_reduced_n_oracle_C | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | ls_best_tuned_cv5 | 1 | 51.9614 | 0.0000 | [n/a, n/a] |
| wall_time_seconds | ls_fixed_eps0p1_cv5 | 1 | 51.9614 | 0.0000 | [n/a, n/a] |

16 further reported metrics are summarised in analysis/quantitative/R006.yaml: calibration_slope_vs_true_logit_at_fixed_C, implied_logloss_cost_of_C_gap, logloss_curvature_at_oracle_log10C, paired_logloss_diff_sd_fixed_eps_minus_tuned_eps, r002_replicates_completed_per_cell_arm, r002_reproduction_max_abs_logloss_diff, selected_minus_oracle_log10C, selected_minus_reduced_n_oracle_log10C, signed_error_slope_per_decade_C, signed_error_vs_true_prob_at_fixed_C, so_minus_sc_contrast_simulated_mde, so_minus_sc_signed_error_at_tuned_C, so_minus_sc_signed_error_constrained, so_minus_sc_signed_error_slope, soft_target_newton_max_abs_prob_diff, tuned_minus_oracle_test_logloss

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

**Table 8: R008, E11@v2 (validation) testing H1@v3, seeds [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
| calibration_slope_vs_true_logit | eps0_constrained_l2 | 20 | 1.0547 | 0.0529 | [1.0299, 1.0795] |
| calibration_slope_vs_true_logit | eps0_tuned | 20 | 0.6172 | 0.0395 | [0.5987, 0.6357] |
| calibration_slope_vs_true_logit | eps0_tuned_null_split | 20 | 0.6175 | 0.0439 | [0.5969, 0.6380] |
| classwise_signed_error_true | eps0_constrained_l2 | 20 | 0.0216 | 0.0038 | [0.0198, 0.0234] |
| classwise_signed_error_true | eps0_tuned | 20 | 0.0204 | 0.0026 | [0.0192, 0.0216] |
| classwise_signed_error_true | eps0_tuned_null_split | 20 | 0.0205 | 0.0028 | [0.0192, 0.0218] |
| impl_check_max_abs_coef_diff | eps0_constrained_l2 | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| impl_check_max_abs_coef_diff | eps0_tuned | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| impl_check_max_abs_coef_diff | eps0_tuned_null_split | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| mean_abs_binned_signed_error_true | eps0_constrained_l2 | 20 | 0.0574 | 0.0028 | [0.0561, 0.0586] |
| mean_abs_binned_signed_error_true | eps0_tuned | 20 | 0.0262 | 0.0033 | [0.0246, 0.0277] |
| mean_abs_binned_signed_error_true | eps0_tuned_null_split | 20 | 0.0258 | 0.0026 | [0.0245, 0.0270] |
| mse_true_overconfident_part | eps0_constrained_l2 | 20 | 0.0418 | 0.0018 | [0.0410, 0.0427] |
| mse_true_overconfident_part | eps0_tuned | 20 | 0.0242 | 0.0011 | [0.0237, 0.0248] |
| mse_true_overconfident_part | eps0_tuned_null_split | 20 | 0.0242 | 0.0010 | [0.0238, 0.0247] |
| mse_true_underconfident_part | eps0_constrained_l2 | 20 | 0.0068 | 0.0003 | [0.0067, 0.0070] |
| mse_true_underconfident_part | eps0_tuned | 20 | 0.0117 | 0.0007 | [0.0113, 0.0120] |
| mse_true_underconfident_part | eps0_tuned_null_split | 20 | 0.0115 | 0.0011 | [0.0110, 0.0120] |
| null_split_delta_logloss_sd | eps0_constrained_l2 | 20 | nan | nan | [nan, nan] |
| null_split_delta_logloss_sd | eps0_tuned | 20 | nan | nan | [nan, nan] |
| null_split_delta_logloss_sd | eps0_tuned_null_split | 20 | 0.0251 | 0.0161 | [0.0176, 0.0326] |
| posterior_check_max_abs_dev | eps0_constrained_l2 | 20 | 0.0046 | 0.0007 | [0.0042, 0.0049] |
| posterior_check_max_abs_dev | eps0_tuned | 20 | 0.0046 | 0.0007 | [0.0042, 0.0049] |
| posterior_check_max_abs_dev | eps0_tuned_null_split | 20 | 0.0046 | 0.0007 | [0.0042, 0.0049] |
| power_detect_s_above_delta_oc | eps0_constrained_l2 | 20 | nan | nan | [nan, nan] |
| power_detect_s_above_delta_oc | eps0_tuned | 20 | nan | nan | [nan, nan] |
| power_detect_s_above_delta_oc | eps0_tuned_null_split | 20 | nan | nan | [nan, nan] |
| power_equivalence_zero_diff | eps0_constrained_l2 | 20 | nan | nan | [nan, nan] |
| power_equivalence_zero_diff | eps0_tuned | 20 | nan | nan | [nan, nan] |
| power_equivalence_zero_diff | eps0_tuned_null_split | 20 | nan | nan | [nan, nan] |
| power_ls_better_0p01 | eps0_constrained_l2 | 20 | nan | nan | [nan, nan] |
| power_ls_better_0p01 | eps0_tuned | 20 | nan | nan | [nan, nan] |
| power_ls_better_0p01 | eps0_tuned_null_split | 20 | nan | nan | [nan, nan] |
| selected_l2_grid_edge_fraction | eps0_constrained_l2 | 20 | 0.0228 | 0.0217 | [0.0126, 0.0329] |
| selected_l2_grid_edge_fraction | eps0_tuned | 20 | 0.0363 | 0.0210 | [0.0265, 0.0461] |
| selected_l2_grid_edge_fraction | eps0_tuned_null_split | 20 | 0.0378 | 0.0257 | [0.0258, 0.0499] |
| signed_conf_minus_acc | eps0_constrained_l2 | 20 | 0.0533 | 0.0032 | [0.0517, 0.0548] |
| signed_conf_minus_acc | eps0_tuned | 20 | -0.0003 | 0.0033 | [-0.0018, 0.0013] |
| signed_conf_minus_acc | eps0_tuned_null_split | 20 | 0.0005 | 0.0048 | [-0.0017, 0.0027] |
| signed_top_label_error_true | eps0_constrained_l2 | 20 | 0.0532 | 0.0033 | [0.0517, 0.0548] |
| signed_top_label_error_true | eps0_tuned | 20 | -0.0003 | 0.0033 | [-0.0018, 0.0013] |
| signed_top_label_error_true | eps0_tuned_null_split | 20 | 0.0005 | 0.0048 | [-0.0018, 0.0027] |

203 further reported metrics are summarised in analysis/quantitative/R008.yaml: cell_F1_0__lr__null_delta_logloss, cell_F1_0__lr__signed_top_label_error_true, cell_F1_0__lr__test_logloss, cell_F1_0__mlp__null_delta_logloss, cell_F1_0__mlp__signed_top_label_error_true, cell_F1_0__mlp__test_logloss, cell_F1_1__lr__null_delta_logloss, cell_F1_1__lr__signed_top_label_error_true, cell_F1_1__lr__test_logloss, cell_F1_1__mlp__signed_top_label_error_true, cell_F1_1__mlp__test_logloss, cell_F1_2__lr__null_delta_logloss, cell_F1_2__lr__signed_top_label_error_true, cell_F1_2__lr__test_logloss, cell_F1_2__mlp__null_delta_logloss, cell_F1_2__mlp__signed_top_label_error_true, cell_F1_2__mlp__test_logloss, cell_F1_3__lr__null_delta_logloss, cell_F1_3__lr__signed_top_label_error_true, cell_F1_3__lr__test_logloss, cell_F1_3__mlp__signed_top_label_error_true, cell_F1_3__mlp__test_logloss, cell_F1_4__lr__null_delta_logloss, cell_F1_4__lr__signed_top_label_error_true, cell_F1_4__lr__test_logloss, cell_F1_4__mlp__signed_top_label_error_true, cell_F1_4__mlp__test_logloss, cell_F1_5__lr__null_delta_logloss, cell_F1_5__lr__signed_top_label_error_true, cell_F1_5__lr__test_logloss, and 173 more

| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |
|---|---|---|---|---|---|---|---|
| signed_top_label_error_true (primary) | eps0_tuned vs eps0_tuned_null_split | -0.0008 | [-0.0034, 0.0019] | 1.0000 | -0.18 | no | no |
| signed_top_label_error_true (primary) | eps0_tuned vs eps0_constrained_l2 | -0.0535 | [-0.0556, -0.0514] | 0.0000 | -16.01 | yes | yes |
| calibration_slope_vs_true_logit | eps0_tuned vs eps0_tuned_null_split | -0.0003 | [-0.0270, 0.0265] | 1.0000 | -0.01 | no | no |
| calibration_slope_vs_true_logit | eps0_tuned vs eps0_constrained_l2 | -0.4375 | [-0.4674, -0.4075] | 0.0000 | -9.19 | yes | yes |
| mse_true_overconfident_part | eps0_tuned vs eps0_tuned_null_split | -0.0000 | [-0.0007, 0.0007] | 1.0000 | -0.02 | no | no |
| mse_true_overconfident_part | eps0_tuned vs eps0_constrained_l2 | -0.0176 | [-0.0186, -0.0166] | 0.0000 | -11.46 | yes | yes |
| mse_true_underconfident_part | eps0_tuned vs eps0_tuned_null_split | 0.0002 | [-0.0004, 0.0008] | 1.0000 | 0.20 | no | no |
| mse_true_underconfident_part | eps0_tuned vs eps0_constrained_l2 | 0.0049 | [0.0045, 0.0052] | 0.0000 | 8.99 | yes | yes |
| mean_abs_binned_signed_error_true | eps0_tuned vs eps0_tuned_null_split | 0.0004 | [-0.0015, 0.0023] | 1.0000 | 0.14 | no | no |
| mean_abs_binned_signed_error_true | eps0_tuned vs eps0_constrained_l2 | -0.0312 | [-0.0331, -0.0292] | 0.0000 | -10.03 | yes | yes |
| classwise_signed_error_true | eps0_tuned vs eps0_tuned_null_split | -0.0002 | [-0.0019, 0.0016] | 1.0000 | -0.06 | no | no |
| classwise_signed_error_true | eps0_tuned vs eps0_constrained_l2 | -0.0012 | [-0.0033, 0.0008] | 1.0000 | -0.38 | no | no |
| signed_conf_minus_acc | eps0_tuned vs eps0_tuned_null_split | -0.0007 | [-0.0034, 0.0019] | 1.0000 | -0.18 | no | no |
| signed_conf_minus_acc | eps0_tuned vs eps0_constrained_l2 | -0.0535 | [-0.0556, -0.0514] | 0.0000 | -16.01 | yes | yes |
| null_split_delta_logloss_sd | eps0_tuned vs eps0_tuned_null_split | nan | [nan, nan] | 1.0000 | nan | no | no |
| null_split_delta_logloss_sd | eps0_tuned vs eps0_constrained_l2 | nan | [nan, nan] | 1.0000 | nan | no | no |
| power_ls_better_0p01 | eps0_tuned vs eps0_tuned_null_split | nan | [nan, nan] | 1.0000 | nan | no | no |
| power_ls_better_0p01 | eps0_tuned vs eps0_constrained_l2 | nan | [nan, nan] | 1.0000 | nan | no | no |
| power_equivalence_zero_diff | eps0_tuned vs eps0_tuned_null_split | nan | [nan, nan] | 1.0000 | nan | no | no |
| power_equivalence_zero_diff | eps0_tuned vs eps0_constrained_l2 | nan | [nan, nan] | 1.0000 | nan | no | no |
| power_detect_s_above_delta_oc | eps0_tuned vs eps0_tuned_null_split | nan | [nan, nan] | 1.0000 | nan | no | no |
| power_detect_s_above_delta_oc | eps0_tuned vs eps0_constrained_l2 | nan | [nan, nan] | 1.0000 | nan | no | no |
| posterior_check_max_abs_dev | eps0_tuned vs eps0_tuned_null_split | 0.0000 | [-0.0005, 0.0005] | 1.0000 | 0.00 | no | no |
| posterior_check_max_abs_dev | eps0_tuned vs eps0_constrained_l2 | 0.0000 | [-0.0005, 0.0005] | 1.0000 | 0.00 | no | no |
| impl_check_max_abs_coef_diff | eps0_tuned vs eps0_tuned_null_split | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| impl_check_max_abs_coef_diff | eps0_tuned vs eps0_constrained_l2 | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| selected_l2_grid_edge_fraction | eps0_tuned vs eps0_tuned_null_split | -0.0015 | [-0.0165, 0.0135] | 1.0000 | -0.06 | no | no |
| selected_l2_grid_edge_fraction | eps0_tuned vs eps0_constrained_l2 | 0.0135 | [-0.0001, 0.0272] | 1.0000 | 0.62 | no | yes |

**Table 9: R009, E12@v2 (primary) testing H1@v3, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
| accuracy_delta | eps0_tuned | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| accuracy_delta | eps0_tuned_ts | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| accuracy_delta | ls_tuned_matched12 | 20 | -0.0001 | 0.0003 | [-0.0002, 0.0000] |
| accuracy_delta | ls_tuned_nested | 20 | -0.0001 | 0.0004 | [-0.0002, 0.0001] |
| brier_score | eps0_tuned | 20 | 0.0967 | 0.0008 | [0.0963, 0.0970] |
| brier_score | eps0_tuned_ts | 20 | 0.0964 | 0.0008 | [0.0961, 0.0968] |
| brier_score | ls_tuned_matched12 | 20 | 0.0968 | 0.0009 | [0.0964, 0.0973] |
| brier_score | ls_tuned_nested | 20 | 0.0966 | 0.0009 | [0.0961, 0.0970] |
| calibration_component_true_delta | eps0_tuned | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| calibration_component_true_delta | eps0_tuned_ts | 20 | -0.0008 | 0.0020 | [-0.0017, 0.0002] |
| calibration_component_true_delta | ls_tuned_matched12 | 20 | 0.0001 | 0.0022 | [-0.0009, 0.0011] |
| calibration_component_true_delta | ls_tuned_nested | 20 | -0.0011 | 0.0013 | [-0.0017, -0.0005] |
| ece_equal_mass | eps0_tuned | 20 | 0.0176 | 0.0028 | [0.0163, 0.0189] |
| ece_equal_mass | eps0_tuned_ts | 20 | 0.0151 | 0.0021 | [0.0142, 0.0161] |
| ece_equal_mass | ls_tuned_matched12 | 20 | 0.0197 | 0.0027 | [0.0184, 0.0210] |
| ece_equal_mass | ls_tuned_nested | 20 | 0.0160 | 0.0024 | [0.0149, 0.0171] |
| kl_true_to_pred_delta | eps0_tuned | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| kl_true_to_pred_delta | eps0_tuned_ts | 20 | -0.0007 | 0.0020 | [-0.0017, 0.0002] |
| kl_true_to_pred_delta | ls_tuned_matched12 | 20 | 0.0005 | 0.0021 | [-0.0004, 0.0015] |
| kl_true_to_pred_delta | ls_tuned_nested | 20 | -0.0009 | 0.0015 | [-0.0016, -0.0001] |
| refinement_component_true_delta | eps0_tuned | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| refinement_component_true_delta | eps0_tuned_ts | 20 | 0.0000 | 0.0001 | [-0.0000, 0.0001] |
| refinement_component_true_delta | ls_tuned_matched12 | 20 | 0.0004 | 0.0005 | [0.0002, 0.0007] |
| refinement_component_true_delta | ls_tuned_nested | 20 | 0.0002 | 0.0007 | [-0.0001, 0.0005] |
| selected_eps | eps0_tuned | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_eps | eps0_tuned_ts | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_eps | ls_tuned_matched12 | 20 | 0.0418 | 0.0048 | [0.0395, 0.0440] |
| selected_eps | ls_tuned_nested | 20 | 0.0150 | 0.0053 | [0.0125, 0.0175] |
| selected_param_grid_edge_fraction | eps0_tuned | 20 | 0.0125 | 0.0196 | [0.0033, 0.0217] |
| selected_param_grid_edge_fraction | eps0_tuned_ts | 20 | 0.0125 | 0.0196 | [0.0033, 0.0217] |
| selected_param_grid_edge_fraction | ls_tuned_matched12 | 20 | 0.5896 | 0.0638 | [0.5597, 0.6194] |
| selected_param_grid_edge_fraction | ls_tuned_nested | 20 | 0.0500 | 0.0320 | [0.0350, 0.0650] |
| signed_top_label_error_true | eps0_tuned | 20 | -0.0027 | 0.0042 | [-0.0047, -0.0008] |
| signed_top_label_error_true | eps0_tuned_ts | 20 | 0.0029 | 0.0035 | [0.0013, 0.0045] |
| signed_top_label_error_true | ls_tuned_matched12 | 20 | -0.0110 | 0.0041 | [-0.0129, -0.0091] |
| signed_top_label_error_true | ls_tuned_nested | 20 | -0.0040 | 0.0036 | [-0.0056, -0.0023] |
| test_log_loss_delta_ls_minus_eps0 | eps0_tuned | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| test_log_loss_delta_ls_minus_eps0 | eps0_tuned_ts | 20 | -0.0006 | 0.0022 | [-0.0016, 0.0004] |
| test_log_loss_delta_ls_minus_eps0 | ls_tuned_matched12 | 20 | 0.0005 | 0.0022 | [-0.0005, 0.0015] |
| test_log_loss_delta_ls_minus_eps0 | ls_tuned_nested | 20 | -0.0008 | 0.0015 | [-0.0016, -0.0001] |
| test_log_loss_delta_ls_minus_eps0_ts | eps0_tuned | 20 | 0.0006 | 0.0022 | [-0.0004, 0.0016] |
| test_log_loss_delta_ls_minus_eps0_ts | eps0_tuned_ts | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| test_log_loss_delta_ls_minus_eps0_ts | ls_tuned_matched12 | 20 | 0.0011 | 0.0031 | [-0.0003, 0.0026] |
| test_log_loss_delta_ls_minus_eps0_ts | ls_tuned_nested | 20 | -0.0002 | 0.0014 | [-0.0009, 0.0004] |

253 further reported metrics are summarised in analysis/quantitative/R009.yaml: cell::LR::F1_n1000_d20::d_acc, cell::LR::F1_n1000_d20::d_cal, cell::LR::F1_n1000_d20::d_ll_eps0, cell::LR::F1_n1000_d20::d_ll_ts, cell::LR::F1_n1000_d20::d_ref, cell::LR::F1_n1000_d20::ll, cell::LR::F1_n1000_d20::sel_eps, cell::LR::F1_n1000_d20::signed, cell::LR::F1_n1000_d5::d_acc, cell::LR::F1_n1000_d5::d_cal, cell::LR::F1_n1000_d5::d_ll_eps0, cell::LR::F1_n1000_d5::d_ll_ts, cell::LR::F1_n1000_d5::d_ref, cell::LR::F1_n1000_d5::ll, cell::LR::F1_n1000_d5::sel_eps, cell::LR::F1_n1000_d5::signed, cell::LR::F1_n100_d20::d_acc, cell::LR::F1_n100_d20::d_cal, cell::LR::F1_n100_d20::d_ll_eps0, cell::LR::F1_n100_d20::d_ll_ts, cell::LR::F1_n100_d20::d_ref, cell::LR::F1_n100_d20::ll, cell::LR::F1_n100_d20::sel_eps, cell::LR::F1_n100_d20::signed, cell::LR::F1_n100_d5::d_acc, cell::LR::F1_n100_d5::d_cal, cell::LR::F1_n100_d5::d_ll_eps0, cell::LR::F1_n100_d5::d_ll_ts, cell::LR::F1_n100_d5::d_ref, cell::LR::F1_n100_d5::ll, and 223 more

| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |
|---|---|---|---|---|---|---|---|
| test_log_loss_delta_ls_minus_eps0 (primary) | ls_tuned_nested vs eps0_tuned | -0.0008 | [-0.0016, -0.0001] | 0.4858 | -0.77 | no | no |
| test_log_loss_delta_ls_minus_eps0 (primary) | ls_tuned_nested vs eps0_tuned_ts | -0.0002 | [-0.0015, 0.0010] | 1.0000 | -0.12 | no | no |
| test_log_loss_delta_ls_minus_eps0 (primary) | ls_tuned_nested vs ls_tuned_matched12 | -0.0014 | [-0.0026, -0.0002] | 0.5201 | -0.72 | no | no |
| test_log_loss_delta_ls_minus_eps0_ts | ls_tuned_nested vs eps0_tuned | -0.0008 | [-0.0020, 0.0003] | 1.0000 | -0.45 | no | no |
| test_log_loss_delta_ls_minus_eps0_ts | ls_tuned_nested vs eps0_tuned_ts | -0.0002 | [-0.0009, 0.0004] | 1.0000 | -0.23 | no | no |
| test_log_loss_delta_ls_minus_eps0_ts | ls_tuned_nested vs ls_tuned_matched12 | -0.0014 | [-0.0029, 0.0002] | 1.0000 | -0.57 | no | no |
| kl_true_to_pred_delta | ls_tuned_nested vs eps0_tuned | -0.0009 | [-0.0016, -0.0001] | 0.4858 | -0.78 | no | yes |
| kl_true_to_pred_delta | ls_tuned_nested vs eps0_tuned_ts | -0.0001 | [-0.0013, 0.0010] | 1.0000 | -0.06 | no | no |
| kl_true_to_pred_delta | ls_tuned_nested vs ls_tuned_matched12 | -0.0014 | [-0.0026, -0.0002] | 0.4858 | -0.75 | no | yes |
| calibration_component_true_delta | ls_tuned_nested vs eps0_tuned | -0.0011 | [-0.0017, -0.0005] | 0.0413 | -1.13 | yes | yes |
| calibration_component_true_delta | ls_tuned_nested vs eps0_tuned_ts | -0.0003 | [-0.0014, 0.0008] | 1.0000 | -0.18 | no | no |
| calibration_component_true_delta | ls_tuned_nested vs ls_tuned_matched12 | -0.0012 | [-0.0023, -0.0000] | 0.8258 | -0.65 | no | yes |
| refinement_component_true_delta | ls_tuned_nested vs eps0_tuned | 0.0002 | [-0.0001, 0.0005] | 1.0000 | 0.46 | no | no |
| refinement_component_true_delta | ls_tuned_nested vs eps0_tuned_ts | 0.0002 | [-0.0001, 0.0005] | 1.0000 | 0.41 | no | no |
| refinement_component_true_delta | ls_tuned_nested vs ls_tuned_matched12 | -0.0002 | [-0.0006, 0.0002] | 1.0000 | -0.35 | no | no |
| signed_top_label_error_true | ls_tuned_nested vs eps0_tuned | -0.0012 | [-0.0037, 0.0012] | 1.0000 | -0.32 | no | no |
| signed_top_label_error_true | ls_tuned_nested vs eps0_tuned_ts | -0.0069 | [-0.0091, -0.0046] | 0.0000 | -1.91 | yes | no |
| signed_top_label_error_true | ls_tuned_nested vs ls_tuned_matched12 | 0.0070 | [0.0046, 0.0095] | 0.0000 | 1.80 | yes | no |
| brier_score | ls_tuned_nested vs eps0_tuned | -0.0001 | [-0.0006, 0.0004] | 1.0000 | -0.12 | no | no |
| brier_score | ls_tuned_nested vs eps0_tuned_ts | 0.0001 | [-0.0004, 0.0006] | 1.0000 | 0.13 | no | no |
| brier_score | ls_tuned_nested vs ls_tuned_matched12 | -0.0003 | [-0.0009, 0.0003] | 1.0000 | -0.31 | no | no |
| ece_equal_mass | ls_tuned_nested vs eps0_tuned | -0.0016 | [-0.0033, 0.0001] | 1.0000 | -0.59 | no | yes |
| ece_equal_mass | ls_tuned_nested vs eps0_tuned_ts | 0.0009 | [-0.0006, 0.0023] | 1.0000 | 0.37 | no | no |
| ece_equal_mass | ls_tuned_nested vs ls_tuned_matched12 | -0.0037 | [-0.0053, -0.0021] | 0.0014 | -1.41 | yes | yes |
| accuracy_delta | ls_tuned_nested vs eps0_tuned | -0.0001 | [-0.0002, 0.0001] | 1.0000 | -0.19 | no | no |
| accuracy_delta | ls_tuned_nested vs eps0_tuned_ts | -0.0001 | [-0.0002, 0.0001] | 1.0000 | -0.19 | no | no |
| accuracy_delta | ls_tuned_nested vs ls_tuned_matched12 | 0.0001 | [-0.0002, 0.0003] | 1.0000 | 0.17 | no | no |
| selected_eps | ls_tuned_nested vs eps0_tuned | 0.0150 | [0.0125, 0.0175] | 0.0000 | 3.89 | yes | yes |
| selected_eps | ls_tuned_nested vs eps0_tuned_ts | 0.0150 | [0.0125, 0.0175] | 0.0000 | 3.89 | yes | yes |
| selected_eps | ls_tuned_nested vs ls_tuned_matched12 | -0.0268 | [-0.0300, -0.0235] | 0.0000 | -5.17 | yes | yes |
| selected_param_grid_edge_fraction | ls_tuned_nested vs eps0_tuned | 0.0375 | [0.0204, 0.0546] | 0.0025 | 1.39 | yes | yes |
| selected_param_grid_edge_fraction | ls_tuned_nested vs eps0_tuned_ts | 0.0375 | [0.0204, 0.0546] | 0.0025 | 1.39 | yes | yes |
| selected_param_grid_edge_fraction | ls_tuned_nested vs ls_tuned_matched12 | -0.5396 | [-0.5723, -0.5069] | 0.0000 | -10.48 | yes | yes |

**Table 10: R010, E7@v2 (primary) testing H3@v2, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
| calibration_slope_vs_true_logit | l2_eps0_best | 30 | 0.7631 | 0.0808 | [0.7330, 0.7933] |
| calibration_slope_vs_true_logit | l2_eps0_best_slope_shrink | 30 | 0.8402 | 0.3319 | [0.7163, 0.9642] |
| calibration_slope_vs_true_logit | l2_eps0_raw | 30 | 0.7702 | 0.0732 | [0.7429, 0.7976] |
| calibration_slope_vs_true_logit | ls_fixed_eps_0 | 30 | 0.9996 | 0.0030 | [0.9985, 1.0007] |
| calibration_slope_vs_true_logit | ls_fixed_eps_0.01 | 30 | 0.9648 | 0.0028 | [0.9637, 0.9658] |
| calibration_slope_vs_true_logit | ls_fixed_eps_0.02 | 30 | 0.9331 | 0.0026 | [0.9322, 0.9341] |
| calibration_slope_vs_true_logit | ls_fixed_eps_0.05 | 30 | 0.8523 | 0.0022 | [0.8515, 0.8531] |
| calibration_slope_vs_true_logit | ls_fixed_eps_0.1 | 30 | 0.7474 | 0.0018 | [0.7468, 0.7481] |
| calibration_slope_vs_true_logit | ls_fixed_eps_0.2 | 30 | 0.5956 | 0.0013 | [0.5952, 0.5961] |
| calibration_slope_vs_true_logit | ls_joint_tuned_nested | 30 | 0.7519 | 0.0781 | [0.7227, 0.7810] |
| calibration_slope_vs_true_logit | ls_joint_tuned_raw | 30 | 0.7536 | 0.0738 | [0.7260, 0.7811] |
| kl_true_to_predicted | l2_eps0_best | 30 | 0.0806 | 0.0169 | [0.0743, 0.0869] |
| kl_true_to_predicted | l2_eps0_best_slope_shrink | 30 | 0.0871 | 0.0389 | [0.0726, 0.1017] |
| kl_true_to_predicted | l2_eps0_raw | 30 | 0.0751 | 0.0149 | [0.0695, 0.0806] |
| kl_true_to_predicted | ls_fixed_eps_0 | 30 | 0.0001 | 0.0000 | [0.0000, 0.0001] |
| kl_true_to_predicted | ls_fixed_eps_0.01 | 30 | 0.0002 | 0.0000 | [0.0002, 0.0002] |
| kl_true_to_predicted | ls_fixed_eps_0.02 | 30 | 0.0007 | 0.0000 | [0.0007, 0.0008] |
| kl_true_to_predicted | ls_fixed_eps_0.05 | 30 | 0.0036 | 0.0001 | [0.0036, 0.0036] |
| kl_true_to_predicted | ls_fixed_eps_0.1 | 30 | 0.0111 | 0.0001 | [0.0111, 0.0112] |
| kl_true_to_predicted | ls_fixed_eps_0.2 | 30 | 0.0310 | 0.0001 | [0.0310, 0.0311] |
| kl_true_to_predicted | ls_joint_tuned_nested | 30 | 0.0809 | 0.0168 | [0.0746, 0.0871] |
| kl_true_to_predicted | ls_joint_tuned_raw | 30 | 0.0786 | 0.0162 | [0.0726, 0.0847] |
| mse_true_per_arm | l2_eps0_best | 30 | 0.0266 | 0.0031 | [0.0254, 0.0278] |
| mse_true_per_arm | l2_eps0_best_slope_shrink | 30 | 0.0269 | 0.0033 | [0.0257, 0.0282] |
| mse_true_per_arm | l2_eps0_raw | 30 | 0.0251 | 0.0030 | [0.0239, 0.0262] |
| mse_true_per_arm | ls_fixed_eps_0 | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| mse_true_per_arm | ls_fixed_eps_0.01 | 30 | 0.0001 | 0.0000 | [0.0001, 0.0001] |
| mse_true_per_arm | ls_fixed_eps_0.02 | 30 | 0.0001 | 0.0000 | [0.0001, 0.0002] |
| mse_true_per_arm | ls_fixed_eps_0.05 | 30 | 0.0007 | 0.0000 | [0.0007, 0.0007] |
| mse_true_per_arm | ls_fixed_eps_0.1 | 30 | 0.0024 | 0.0000 | [0.0024, 0.0025] |
| mse_true_per_arm | ls_fixed_eps_0.2 | 30 | 0.0078 | 0.0000 | [0.0078, 0.0078] |
| mse_true_per_arm | ls_joint_tuned_nested | 30 | 0.0264 | 0.0031 | [0.0253, 0.0276] |
| mse_true_per_arm | ls_joint_tuned_raw | 30 | 0.0256 | 0.0029 | [0.0245, 0.0267] |
| nonconverged_fit_fraction | l2_eps0_best | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction | l2_eps0_best_slope_shrink | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction | l2_eps0_raw | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction | ls_fixed_eps_0 | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction | ls_fixed_eps_0.01 | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction | ls_fixed_eps_0.02 | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction | ls_fixed_eps_0.05 | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction | ls_fixed_eps_0.1 | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction | ls_fixed_eps_0.2 | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction | ls_joint_tuned_nested | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| nonconverged_fit_fraction | ls_joint_tuned_raw | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_delta_mse_true_raw_diagnostic | l2_eps0_best | 30 | 0.0015 | 0.0020 | [0.0008, 0.0023] |
| paired_delta_mse_true_raw_diagnostic | l2_eps0_best_slope_shrink | 30 | 0.0018 | 0.0023 | [0.0010, 0.0027] |
| paired_delta_mse_true_raw_diagnostic | l2_eps0_raw | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_delta_mse_true_raw_diagnostic | ls_joint_tuned_nested | 30 | 0.0013 | 0.0020 | [0.0006, 0.0021] |
| paired_delta_mse_true_raw_diagnostic | ls_joint_tuned_raw | 30 | 0.0006 | 0.0021 | [-0.0002, 0.0013] |
| paired_delta_mse_true_vs_best_eps0 | l2_eps0_best | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_delta_mse_true_vs_best_eps0 | l2_eps0_best_slope_shrink | 30 | 0.0003 | 0.0008 | [0.0000, 0.0006] |
| paired_delta_mse_true_vs_best_eps0 | l2_eps0_raw | 30 | -0.0015 | 0.0020 | [-0.0023, -0.0008] |
| paired_delta_mse_true_vs_best_eps0 | ls_joint_tuned_nested | 30 | -0.0002 | 0.0013 | [-0.0007, 0.0003] |
| paired_delta_mse_true_vs_best_eps0 | ls_joint_tuned_raw | 30 | -0.0010 | 0.0016 | [-0.0016, -0.0003] |
| paired_delta_mse_true_vs_slope_shrink | l2_eps0_best | 30 | -0.0003 | 0.0008 | [-0.0006, -0.0000] |
| paired_delta_mse_true_vs_slope_shrink | l2_eps0_best_slope_shrink | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_delta_mse_true_vs_slope_shrink | l2_eps0_raw | 30 | -0.0018 | 0.0023 | [-0.0027, -0.0010] |
| paired_delta_mse_true_vs_slope_shrink | ls_joint_tuned_nested | 30 | -0.0005 | 0.0015 | [-0.0010, 0.0001] |
| paired_delta_mse_true_vs_slope_shrink | ls_joint_tuned_raw | 30 | -0.0013 | 0.0019 | [-0.0020, -0.0006] |
| selected_c_grid_edge_fraction | l2_eps0_best | 30 | 0.0167 | 0.0297 | [0.0056, 0.0278] |
| selected_c_grid_edge_fraction | l2_eps0_best_slope_shrink | 30 | 0.0167 | 0.0297 | [0.0056, 0.0278] |
| selected_c_grid_edge_fraction | l2_eps0_raw | 30 | 0.0259 | 0.0349 | [0.0129, 0.0390] |
| selected_c_grid_edge_fraction | ls_joint_tuned_nested | 30 | 0.0241 | 0.0348 | [0.0111, 0.0371] |
| selected_c_grid_edge_fraction | ls_joint_tuned_raw | 30 | 0.0704 | 0.0618 | [0.0473, 0.0934] |
| selected_eps_and_penalty_distribution | l2_eps0_best | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_eps_and_penalty_distribution | l2_eps0_best_slope_shrink | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_eps_and_penalty_distribution | l2_eps0_raw | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_eps_and_penalty_distribution | ls_fixed_eps_0 | 30 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| selected_eps_and_penalty_distribution | ls_fixed_eps_0.01 | 30 | 0.0100 | 0.0000 | [0.0100, 0.0100] |
| selected_eps_and_penalty_distribution | ls_fixed_eps_0.02 | 30 | 0.0200 | 0.0000 | [0.0200, 0.0200] |
| selected_eps_and_penalty_distribution | ls_fixed_eps_0.05 | 30 | 0.0500 | 0.0000 | [0.0500, 0.0500] |
| selected_eps_and_penalty_distribution | ls_fixed_eps_0.1 | 30 | 0.1000 | 0.0000 | [0.1000, 0.1000] |
| selected_eps_and_penalty_distribution | ls_fixed_eps_0.2 | 30 | 0.2000 | 0.0000 | [0.2000, 0.2000] |
| selected_eps_and_penalty_distribution | ls_joint_tuned_nested | 30 | 0.0424 | 0.0160 | [0.0364, 0.0484] |
| selected_eps_and_penalty_distribution | ls_joint_tuned_raw | 30 | 0.0514 | 0.0174 | [0.0449, 0.0579] |
| signed_top_label_error | l2_eps0_best | 30 | -0.0295 | 0.0112 | [-0.0337, -0.0254] |
| signed_top_label_error | l2_eps0_best_slope_shrink | 30 | -0.0250 | 0.0123 | [-0.0296, -0.0204] |
| signed_top_label_error | l2_eps0_raw | 30 | -0.0299 | 0.0100 | [-0.0337, -0.0262] |
| signed_top_label_error | ls_fixed_eps_0 | 30 | -0.0000 | 0.0004 | [-0.0002, 0.0001] |
| signed_top_label_error | ls_fixed_eps_0.01 | 30 | -0.0045 | 0.0004 | [-0.0047, -0.0044] |
| signed_top_label_error | ls_fixed_eps_0.02 | 30 | -0.0089 | 0.0004 | [-0.0091, -0.0088] |
| signed_top_label_error | ls_fixed_eps_0.05 | 30 | -0.0214 | 0.0003 | [-0.0215, -0.0212] |
| signed_top_label_error | ls_fixed_eps_0.1 | 30 | -0.0405 | 0.0003 | [-0.0406, -0.0404] |
| signed_top_label_error | ls_fixed_eps_0.2 | 30 | -0.0749 | 0.0003 | [-0.0750, -0.0748] |
| signed_top_label_error | ls_joint_tuned_nested | 30 | -0.0315 | 0.0114 | [-0.0357, -0.0273] |
| signed_top_label_error | ls_joint_tuned_raw | 30 | -0.0314 | 0.0099 | [-0.0351, -0.0277] |
| test_accuracy_vs_bayes_label | l2_eps0_best | 30 | 0.8528 | 0.0123 | [0.8481, 0.8574] |
| test_accuracy_vs_bayes_label | l2_eps0_best_slope_shrink | 30 | 0.8528 | 0.0123 | [0.8481, 0.8574] |
| test_accuracy_vs_bayes_label | l2_eps0_raw | 30 | 0.8549 | 0.0132 | [0.8500, 0.8599] |
| test_accuracy_vs_bayes_label | ls_fixed_eps_0 | 30 | 0.9956 | 0.0004 | [0.9955, 0.9958] |
| test_accuracy_vs_bayes_label | ls_fixed_eps_0.01 | 30 | 0.9956 | 0.0004 | [0.9955, 0.9958] |
| test_accuracy_vs_bayes_label | ls_fixed_eps_0.02 | 30 | 0.9956 | 0.0004 | [0.9955, 0.9958] |
| test_accuracy_vs_bayes_label | ls_fixed_eps_0.05 | 30 | 0.9956 | 0.0004 | [0.9955, 0.9958] |
| test_accuracy_vs_bayes_label | ls_fixed_eps_0.1 | 30 | 0.9956 | 0.0004 | [0.9955, 0.9958] |
| test_accuracy_vs_bayes_label | ls_fixed_eps_0.2 | 30 | 0.9956 | 0.0004 | [0.9954, 0.9957] |
| test_accuracy_vs_bayes_label | ls_joint_tuned_nested | 30 | 0.8540 | 0.0118 | [0.8496, 0.8584] |
| test_accuracy_vs_bayes_label | ls_joint_tuned_raw | 30 | 0.8544 | 0.0125 | [0.8497, 0.8590] |

503 further reported metrics are summarised in analysis/quantitative/R010.yaml: calibration_slope_vs_true_logit@F_aniso_n100_tau1, calibration_slope_vs_true_logit@F_aniso_n100_tau2.5, calibration_slope_vs_true_logit@F_aniso_n100_tau4, calibration_slope_vs_true_logit@F_aniso_n30_tau1, calibration_slope_vs_true_logit@F_aniso_n30_tau2.5, calibration_slope_vs_true_logit@F_aniso_n30_tau4, calibration_slope_vs_true_logit@F_aniso_n500_tau1, calibration_slope_vs_true_logit@F_aniso_n500_tau2.5, calibration_slope_vs_true_logit@F_aniso_n500_tau4, calibration_slope_vs_true_logit@F_iso_n100_tau1, calibration_slope_vs_true_logit@F_iso_n100_tau2.5, calibration_slope_vs_true_logit@F_iso_n100_tau4, calibration_slope_vs_true_logit@F_iso_n30_tau1, calibration_slope_vs_true_logit@F_iso_n30_tau2.5, calibration_slope_vs_true_logit@F_iso_n30_tau4, calibration_slope_vs_true_logit@F_iso_n500_tau1, calibration_slope_vs_true_logit@F_iso_n500_tau2.5, calibration_slope_vs_true_logit@F_iso_n500_tau4, calibration_slope_vs_true_logit@F_mean, calibration_slope_vs_true_logit@M2_aniso_tau1, calibration_slope_vs_true_logit@M2_aniso_tau2.5, calibration_slope_vs_true_logit@M2_aniso_tau4, calibration_slope_vs_true_logit@M2_iso_tau1, calibration_slope_vs_true_logit@M2_iso_tau2.5, calibration_slope_vs_true_logit@M2_iso_tau4, empirical_over_sandwich_sd_ratio, empirical_over_sandwich_sd_ratio@M2_aniso_tau1, empirical_over_sandwich_sd_ratio@M2_aniso_tau2.5, empirical_over_sandwich_sd_ratio@M2_aniso_tau4, empirical_over_sandwich_sd_ratio@M2_iso_tau1, and 473 more

| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |
|---|---|---|---|---|---|---|---|
| paired_delta_mse_true_vs_best_eps0 (primary) | ls_joint_tuned_nested vs l2_eps0_best | -0.0002 | [-0.0007, 0.0003] | 1.0000 | -0.18 | no | no |
| paired_delta_mse_true_vs_best_eps0 (primary) | ls_joint_tuned_nested vs l2_eps0_best_slope_shrink | -0.0005 | [-0.0011, 0.0001] | 1.0000 | -0.45 | no | no |
| paired_delta_mse_true_vs_best_eps0 (primary) | ls_joint_tuned_nested vs l2_eps0_raw | 0.0013 | [0.0005, 0.0022] | 0.1530 | 0.77 | no | yes |
| paired_delta_mse_true_vs_best_eps0 (primary) | ls_joint_tuned_nested vs ls_joint_tuned_raw | 0.0008 | [0.0000, 0.0016] | 1.0000 | 0.52 | no | no |
| paired_delta_mse_true_vs_slope_shrink | ls_joint_tuned_nested vs l2_eps0_best | -0.0002 | [-0.0008, 0.0004] | 1.0000 | -0.14 | no | no |
| paired_delta_mse_true_vs_slope_shrink | ls_joint_tuned_nested vs l2_eps0_best_slope_shrink | -0.0005 | [-0.0010, 0.0001] | 1.0000 | -0.46 | no | no |
| paired_delta_mse_true_vs_slope_shrink | ls_joint_tuned_nested vs l2_eps0_raw | 0.0013 | [0.0004, 0.0023] | 0.3221 | 0.70 | no | yes |
| paired_delta_mse_true_vs_slope_shrink | ls_joint_tuned_nested vs ls_joint_tuned_raw | 0.0008 | [-0.0001, 0.0017] | 1.0000 | 0.46 | no | no |
| paired_delta_mse_true_raw_diagnostic | ls_joint_tuned_nested vs l2_eps0_best | -0.0002 | [-0.0012, 0.0009] | 1.0000 | -0.08 | no | no |
| paired_delta_mse_true_raw_diagnostic | ls_joint_tuned_nested vs l2_eps0_best_slope_shrink | -0.0005 | [-0.0016, 0.0006] | 1.0000 | -0.23 | no | no |
| paired_delta_mse_true_raw_diagnostic | ls_joint_tuned_nested vs l2_eps0_raw | 0.0013 | [0.0006, 0.0021] | 0.0342 | 0.95 | yes | yes |
| paired_delta_mse_true_raw_diagnostic | ls_joint_tuned_nested vs ls_joint_tuned_raw | 0.0008 | [-0.0003, 0.0018] | 1.0000 | 0.38 | no | no |
| mse_true_per_arm | ls_joint_tuned_nested vs l2_eps0_best | -0.0002 | [-0.0018, 0.0014] | 1.0000 | -0.05 | no | no |
| mse_true_per_arm | ls_joint_tuned_nested vs l2_eps0_best_slope_shrink | -0.0005 | [-0.0021, 0.0012] | 1.0000 | -0.15 | no | no |
| mse_true_per_arm | ls_joint_tuned_nested vs l2_eps0_raw | 0.0013 | [-0.0002, 0.0029] | 1.0000 | 0.44 | no | no |
| mse_true_per_arm | ls_joint_tuned_nested vs ls_joint_tuned_raw | 0.0008 | [-0.0008, 0.0023] | 1.0000 | 0.26 | no | no |
| calibration_slope_vs_true_logit | ls_joint_tuned_nested vs l2_eps0_best | -0.0113 | [-0.0523, 0.0298] | 1.0000 | -0.14 | no | no |
| calibration_slope_vs_true_logit | ls_joint_tuned_nested vs l2_eps0_best_slope_shrink | -0.0884 | [-0.2151, 0.0384] | 1.0000 | -0.36 | no | no |
| calibration_slope_vs_true_logit | ls_joint_tuned_nested vs l2_eps0_raw | -0.0184 | [-0.0575, 0.0207] | 1.0000 | -0.24 | no | no |
| calibration_slope_vs_true_logit | ls_joint_tuned_nested vs ls_joint_tuned_raw | -0.0017 | [-0.0410, 0.0376] | 1.0000 | -0.02 | no | no |
| signed_top_label_error | ls_joint_tuned_nested vs l2_eps0_best | -0.0020 | [-0.0078, 0.0038] | 1.0000 | -0.17 | no | no |
| signed_top_label_error | ls_joint_tuned_nested vs l2_eps0_best_slope_shrink | -0.0065 | [-0.0127, -0.0004] | 1.0000 | -0.55 | no | yes |
| signed_top_label_error | ls_joint_tuned_nested vs l2_eps0_raw | -0.0016 | [-0.0071, 0.0039] | 1.0000 | -0.15 | no | no |
| signed_top_label_error | ls_joint_tuned_nested vs ls_joint_tuned_raw | -0.0001 | [-0.0056, 0.0054] | 1.0000 | -0.01 | no | no |
| kl_true_to_predicted | ls_joint_tuned_nested vs l2_eps0_best | 0.0003 | [-0.0084, 0.0090] | 1.0000 | 0.02 | no | no |
| kl_true_to_predicted | ls_joint_tuned_nested vs l2_eps0_best_slope_shrink | -0.0063 | [-0.0219, 0.0094] | 1.0000 | -0.21 | no | no |
| kl_true_to_predicted | ls_joint_tuned_nested vs l2_eps0_raw | 0.0058 | [-0.0024, 0.0140] | 1.0000 | 0.36 | no | no |
| kl_true_to_predicted | ls_joint_tuned_nested vs ls_joint_tuned_raw | 0.0022 | [-0.0063, 0.0107] | 1.0000 | 0.13 | no | no |
| test_accuracy_vs_bayes_label | ls_joint_tuned_nested vs l2_eps0_best | 0.0012 | [-0.0050, 0.0075] | 1.0000 | 0.10 | no | no |
| test_accuracy_vs_bayes_label | ls_joint_tuned_nested vs l2_eps0_best_slope_shrink | 0.0012 | [-0.0050, 0.0075] | 1.0000 | 0.10 | no | no |
| test_accuracy_vs_bayes_label | ls_joint_tuned_nested vs l2_eps0_raw | -0.0009 | [-0.0074, 0.0056] | 1.0000 | -0.07 | no | no |
| test_accuracy_vs_bayes_label | ls_joint_tuned_nested vs ls_joint_tuned_raw | -0.0004 | [-0.0067, 0.0059] | 1.0000 | -0.03 | no | no |
| nonconverged_fit_fraction | ls_joint_tuned_nested vs l2_eps0_best | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| nonconverged_fit_fraction | ls_joint_tuned_nested vs l2_eps0_best_slope_shrink | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| nonconverged_fit_fraction | ls_joint_tuned_nested vs l2_eps0_raw | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| nonconverged_fit_fraction | ls_joint_tuned_nested vs ls_joint_tuned_raw | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| selected_c_grid_edge_fraction | ls_joint_tuned_nested vs l2_eps0_best | 0.0074 | [-0.0093, 0.0241] | 1.0000 | 0.23 | no | no |
| selected_c_grid_edge_fraction | ls_joint_tuned_nested vs l2_eps0_best_slope_shrink | 0.0074 | [-0.0093, 0.0241] | 1.0000 | 0.23 | no | no |
| selected_c_grid_edge_fraction | ls_joint_tuned_nested vs l2_eps0_raw | -0.0019 | [-0.0199, 0.0162] | 1.0000 | -0.05 | no | no |
| selected_c_grid_edge_fraction | ls_joint_tuned_nested vs ls_joint_tuned_raw | -0.0463 | [-0.0724, -0.0202] | 0.0342 | -0.91 | yes | yes |
| selected_eps_and_penalty_distribution | ls_joint_tuned_nested vs l2_eps0_best | 0.0424 | [0.0364, 0.0484] | 0.0000 | 3.70 | yes | yes |
| selected_eps_and_penalty_distribution | ls_joint_tuned_nested vs l2_eps0_best_slope_shrink | 0.0424 | [0.0364, 0.0484] | 0.0000 | 3.70 | yes | yes |
| selected_eps_and_penalty_distribution | ls_joint_tuned_nested vs l2_eps0_raw | 0.0424 | [0.0364, 0.0484] | 0.0000 | 3.70 | yes | yes |
| selected_eps_and_penalty_distribution | ls_joint_tuned_nested vs ls_joint_tuned_raw | -0.0090 | [-0.0177, -0.0004] | 1.0000 | -0.53 | no | yes |

**Table 11: R011, E6@v2 (validation) testing H3@v2, seeds [0, 1, 2]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
| rel_norm_dev_quadrature_ref_vs_c_times_wstar | closed_form_c_eps_tau_sigma_prime | 3 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| rel_norm_dev_quadrature_ref_vs_c_times_wstar | soft_target_lr_quadrature_reference | 3 | 0.0000 | 0.0000 | [0.0000, 0.0000] |

10 further reported metrics are summarised in analysis/quantitative/R011.yaml: discrimination_ratio_abs_1_minus_c_over_tol, intercept_abs_value_reference, max_abs_mean_sigma_d2, mc_ref_rel_dev_over_sandwich_se, mean_abs_gap_sigma_w_eps_vs_p_ls, min_derivative_f_c, one_minus_cosine_quadrature_ref_wstar, rel_norm_dev_quadrature_ref_vs_c_sigma_d2_variant, sklearn_abs_logloss_diff, sklearn_max_relative_coef_diff

| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |
|---|---|---|---|---|---|---|---|
| rel_norm_dev_quadrature_ref_vs_c_times_wstar (primary) | soft_target_lr_quadrature_reference vs closed_form_c_eps_tau_sigma_prime | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |

**Table 12: R012, E13@v2 (ablation) testing H1@v3, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]**

| metric | arm | n | mean | std | 95% CI |
|---|---|---|---|---|---|
| calibration_slope_vs_true_logit | eps0_tuned | 20 | 0.7152 | 0.0488 | [0.6924, 0.7381] |
| calibration_slope_vs_true_logit | ls_fixed_eps_0p05_l2fixed | 20 | 0.5035 | 0.0076 | [0.5000, 0.5071] |
| calibration_slope_vs_true_logit | ls_fixed_eps_0p05_l2retuned | 20 | 0.5195 | 0.0083 | [0.5156, 0.5234] |
| calibration_slope_vs_true_logit | ls_fixed_eps_0p1_l2fixed | 20 | 0.4269 | 0.0056 | [0.4243, 0.4296] |
| calibration_slope_vs_true_logit | ls_fixed_eps_0p1_l2retuned | 20 | 0.4386 | 0.0057 | [0.4360, 0.4413] |
| calibration_slope_vs_true_logit | ls_fixed_eps_0p2_l2fixed | 20 | 0.3279 | 0.0035 | [0.3262, 0.3295] |
| calibration_slope_vs_true_logit | ls_fixed_eps_0p2_l2retuned | 20 | 0.3340 | 0.0036 | [0.3323, 0.3357] |
| paired_delta_signed_error_eps0p2_minus_eps0p05 | eps0_tuned | 20 | -0.0668 | 0.0012 | [-0.0674, -0.0662] |
| paired_delta_signed_error_eps0p2_minus_eps0p05 | ls_fixed_eps_0p05_l2fixed | 20 | -0.0668 | 0.0012 | [-0.0674, -0.0662] |
| paired_delta_signed_error_eps0p2_minus_eps0p05 | ls_fixed_eps_0p05_l2retuned | 20 | -0.0668 | 0.0012 | [-0.0674, -0.0662] |
| paired_delta_signed_error_eps0p2_minus_eps0p05 | ls_fixed_eps_0p1_l2fixed | 20 | -0.0668 | 0.0012 | [-0.0674, -0.0662] |
| paired_delta_signed_error_eps0p2_minus_eps0p05 | ls_fixed_eps_0p1_l2retuned | 20 | -0.0668 | 0.0012 | [-0.0674, -0.0662] |
| paired_delta_signed_error_eps0p2_minus_eps0p05 | ls_fixed_eps_0p2_l2fixed | 20 | -0.0668 | 0.0012 | [-0.0674, -0.0662] |
| paired_delta_signed_error_eps0p2_minus_eps0p05 | ls_fixed_eps_0p2_l2retuned | 20 | -0.0668 | 0.0012 | [-0.0674, -0.0662] |
| paired_delta_signed_error_vs_eps0 | eps0_tuned | 20 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| paired_delta_signed_error_vs_eps0 | ls_fixed_eps_0p05_l2fixed | 20 | -0.0280 | 0.0010 | [-0.0284, -0.0275] |
| paired_delta_signed_error_vs_eps0 | ls_fixed_eps_0p05_l2retuned | 20 | -0.0239 | 0.0014 | [-0.0245, -0.0232] |
| paired_delta_signed_error_vs_eps0 | ls_fixed_eps_0p1_l2fixed | 20 | -0.0518 | 0.0015 | [-0.0524, -0.0511] |
| paired_delta_signed_error_vs_eps0 | ls_fixed_eps_0p1_l2retuned | 20 | -0.0480 | 0.0018 | [-0.0488, -0.0471] |
| paired_delta_signed_error_vs_eps0 | ls_fixed_eps_0p2_l2fixed | 20 | -0.0948 | 0.0021 | [-0.0957, -0.0938] |
| paired_delta_signed_error_vs_eps0 | ls_fixed_eps_0p2_l2retuned | 20 | -0.0921 | 0.0024 | [-0.0933, -0.0910] |
| selected_l2_retuned | eps0_tuned | 20 | -0.1917 | 0.1261 | [-0.2507, -0.1326] |
| selected_l2_retuned | ls_fixed_eps_0p05_l2fixed | 20 | -0.1917 | 0.1261 | [-0.2507, -0.1326] |
| selected_l2_retuned | ls_fixed_eps_0p05_l2retuned | 20 | -2.6096 | 0.0959 | [-2.6545, -2.5647] |
| selected_l2_retuned | ls_fixed_eps_0p1_l2fixed | 20 | -0.1917 | 0.1261 | [-0.2507, -0.1326] |
| selected_l2_retuned | ls_fixed_eps_0p1_l2retuned | 20 | -2.8500 | 0.0595 | [-2.8778, -2.8222] |
| selected_l2_retuned | ls_fixed_eps_0p2_l2fixed | 20 | -0.1917 | 0.1261 | [-0.2507, -0.1326] |
| selected_l2_retuned | ls_fixed_eps_0p2_l2retuned | 20 | -2.8833 | 0.0248 | [-2.8950, -2.8717] |
| spearman_signed_error_vs_eps | eps0_tuned | 20 | -1.0000 | 0.0000 | [-1.0000, -1.0000] |
| spearman_signed_error_vs_eps | ls_fixed_eps_0p05_l2fixed | 20 | -1.0000 | 0.0000 | [-1.0000, -1.0000] |
| spearman_signed_error_vs_eps | ls_fixed_eps_0p05_l2retuned | 20 | -1.0000 | 0.0000 | [-1.0000, -1.0000] |
| spearman_signed_error_vs_eps | ls_fixed_eps_0p1_l2fixed | 20 | -1.0000 | 0.0000 | [-1.0000, -1.0000] |
| spearman_signed_error_vs_eps | ls_fixed_eps_0p1_l2retuned | 20 | -1.0000 | 0.0000 | [-1.0000, -1.0000] |
| spearman_signed_error_vs_eps | ls_fixed_eps_0p2_l2fixed | 20 | -1.0000 | 0.0000 | [-1.0000, -1.0000] |
| spearman_signed_error_vs_eps | ls_fixed_eps_0p2_l2retuned | 20 | -1.0000 | 0.0000 | [-1.0000, -1.0000] |
| test_log_loss | eps0_tuned | 20 | 0.2488 | 0.0019 | [0.2479, 0.2497] |
| test_log_loss | ls_fixed_eps_0p05_l2fixed | 20 | 0.2603 | 0.0011 | [0.2598, 0.2609] |
| test_log_loss | ls_fixed_eps_0p05_l2retuned | 20 | 0.2591 | 0.0010 | [0.2586, 0.2596] |
| test_log_loss | ls_fixed_eps_0p1_l2fixed | 20 | 0.2759 | 0.0014 | [0.2752, 0.2765] |
| test_log_loss | ls_fixed_eps_0p1_l2retuned | 20 | 0.2741 | 0.0013 | [0.2735, 0.2747] |
| test_log_loss | ls_fixed_eps_0p2_l2fixed | 20 | 0.3107 | 0.0019 | [0.3098, 0.3116] |
| test_log_loss | ls_fixed_eps_0p2_l2retuned | 20 | 0.3089 | 0.0019 | [0.3080, 0.3098] |

108 further reported metrics are summarised in analysis/quantitative/R012.yaml: calibration_slope_vs_true_logit|LR::F1_n1000_d20, calibration_slope_vs_true_logit|LR::F1_n1000_d5, calibration_slope_vs_true_logit|LR::F1_n3000_d20, calibration_slope_vs_true_logit|LR::F1_n3000_d5, calibration_slope_vs_true_logit|LR::F1_n300_d5, calibration_slope_vs_true_logit|LR::F2_n1000_d20, calibration_slope_vs_true_logit|LR::F2_n1000_d5, calibration_slope_vs_true_logit|LR::F2_n3000_d20, calibration_slope_vs_true_logit|LR::F2_n3000_d5, calibration_slope_vs_true_logit|LR::F2_n300_d20, calibration_slope_vs_true_logit|LR::F2_n300_d5, calibration_slope_vs_true_logit|LR::F3_n1000_d20, calibration_slope_vs_true_logit|LR::F3_n1000_d5, calibration_slope_vs_true_logit|LR::F3_n3000_d20, calibration_slope_vs_true_logit|LR::F3_n3000_d5, calibration_slope_vs_true_logit|LR::F3_n300_d5, calibration_slope_vs_true_logit|MLP::F2_n1000_d5, calibration_slope_vs_true_logit|MLP::F2_n300_d5, paired_delta_signed_error_eps0p2_minus_eps0p05|LR::F1_n1000_d20, paired_delta_signed_error_eps0p2_minus_eps0p05|LR::F1_n1000_d5, paired_delta_signed_error_eps0p2_minus_eps0p05|LR::F1_n3000_d20, paired_delta_signed_error_eps0p2_minus_eps0p05|LR::F1_n3000_d5, paired_delta_signed_error_eps0p2_minus_eps0p05|LR::F1_n300_d5, paired_delta_signed_error_eps0p2_minus_eps0p05|LR::F2_n1000_d20, paired_delta_signed_error_eps0p2_minus_eps0p05|LR::F2_n1000_d5, paired_delta_signed_error_eps0p2_minus_eps0p05|LR::F2_n3000_d20, paired_delta_signed_error_eps0p2_minus_eps0p05|LR::F2_n3000_d5, paired_delta_signed_error_eps0p2_minus_eps0p05|LR::F2_n300_d20, paired_delta_signed_error_eps0p2_minus_eps0p05|LR::F2_n300_d5, paired_delta_signed_error_eps0p2_minus_eps0p05|LR::F3_n1000_d20, and 78 more

| metric | method vs baseline | diff | 95% CI | p (Holm) | g | significant | practical |
|---|---|---|---|---|---|---|---|
| paired_delta_signed_error_vs_eps0 (primary) | ls_fixed_eps_0p1_l2fixed vs eps0_tuned | -0.0518 | [-0.0524, -0.0511] | 0.0000 | -49.43 | yes | yes |
| paired_delta_signed_error_vs_eps0 (primary) | ls_fixed_eps_0p1_l2fixed vs ls_fixed_eps_0p05_l2fixed | -0.0238 | [-0.0246, -0.0230] | 0.0000 | -18.71 | yes | yes |
| paired_delta_signed_error_vs_eps0 (primary) | ls_fixed_eps_0p1_l2fixed vs ls_fixed_eps_0p2_l2fixed | 0.0430 | [0.0419, 0.0442] | 0.0000 | 23.46 | yes | yes |
| paired_delta_signed_error_eps0p2_minus_eps0p05 | ls_fixed_eps_0p1_l2fixed vs eps0_tuned | 0.0000 | [-0.0008, 0.0008] | 1.0000 | 0.00 | no | no |
| paired_delta_signed_error_eps0p2_minus_eps0p05 | ls_fixed_eps_0p1_l2fixed vs ls_fixed_eps_0p05_l2fixed | 0.0000 | [-0.0008, 0.0008] | 1.0000 | 0.00 | no | no |
| paired_delta_signed_error_eps0p2_minus_eps0p05 | ls_fixed_eps_0p1_l2fixed vs ls_fixed_eps_0p2_l2fixed | 0.0000 | [-0.0008, 0.0008] | 1.0000 | 0.00 | no | no |
| spearman_signed_error_vs_eps | ls_fixed_eps_0p1_l2fixed vs eps0_tuned | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| spearman_signed_error_vs_eps | ls_fixed_eps_0p1_l2fixed vs ls_fixed_eps_0p05_l2fixed | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| spearman_signed_error_vs_eps | ls_fixed_eps_0p1_l2fixed vs ls_fixed_eps_0p2_l2fixed | 0.0000 | [0.0000, 0.0000] | 1.0000 | n/a | no | no |
| calibration_slope_vs_true_logit | ls_fixed_eps_0p1_l2fixed vs eps0_tuned | -0.2883 | [-0.3112, -0.2654] | 0.0000 | -8.14 | yes | yes |
| calibration_slope_vs_true_logit | ls_fixed_eps_0p1_l2fixed vs ls_fixed_eps_0p05_l2fixed | -0.0766 | [-0.0809, -0.0723] | 0.0000 | -11.24 | yes | yes |
| calibration_slope_vs_true_logit | ls_fixed_eps_0p1_l2fixed vs ls_fixed_eps_0p2_l2fixed | 0.0990 | [0.0960, 0.1021] | 0.0000 | 20.78 | yes | yes |
| test_log_loss | ls_fixed_eps_0p1_l2fixed vs eps0_tuned | 0.0271 | [0.0260, 0.0281] | 0.0000 | 16.18 | yes | yes |
| test_log_loss | ls_fixed_eps_0p1_l2fixed vs ls_fixed_eps_0p05_l2fixed | 0.0155 | [0.0147, 0.0163] | 0.0000 | 12.07 | yes | yes |
| test_log_loss | ls_fixed_eps_0p1_l2fixed vs ls_fixed_eps_0p2_l2fixed | -0.0348 | [-0.0359, -0.0338] | 0.0000 | -20.80 | yes | yes |
| selected_l2_retuned | ls_fixed_eps_0p1_l2fixed vs eps0_tuned | 0.0000 | [-0.0807, 0.0807] | 1.0000 | 0.00 | no | no |
| selected_l2_retuned | ls_fixed_eps_0p1_l2fixed vs ls_fixed_eps_0p05_l2fixed | 0.0000 | [-0.0807, 0.0807] | 1.0000 | 0.00 | no | no |
| selected_l2_retuned | ls_fixed_eps_0p1_l2fixed vs ls_fixed_eps_0p2_l2fixed | 0.0000 | [-0.0807, 0.0807] | 1.0000 | 0.00 | no | no |

## 6. Discussion

What the data support. In the pilot settings, tuned L2 plus post-hoc scaling leaves no detectable room for LS in pooled log loss, and LS shrinkage lowers calibration slope in the direction predicted by p_LS=(1-eps)P(y|x)+eps/K. These agree with the expectation that LS acts like a shrinkage that post-hoc scaling can undo, as suggested by [P042, P061], but that interpretation is not tested directly here. The small R009 gains versus untuned eps=0 and the calibration-component difference suggest LS may substitute for part of tuning, but they vanish against eps0+TS. Practitioner implication (preliminary). If L2 is tuned by log loss and temperature scaling is available, the current evidence gives no reason to add LS. Where L2 is constrained, overconfidence appears by construction and any shrinkage helps. LS may then be a cheap substitute, though this was not compared against TS in those cells. What is not established. No H1 component is tested. Regimes where LS could win (small n with high d, near-separable data, label noise, wide MLPs without early stopping) were not run on the real grid. The R003 finite-sample behaviour, with negative point estimates at small n, is a lead worth testing on held-out classified cells, but no CI excludes 0. The inconsistent sign of the tuned-baseline error across R001, R002 and R003 needs a common definition before the overconfidence premise can be called false. Transfer to real data is untested. Digits and real-data MLP results were omitted.

## 7. Limitations

1. No run confirms or refutes any H1 component. All twelve runs are labelled inconclusive, and several were validation or gate runs. 2. Stand-in or reconstructed cells and generators were used (R001, R002, R006, R007, R009, R012) because the frozen E1/E11 grids were unavailable. Results are not comparable across runs, and generator specifications are not in the paper, so the runs cannot be reproduced from it. No frozen configuration hash exists. 3. The R001 and R012 MLPs have no early stopping, so no run tests H5 or the 'tuned L2 plus early stopping' baseline. MLP protocols differ across runs. 4. The cross-seed aggregators (Holm families, TOST, permutation tests, power, R002 gate statistics, R004 bootstrap, R005 seed pooling) were not completed. Holm p-values in R005 come from a single seed bootstrap over 5 folds, and fold dependence (Nadeau-Bengio) was not handled. 5. R002 has n=8 per cell in the tables despite 10 seeds, and the handling of failing checks is undocumented, so post-hoc exclusion cannot be ruled out. Firing proxies are undefined and conflict with signed-error summaries. 6. The gate null rests on absence of evidence, with no equivalence margin; pooling mixes heterogeneous cells. 7. The -0.01 log-loss margin and -1 pp accuracy margin are not justified and no power analysis supports them. R006's MDE (0.0294) conflicts with R001's precision and is unresolved. 8. De-smoothing results contradict across runs (R001 vs R005), and Table 1 ablation rows repeat identical values across arms (a reporting defect). 9. Missing baselines: Platt, vector scaling and isotonic arms; tuning budgets for the eps dimension are not equalised. 10. Coverage gaps: no tests of H6 (class imbalance, label noise), the leakage guard H7(a), the d=30 robustness or the misspecified generator; real data covers only LR on breast_cancer, iris and wine without intervals. 11. Calibration measurement is coarse (top-label signed error and 15-bin equal-mass ECE; no ICI [P012], classwise ECE or bin sensitivity). 12. The hypothesis registry was revised several times and runs ran on earlier versions; secondary hypotheses are exploratory. Digits is image pixels, not tabular. The novelty claim awaits a documented literature search, and the work is incremental in significance. 13. Some deviation details are truncated in the run record, so the deviation list in Methods may be incomplete.

## 8. Conclusion

In pilot runs on synthetic generators, tuned L2 with post-hoc scaling leaves no detectable room for label smoothing in pooled log loss (+0.0009, CI [-0.0006, 0.0025]). LS shrinks calibration slope as expected, and no finite-sample regime where LS lowers error against the truth was found. These results are inconclusive: the generators are reconstructions or self-defined, the regime-resolved grid was not run, and the aggregation was incomplete. The title question therefore remains open. The next steps are to freeze the specification and hash, run the real regime grid with held-out overconfidence classification, add early stopping, Platt, vector scaling and isotonic baselines, equalise tuning budgets, and release the aggregators.

## Claim-Evidence Map

| claim | kind | importance | evidence state | traceability |
|---|---|---|---|---|
| C48: Contribution: a pre-registered, regime-resolved, finite-sample comparison design that sets label smoothing against tuned L2/early stopping and post-hoc scali... | contribution | central | HYPOTHESIS | H1, H2, H7, R002, R006, R007, R008, R009, P001, P042, P018, P037; 0 experimental, 4 literature, untested: literature can motivate this claim but not establish it |
| C49: Finding (premise test, partial): the deep-network overconfidence premise is not supported for LR under the tuned eps=0 arm, since no strongly-overconfident (... | finding | central | HYPOTHESIS | H1, R002, P001, P042; 0 experimental, 2 literature, untested: literature can motivate this claim but not establish it |
| C50: Finding (primary comparison, inconclusive): the pooled difference in test log loss between best-tuned LS and best-tuned eps=0 (each with post-hoc correction)... | finding | central | HYPOTHESIS | H2, H1, R001, R009, P042, P061, P018; 0 experimental, 3 literature, untested: literature can motivate this claim but not establish it |
| C51: Finding (mechanism, descriptive): LS induces strong underconfidence relative to ground-truth probabilities in LR. At n=100,000 the calibration slope is about... | finding | supporting | HYPOTHESIS | H3, H1, R010, R012, R011, P001; 0 experimental, 1 literature, untested: literature can motivate this claim but not establish it |
| C52: Finding (finite-sample crossover, not demonstrated): in none of the 8 n x signal cells does the paired CI of D = MSE(ls_best_lr) - MSE(eps0_best_lr) lie enti... | finding | supporting | HYPOTHESIS | H3, R003, P062, P053; 0 experimental, 2 literature, untested: literature can motivate this claim but not establish it |
| C53: Finding (accuracy): in synthetic LR cells the paired accuracy differences are tiny (|mean| <= 0.14 pp) with intervals well inside the -1 pp margin, so part (... | finding | supporting | HYPOTHESIS | H4, R004; 0 experimental, 0 literature |
| C54: Finding (de-smoothing ablation): de-smoothing LS predictions changes log loss by -0.0106 relative to raw LS (CI [-0.0170, -0.0041], Holm p=0.0583). The sign... | finding | supporting | HYPOTHESIS | H2, R005, P042, P061; 0 experimental, 2 literature, untested: literature can motivate this claim but not establish it |
| C55: Validation: the numerical gate for the population-shrinkage solution (E6) looks favourable, with a relative norm deviation of 0.0000 (CI [0,0]) between the q... | finding | supporting | HYPOTHESIS | H3, H7, R011, R007; 0 experimental, 0 literature |
| C56: Background: LS is reported to improve calibration and generalisation in deep networks [P001], and modern neural networks are overconfident but are corrected... | background | supporting | SUPPORTED | H1, H2, P001, P042, P018, P037, P012, P061; 0 experimental, 6 literature |
| C57: Background: finite-sample behaviour and imbalance handling in tabular classifiers are known to be sensitive to sample size and to class-imbalance corrections... | background | supporting | SUPPORTED | H6, H3, P051, P062, P053, P052, P036, P022; 0 experimental, 6 literature |
| C58: Limitation: no run in the evidence base confirms or refutes any H1 component (P, A, B, C). All runs are labelled inconclusive. Several were validation or gat... | limitation | supporting | HYPOTHESIS | H1, H2, H3, H4, R002, R006, R007, R008, R009, R001, R004, R010, R012; 0 experimental, 0 literature |
| C59: Limitation: model and data coverage is incomplete. The MLP in R001 uses fixed 150 epochs with no early stopping, so no run tests H5 (LS substituting for earl... | limitation | supporting | HYPOTHESIS | H5, H6, H7, H3, R001, R003; 0 experimental, 0 literature |
| C60: Limitation: the hypothesis registry was revised repeatedly (H1 v1-v3, H2 v1, H3 v1-v2, H4 v1, H5 v1, H6 v1 and H7 v1 are superseded), with the current versio... | limitation | supporting | HYPOTHESIS | H1, H2, H3, H4, H5, H6, H7; 0 experimental, 0 literature |

## References

[P001] Rafael Rios Müller et al. (2019). When Does Label Smoothing Help?. *arXiv (Cornell University)*. https://doi.org/10.48550/arxiv.1906.02629
[P012] Peter C. Austin and Ewout Willem Steyerberg (2019). The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models. *Statistics in Medicine*. https://doi.org/10.1002/sim.8281
[P018] Telmo M. Silva Filho et al. (2023). Classifier calibration: a survey on how to assess and improve predicted class probabilities. *Machine Learning*. https://doi.org/10.1007/s10994-023-06336-7
[P022] K. Ruwani M. Fernando and Chris P. Tsokos (2021). Dynamically Weighted Balanced Loss: Class Imbalanced Learning and Confidence Calibration of Deep Neural Networks. *IEEE Transactions on Neural Networks and Learning Systems*. https://doi.org/10.1109/tnnls.2020.3047335
[P036] Yangdi Wang et al. (2024). Generalizing Few Data to Unseen Domains Flexibly Based on Label Smoothing Integrated with Distributionally Robust Optimization. *arXiv*. https://arxiv.org/abs/2408.05082v1
[P037] Muthu Chidambaram and Rong Ge (2024). Reassessing How to Compare and Improve the Calibration of Machine Learning Models. *arXiv*. https://arxiv.org/abs/2406.04068v2
[P042] Chuan Guo et al. (2017). On Calibration of Modern Neural Networks. *arXiv (Cornell University)*. https://doi.org/10.48550/arxiv.1706.04599
[P051] Ruben van den Goorbergh et al. (2022). The harm of class imbalance corrections for risk prediction models: illustration and simulation using logistic regression. *Journal of the American Medical Informatics Association*. https://doi.org/10.1093/jamia/ocac093
[P052] Yanan Hu et al. (2025). Beyond Comparing Machine Learning and Logistic Regression in Clinical Prediction Modelling: Shifting from Model Debate to Data Quality. *Journal of Medical Internet Research*. https://doi.org/10.2196/77721
[P053] Scott G. Silvey and Jinze Liu (2024). Sample Size Requirements for Popular Classification Algorithms in Tabular Clinical Data: Empirical Study. *Journal of Medical Internet Research*. https://doi.org/10.2196/60231
[P061] Ramakrishna Rao Pasupuleti (2026). Dual-Parameter Distribution Correction for Neural-Network Calibration: A Shrinkage-and-Scaling Framework With Multi-Seed Empirical Validation. *Research Square*. https://doi.org/10.21203/rs.3.rs-9511149/v1
[P062] Maarten van Smeden et al. (2018). Sample size for binary logistic prediction models: Beyond events per variable criteria. *Statistical Methods in Medical Research*. https://doi.org/10.1177/0962280218784726
