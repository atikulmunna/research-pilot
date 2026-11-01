# Example: label smoothing and calibration of small classifiers

A complete, real Research Pilot project, published as the system produced it. Only this README was written by hand; the other files are copied unchanged from the project folder.

| File | What it is |
|---|---|
| [manuscript.md](manuscript.md) | The final manuscript (about 22,000 words, including result tables and the claim-evidence map) |
| [red_team_review_round4.yaml](red_team_review_round4.yaml) | The last simulated peer review |
| [reproducibility.md](reproducibility.md) | Every run with its seeds, code version, executor and all recorded deviations |

## The project

- **Question:** does label smoothing improve the calibration of logistic regression and small MLPs on tabular classification data without hurting accuracy?
- **Constraints:** CPU only; Python with numpy and scikit-learn; scikit-learn's bundled datasets or synthetic data, no downloads; each experiment under 10 minutes.
- **Seed paper:** Müller, Kornblith and Hinton, "When does label smoothing help?" (arXiv 1906.02629).
- **Run:** several sessions on 2026-10-03, 2026-10-04 and 2026-11-01, 104 orchestrator steps and 148 model calls. The first and last parts ran on Anthropic models (Haiku 4.5 for lite work, Sonnet 5.5 for standard work and coding, Opus 5.5 for strong work); the middle part ran on OpenRouter (GPT-6 Luna for lite work, Claude Sonnet 5.5 for standard work and coding, Claude Opus 5.5 for strong work).
- **Cost:** $23.02 recorded: $11.04 strong tier, $7.67 standard, $2.68 coding, $1.63 lite. Two calls were billed but not recorded because the process was cut off mid-call (once when the project drive disconnected, once when the machine ran low on memory), about $0.55 to $0.90 in all.

The swarm searched and extracted 63 papers, found 4 research gaps, revised the proposal twice, wrote 7 hypotheses (17 versions in total) and designed 23 experiments, of which 12 ran.

## The experiments

Every run used the manual executor: the swarm wrote the script and a person reviewed and ran it. Eight of the twelve needed a reviewer's intervention first, each recorded as a deviation.

| Run | Experiment | Tests | Verdict | Main result | Reviewer intervention |
|---|---|---|---|---|---|
| R001 | E2, primary | H2 | inconclusive | Best-tuned label smoothing and best-tuned eps=0 did not differ in log loss: +0.0009, CI [-0.0006, 0.0025]. Both beat raw eps=0. | none |
| R002 | E1, gate | H1 | inconclusive | Tuned eps=0 logistic regression was not overconfident (-0.0027, CI [-0.0101, 0.0046]); a deliberately under-regularised arm was (+0.0249, CI [0.0194, 0.0303]). | A self-check failed on optimiser tolerance, not on an error; the solver was changed and the check passed unchanged. |
| R003 | E4, primary | H3 | inconclusive | No cell where label smoothing was better. At n=3000 with strong signal it was worse in all 20 seeds, as the population analysis predicts. | none |
| R004 | E5, primary | H4 | inconclusive | Accuracy non-inferior: -0.03 percentage points, one-sided bound -0.07 against a -1 point margin. The accuracy-calibration trade-off part could not be judged. | The label-smoothing arm could select eps=0, contradicting the spec's grid; restored to the spec. |
| R005 | E3, ablation | H2 | inconclusive | Removing the smoothing after training lowered log loss in all three seeds (pooled over ten cells), the opposite of R001, but about 21% of probabilities needed clipping, far above the 5% the protocol treats as reliable. | A mistyped dictionary key crashed every seed; fixed. |
| R006 | E8, diagnostic | H1 | inconclusive | Inner cross-validation chose C about half a decade below the oracle; three cells were underpowered even at 20 replicates. | The script looped over all 20 replicates itself, so it was run once rather than once per seed. |
| R007 | E9, diagnostic | H1 | inconclusive | No duplicate replicates or non-converged fits; the script left its main aggregate diagnostics uncomputed. | Two self-check test cases contradicted the hypothesis's own decision rules; corrected. |
| R008 | E11, gate | H1 | inconclusive | No tuned eps=0 logistic regression cell was measurably overconfident; three tuned MLP cells were (95% lower bounds above +0.01). Power estimates came out undefined, so no cell could be called powered. | none |
| R009 | E12, primary | H1 | inconclusive | Nested-tuned label smoothing against tuned eps=0: -0.0008 log loss, CI [-0.0016, -0.0001], Holm p=0.49; against eps=0 with temperature scaling: -0.0002, CI [-0.0015, 0.0010]. The engineer could not see E11's cells, so it rebuilt a 30-cell grid and the pilot-conditioned tests were not possible. | none |
| R010 | E7, primary | H3 | inconclusive | At n=100,000 the calibration slope fell from about 1.0 at eps=0 to 0.747 at eps=0.1 and 0.596 at eps=0.2: smoothing makes logistic regression underconfident, as the population solution predicts. No finite-sample cell established a log-loss advantage of at least the 0.001 margin. | The reference solution came from one fixed Monte Carlo sample, off by up to 0.37 replicate SDs; replaced with exact quadrature. |
| R011 | E6, gate | H3 | inconclusive | The quadrature solution matched the closed-form shrinkage to within 1e-4, but at the largest signal strength two precision checks missed their tolerances (1e-6 against 1e-8, 6e-8 against 1e-10). | numpy returned NaN for a third of the 400-node quadrature weights, and a zero standard error at eps=0 made a z-score 0/0; fixed with 300 nodes and a floored SE. |
| R012 | E13, ablation | H1 | inconclusive | With L2 held at eps=0's choice, smoothing lowered the signed top-label error against the true probabilities by 0.052 at eps=0.1 and 0.095 at eps=0.2 (tight CIs, 18 LR and 2 MLP cells), and raised log loss. Retuning L2 chose penalties about 2.7 orders of magnitude weaker on average but absorbed little of the shift (0.048 and 0.092). Not a formal confirmation, because the calibrated cells were picked using the same replicates. | The script looked for fields that R008 and R009 never wrote and stopped at load. It was rewritten to import E12's own code, which reproduced E12's baseline exactly on every seed. |

## How it ended

The final red-team review recommends **reject**: 1 critical, 15 major and 4 minor issues, with 1 earlier issue resolved. The critical one is unchanged from earlier rounds: H1, the paper's central question (does label smoothing help where the tuned baseline is overconfident?), was never tested on the regime grid the hypothesis defines. Four of the eight completion checks pass: manuscript written, limitations documented, citations verified, results reproducible.

With the 12-run limit reached, the planner rescoped the proposal to a protocol paper with pilot results. The manuscript now says so in its title and abstract and labels every hypothesis as untested. The budget's endgame reserve then forced the final manuscript and review. The outcome is a **draft with open issues**.

## Errors found on review

The manuscript and review have known errors that a human reader caught; they are left in place so that the example shows real output.

- **Limitation 3 says the R012 MLPs have no early stopping.** R012 reuses E12's MLP, which stops early on hard-label validation log loss (patience 15). R001's MLP does lack it.
- **Review issue RT4-5 compares two different quantities.** It calls the R010 and R012 results inconsistent because the slope drop at eps=0.1 (about 0.25) is about five times the R012 drop (0.052). The R012 figure is a signed error in probability units, not a slope, so the two numbers are not comparable. The issue's other point, that the manuscript does not say which cells each table covers, stands.

Read generated manuscripts the way you would read a capable but unsupervised collaborator's draft: the numbers come from code, but the prose summarising them needs checking.
