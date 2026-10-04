# Example: label smoothing and calibration of small classifiers

A complete, real Research Pilot project, published as the system produced it. Only this README was written by hand; the other files are copied unchanged from the project folder.

| File | What it is |
|---|---|
| [manuscript.md](manuscript.md) | The final manuscript (about 26,000 words, including result tables and the claim-evidence map) |
| [red_team_review_round3.yaml](red_team_review_round3.yaml) | The last simulated peer review |
| [reproducibility.md](reproducibility.md) | Every run with its seeds, code version, executor and all recorded deviations |

## The project

- **Question:** does label smoothing improve the calibration of logistic regression and small MLPs on tabular classification data without hurting accuracy?
- **Constraints:** CPU only; Python with numpy and scikit-learn; scikit-learn's bundled datasets or synthetic data, no downloads; each experiment under 10 minutes.
- **Seed paper:** Müller, Kornblith and Hinton, "When does label smoothing help?" (arXiv 1906.02629).
- **Run:** three sessions on 2026-10-03 and 2026-10-04, 78 orchestrator steps and 105 successful model calls. The first part ran on Anthropic models (Haiku 4.5, Sonnet 5.5, Opus 5.5); the rest on OpenRouter (GPT-6 Luna for lite work, Claude Sonnet 5.5 for standard work and coding, Claude Opus 5.5 for strong work).
- **Cost:** $13.67 recorded: $9.02 strong tier, $2.82 standard, $1.52 coding, $0.32 lite. About $0.30 to $0.50 more was billed but not recorded when the machine's project drive disconnected mid-call.

The swarm searched and extracted 59 papers, found 4 research gaps, revised the proposal once, wrote 4 hypotheses (7 versions in total) and designed 13 experiments, of which 7 ran.

## The experiments

Every run used the manual executor: the swarm wrote the script and a person reviewed and ran it. Five of the seven needed a reviewer's intervention first, each recorded as a deviation.

| Run | Experiment | Tests | Verdict | Main result | Reviewer intervention |
|---|---|---|---|---|---|
| R001 | E2, primary | H2 | inconclusive | Best-tuned label smoothing and best-tuned eps=0 did not differ in log loss: +0.0009, CI [-0.0006, 0.0025]. Both beat raw eps=0. | none |
| R002 | E1, gate | H1 | inconclusive | Tuned eps=0 logistic regression was not overconfident (-0.0027, CI [-0.0101, 0.0046]); a deliberately under-regularised arm was (+0.0249, CI [0.0194, 0.0303]). | A self-check failed on optimiser tolerance, not on an error; the solver was changed and the check passed unchanged. |
| R003 | E4, primary | H3 | inconclusive | No cell where label smoothing was better. At n=3000 with strong signal it was worse in all 20 seeds, as the population analysis predicts. | none |
| R004 | E5, primary | H4 | inconclusive | Accuracy non-inferior: -0.03 percentage points, one-sided bound -0.07 against a -1 point margin. The accuracy-calibration trade-off part could not be judged. | The label-smoothing arm could select eps=0, contradicting the spec's grid; restored to the spec. |
| R005 | E3, ablation | H2 | inconclusive | Removing the smoothing after training lowered log loss in all three seeds (pooled over ten cells), the opposite of R001, but about 21% of probabilities needed clipping, far above the 5% the protocol treats as reliable. | A mistyped dictionary key crashed every seed; fixed. |
| R006 | E8, diagnostic | H1 | inconclusive | Inner cross-validation chose C about half a decade below the oracle; three cells were underpowered even at 20 replicates. | The script looped over all 20 replicates itself, so it was run once rather than once per seed. |
| R007 | E9, diagnostic | H1 | inconclusive | No duplicate replicates or non-converged fits; the script left its main aggregate diagnostics uncomputed. | Two self-check test cases contradicted the hypothesis's own decision rules; corrected. |

## How it ended

The final red-team review recommends **reject**: 1 critical, 12 major and 4 minor issues. The critical one is that H1, the paper's central question (does label smoothing help where the tuned baseline is overconfident?), was never directly tested. Four of the eight completion checks pass: manuscript written, limitations documented, citations verified, results reproducible.

In the third session the planner designed primary tests of H1 (E10, then E12 after revising H1), but the OpenRouter credit ran out before any of them could run. The user stopped the project there, so the outcome is a **draft with open issues**. E12 is approved and recorded in the decision log as the next run.

## Errors found on review

The manuscript has known errors that a human reader caught; they are left in place so that the example shows real output.

- **The abstract misstates R003.** It says the mechanism test "did not show LS-induced probability error in any of 8 cells". The run did show such error at n=3000 with strong signal (worse in all 20 seeds); what it did not show is any cell where label smoothing helped. The interpretation, the Results section and Table 3 are correct.
- **The abstract counts five completed runs.** Seven ran; the body reports the two diagnostic runs, R006 and R007, as validation runs.

Read generated manuscripts the way you would read a capable but unsupervised collaborator's draft: the numbers come from code, but the prose summarising them needs checking.
