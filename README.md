# Research Pilot

Research Pilot takes a research idea to a tested, evidence-mapped manuscript. Twelve specialised agents work on one shared research state (literature base, evidence graph, hypothesis and experiment registries, decision log, adaptive roadmap) under a control-plane orchestrator. The loop does not stop at the first draft: results feed back into planning, and a red-team reviewer's objections send the work back for more literature, new experiments, revised hypotheses or a pivot.

Model calls are **routed by task difficulty, not by agent**. Searching, extraction and summaries run on a lite model at low reasoning effort; gap analysis, interpretation, planning, drafting and experiment code run on a standard model; only the hardest judgments (novelty, critique, hypothesis and experiment design, peer review) go to a strong model at high effort; deduplication, citation graphs, statistics and citation formatting are plain code.

![Research Pilot architecture: twelve agents, the orchestrator, the shared research state, the evidence graph and the experiment registry](resources/architecture.png)

To see what a real project produces, look at the [label-smoothing example](examples/label-smoothing/README.md): its manuscript, final review and reproducibility record, published as generated.

This README is the user manual. If you are new, read [What you need](#what-you-need), [Install and configure](#install-and-configure) and [Your first project](#your-first-project), then keep [Running experiments](#running-experiments) open while your project runs.

## Contents

- [What you need](#what-you-need)
- [Install and configure](#install-and-configure)
- [Try it offline first](#try-it-offline-first)
- [Your first project](#your-first-project)
- [Running experiments](#running-experiments)
- [Budgets and costs](#budgets-and-costs)
- [Reading the results](#reading-the-results)
- [Continuing, pausing and recovering](#continuing-pausing-and-recovering)
- [How a project runs](#how-a-project-runs)
- [Routing by difficulty](#routing-by-difficulty)
- [What the code guarantees](#what-the-code-guarantees)
- [Command reference](#command-reference)
- [Dashboard and API](#dashboard-and-api)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Limitations](#limitations)
- [Previous version](#previous-version)

## What you need

- **Python 3.10 or newer.**
- **A model provider key**: an [Anthropic](https://console.anthropic.com) key, an [OpenRouter](https://openrouter.ai) key, or both. Tiers can mix providers.
- **A budget.** Real projects cost dollars, not cents; see [Budgets and costs](#budgets-and-costs). Every run is capped by `MAX_COST_USD`.
- **Time to review experiment code.** By default the swarm writes each experiment as a Python script and pauses so that you can review and run it yourself. Plan to spend some minutes per experiment, plus the compute the experiments need on your machine.
- Optional: an OpenAlex contact email (faster, politer literature search), a Semantic Scholar key, and a Tavily or SerpAPI key for web search.

## Install and configure

```bash
git clone https://github.com/atikulmunna/research-pilot.git
cd research-pilot
python -m venv .venv
. .venv/Scripts/activate          # Windows; use .venv/bin/activate elsewhere
pip install -e .[dev]
cp .env.example .env
```

Then edit `.env`. Settings can also come from environment variables, which override `.env`. The minimum for a real project:

```bash
ANTHROPIC_API_KEY=sk-ant-...      # or OPENROUTER_API_KEY=... with OpenRouter models (below)
MAX_COST_USD=7.00                 # hard cap for this project, in dollars, counted across resumes
OPENALEX_MAILTO=you@example.org   # optional, recommended
```

With only an Anthropic key, the defaults use Claude Haiku 4.5 for lite work, Claude Sonnet 5.5 for standard and coding work, and Claude Opus 5.5 for very-high-difficulty work. To use OpenRouter instead, or to mix providers, set the per-tier provider and model. This is the recommended mix, chosen on the Artificial Analysis Intelligence Index and price (October 2026):

| Tier | Provider and model | Index | Price per MTok (in / out) |
|---|---|---|---|
| Lite | OpenRouter, `openai/gpt-6-luna` | 38 | $0.10 / $0.50 |
| Standard and coding | OpenRouter, `anthropic/claude-sonnet-5.5` | 56 | $2 / $10 |
| Strong | Anthropic, `claude-opus-5-5` at medium effort | 58 | $4 / $20 |

```bash
LLM_LITE_PROVIDER=openrouter
LLM_LITE_MODEL=openai/gpt-6-luna
LLM_STANDARD_PROVIDER=openrouter
LLM_STANDARD_MODEL=anthropic/claude-sonnet-5.5
LLM_STRONG_REASONING_EFFORT=medium
LLM_FALLBACK_PROVIDER=openrouter               # if the Anthropic account runs out of credit,
LLM_FALLBACK_MODEL=anthropic/claude-opus-5.5   # strong work continues on OpenRouter
```

Check what you configured before spending anything:

```bash
research-pilot routing        # every task kind with its difficulty, tier, provider, model and effort
```

## Try it offline first

The mock provider returns deterministic, clearly marked placeholder output, the literature is synthetic and experiments are simulated. A full project runs in about a minute and costs nothing:

```bash
LLM_PROVIDER=mock LITERATURE_PROVIDERS=mock EXPERIMENT_EXECUTOR=simulated \
  research-pilot new "retrieval augmented generation under distribution shift" --run
research-pilot status latest
research-pilot serve          # then open http://127.0.0.1:8000/dashboard
```

On Windows PowerShell, set the variables first (`$env:LLM_PROVIDER="mock"` and so on) or put them in `.env`. Mock output proves the control flow only; nothing it produces is evidence.

## Your first project

### 1. Describe the idea

A project starts from a topic. Everything else is optional but improves the result:

| Input | Flag | Advice |
|---|---|---|
| Topic | positional | A short phrase; it also names the project folder. |
| Research question | `--question` | One answerable question. The swarm tests hypotheses against it. |
| Proposal | `--proposal TEXT` or `--proposal @file.md` | Your plan, assumptions and what you expect. It is critiqued, revised or killed, so be concrete. |
| Seed papers | `--seed-paper ID` (repeatable) | DOIs, arXiv ids or titles of papers you already know matter. |
| Constraints | `--constraint TEXT` (repeatable) | Compute, data, time or method limits, such as "CPU only" or "public datasets only". Experiments are designed to fit them. |
| Title | `--title` | A short title for the manuscript. |

### 2. Create and start the project

```bash
research-pilot new "Label smoothing and calibration of small classifiers" \
  --question "Does label smoothing improve calibration of logistic regression and small MLPs on tabular data?" \
  --proposal @proposal.md \
  --seed-paper 1906.02629 \
  --constraint "CPU only" \
  --run
```

Without `--run` the project is only created; start it later with `research-pilot run <project>`. Commands take a project id, a unique prefix of it, or `latest`.

### 3. Watch it work

The terminal shows each agent's task and its findings as they happen. For a fuller picture, run `research-pilot serve` in another terminal and open `http://127.0.0.1:8000/dashboard`: it shows the active agent, spend by tier, the roadmap, hypotheses and experiments, claims and evidence, reviews and the manuscript. `research-pilot status <project>` prints a summary at any time.

### 4. Run the experiments when it pauses

When the swarm has an approved experiment, it writes `run.py` and stops with status `awaiting_experiments`. Follow [Running experiments](#running-experiments): review the script, run it, ingest the results and resume with `research-pilot run <project>`. A project usually pauses several times.

### 5. Read the outcome

When it stops, the terminal prints the outcome, the cost by tier and the path of the manuscript. [Reading the results](#reading-the-results) explains the outcome, the completion checks and where every artifact is.

## Running experiments

### The script contract

For each approved experiment, the Experiment Engineer writes a self-contained `projects/<project>/experiments/runs/R###/run.py` and a `README.md` with the exact commands for that run. Every script follows one contract:

```bash
python run.py --seed N --output-dir DIR
```

- It prints one line per arm: `RESULT_JSON: {"arm": "...", "seed": N, "metrics": {"metric_name": 0.123, ...}}`. Only these lines are ingested.
- **One invocation is one independent replicate.** Data, splits and initialisation all come from the seed, and the swarm computes the statistics across seeds. A script must not loop over replicate seeds internally.
- All seeds share the output directory, so artifact file names include the seed.
- It exits with a non-zero status on failure and never prints made-up results.

### How runs are executed

| `EXPERIMENT_EXECUTOR` | Behaviour |
|---|---|
| `manual` (default) | The project pauses as `awaiting_experiments`. You review, run and ingest; this is the safe default. |
| `subprocess` | Runs generated code locally, without asking, with a timeout per seed (`EXPERIMENT_TIMEOUT_S`). Failed runs are repaired by the coding tier up to `EXPERIMENT_MAX_REPAIRS` times. Not a sandbox: enable it only for code you are willing to run unseen. |
| `simulated` | Deterministic synthetic numbers for demos and tests, always flagged synthetic and never counted as evidence. |

### The manual workflow, step by step

```bash
research-pilot experiments list <project>                 # find the run waiting for you, e.g. R004
cd projects/<project>/experiments/runs/R004
# 1. Read README.md (objective, seeds, commands) and review run.py (checklist below).
# 2. Try one seed, timed, into a scratch file:
python run.py --seed 0 --output-dir artifacts > try.out
# 3. If it is correct, run every seed listed in README.md, appending to results.jsonl:
python run.py --seed 0 --output-dir artifacts >> results.jsonl
python run.py --seed 1 --output-dir artifacts >> results.jsonl
# ...
# 4. Ingest and resume:
cd -
research-pilot experiments ingest <project> R004 projects/<project>/experiments/runs/R004/results.jsonl
research-pilot run <project>
```

Seeds are independent, so you can run them in parallel. `ingest` accepts the script's stdout (`RESULT_JSON` lines), plain JSON lines, a JSON list of `{arm, seed, metrics}` records, or `{"records": [...]}`. It warns about arms, seeds or metrics that are missing and records each warning on the run. Ingested results are final, so check the output before you ingest it.

If you trust a script and want it run for you, `research-pilot experiments execute <project> R004` runs every seed with the configured Python (`EXPERIMENT_PYTHON`) and ingests the results.

### Review checklist

Generated code needs a human reviewer. On the label-smoothing project, five of seven scripts needed a fix before they produced valid results. Check at least:

1. **Safety.** No network access, no subprocesses, no deletion, and no writes outside `--output-dir`.
2. **Fidelity to the spec.** Arms, metric names, tuning grids and seeds match the approved experiment (`research-pilot show <project> experiments`). One script let the label-smoothing arm select eps=0, which turned it into the baseline it was compared with.
3. **One replicate per seed.** Another script looped over all twenty replicates inside every invocation; running it per seed would have counted the same data twenty times.
4. **Self-checks that stop the script.** Run one seed first. Gates written by the model can fail for the wrong reason: one failed on optimiser tolerance rather than on an error, one had a z-score threshold that 10% of seeds failed by chance, and one had hand-written test cases that contradicted the hypothesis's own decision rules.
5. **Plain bugs.** One script crashed on a mistyped dictionary key.
6. **Runtime.** Time one seed and multiply.

### Fixing a script

Keep the experimental protocol unchanged, and record every fix so that it reaches the manuscript:

```bash
cp run.py run.fixed.py            # edit the copy, never run.py itself
research-pilot experiments fix <project> R004 run.fixed.py \
  --reason "the LS arm could select eps=0; restricted it to the spec's eps grid"
```

`fix` keeps the original as `run.attempt0.py`, installs your copy as `run.py`, updates the code version and records the reason on the run as a deviation. The Paper Architect must disclose every deviation in the Methods section, and the reproducibility record lists them in full. If the protocol itself is wrong, do not patch it: let the run fail or report it, and the planner can revise the design.

## Budgets and costs

Every project has hard budgets, and all of them are **cumulative across resumes**:

| Budget | Setting | When it is reached |
|---|---|---|
| Cost | `MAX_COST_USD` | The project stops with outcome `budget_exhausted`. When $0.75 is left, the orchestrator writes and reviews the manuscript before anything else, so a stop always leaves a reviewed draft. |
| Steps | `MAX_STEPS`, or `run --max-steps N` | The project stops with outcome `step_limit`. A step is one orchestrator action; two steps before the limit the manuscript is written. |
| Experiment runs | `MAX_EXPERIMENT_RUNS`, `MAX_RUNS_PER_SPEC` | The planner stops proposing runs. |
| Review rounds | `MAX_REVIEW_ROUNDS` | Further red-team rounds stop; a final review still runs at the end. |
| Tokens, time | `MAX_TOTAL_TOKENS`, `MAX_SECONDS` | The project stops; at 70% of a token budget the manuscript is written first. |

The cost cap is checked between steps, so the last step can go slightly over it. A model call that is in flight when the process is killed may be billed by the provider without being recorded.

`MAX_COST_USD` is the project's cap, not your provider balance. Keep a few dollars more in your provider account than the project may still spend: OpenRouter rejects a request with HTTP 402 when the balance is too low, and it can count a call's full output allowance (`max_tokens`) when checking, which for a strong-tier call can exceed a dollar. If that happens mid-run, the step fails cleanly and nothing is charged; see [Troubleshooting](#troubleshooting).

What it costs in practice: on the label-smoothing project, which ran mostly on the recommended mix, a first reviewed draft with two completed experiments cost **$6.75**, and seven experiments with three review rounds cost **$10.76**. Most of the spend was experiment design and critique on the strong tier: a later session spent $2.91 revising one hypothesis and designing three experiments, and the provider credit ran out before any of them ran. The cheapest saving is to move those tasks to the standard tier:

```bash
LLM_TASK_OVERRIDES=critique.design=standard,hypothesis.design=standard
```

`research-pilot metrics <project>` shows calls, tokens and cost by tier and by task.

## Reading the results

### Status and outcome

`research-pilot status <project>` prints the status, the outcome, counts, cost by tier and the completion checks.

| Status | Meaning |
|---|---|
| `created` | Not started yet. |
| `running` | Working now. |
| `paused` | Stopped by the user, or reopened and waiting for `run`. |
| `awaiting_experiments` | Waiting for you to run and ingest an experiment. |
| `completed` | Stopped for good; see the outcome. |
| `failed` | A step raised an error; `research-pilot run <project>` retries it. |

| Outcome | Meaning |
|---|---|
| `publication_ready` | All eight completion checks pass. |
| `draft_with_open_issues` | A manuscript exists, but some completion checks fail. |
| `budget_exhausted`, `step_limit` | A budget stopped the project; the manuscript reflects the evidence so far. |
| `killed` | The planner abandoned the research idea, with its reasons in the decision log. |
| `no_manuscript` | The project stopped before a manuscript existed. |

The eight completion checks are: manuscript written, central claims supported, claims within their evidence, required experiments complete, reviewer objections addressed, limitations documented, citations verified, results reproducible.

### Where to look

| What | Command | File |
|---|---|---|
| Manuscript | `research-pilot export <project> --to html` | `paper/manuscript/manuscript.md` |
| Claims and their evidence state | `research-pilot show <project> claims` | `paper/claims.yaml` |
| Red-team reviews | `research-pilot show <project> reviews` | `reviews/red_team/review_##.yaml` |
| Hypotheses and verdicts | `research-pilot show <project> hypotheses` | `hypotheses/hypotheses.yaml`, `analysis/interpretations/` |
| Experiments and runs | `research-pilot experiments list <project>` | `experiments/` |
| Statistics per run | `research-pilot show <project> analysis` | `analysis/quantitative/` |
| Weak spots in the evidence | `research-pilot evidence <project>` | `evidence/evidence_graph.yaml` |
| Why the swarm did what it did | `research-pilot show <project> decisions` | `decisions/decision_log.yaml` |
| Guard violations | `research-pilot show <project> violations` | `logs/violations.jsonl` |
| Reproducibility record | | `paper/supplementary/reproducibility.md` |

Claim states come from the evidence graph, not from the model: `SUPPORTED` needs two supporting results, `PARTIALLY_SUPPORTED` one, a contribution with only literature behind it stays a `HYPOTHESIS`, and contradicting evidence gives `CONTRADICTED`. A guard violation means a model referred to a paper, hypothesis, experiment or run that does not exist; the reference was dropped and logged.

### Viewing and sharing results

Results stay on your machine: `projects/` is listed in `.gitignore`, so nothing a project produces is committed or pushed. To look at a project:

- **Dashboard:** `research-pilot serve`, then open `http://127.0.0.1:8000/dashboard` for the manuscript, claims, evidence, experiments, decisions and spend.
- **Manuscript as a web page:** `research-pilot export <project> --to html --output paper.html`, then open the file in a browser. The Markdown source is `projects/<project>/paper/manuscript/manuscript.md`.
- **Everything else:** the YAML and Markdown files in the project folder, described below.

The [label-smoothing example](examples/label-smoothing/README.md) shows what these files look like for a finished project. To share a project, send the exported HTML, or zip the project folder; anyone with Research Pilot installed can open a copied folder by placing it in their own `WORKSPACE_DIR`.

### The project folder

Each project is a resumable folder in `WORKSPACE_DIR` (default `./projects`), readable as YAML and Markdown:

```text
projects/<project-id>/
  project.yaml                 status, phase, budget use, completion checks
  field/                       field_map.yaml, terminology.yaml
  literature/                  papers/P###.yaml, clusters/, citations/, literature_map.yaml
  gaps/research_gaps.yaml
  proposal/                    proposal.yaml (every version), criticisms.yaml, novelty_analysis.yaml
  hypotheses/hypotheses.yaml   every version, with revision reasons
  experiments/                 specs/E#.v#.yaml, runs/R###/, results/, artifacts/
  analysis/                    quantitative/, interpretations/, errors/
  decisions/decision_log.yaml
  roadmap/roadmap.yaml         the orchestrator's action queue
  paper/                       claims.yaml, tables/, manuscript/manuscript.md, qa.yaml, supplementary/
  reviews/red_team/            review_##.yaml
  evidence/evidence_graph.yaml
  logs/                        activity.jsonl, llm_calls.jsonl, violations.jsonl
```

## Continuing, pausing and recovering

- **Resume** a paused, failed or awaiting project with `research-pilot run <project>`.
- **Continue a finished project**, for example after a budget stop, with `research-pilot run <project> --reopen`. It keeps all state, designs experiments for hypotheses that have none, then hands control to the planner. Raise the cumulative budgets first: `MAX_COST_USD` in `.env` or the environment, and `--max-steps` on the command line.
- **Pause** a project started through the API with `POST /api/v1/projects/{id}/cancel`; it stops after the current step with status `paused`.
- **Stop** a terminal run with Ctrl+C. State is written atomically, so nothing is corrupted; the next `research-pilot run <project>` re-queues the interrupted step and runs it again. The same applies after a crash or power loss.

## How a project runs

1. **Field map.** The Field Scout maps subfields, methods, datasets, metrics and open questions.
2. **Literature.** Literature Intelligence searches OpenAlex and arXiv (Semantic Scholar optional), follows citation chains, extracts every paper and clusters methods. The Gap & Novelty Analyst finds gaps and can ask for another round.
3. **Proposal validation.** Novelty is judged against the searched corpus, the Proposal Critic attacks the idea, and the planner revises, pivots or kills it.
4. **Hypotheses.** Explicit, falsifiable hypotheses, critiqued and refined before any experiment runs.
5. **Experiment design.** The smallest experiments that could falsify each hypothesis, critiqued, revised as new versions, then frozen.
6. **Research loop.** Run, quantify, interpret, re-plan. The planner ranks candidate actions by `scientific value x uncertainty reduction x feasibility / cost` and can search more literature, replicate, ablate, test robustness, investigate a failure, revise a hypothesis or the proposal, write, request review, stop or kill.
7. **Paper.** The Paper Architect maps every claim to evidence and drafts the manuscript. Unresolved red-team issues go back to the planner until the completion checks pass or the review budget runs out.

## Routing by difficulty

Every model call is a registered task kind with a difficulty and a capability. The router derives the tier from both and the reasoning effort from the difficulty; agents never pick models. `research-pilot routing` prints the table for your configuration.

| Work | Difficulty | Tier | Default model | Effort |
|---|---|---|---|---|
| Search queries, field map, paper extraction and summaries, clustering, rewriting | Low to low-medium | Lite | Claude Haiku 4.5 | low |
| Claim verification against full text | Medium | Lite | Claude Haiku 4.5 | medium |
| Paper drafting | Medium-high | Standard | Claude Sonnet 5.5 | medium |
| Experiment implementation and repair | Medium-high | Coding | Claude Sonnet 5.5 | medium |
| Gap discovery, interpretation, planning, claim-evidence mapping, proposal revision | High | Standard | Claude Sonnet 5.5 | high |
| Novelty, proposal and design critique, hypothesis and experiment design, review | Very high | Strong | Claude Opus 5.5 | high |
| Deduplication, citation graph, data processing, statistics, citation formatting | Low | Code | none | none |

- Tasks at or above `LLM_STANDARD_MIN_DIFFICULTY` (default `medium_high`) use the standard tier and tasks at or above `LLM_STRONG_MIN_DIFFICULTY` (default `very_high`) the strong tier. Coding tasks use the coding tier, which falls back to the standard model.
- Opus is kept for the very-high work where nothing checks the model afterwards. Standard-tier output is backed by code checks: interpretation verdicts are capped by the statistics, plans are validated before they run, and drafts go through citation and number checks.
- If a lite or standard answer cannot be parsed or validated, the call is retried once on the next tier up. Escalations are counted.
- `LLM_TASK_OVERRIDES` moves single tasks between tiers, for example `critique.design=standard,hypothesis.design=standard` to keep design critique and hypothesis design off the strong tier.
- If a call fails, it is retried on `LLM_FALLBACK_PROVIDER` and `LLM_FALLBACK_MODEL` when they are set.
- Tokens and cost are tracked per tier and per task (`research-pilot metrics`). Claude prices are built in, and OpenRouter reports its own costs, so cost budgets work with both.

On Claude Opus 5.5 and Sonnet 5.5 the effort maps to `output_config.effort` with adaptive thinking, and server-side refusal fallbacks (`fallbacks: "default"`) are enabled so a declined request is retried on Anthropic's recommended model. On Claude Haiku 4.5, low effort runs without thinking and medium or high effort uses a thinking budget.

## What the code guarantees

The safety rules are enforced in code rather than left to prompts.

- **No fabricated evidence.** Papers enter the literature base only from a literature provider with a traceable identifier. References to unknown papers, gaps, hypotheses, experiments or runs are stripped and logged as violations. Ids, provenance and evidence states are owned by code and never requested from a model.
- **Claims earn their state.** `SUPPORTED`, `PARTIALLY_SUPPORTED`, `HYPOTHESIS`, `SPECULATION`, `CONTRADICTED` and `UNKNOWN` are derived from the evidence graph. A contribution needs experimental support: without it the claim stays a hypothesis however much literature backs it; one result gives partial support and two give support. Statements tagged evidence-backed without valid evidence ids are downgraded to inference.
- **Statistics are code.** Welch's t-test, confidence intervals, Hedges' g and Holm correction are computed in Python and checked against SciPy. The interpreter cannot call a hypothesis supported unless every primary comparison is significant in the expected direction after correction. The analysis flags too few seeds, unstable seeds, best-seed dependence, implausibly large effects, perfect scores and multiple-comparison effects.
- **No silent protocol changes.** Experiment specs are versioned and frozen on approval; any change is a new version with a change log. Code repairs, reviewer fixes and declared deviations are recorded on the run and disclosed in the manuscript.
- **No cherry-picking.** Runs are append-only, failed attempts are kept, results are written once.
- **No retrospective rewriting.** A revised hypothesis is a new version with its reason, triggering evidence and decision; the original stays on record and its results do not carry over.
- **Synthetic is never evidence.** Mock papers and simulated or mock-generated results are flagged and ignored by the evidence graph.
- **Every decision is logged** with its reason, evidence and the alternatives considered.

## Command reference

`PROJECT` accepts a project id, a unique prefix, or `latest` (the default where it is optional).

| Command | What it does |
|---|---|
| `research-pilot new TOPIC [--question] [--proposal TEXT\|@file] [--seed-paper ID]... [--constraint TEXT]... [--title] [--run] [--max-steps N] [--quiet]` | Create a project, and with `--run` start it. |
| `research-pilot run [PROJECT] [--max-steps N] [--reopen] [--quiet]` | Run or resume. `--max-steps` sets the total step limit; `--reopen` continues a finished project. |
| `research-pilot projects [--json]` | List projects in the workspace. |
| `research-pilot status [PROJECT] [--json]` | Status, outcome, counts, cost by tier and completion checks. |
| `research-pilot show PROJECT SECTION [--json]` | Print one part of the research state. Sections: `field`, `papers`, `literature`, `gaps`, `proposal`, `novelty`, `critique`, `hypotheses`, `experiments`, `analysis`, `decisions`, `roadmap`, `claims`, `reviews`, `qa`, `manuscript`, `activity`, `violations`. |
| `research-pilot evidence [PROJECT] [--graph] [--json]` | Single-support claims, contradicted and untested claims, unsupported statements and uncertainty hotspots; `--graph` adds every node and edge. |
| `research-pilot metrics [PROJECT] [--json]` | Model calls, tokens and cost by tier and by task. |
| `research-pilot routing [--json]` | The resolved routing table for your configuration. |
| `research-pilot export [PROJECT] --to md\|html\|pdf [--output PATH]` | Export the manuscript. PDF export is plain text; prefer HTML for formatting. |
| `research-pilot experiments list [PROJECT] [--json]` | Runs with their status, seeds, code version and folder. |
| `research-pilot experiments ingest PROJECT RUN FILE` | Ingest results of a run you executed. |
| `research-pilot experiments fix PROJECT RUN SCRIPT --reason TEXT` | Install your fixed copy of a waiting run's script, keeping the original and logging the reason. |
| `research-pilot experiments execute PROJECT RUN [--timeout S]` | Run a reviewed script for every seed and ingest the results. |
| `research-pilot serve [--host] [--port]` | Start the API and the dashboard. |

## Dashboard and API

`research-pilot serve` starts the API and the dashboard at `http://127.0.0.1:8000/dashboard`. The dashboard draws the twelve agents with their feedback loops, highlights the active agent and each agent's tier mix, and shows spend by tier, completion checks, the roadmap, claims and evidence, hypotheses and experiments, literature, decisions, critique and review, the manuscript and the activity log. It reads the same `WORKSPACE_DIR` as the CLI, so projects started from the terminal appear there too.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/projects` | Create a project (`topic`, `question`, `proposal`, `seed_papers`, `constraints`, `title`, `autorun`, `max_steps`) |
| GET | `/api/v1/projects` | List projects |
| GET | `/api/v1/projects/{id}` | Overview: status, counts, agents, next actions, evidence summary |
| POST | `/api/v1/projects/{id}/run`, `/cancel` | Resume or pause |
| GET | `/api/v1/projects/{id}/state/{section}` | Any state section (same names as `show`) |
| GET | `/api/v1/projects/{id}/evidence?graph=true` | Evidence graph queries, optionally the whole graph |
| GET | `/api/v1/projects/{id}/activity`, `/metrics` | Task log; usage by tier and task |
| GET | `/api/v1/projects/{id}/manuscript?format=md` or `html` | Manuscript |
| POST | `/api/v1/projects/{id}/runs/{run_id}/results` | Ingest results of a manually executed run (`text`, or `records`) |
| GET | `/api/v1/routing` | Resolved routing table |

`API_AUTH_TOKEN` turns on an `X-API-Key` check for `/api/v1/` routes and `API_RATE_LIMIT_PER_MINUTE` a per-client limit. The server binds to `127.0.0.1` by default; set a token before exposing it on a network.

## Configuration reference

Settings come from the environment or `.env`; `.env.example` lists them all with comments.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL` | `anthropic`, `claude-opus-5-5` | Default route (`anthropic`, `openrouter`, `mock`), used by the strong tier and by any tier left empty |
| `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` | empty | Provider keys |
| `LLM_{LITE,STANDARD,STRONG,CODING}_PROVIDER`, `_MODEL` | empty | Per-tier model; on anthropic, lite defaults to `claude-haiku-4-5` and standard and coding to `claude-sonnet-5-5` |
| `LLM_{TIER}_REASONING_EFFORT` | `auto` | `auto` follows difficulty; `low`, `medium`, `high` fix it; `none` sends nothing |
| `LLM_{TIER}_MAX_TOKENS`, `_TEMPERATURE` | `0` (64,000), empty | Output cap and sampling temperature per tier (Anthropic ignores temperature) |
| `LLM_STANDARD_MIN_DIFFICULTY`, `LLM_STRONG_MIN_DIFFICULTY` | `medium_high`, `very_high` | Difficulty thresholds for the standard and strong tiers |
| `LLM_TASK_OVERRIDES` | empty | `task=tier` pairs, comma separated |
| `LLM_ESCALATE_ON_FAILURE` | `true` | Retry an unusable lite or standard answer one tier up |
| `LLM_FALLBACK_PROVIDER`, `LLM_FALLBACK_MODEL` | empty | Model to retry on when a call fails |
| `LLM_PRICING` | empty | Extra prices for models without built-in prices |
| `LLM_RETRY_MAX_ATTEMPTS`, `LLM_REQUEST_TIMEOUT_S` | `4`, `600` | Retries for rate limits and server errors; seconds per call |
| `LITERATURE_PROVIDERS` | `openalex,arxiv` | Also `semantic_scholar` (set `SEMANTIC_SCHOLAR_API_KEY`) or `mock` |
| `OPENALEX_MAILTO`, `OPENALEX_API_KEY` | empty | Contact email for OpenAlex's polite pool; optional key |
| `LITERATURE_MAX_PAPERS`, `LITERATURE_RESULTS_PER_QUERY` | `50`, `8` | Size of the literature base and of each search |
| `LITERATURE_FULLTEXT_TOP_K` | `0` | Read open-access PDFs of the top papers and verify their claims |
| `WEB_SEARCH_PROVIDER` | `none` | `tavily` or `serpapi` (with `TAVILY_API_KEY` or `SERPAPI_API_KEY`) for recent web snippets |
| `WORKSPACE_DIR` | `./projects` | Where projects live |
| `EXPERIMENT_EXECUTOR` | `manual` | `manual`, `subprocess` or `simulated` |
| `EXPERIMENT_TIMEOUT_S`, `EXPERIMENT_PYTHON`, `EXPERIMENT_MAX_REPAIRS` | `900`, current, `1` | Seconds per seed, interpreter, and automatic repairs (subprocess executor) |
| `DEFAULT_SEEDS`, `SIGNIFICANCE_ALPHA` | `3`, `0.05` | Seeds when a spec names none; significance level before Holm correction |
| `MAX_STEPS`, `MAX_EXPERIMENT_RUNS`, `MAX_RUNS_PER_SPEC`, `MAX_REVIEW_ROUNDS` | `60`, `8`, `2`, `2` | Orchestration budgets, counted per project across resumes |
| `MAX_LITERATURE_ROUNDS`, `MAX_PROPOSAL_REVISIONS` | `2`, `1` | Literature rounds and proposal revisions |
| `MAX_COST_USD`, `MAX_TOTAL_TOKENS`, `MAX_SECONDS` | `0` (off) | Hard budgets, counted per project across resumes |
| `ALLOW_SYNTHETIC_EVIDENCE` | `false` | Count synthetic results as evidence (for demos only) |
| `API_AUTH_TOKEN`, `API_RATE_LIMIT_PER_MINUTE` | empty, `0` | API key check and per-client rate limit |

## Troubleshooting

**Every Anthropic call fails with "Connection error".** An old `brotli` package (some Anaconda installs ship 1.0.9) breaks response decoding in the HTTP client. Install `brotli>=1.2.0` in the environment that runs Research Pilot.

**A step fails with "402 Client Error: Payment Required" from OpenRouter.** The account balance is too low for the request; OpenRouter can count the request's full output allowance (`max_tokens`) when checking. Nothing was charged. Add credit, or lower the output cap of the tier that failed (for example `LLM_STRONG_MAX_TOKENS=16000`), then `research-pilot run <project>` to retry the step. To end the project instead, keep the current manuscript; `research-pilot status <project>` shows what it covers.

**The project stopped with `budget_exhausted` or `step_limit`.** Raise `MAX_COST_USD` (it counts everything spent so far) or pass a larger `--max-steps` (the total, not the remainder), then `research-pilot run <project> --reopen`.

**The project is `failed`.** Run `research-pilot run <project>` to retry the failed step. The cause is in the terminal output and in `logs/activity.jsonl`; model errors are also in `logs/llm_calls.jsonl`.

**The run was interrupted** (Ctrl+C, crash, sleep, a disconnected drive). Run it again; the interrupted step is redone. A model call that was in flight may have been billed without being recorded, so check your provider's dashboard if the budget is tight.

**"R### is not awaiting execution"** from `fix` or `execute`. Only runs waiting for execution can be changed or executed; `research-pilot experiments list` shows each run's status. Ingest also accepts a failed run.

**Ingest warns about missing seeds or metrics.** A seed crashed or a metric name differs from the spec. The warnings are recorded on the run. Ingested results are final, so fix and rerun before ingesting when you can.

**A model refused or ran out of output tokens.** Refusals are retried on the fallback model; truncation raises an error naming the limit. Raise `LLM_{TIER}_MAX_TOKENS` for that tier, or set a fallback model.

**Literature warnings** ("could not resolve seed", rate limits). Provider failures become warnings and the run continues. Set `OPENALEX_MAILTO`, add `SEMANTIC_SCHOLAR_API_KEY`, or give seed papers as DOIs.

**Guard violations appear in the output.** A model cited a paper, hypothesis, experiment or run that does not exist (or, for evidence, something that is not a run or a paper). The reference was dropped; nothing invented reaches the state. `research-pilot show <project> violations` lists them.

**The dashboard shows no projects.** Start `serve` from the same folder, or with the same `WORKSPACE_DIR`, as the CLI.

## Development

```bash
pytest -q
ruff check src tests
```

The suite covers routing, the Anthropic and OpenRouter providers, parsing, repair, escalation and fallback, statistics against SciPy reference values, the evidence graph, the state safety rules, literature parsing, experiment execution and analysis, planning, export, the API and the CLI. Integration tests run the full swarm offline (including the publication-ready path, manual pause and resume, recovery from a failed step and the interpreter guard). CI runs a secret scan, ruff, the tests and an offline end-to-end project on Python 3.10 to 3.12. The lint rules are listed in `pyproject.toml` so that new ruff releases cannot change them silently.

## Limitations

- Manuscript quality depends on the strong model; the mock provider only proves the control flow.
- Generated experiment scripts need a human reviewer. Coverage checks confirm that a script reported every arm, seed and metric, but nothing checks automatically that it implements its spec.
- Semantic Scholar rate-limits anonymous use and arXiv can be slow; provider failures become warnings and the run continues.
- The `subprocess` executor is not a sandbox.
- PDF export is plain text; use the HTML export for formatted output.

## Previous version

v1 was a linear LangGraph web-search report generator: it decomposed a query into sub-questions, searched the web with Tavily or SerpAPI, synthesised sections and exported a cited report, with quality gates and a benchmark harness. v2 replaces that pipeline with the closed research loop above. The v1 documentation is archived in [resources/README_V1.md](resources/README_V1.md).
