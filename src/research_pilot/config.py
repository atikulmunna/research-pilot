from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Default route: anthropic, openrouter or mock. Any tier left unconfigured falls back to it,
    # except on anthropic, where lite defaults to claude-haiku-4-5 and standard to claude-sonnet-5-5.
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-5-5"

    # Lite tier: low-difficulty work (query generation, extraction, summaries, rewriting).
    llm_lite_provider: str = ""
    llm_lite_model: str = ""
    llm_lite_reasoning_effort: str = "auto"
    llm_lite_temperature: str = ""
    llm_lite_max_tokens: int = 0

    # Standard tier: high and medium-high work (gaps, interpretation, planning, drafting).
    llm_standard_provider: str = ""
    llm_standard_model: str = ""
    llm_standard_reasoning_effort: str = "auto"
    llm_standard_temperature: str = ""
    llm_standard_max_tokens: int = 0

    # Strong tier: very-high-difficulty reasoning (novelty, critique, hypotheses, design, review).
    llm_strong_provider: str = ""
    llm_strong_model: str = ""
    llm_strong_reasoning_effort: str = "auto"
    llm_strong_temperature: str = ""
    llm_strong_max_tokens: int = 0

    # Coding tier: experiment implementation and debugging. Falls back to the standard tier.
    llm_coding_provider: str = ""
    llm_coding_model: str = ""
    llm_coding_reasoning_effort: str = "auto"
    llm_coding_temperature: str = ""
    llm_coding_max_tokens: int = 0

    # Tasks at or above these difficulties go to the standard and strong tiers.
    llm_standard_min_difficulty: str = "medium_high"
    llm_strong_min_difficulty: str = "very_high"
    # Retry a lite or standard task once on the next tier up when its output cannot be parsed or validated.
    llm_escalate_on_failure: bool = True
    # Per-task tier overrides, e.g. "paper.drafting=lite,literature.clustering=strong".
    llm_task_overrides: str = ""
    # Optional price table (USD per million tokens) used when a provider does not report cost,
    # e.g. {"openai/gpt-5": [1.25, 10.0]}.
    llm_pricing: str = ""

    llm_route_fallback_enabled: bool = True
    llm_fallback_provider: str = ""
    llm_fallback_model: str = ""
    llm_retry_max_attempts: int = 4
    llm_retry_base_delay_s: float = 1.0
    llm_retry_max_delay_s: float = 8.0
    llm_request_timeout_s: float = 600.0

    anthropic_api_key: str = ""
    openrouter_api_key: str = ""

    # Literature sources: any of openalex, semantic_scholar, arxiv, mock (comma separated).
    literature_providers: str = "openalex,arxiv"
    literature_results_per_query: int = 8
    literature_max_papers: int = 50
    literature_citation_seed_k: int = 3
    literature_citation_limit: int = 5
    literature_fulltext_top_k: int = 0
    literature_extraction_batch: int = 4
    openalex_api_key: str = ""
    openalex_mailto: str = ""
    semantic_scholar_api_key: str = ""

    # Optional web search used only by the Field Scout for recent developments.
    web_search_provider: str = "none"
    tavily_api_key: str = ""
    serpapi_api_key: str = ""

    # Orchestration limits.
    workspace_dir: str = "./projects"
    max_steps: int = 60
    max_literature_rounds: int = 2
    max_proposal_revisions: int = 1
    max_review_rounds: int = 2
    max_experiment_runs: int = 8
    max_runs_per_spec: int = 2
    max_total_tokens: int = 0
    max_cost_usd: float = 0.0
    max_seconds: float = 0.0
    default_seeds: int = 3
    significance_alpha: float = 0.05
    allow_synthetic_evidence: bool = False

    # Experiment execution: manual (default), subprocess, or simulated.
    experiment_executor: str = "manual"
    experiment_timeout_s: int = 900
    experiment_python: str = ""
    experiment_max_repairs: int = 1

    api_auth_token: str = ""
    api_rate_limit_per_minute: int = 0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
