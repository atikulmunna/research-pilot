"""Optional web search used by the Field Scout to spot recent developments.

Web results only inform the field map. They never enter the literature base or the
evidence graph, because they are not traceable scholarly sources.
"""

from typing import Dict, List

import requests


class WebSearch:
    def __init__(self, provider: str = "none", tavily_api_key: str = "", serpapi_api_key: str = ""):
        self.provider = (provider or "none").strip().lower()
        self.tavily_api_key = tavily_api_key
        self.serpapi_api_key = serpapi_api_key

    @property
    def enabled(self) -> bool:
        return self.provider not in {"", "none"}

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        if self.provider in {"", "none"}:
            return []
        if self.provider == "mock":
            return [
                {"title": f"Recent development {i + 1} in {query}", "url": f"mock://web/{i + 1}", "snippet": f"News about {query}."}
                for i in range(max_results)
            ]
        if self.provider == "tavily":
            return self._tavily(query, max_results)
        if self.provider == "serpapi":
            return self._serpapi(query, max_results)
        raise ValueError(f"Unsupported web search provider: {self.provider}")

    def _tavily(self, query: str, max_results: int) -> List[Dict[str, str]]:
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required for tavily provider.")
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": self.tavily_api_key, "query": query, "max_results": max_results, "search_depth": "basic"},
            timeout=30,
        )
        response.raise_for_status()
        return [
            {"title": row.get("title", ""), "url": row.get("url", ""), "snippet": row.get("content", "")}
            for row in response.json().get("results", [])
        ]

    def _serpapi(self, query: str, max_results: int) -> List[Dict[str, str]]:
        if not self.serpapi_api_key:
            raise ValueError("SERPAPI_API_KEY is required for serpapi provider.")
        response = requests.get(
            "https://serpapi.com/search.json",
            params={"q": query, "engine": "google", "api_key": self.serpapi_api_key, "num": max_results},
            timeout=30,
        )
        response.raise_for_status()
        return [
            {"title": row.get("title", ""), "url": row.get("link", ""), "snippet": row.get("snippet", "")}
            for row in response.json().get("organic_results", [])[:max_results]
        ]
