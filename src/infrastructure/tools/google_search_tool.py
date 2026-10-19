import os
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx


class _DuckDuckGoResultsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())

        if tag == "a" and classes.intersection({"result__a", "result-link"}):
            self._current = {
                "title": "",
                "url": self._target_url(attributes.get("href") or ""),
                "snippet": "",
            }
            self._capture = "title"
            self._text = []
        elif tag == "a" and classes.intersection(
            {"result__snippet", "result-snippet"}
        ):
            self._capture = "snippet"
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._capture:
            return

        text = " ".join(" ".join(self._text).split())
        if self._current is not None:
            self._current[self._capture] = text
            if self._capture == "title":
                self.results.append(self._current)
                self._current = None
        elif self.results and self._capture == "snippet":
            self.results[-1]["snippet"] = text

        self._capture = None
        self._text = []

    @staticmethod
    def _target_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.path == "/l/":
            target = parse_qs(parsed.query).get("uddg", [])
            if target:
                return unquote(target[0])
        return url


class GoogleSearchTool:
    """Recherche web avec SerpApi si configurée, sinon DuckDuckGo HTML."""

    name = "google_search"
    description = (
        "Recherche sur le web via Google/SerpApi si configuré, sinon DuckDuckGo. "
        "Retourne des titres, extraits et sources."
    )
    args_schema = {
        "query": "La requete de recherche (mots-cles).",
    }
    return_direct = False
    trigger_words: tuple[str, ...] = (
        "cherche",
        "google",
        "recherche",
        "search",
        "trouve",
    )

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")

    async def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "Erreur : Requete vide."

        if self.api_key:
            return await self._search_serpapi(query)
        return await self._search_duckduckgo(query)

    async def _search_serpapi(self, query: str) -> str:
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": 5,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://serpapi.com/search", params=params
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return f"Echec de la recherche Google : HTTP {exc.response.status_code}."
        except httpx.RequestError as exc:
            return f"Echec de la recherche Google : {exc}."

        data = response.json()
        results = []
        answer_box = data.get("answer_box", {})
        answer = (
            answer_box.get("answer")
            or answer_box.get("snippet")
            or answer_box.get("title")
        )
        if answer:
            results.append(f"REPONSE DIRECTE : {answer}")

        for index, item in enumerate(data.get("organic_results", [])[:5], start=1):
            results.append(self._format_result(
                index,
                item.get("title", "Sans titre"),
                item.get("snippet", ""),
                item.get("link", ""),
            ))

        if results:
            return "\n\n".join(results)
        return f"Aucun resultat trouve pour '{query}'."

    async def _search_duckduckgo(self, query: str) -> str:
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AgentPlatform/0.1)"},
            ) as client:
                response = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return (
                "Echec de la recherche web (DuckDuckGo) : "
                f"HTTP {exc.response.status_code}."
            )
        except httpx.RequestError as exc:
            return f"Echec de la recherche web (DuckDuckGo) : {exc}."

        if response.status_code == 202:
            return "Echec de la recherche web : DuckDuckGo a demande une verification."

        parser = _DuckDuckGoResultsParser()
        parser.feed(response.text)
        results = [
            self._format_result(
                index,
                item["title"],
                item["snippet"],
                item["url"],
            )
            for index, item in enumerate(parser.results[:5], start=1)
            if item["title"]
        ]
        if not results:
            return (
                "Aucun resultat exploitable trouve via DuckDuckGo "
                f"pour '{query}'."
            )
        return "Recherche web (DuckDuckGo, sans SerpApi) :\n\n" + "\n\n".join(results)

    @staticmethod
    def _format_result(index: int, title: str, snippet: str, url: str) -> str:
        return (
            f"[{index}] {title}\n"
            f"Extrait : {snippet or 'Aucun extrait fourni.'}\n"
            f"Source : {url or 'URL indisponible.'}"
        )
