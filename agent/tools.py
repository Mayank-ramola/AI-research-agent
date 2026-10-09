"""Tools the agent can call: web search (Tavily) and page reading."""
import ipaddress
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .sources import SourceStore

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Returns a numbered list of results (id, title, url, snippet).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A focused search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read the text of one search result, by its source id, to check details.",
            "parameters": {
                "type": "object",
                "properties": {"source_id": {"type": "integer", "description": "Id from a web_search result"}},
                "required": ["source_id"],
            },
        },
    },
]


def web_search(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    """Call the Tavily search API and return [{title, url, content}]."""
    resp = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query, "max_results": max_results, "search_depth": "basic"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def _is_public_http_url(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    host = p.hostname
    if host == "localhost" or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local)
    except ValueError:
        return True


def read_page(url: str, max_chars: int = 5000) -> str:
    """Download a page and return its main text (scripts, menus and footers removed)."""
    if not _is_public_http_url(url):
        raise ValueError("Only public http(s) URLs can be read.")
    resp = requests.get(url, headers=HEADERS, timeout=12)
    resp.raise_for_status()
    if "html" not in resp.headers.get("content-type", ""):
        raise ValueError("The page is not HTML.")
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    return text[:max_chars]


class Toolbox:
    """Runs tool calls requested by the model and records every source it sees."""

    def __init__(self, settings, store: SourceStore, search_fn=web_search, read_fn=read_page):
        self.settings = settings
        self.store = store
        self.search_fn = search_fn
        self.read_fn = read_fn

    def call(self, name: str, args: dict) -> str:
        try:
            if name == "web_search":
                return self._search(str(args.get("query", "")).strip())
            if name == "read_page":
                return self._read(int(args.get("source_id", 0)))
            return f"Unknown tool: {name}"
        except Exception as exc:  # tool errors go back to the model so it can adapt
            return f"Tool error: {exc}"

    def _search(self, query: str) -> str:
        if not query:
            return "Tool error: empty query"
        results = self.search_fn(query, self.settings.tavily_api_key, self.settings.search_results)
        lines = []
        for r in results:
            s = self.store.add(r.get("title", ""), r.get("url", ""), (r.get("content") or "")[:500])
            lines.append(f"[{s.id}] {s.title} | {s.url}\n    {s.snippet}")
        return "\n".join(lines) or "No results."

    def _read(self, source_id: int) -> str:
        source = self.store.get(source_id)
        if not source:
            return f"Tool error: no source with id {source_id}. Use an id from web_search."
        if not source.content:
            source.content = self.read_fn(source.url, self.settings.max_page_chars)
        return f"[{source.id}] {source.title}\n{source.content}"
