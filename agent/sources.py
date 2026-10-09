"""Source bookkeeping: every page the agent sees gets one stable id, duplicates are merged."""
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlparse

TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid", "ref"}


def normalize_url(url: str) -> str:
    """Lower-case host, drop www, fragments, tracking params and trailing slashes."""
    p = urlparse(url.strip())
    host = p.netloc.lower().removeprefix("www.")
    query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query) if k.lower() not in TRACKING))
    path = p.path.rstrip("/")
    return f"{host}{path}" + (f"?{query}" if query else "")


def _title_key(title: str, url: str) -> str:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return host + "|" + re.sub(r"\W+", " ", title.lower()).strip()


@dataclass
class Source:
    id: int
    title: str
    url: str
    snippet: str = ""
    content: str = ""


class SourceStore:
    def __init__(self):
        self._by_id: dict[int, Source] = {}
        self._by_url: dict[str, int] = {}
        self._by_title: dict[str, int] = {}
        self.duplicates_skipped = 0

    def add(self, title: str, url: str, snippet: str = "") -> Source:
        key = normalize_url(url)
        tkey = _title_key(title, url)
        existing = self._by_url.get(key) or self._by_title.get(tkey)
        if existing:
            self.duplicates_skipped += 1
            source = self._by_id[existing]
            if len(snippet) > len(source.snippet):
                source.snippet = snippet
            return source
        source = Source(id=len(self._by_id) + 1, title=title.strip() or url, url=url.strip(), snippet=snippet.strip())
        self._by_id[source.id] = source
        self._by_url[key] = source.id
        self._by_title[tkey] = source.id
        return source

    def get(self, source_id: int):
        return self._by_id.get(source_id)

    def all(self) -> list[Source]:
        return list(self._by_id.values())

    def ids(self) -> set[int]:
        return set(self._by_id)
