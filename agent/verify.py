"""Citation checks that run on the finished report, without calling an LLM."""
import re

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _ids_in(match: re.Match) -> list[int]:
    return [int(n) for n in re.split(r"\s*,\s*", match.group(1))]


def extract_citation_ids(text: str) -> list[int]:
    """Citation ids in order of first appearance."""
    seen: list[int] = []
    for m in CITATION.finditer(text):
        for n in _ids_in(m):
            if n not in seen:
                seen.append(n)
    return seen


def strip_invalid(text: str, valid_ids: set[int]) -> tuple[str, int]:
    """Remove citations that point to ids that do not exist. Returns (text, removed_count)."""
    removed = 0

    def fix(m: re.Match) -> str:
        nonlocal removed
        ids = _ids_in(m)
        good = [n for n in ids if n in valid_ids]
        removed += len(ids) - len(good)
        return "".join(f"[{n}]" for n in good)

    return CITATION.sub(fix, text), removed


def renumber(text: str) -> tuple[str, dict[int, int]]:
    """Renumber citations 1..k in order of appearance. Returns (text, {old_id: new_id})."""
    mapping = {old: new for new, old in enumerate(extract_citation_ids(text), start=1)}
    return CITATION.sub(lambda m: "".join(f"[{mapping[n]}]" for n in _ids_in(m)), text), mapping


def find_uncited_sentences(text: str, min_len: int = 60) -> list[str]:
    """Long sentences in the body that carry no citation (headings and short lines are ignored)."""
    flagged = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if len(sentence) >= min_len and not CITATION.search(sentence):
                flagged.append(sentence)
    return flagged
