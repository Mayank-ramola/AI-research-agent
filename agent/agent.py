"""The research agent: plan -> search -> read -> summarise -> review -> write -> verify.

`ResearchAgent.run()` is a generator that yields one event dict per step, so a UI can show progress live.
"""
import json
from typing import Iterator

from .config import Settings
from .sources import SourceStore
from .tools import TOOL_SCHEMAS, Toolbox, read_page, web_search
from .verify import find_uncited_sentences, renumber, strip_invalid

PLAN_SYSTEM = (
    "You are a research planner. Break the user's topic into {n} specific, searchable sub-questions "
    "that together cover it. Reply only with JSON: {{\"sub_questions\": [\"...\"]}}"
)

RESEARCH_SYSTEM = (
    "You are a careful research assistant answering one question. Use web_search to find sources, then "
    "read_page on the one or two most relevant results to check details. When you have enough, reply "
    "with 3 to 6 short factual bullet points. Every bullet must end with the source id(s) it came from, "
    "like [3] or [3][5]. Use only facts you saw in tool results, never your own memory. If sources "
    "disagree or nothing relevant was found, say so."
)

REVIEW_SYSTEM = (
    "You review research notes for a report on the given topic. Decide whether the notes are enough to "
    "write a factual report. If there are important gaps, give up to 2 follow-up questions. "
    "Reply only with JSON: {\"enough\": true or false, \"follow_up_questions\": [\"...\"]}"
)

WRITE_SYSTEM = (
    "Write a clear research report in Markdown: a '# ' title, a short summary paragraph, then '## ' sections. "
    "Use ONLY the notes provided. After every factual sentence add the supporting source id in square "
    "brackets, like [2], or [2][5] for two sources. Never invent ids that are not in the source list. "
    "If the notes do not answer something, say the evidence was not found. Do not add a Sources section."
)

FACTCHECK_SYSTEM = (
    "You are a strict fact-checker. Compare the report with the research notes. List claims in the report "
    "that the notes do not support or that cite the wrong source. "
    "Reply only with JSON: {\"unsupported_claims\": [\"claim text\"]}. Use an empty list if all claims are supported."
)

REVISE_SYSTEM = (
    "Revise the report: remove or soften every claim listed as unsupported, keeping everything else, the "
    "structure and the [n] citations. Reply with the full revised report only."
)


class StepLimitReached(Exception):
    """Raised when the agent has used its step budget."""


def _format_notes(notes: list[dict]) -> str:
    return "\n\n".join(f"Question: {n['question']}\n{n['notes']}" for n in notes if n["notes"].strip())


class ResearchAgent:
    def __init__(self, llm, settings: Settings, search_fn=web_search, read_fn=read_page):
        self.llm = llm
        self.settings = settings
        self.search_fn = search_fn
        self.read_fn = read_fn
        self.steps = 0
        self.store = SourceStore()
        self.toolbox = Toolbox(settings, self.store, search_fn, read_fn)

    def _tick(self) -> None:
        self.steps += 1
        if self.steps > self.settings.max_steps:
            raise StepLimitReached()

    # ------------------------------------------------------------------ run
    def run(self, topic: str) -> Iterator[dict]:
        self.steps = 0
        self.store = SourceStore()
        self.toolbox = Toolbox(self.settings, self.store, self.search_fn, self.read_fn)
        s = self.settings

        yield {"type": "start", "topic": topic}
        notes: list[dict] = []
        limit_hit = False

        try:
            questions = self._plan(topic)
        except StepLimitReached:
            questions, limit_hit = [topic], True
        yield {"type": "plan", "questions": questions}

        pending = questions
        for round_no in range(1, s.max_rounds + 1):
            if limit_hit:
                break
            for question in pending:
                try:
                    yield from self._research(question, notes)
                except StepLimitReached:
                    limit_hit = True
                    break
            if limit_hit or round_no == s.max_rounds:
                break
            try:
                review = self._review(topic, notes)
            except StepLimitReached:
                limit_hit = True
                break
            yield {"type": "review", "round": round_no, **review}
            if review["enough"] or not review["follow_up_questions"]:
                break
            pending = review["follow_up_questions"]

        if limit_hit:
            yield {"type": "limit", "steps": self.steps}

        yield from self._write_and_verify(topic, notes)
        yield {"type": "done", "steps": self.steps}

    # ----------------------------------------------------------------- steps
    def _plan(self, topic: str) -> list[str]:
        self._tick()
        data = self.llm.json(
            [
                {"role": "system", "content": PLAN_SYSTEM.format(n=self.settings.max_sub_questions)},
                {"role": "user", "content": f"Topic: {topic}"},
            ]
        )
        questions = [q.strip() for q in data.get("sub_questions", []) if isinstance(q, str) and q.strip()]
        return questions[: self.settings.max_sub_questions] or [topic]

    def _research(self, question: str, notes: list[dict]) -> Iterator[dict]:
        yield {"type": "question", "question": question}
        messages = [
            {"role": "system", "content": RESEARCH_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ]
        used = 0
        final = ""
        try:
            while True:
                self._tick()
                tools_allowed = used < self.settings.max_tool_calls
                msg = self.llm.chat(messages, tools=TOOL_SCHEMAS if tools_allowed else None)
                if not msg.tool_calls or not tools_allowed:
                    final = msg.content
                    break
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    if used >= self.settings.max_tool_calls:
                        result = "Tool budget used up. Write your findings now."
                    else:
                        self._tick()
                        result = self.toolbox.call(tc.name, tc.arguments)
                        used += 1
                    yield {
                        "type": "tool",
                        "name": tc.name,
                        "args": tc.arguments,
                        "preview": result[:300],
                        "sources_so_far": len(self.store.all()),
                    }
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        except StepLimitReached:
            raise
        except Exception as exc:  # tool calling failed: fall back to plain search + summary
            yield {"type": "fallback", "reason": str(exc)[:200]}
            final = self._fallback(question)

        notes.append({"question": question, "notes": final})
        yield {"type": "notes", "question": question, "notes": final}

    def _fallback(self, question: str) -> str:
        self._tick()
        results = self.toolbox.call("web_search", {"query": question})
        self._tick()
        msg = self.llm.chat(
            [
                {"role": "system", "content": RESEARCH_SYSTEM},
                {"role": "user", "content": f"Question: {question}\n\nSearch results:\n{results}"},
            ]
        )
        return msg.content

    def _review(self, topic: str, notes: list[dict]) -> dict:
        self._tick()
        data = self.llm.json(
            [
                {"role": "system", "content": REVIEW_SYSTEM},
                {"role": "user", "content": f"Topic: {topic}\n\nNotes:\n{_format_notes(notes)}"},
            ]
        )
        follow = [q.strip() for q in data.get("follow_up_questions", []) if isinstance(q, str) and q.strip()]
        return {"enough": bool(data.get("enough", True)), "follow_up_questions": follow[:2]}

    # ---------------------------------------------------------- write + verify
    def _write_and_verify(self, topic: str, notes: list[dict]) -> Iterator[dict]:
        sources = self.store.all()
        if not sources or not _format_notes(notes):
            yield {
                "type": "report",
                "markdown": f"# {topic}\n\nNo usable sources were found, so no report could be written.",
                "stats": {"sources_found": len(sources), "sources_cited": 0},
                "sources": [],
            }
            return

        source_list = "\n".join(f"[{src.id}] {src.title} | {src.url}" for src in sources)
        draft = self.llm.chat(
            [
                {"role": "system", "content": WRITE_SYSTEM},
                {"role": "user", "content": f"Topic: {topic}\n\nNotes:\n{_format_notes(notes)}\n\nSources:\n{source_list}"},
            ],
            max_tokens=2500,
        ).content
        yield {"type": "draft", "chars": len(draft)}

        report, invalid_removed = strip_invalid(draft, self.store.ids())

        unsupported: list[str] = []
        try:
            check = self.llm.json(
                [
                    {"role": "system", "content": FACTCHECK_SYSTEM},
                    {"role": "user", "content": f"Notes:\n{_format_notes(notes)}\n\nReport:\n{report}"},
                ]
            )
            unsupported = [c for c in check.get("unsupported_claims", []) if isinstance(c, str) and c.strip()]
        except Exception:
            pass  # a failed fact-check should not lose the report

        if unsupported:
            revised = self.llm.chat(
                [
                    {"role": "system", "content": REVISE_SYSTEM},
                    {
                        "role": "user",
                        "content": "Unsupported claims:\n- " + "\n- ".join(unsupported) + f"\n\nReport:\n{report}",
                    },
                ],
                max_tokens=2500,
            ).content
            report, extra = strip_invalid(revised, self.store.ids())
            invalid_removed += extra

        uncited = find_uncited_sentences(report)
        report, mapping = renumber(report)
        used = [self.store.get(old) for old in mapping]
        sources_md = "\n".join(f"{new}. [{src.title}]({src.url})" for new, src in enumerate(used, start=1))
        markdown = report.rstrip() + ("\n\n## Sources\n" + sources_md if used else "")

        stats = {
            "sources_found": len(sources),
            "sources_cited": len(used),
            "duplicates_skipped": self.store.duplicates_skipped,
            "invalid_citations_removed": invalid_removed,
            "unsupported_claims_fixed": len(unsupported),
            "uncited_sentences": len(uncited),
            "steps": self.steps,
        }
        yield {"type": "verify", "stats": stats, "uncited": uncited[:5]}
        yield {
            "type": "report",
            "markdown": markdown,
            "stats": stats,
            "sources": [{"id": i, "title": s.title, "url": s.url} for i, s in enumerate(used, start=1)],
        }
