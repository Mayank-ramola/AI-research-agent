from agent.agent import ResearchAgent
from agent.config import Settings
from agent.llm import Message, ToolCall


class FakeLLM:
    """Returns scripted answers in order, so the agent loop can be tested without any API."""

    def __init__(self, chats, jsons):
        self.chats = list(chats)
        self.jsons = list(jsons)
        self.chat_calls = 0

    def chat(self, messages, tools=None, **kwargs):
        self.chat_calls += 1
        return self.chats.pop(0)

    def json(self, messages, **kwargs):
        return self.jsons.pop(0)


def fake_search(query, api_key, max_results):
    return [
        {"title": "Solar basics", "url": "https://example.com/solar", "content": "Solar panels convert light."},
        {"title": "Solar basics", "url": "https://www.example.com/solar/?utm_source=x", "content": "Duplicate page."},
    ]


def fake_read(url, max_chars):
    return "Solar panels convert sunlight into electricity using photovoltaic cells."


def settings(**kw):
    return Settings(groq_api_key="x", tavily_api_key="y", **kw)


def test_full_run_cites_only_real_sources_and_dedupes():
    chats = [
        Message(tool_calls=[ToolCall("1", "web_search", {"query": "how solar works"})]),
        Message(tool_calls=[ToolCall("2", "read_page", {"source_id": 1})]),
        Message(content="- Panels use photovoltaic cells [1]."),
        # report with one valid and one made-up citation, plus a claim that gets flagged
        Message(content="# Solar\nSolar panels use photovoltaic cells to make electricity [1][8]. Unsupported extra claim [1]."),
        Message(content="# Solar\nSolar panels use photovoltaic cells to make electricity [1]."),  # revision
    ]
    jsons = [
        {"sub_questions": ["How does solar power work?"]},
        {"unsupported_claims": ["Unsupported extra claim"]},
    ]
    agent = ResearchAgent(FakeLLM(chats, jsons), settings(max_rounds=1), fake_search, fake_read)
    events = list(agent.run("solar power"))

    kinds = [e["type"] for e in events]
    assert kinds[:3] == ["start", "plan", "question"]
    assert kinds.count("tool") == 2
    report = next(e for e in events if e["type"] == "report")
    assert "[8]" not in report["markdown"]
    assert "## Sources" in report["markdown"]
    assert report["stats"]["sources_found"] == 1          # duplicate merged
    assert report["stats"]["duplicates_skipped"] == 1
    assert report["stats"]["unsupported_claims_fixed"] == 1
    assert report["stats"]["sources_cited"] == 1


def test_review_adds_follow_up_round():
    chats = [
        Message(content="- Round one note [1]."),   # q1 (no tools used)
        Message(content="- Round two note [1]."),   # follow-up question
        Message(content="# T\nBody sentence that is cited properly [1]."),
    ]
    jsons = [
        {"sub_questions": ["Q1"]},
        {"enough": False, "follow_up_questions": ["Follow-up?"]},
        {"unsupported_claims": []},
    ]
    agent = ResearchAgent(FakeLLM(chats, jsons), settings(max_rounds=2), fake_search, fake_read)
    agent.store.add("Pre-existing", "https://example.com/pre")  # replaced in run(); proves store resets
    events = list(agent.run("topic"))
    questions = [e["question"] for e in events if e["type"] == "question"]
    assert questions == ["Q1", "Follow-up?"]
    assert any(e["type"] == "review" and not e["enough"] for e in events)


def test_step_limit_stops_research_but_still_reports():
    chats = [
        Message(tool_calls=[ToolCall("1", "web_search", {"query": "a"})]),
        Message(tool_calls=[ToolCall("2", "web_search", {"query": "b"})]),  # never reached
        Message(content="# T\nA cited sentence that is long enough to be checked properly [1]."),
    ]
    jsons = [{"sub_questions": ["Q1", "Q2"]}, {"unsupported_claims": []}]
    # plan=1 step, llm call=2, search tool=3, next llm call=4 > limit of 3
    agent = ResearchAgent(FakeLLM(chats[:1] + chats[2:], jsons), settings(max_steps=3), fake_search, fake_read)
    events = list(agent.run("topic"))
    assert any(e["type"] == "limit" for e in events)
    assert events[-1]["type"] == "done"
    assert any(e["type"] == "report" for e in events)


def test_tool_budget_per_question_is_enforced():
    many = [Message(tool_calls=[ToolCall(str(i), "web_search", {"query": f"q{i}"})]) for i in range(6)]
    chats = many[:2] + [Message(content="- note [1].")] + [Message(content="# T\nLong cited sentence for the report body [1].")]
    jsons = [{"sub_questions": ["Q"]}, {"unsupported_claims": []}]
    agent = ResearchAgent(FakeLLM(chats, jsons), settings(max_tool_calls=2, max_rounds=1), fake_search, fake_read)
    events = list(agent.run("topic"))
    assert sum(1 for e in events if e["type"] == "tool") == 2


def test_tool_calling_failure_falls_back_to_plain_search():
    class BrokenToolLLM(FakeLLM):
        def chat(self, messages, tools=None, **kwargs):
            if tools:
                raise RuntimeError("tool call malformed")
            return super().chat(messages, tools, **kwargs)

    chats = [Message(content="- fallback note [1]."), Message(content="# T\nLong cited sentence for the report body [1].")]
    jsons = [{"sub_questions": ["Q"]}, {"unsupported_claims": []}]
    agent = ResearchAgent(BrokenToolLLM(chats, jsons), settings(max_rounds=1), fake_search, fake_read)
    events = list(agent.run("topic"))
    assert any(e["type"] == "fallback" for e in events)
    assert any(e["type"] == "report" and "Sources" in e["markdown"] for e in events)
