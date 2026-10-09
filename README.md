# AI Research Agent

An autonomous research agent. Give it a topic and it splits it into sub-questions, searches the web, reads pages, reviews its own notes, and writes a Markdown report where every claim cites a source. A Streamlit app shows each step live.

## How it works

```
topic
  -> plan        LLM splits the topic into sub-questions
  -> research    per question, the LLM calls tools in a loop:
                   web_search (Tavily)  ->  read_page (fetch + clean text)
                 then writes bullet notes that cite source ids
  -> review      LLM decides if the notes are enough, or asks follow-up questions (extra round)
  -> write       LLM writes the report using only the notes
  -> verify      citation checks, fact-check pass, revision, renumbering
```

**Tool calling.** The model decides when to call `web_search` and `read_page` using the Groq tool-calling API. If a tool call fails, the agent falls back to a plain search plus summary for that question.

**Step limit.** Every model call and tool call counts as a step. Per-question tool budget (`max_tool_calls`), number of rounds, and a global `max_steps` stop the agent from looping forever. When the limit is hit, it still writes a report from what it found.

**Fewer made-up claims.**
- Every page gets one id; duplicate URLs (www, tracking parameters, trailing slashes) and same-title pages are merged.
- `read_page` only accepts ids from earlier search results, so the model cannot make the server fetch arbitrary URLs, and private/local addresses are blocked.
- Citations to ids that do not exist are removed.
- A second LLM pass compares the report with the notes and lists unsupported claims; the report is then revised to remove or soften them.
- Citations are renumbered in order of appearance and only cited sources appear in the Sources list.
- Long sentences with no citation are counted and flagged in the UI.

## Project structure

```
api/index.py      FastAPI endpoint for Vercel (streams steps as SSE)
public/index.html Web UI for Vercel
streamlit_app.py  Streamlit UI for local use
cli.py            Run from the terminal
agent/
  agent.py        Agent loop (plan, research, review, write, verify)
  tools.py        web_search (Tavily), read_page, tool schemas, Toolbox
  sources.py      Source ids and de-duplication
  verify.py       Citation checks
  llm.py          Groq wrapper (retries, JSON parsing, tool-call parsing)
  config.py       Settings
tests/            pytest tests (no API keys needed; the LLM is faked)
```

## Run locally

You need Python 3.10+ and two free API keys: [Groq](https://console.groq.com) and [Tavily](https://tavily.com).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)

copy .env.example .env          # macOS/Linux: cp .env.example .env
# put your keys in .env

pip install -r requirements-dev.txt
streamlit run streamlit_app.py
```

Terminal version: `python cli.py "How do solid-state batteries compare with lithium-ion?"` (saves to `reports/`).

## Run the tests

```bash
pytest
```

The tests cover URL normalising and de-duplication, citation checking and renumbering, the full agent loop with a scripted fake LLM, the follow-up round, the step limit, the per-question tool budget, and the tool-calling fallback.

## Deploy on Vercel

The Vercel version is a static page (`public/index.html`) plus a FastAPI function (`api/index.py`) that streams the agent's steps live. The original Streamlit UI is kept as `streamlit_app.py` for local use, because Streamlit cannot run on Vercel.

1. Push this repo to GitHub.
2. On vercel.com choose **Add New, Project**, import the repo and leave the framework preset on **Other**. No build command is needed.
3. Under **Environment Variables** add `GROQ_API_KEY` and `TAVILY_API_KEY` (optionally `MODEL`).
4. Recommended for a public URL: also add `ACCESS_CODE`. Visitors must then enter it, or supply their own keys, so strangers cannot spend your API quota.
5. Deploy.

CLI alternative: `npm i -g vercel`, then `vercel` in this folder and `vercel --prod`.

**Time limit.** A run can take 1 to 3 minutes. `vercel.json` sets `maxDuration` to 300 seconds, which needs Fluid Compute (on by default for new projects; on Hobby the cap is 300 s). If runs get cut off, lower sub-questions, rounds or the step limit in Settings.

Other free host: Streamlit Community Cloud with `streamlit_app.py` as the main file and `.streamlit/secrets.toml.example` as the secrets template.

## Limitations

- Report quality depends on the search results and on the model; always open the cited sources for anything important.
- The fact-check pass is another LLM call, so it reduces errors but cannot guarantee there are none.
- Pages behind logins, paywalls or heavy JavaScript cannot be read; the agent then relies on search snippets.
- Free API tiers have rate limits; if you hit them, lower the number of sub-questions or rounds in the sidebar.

## Roadmap

- Save and compare past reports
- Domain allow/deny lists
- Parallel research of sub-questions
