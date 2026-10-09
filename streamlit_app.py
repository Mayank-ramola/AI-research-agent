import os

import streamlit as st

from agent import ResearchAgent, Settings
from agent.llm import LLM

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="wide")

# On Streamlit Community Cloud, keys come from "Secrets"; locally they come from .env.
try:
    for key in ("GROQ_API_KEY", "TAVILY_API_KEY", "MODEL"):
        if key in st.secrets:
            os.environ.setdefault(key, str(st.secrets[key]))
except Exception:
    pass

st.title("AI Research Agent")
st.caption("Give it a topic. It plans sub-questions, searches the web, reads pages, and writes a cited report.")

with st.sidebar:
    st.header("Settings")
    groq_key = st.text_input("Groq API key", type="password", help="Leave empty to use GROQ_API_KEY from .env or Secrets.")
    tavily_key = st.text_input("Tavily API key", type="password", help="Leave empty to use TAVILY_API_KEY.")
    model = st.text_input("Model", value=os.getenv("MODEL", "llama-3.3-70b-versatile"))
    max_sub = st.slider("Sub-questions", 2, 6, 4)
    max_rounds = st.slider("Research rounds", 1, 3, 2, help="Extra rounds answer follow-up questions found in review.")
    max_steps = st.slider("Step limit", 10, 60, 30, help="Hard cap on model calls plus tool calls while researching.")

topic = st.text_input("Research topic", placeholder="e.g. How do solid-state batteries compare with lithium-ion?")
run = st.button("Research", type="primary", disabled=not topic.strip())

TOOL_ICON = {"web_search": "🔎", "read_page": "📄"}

if run:
    settings = Settings.from_env(
        groq_api_key=groq_key,
        tavily_api_key=tavily_key,
        model=model,
        max_sub_questions=max_sub,
        max_rounds=max_rounds,
        max_steps=max_steps,
    )
    if settings.missing_keys():
        st.error("Add these keys in the sidebar or in .env: " + ", ".join(settings.missing_keys()))
    else:
        agent = ResearchAgent(LLM(settings.groq_api_key, settings.model), settings)
        st.session_state.pop("result", None)
        try:
            with st.status("Researching…", expanded=True) as status:
                for event in agent.run(topic.strip()):
                    kind = event["type"]
                    if kind == "plan":
                        st.markdown("**Plan**\n" + "\n".join(f"- {q}" for q in event["questions"]))
                    elif kind == "question":
                        st.markdown(f"**Researching:** {event['question']}")
                    elif kind == "tool":
                        icon = TOOL_ICON.get(event["name"], "🛠️")
                        if event["name"] == "web_search":
                            st.write(f"{icon} Searched: *{event['args'].get('query', '')}* ({event['sources_so_far']} sources so far)")
                        else:
                            src = agent.store.get(int(event["args"].get("source_id", 0) or 0))
                            st.write(f"{icon} Read: {src.title if src else 'unknown source'}")
                    elif kind == "fallback":
                        st.write("⚠️ Tool calling failed for this question, using plain search instead.")
                    elif kind == "notes":
                        with st.expander("Notes for this question"):
                            st.markdown(event["notes"] or "_No notes._")
                    elif kind == "review":
                        verdict = "enough evidence" if event["enough"] else "gaps found, adding follow-up questions"
                        st.write(f"🧐 Review: {verdict}")
                    elif kind == "limit":
                        st.write("⏹️ Step limit reached. Writing the report with what was found.")
                    elif kind == "draft":
                        st.write("✍️ Draft written. Checking citations…")
                    elif kind == "verify":
                        st.write("✅ Citation check finished.")
                    elif kind == "report":
                        st.session_state["result"] = event
                status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"The research run failed: {exc}")

result = st.session_state.get("result")
if result:
    stats = result["stats"]
    cols = st.columns(4)
    cols[0].metric("Sources found", stats.get("sources_found", 0))
    cols[1].metric("Sources cited", stats.get("sources_cited", 0))
    cols[2].metric("Duplicates merged", stats.get("duplicates_skipped", 0))
    cols[3].metric("Bad citations removed", stats.get("invalid_citations_removed", 0))
    if stats.get("uncited_sentences"):
        st.warning(f"{stats['uncited_sentences']} sentence(s) in the report have no citation. Check them before relying on them.")
    st.divider()
    st.markdown(result["markdown"])
    st.download_button("Download report (.md)", result["markdown"], file_name="research-report.md", mime="text/markdown")
