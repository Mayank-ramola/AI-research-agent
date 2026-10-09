"""Vercel entrypoint: a FastAPI app that streams the research agent's steps as Server-Sent Events."""
import hmac
import json
import os
import sys
from pathlib import Path

# Make the `agent` package importable no matter where Vercel runs this file from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent import ResearchAgent, Settings
from agent.llm import LLM

app = FastAPI(title="AI Research Agent")


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    groq_api_key: str = ""
    tavily_api_key: str = ""
    model: str = ""
    access_code: str = ""
    max_sub_questions: int = Field(4, ge=2, le=6)
    max_rounds: int = Field(2, ge=1, le=3)
    max_steps: int = Field(30, ge=10, le=60)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "server_keys": bool(os.getenv("GROQ_API_KEY") and os.getenv("TAVILY_API_KEY")),
        "access_code_required": bool(os.getenv("ACCESS_CODE")),
        "model": os.getenv("MODEL", Settings.model),
    }


@app.post("/api/research")
def research(req: ResearchRequest):
    required = os.getenv("ACCESS_CODE", "")
    # Visitors who bring their own keys do not spend yours, so they skip the access code.
    uses_own_keys = bool(req.groq_api_key.strip() and req.tavily_api_key.strip())
    if required and not uses_own_keys and not hmac.compare_digest(req.access_code.encode(), required.encode()):
        return JSONResponse({"error": "Wrong or missing access code."}, status_code=401)

    settings = Settings.from_env(
        groq_api_key=req.groq_api_key.strip(),
        tavily_api_key=req.tavily_api_key.strip(),
        model=req.model.strip(),
        max_sub_questions=req.max_sub_questions,
        max_rounds=req.max_rounds,
        max_steps=req.max_steps,
    )
    if settings.missing_keys():
        return JSONResponse({"error": "Missing API keys: " + ", ".join(settings.missing_keys())}, status_code=400)

    def stream():
        agent = ResearchAgent(LLM(settings.groq_api_key, settings.model), settings)
        try:
            for event in agent.run(req.topic.strip()):
                if event["type"] == "tool" and event["name"] == "read_page":
                    try:
                        src = agent.store.get(int(event["args"].get("source_id", 0) or 0))
                    except (TypeError, ValueError):
                        src = None
                    event["source_title"] = src.title if src else "unknown source"
                yield _sse(event)
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)[:300]})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
