import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    groq_api_key: str = ""
    tavily_api_key: str = ""
    model: str = "llama-3.3-70b-versatile"
    max_sub_questions: int = 4      # sub-questions created by the planner
    max_rounds: int = 2             # research rounds (round 2+ answers follow-up questions)
    max_tool_calls: int = 4         # tool calls allowed per sub-question
    max_steps: int = 30             # hard limit on LLM calls + tool calls during research
    max_page_chars: int = 5000      # text kept from each page that is read
    search_results: int = 5

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        base = dict(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            model=os.getenv("MODEL", cls.model),
        )
        base.update({k: v for k, v in overrides.items() if v not in (None, "")})
        return cls(**base)

    def missing_keys(self) -> list[str]:
        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        return missing
