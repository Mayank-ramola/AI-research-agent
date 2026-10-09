"""Thin wrapper around the Groq chat API with retries, JSON parsing and tool-call normalisation."""
import json
import re
import time
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    content: str = ""
    tool_calls: list = field(default_factory=list)


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating text around it."""
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


class LLM:
    def __init__(self, api_key: str, model: str):
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model = model

    def chat(self, messages, tools=None, json_mode=False, temperature=0.2, max_tokens=1800) -> Message:
        kwargs = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(3):
            try:
                msg = self.client.chat.completions.create(**kwargs).choices[0].message
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return Message(content=msg.content or "", tool_calls=calls)

    def json(self, messages, **kwargs) -> dict:
        return parse_json(self.chat(messages, json_mode=True, **kwargs).content)
