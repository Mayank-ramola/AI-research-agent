"""Run the agent from the terminal:  python cli.py "your topic" """
import re
import sys
from pathlib import Path

from agent import ResearchAgent, Settings
from agent.llm import LLM


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: python cli.py "topic to research"')
    topic = " ".join(sys.argv[1:])
    settings = Settings.from_env()
    if settings.missing_keys():
        sys.exit("Missing keys in .env: " + ", ".join(settings.missing_keys()))

    agent = ResearchAgent(LLM(settings.groq_api_key, settings.model), settings)
    report = ""
    for event in agent.run(topic):
        kind = event["type"]
        if kind == "plan":
            print("Plan:", *event["questions"], sep="\n  - ")
        elif kind == "tool":
            print(f"  {event['name']}({event['args']}) -> {event['sources_so_far']} sources so far")
        elif kind == "review":
            print(f"Review: enough={event['enough']} follow-ups={event['follow_up_questions']}")
        elif kind == "limit":
            print("Step limit reached; writing the report with what was found.")
        elif kind == "verify":
            print("Checks:", event["stats"])
        elif kind == "report":
            report = event["markdown"]

    out = Path("reports")
    out.mkdir(exist_ok=True)
    path = out / (re.sub(r"\W+", "-", topic.lower()).strip("-")[:60] + ".md")
    path.write_text(report, encoding="utf-8")
    print(f"\nSaved {path}\n")
    print(report)


if __name__ == "__main__":
    main()
