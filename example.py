"""A runnable demo — no API keys needed.

    python example.py
    replai ui
"""
import time

import replai

replai.init(instrument=False)


@replai.tool
def web_search(query: str):
    time.sleep(0.15)
    return [f"result about {query}", "another result"]


@replai.trace(name="plan")
def plan(goal: str):
    replai.record_llm_call(
        model="claude-opus-4",
        input=[{"role": "user", "content": goal}],
        output="I'll search the web, then summarize.",
        tokens_in=14, tokens_out=9,
    )
    return "search then summarize"


def agent(goal: str):
    with replai.run("demo-agent", goal=goal):
        plan(goal)
        hits = web_search("python sqlite tutorial")
        replai.record_llm_call(
            model="claude-opus-4",
            input=str(hits),
            output="Here is the summary you asked for.",
            tokens_in=22, tokens_out=18,
        )
        return "done"


if __name__ == "__main__":
    agent("learn how to use sqlite in python")
    print("Recorded a run. Now launch the viewer:\n\n    replai ui\n")
