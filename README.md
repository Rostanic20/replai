# replai

**A local-first debugger for LLM agents.** See exactly what your agent did — every model call, tool call, and decision — step by step. No account, no cloud, no Docker. Just `pip install` and look.

> ⚠️ **Early alpha (v0.1).** The capture engine and local viewer work today. Replay and run-diffing are next on the roadmap.

<!-- TODO: drop a GIF of the timeline view here — this is the most important part of the README. -->

## Why

When an AI agent does the wrong thing, you're usually staring at a wall of logs trying to reconstruct what happened. Production observability platforms exist, but they're heavy — dashboards, servers, accounts — built for *monitoring at scale*, not for the moment you're on your laptop going *"wait, why did it call that tool?"*

`replai` is the other thing: a **debugger** for the dev inner loop. Drop it in, run your agent, and get a clickable, step-by-step timeline of everything it did — locally.

## Install

```sh
pip install "replai[viewer]"
```

## Quickstart

```python
import replai
replai.init()          # auto-captures Anthropic & OpenAI calls

# ... run your agent exactly as you normally would ...
```

Then open the viewer:

```sh
replai ui
```

Want to annotate your own steps?

```python
with replai.run("my-agent"):
    with replai.span("retrieve", type="tool_call") as s:
        s.output = my_retriever(query)
```

Or decorate functions and tools:

```python
@replai.tool
def web_search(query): ...

@replai.trace
def plan(goal): ...
```

### Try it with no API keys

```sh
python example.py
replai ui
```

## How it works

- **Auto-instrumentation** wraps the Anthropic / OpenAI clients, so calls are captured with zero code changes.
- **`@replai.trace` / `@replai.tool` / `replai.span()`** annotate your own functions and tool calls. Spans nest automatically.
- Everything is stored in a local **SQLite** file (`~/.replai/replai.db`). Nothing leaves your machine.
- A small **FastAPI** viewer renders each run as a step-by-step timeline.

## Roadmap

- [x] Capture engine (LLM + tool + function spans, sync & async)
- [x] Local timeline viewer
- [ ] **Replay** — step through a run; re-run from any step
- [ ] **Diff** — compare two runs, highlight where they diverged
- [ ] Framework adapters (LangChain, LlamaIndex, …)
- [ ] MCP tool-call capture
- [ ] OpenTelemetry GenAI export

## License

MIT
