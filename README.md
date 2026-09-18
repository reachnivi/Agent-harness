# agent-harness

Building an agent harness from scratch, to learn how one works.

The whole idea in one line: **an agent is a `while` loop over a list of messages.** Each stage
adds exactly one concept to that loop, and every stage runs on its own.

## Setup

Runs against **local models via Ollama** — free, offline, no API key. Ollama v0.14+ implements
Anthropic's Messages API, so the official `anthropic` SDK talks to it directly. Nothing here is
Ollama-specific; you're learning the real API with a local model behind it.

```bash
ollama serve                 # in another terminal
ollama pull qwen3-coder      # must be a TOOL-CAPABLE model

cp .env.example .env
uv sync
uv run stages/s00_hello.py   # preflight + one API call
```

The model **must support tool calling** or stages 2+ silently do nothing — it just answers in
prose and never emits a `tool_use` block, which looks exactly like a broken loop. Stage 0's
preflight checks this for you. Known-good: `qwen3-coder`, `qwen2.5`, `llama3.1`, `mistral-nemo`.

To run against the hosted Anthropic API instead, edit `.env` — see Option B in `.env.example`.
**No code changes.** That's the point: the harness is the loop, not the provider.

## Stages

| Stage | File | Concept |
|---|---|---|
| 0 | `harness/client.py`, `stages/s00_hello.py` | One endpoint (`POST /v1/messages`) underneath everything. |
| 1 | `stages/s01_chat.py` | The API is stateless — the `messages` list is the agent's entire memory. |
| 2 | `stages/s02_one_tool.py` | The model *requests* a tool call; your code executes it. |
| 3 | `stages/s03_loop.py` | Wrap stage 2 in a loop and you have an agent. **The core lesson.** |
| 4 | `harness/tools.py`, `harness/agent.py` | Separate the loop (policy) from the tools (capability). |
| 5 | — | Streaming. |
| 6 | `harness/permissions.py` | A permission gate. A refusal is a tool *result*, not an exception. |
| 7 | — | Specialize; compare against the SDK's built-in `tool_runner`. |

Stages 0–3 exist. Each file's docstring has a **TRY BREAKING IT** section — that's where most of
the learning is; the code just running teaches you much less than watching it fail on purpose.

`diff stages/s02_one_tool.py stages/s03_loop.py` is a decent explanation of what the loop adds.

### The four rules (stage 3)

1. All `tool_result` blocks for one assistant turn go in **one** user message.
2. **Every** `tool_use` gets a result.
3. Tool failures are **results** (`is_error: True`), not exceptions.
4. **Bound** the loop.

Rules 1 and 3 fail quietly rather than loudly, which is why they're worth memorizing.

## Tests

`tests/` drives the real loop with a scripted fake client — no key, no model, no network,
milliseconds. One test per rule. Run after every refactor:

```bash
uv run pytest -v
```

## Known gaps on Ollama

| Feature | Where it matters |
|---|---|
| `tool_choice` | Nowhere — stages 0–6 never force a tool. |
| Extended thinking | Stage 5: don't pass `thinking=`. |
| `count_tokens` endpoint | Stage 7: token counts are approximations. |
| Prompt caching | Irrelevant at this scale. |
| Server-side tools (`web_search`) | Stage 7: needs a real Anthropic key. |
