# agent-harness

Building an agent harness from scratch to learn how one works.

The whole idea: an agent is a `while` loop over a list of messages. Each stage in
`stages/` adds exactly one concept to that loop, and every stage runs on its own.

| Stage | File | Concept |
|---|---|---|
| 0 | `stages/s00_hello.py` | One API call. Everything is built on `POST /v1/messages`. |
| 1 | `stages/s01_chat.py` | The API is stateless — the `messages` list is the agent's whole memory. |
| 2 | `stages/s02_one_tool.py` | The model requests a tool call; *your code* executes it. |
| 3 | `stages/s03_loop.py` | Wrap stage 2 in a loop and you have an agent. |
| 4 | `harness/` | Separate the loop (policy) from the tools (capability). |
| 5 | — | Streaming. |
| 6 | `harness/permissions.py` | A permission gate. A refusal is a tool *result*, not an exception. |
| 7 | — | Specialize, and compare against the SDK's built-in `tool_runner`. |

`diff stages/s02_one_tool.py stages/s03_loop.py` is a decent explanation of what the
agentic loop actually adds.

## Setup

```bash
cp .env.example .env     # then paste your key from console.anthropic.com
uv sync
uv run stages/s00_hello.py
```

## Tests

`tests/` runs the loop against a scripted fake client — no API key, no tokens spent,
milliseconds. Run it after every refactor:

```bash
uv run pytest -v
```
