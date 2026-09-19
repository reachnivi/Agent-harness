# The plugin pattern, and how `dshpy/` maps to DeepSeek Harness

`dshpy/` is a small Python port of the architecture behind
[`deepseek-ai/deepseek-harness`](https://github.com/deepseek-ai/deepseek-harness) (MIT,
TypeScript), which is built on **Cordis** and whose thesis is *everything is a plugin* — the
model adapter, the tool registry, the session log, **and the agent loop itself**.

This repo now contains both a hand-written harness (`stages/`, `harness/`) and a plugin one
(`dshpy/`), doing the same job. Reading them against each other is the point.

## Cordis in five ideas

| # | Idea | Where it lives here |
|---|---|---|
| 1 | A plugin is an object with `inject` and `apply(ctx)` | every file in `dshpy/plugins/` and `dshpy/services/` |
| 2 | A context is a repository of services; plugins find each other **by key**, never by import | `dshpy/core/context.py` — `Context.__getattr__`, `ctx.provide` |
| 3 | `inject` replaces boot ordering: a plugin waits for its services, and deactivates if one disappears | `Runtime._settle`, `_on_service_lost` |
| 4 | Typed events with dispatch modes — `emit`, `waterfall`, `bail` | `dshpy/core/events.py` |
| 5 | Registrations are reversible effects | `dshpy/core/effects.py` |

### Why lookup-by-key is the whole trick

`from dshpy.plugins.llm_openai import OpenAIAdapter` welds the loop to one provider: the import
is a hard edge, and swapping the provider means editing the consumer.

`ctx.llm` does not. Whoever claimed the key is what you get, decided by configuration at boot.
The loop can be written, read and tested with no provider mounted at all.

The cost is real: you lose "go to definition" on `ctx.llm`. That is the price of a seam, and it
is why `inject` exists — it recovers at boot the guarantee the import used to give at import time.

### Why `waterfall` is the interesting dispatch mode

A `waterfall` listener receives `(*args, next)`:

```python
def listener(exec_, next):
    if not allowed(exec_):
        return Deny("policy")   # short-circuit: nothing below runs
    result = next()             # delegate to the rest of the chain
    return annotate(result)     # ...and you may still transform what comes back
```

Because `next()` is called *inside* the listener, a plugin can run code after the thing it
wraps. That single property is what lets permission policy, timeouts and result rewriting live
in plugins while the agent loop stays ignorant of all three.

## The tool pipeline

Every tool call runs this sequence. Each stage is an extension point; all are optional.

```
tools/pre-execute    (waterfall)  policy: allow / deny / ask      <- reorderable
  guards             (monotonic)  a final deny nothing can undo   <- NOT reorderable
    tools/execute    (waterfall)  around-dispatch: timeout, retry, metrics
      <tool body>
    tools/post-execute (waterfall) transform, block, annotate
tools/result         (emit)       observe the final outcome
```

### The gate/guard distinction, which is not pedantry

`tools/pre-execute` is **reorderable policy**. Listeners compose, and a plugin mounted later —
or one registered with `prepend=True` — can decline to delegate and thereby allow something an
earlier listener questioned. That is correct for policy, whose nature is that the answer
depends on configuration.

`ctx.tools.guard()` is a **monotonic final deny**. Once a guard denies, nothing can allow.

Put preferences in the gate and invariants in the guard. Path confinement belongs in the guard,
because no profile ordering should be able to switch it off. Put an invariant in the gate by
mistake and you are one badly-ordered config away from a bypass — which is how permission
systems grow holes. `tests/test_pipeline.py::test_a_guard_denies_and_cannot_be_overridden_by_later_policy`
pins this down.

## Services in this port

| Key | Provided by | Owns |
|---|---|---|
| `ctx.llm` | `services/llm.py` | the adapter seam + neutral message vocabulary |
| `ctx.tools` | `services/tools.py` | tool registry + the pipeline above |
| `ctx.sessions` | `services/sessions.py` | append-only event log; model history is *derived* |
| `ctx.agent_loop` | `services/agent_loop.py` | the loop — itself just a plugin |

dsh has many more (`ctx.agents`, `ctx.jobs`, `ctx.fs`, `ctx.sandbox`, `ctx.commands`,
`ctx.approval`, …). The four here are enough to show the shape.

## What was deliberately left out

Faithfulness has a cost, and these were judged not worth it for a learning port:

- **`parallel` / `serial` dispatch modes** — the async variants. Mechanical to add; teach nothing new.
- **Bundles, profiles-as-packages, `cordis.patch.yml` layering, HMR** — real dsh composes a
  plugin tree from ordered layers, each patchable by the ones above. That is config plumbing,
  not architectural insight. `profiles/default.py` is the one-layer version.
- **Context forking / isolate realms** — dsh scopes registrations per agent via `agent.ctx`.
- **Streaming** — both adapters are request/response. Streaming would change the adapter
  contract (`AsyncIterable[StreamChunk]`) but not the architecture.
- **A real cancellation signal.** `plugins/timeout.py` uses a daemon thread, so the *call*
  returns on time but the work keeps running. dsh threads an `exec.signal` through to the tool
  body. Honest simplification, flagged in the file.

## Writing a plugin

```python
# dshpy/plugins/my_tool.py
from dshpy.services.tools import define_tool

name = "my-tool"
inject = ["tools"]           # activation waits until ctx.tools exists

def apply(ctx, config=None):
    ctx.effect(ctx.tools.register(define_tool(
        name="roll_dice",
        description="Roll an n-sided die.",
        parameters={"sides": {"type": "number", "required": True}},
        execute=lambda args, exec: str(random.randint(1, int(args["sides"]))),
    )))
```

Add `{"plugin": my_tool}` to `dshpy/profiles/default.py`. Anywhere in the list — `inject`
decides activation order, and `tests/test_profile.py::test_profile_row_order_does_not_matter`
shuffles the rows twelve ways to prove it.

## The two harnesses, side by side

| | `stages/` + `harness/` | `dshpy/` |
|---|---|---|
| Architecture | one hardcoded loop | plugin tree |
| Add a tool | edit `TOOLS` **and** `dispatch()` | one registration in a plugin |
| Add policy | edit the loop | listen on `tools/pre-execute` |
| Change provider | rewrite the loop | change one profile row |
| Tool results | the loop knows the wire format | the adapter does; the loop never learns it |
| SDK | `anthropic` | none — stdlib `urllib` |
| Lines to read before you understand it | ~200 | ~700 |

That last row is the honest cost. The plugin version is three times the code and buys you
nothing at all until the second provider, the second policy, or the second person shows up.
Then it buys you quite a lot. Knowing which situation you are in is the actual skill.
