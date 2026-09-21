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
| `ctx.llm` | `services/llm.py` | the adapter seam, neutral vocabulary, `StreamChunk` protocol |
| `ctx.tools` | `services/tools.py` | tool registry + the pipeline above |
| `ctx.sessions` | `services/sessions.py` | append-only event log; model history is *derived* |
| `ctx.agent_loop` | `services/agent_loop.py` | the loop — itself just a plugin |
| `ctx.tokens` | `plugins/token_meter.py` | usage accounting, via the `llm/stream` waterfall |
| `ctx.fs` | `services/fs.py` | filesystem seam; `fs_local` is the provider |
| `ctx.shell` | `services/shell.py` | subprocess seam; `shell_local` is the provider |

dsh has many more (`ctx.agents`, `ctx.jobs`, `ctx.fs`, `ctx.sandbox`, `ctx.commands`,
`ctx.approval`, …). The four here are enough to show the shape.

## Streaming (phase 6)

`ctx.llm.stream()` returns an iterator of `StreamChunk`, ported near-verbatim from dsh:

```
block-start / text-delta / reasoning-delta / tool-call-delta / block-end / usage / finish
```

Three adapter rules, each fixing a real failure:

1. **`index` ties a delta to its block**, allocated in first-seen order and reused. A response
   interleaves text and several tool calls; without it you cannot tell whose delta you hold.
2. **Tool arguments stay raw JSON strings**, streamed as `arguments_delta`, parsed once at the
   end. Parsing a fragment is how you get a JSONDecodeError halfway through a reply.
3. **`usage` before `finish`, nothing after `finish`.** Consumers treat finish as terminal.

`BlockAssembler` is the one shared fold from chunks back to a `Completion`, so the loop (which
needs a finished message) and a UI (which wants deltas) read the same stream without either
reimplementing accumulation. `plugins/stream_ui.py` is the smallest consumer of the raw side.

**Two SSE dialects, one protocol.** OpenAI sends anonymous `data:` frames with fragmentary
`tool_calls[].function.arguments` and a `[DONE]` sentinel; Anthropic sends *named* events with
an explicit block lifecycle and the tool name on `content_block_start`. Both normalize to the
chunks above inside their adapter. `dshpy/sse.py` is the hand-written parser — the two tests
worth reading are the ones for a frame split across reads and a multi-byte character split
across reads, both valid input that a naive parser mangles.

**Failure normalization is the outermost layer.** An adapter may raise; `llm/stream` listeners
see the raw exception (retry needs it to tell a 503 from a 400); only at the very top is it
turned into a terminal `finish{error|aborted}`. Getting this order wrong — normalizing on the
inside — silently disables retry, which is exactly what happened on the first attempt.

## The filesystem and shell seams (phase 7)

Same three-role shape as `ctx.llm`: a contract (`services/fs.py`), a provider
(`plugins/fs_local.py`), and model-facing consumers (`plugins/tool_fs.py`). Pointing the agent
at a container becomes a config change rather than a fork of the tools.

**The intents are mutable objects, not positional arguments.** `fs/read-intent`,
`fs/write-intent` and `fs/edit-intent` each carry a dataclass that a listener may mutate before
calling `next()`. That is dsh's cooperative-listener idiom, and it is load-bearing: `next()`
takes no arguments, so a guard that needs to impose a version the caller never supplied has no
other way to do it. The first version of `fs.py` used positional args and needed an awkward
helper to work around exactly this.

**Path confinement moved out of `permission.py`.** It belongs next to the filesystem — it must
understand symlinks and roots, and it must cover every `ctx.fs` consumer rather than only tools
whose argument happens to be named `path`. It is still a *guard*, not a gate.

**Read-before-edit** is the version guard in action: `fs_guard` records the version of every
file read and imposes it on the next write. A model working from a copy it read three tool
calls ago will otherwise silently destroy whatever changed in between.

### Two bugs worth keeping in the record

- **Killing the process, not the group.** With `shell=True` the child is `/bin/sh -c ...`, and
  the shell may *fork* the real command. Terminating the shell leaves a grandchild holding the
  stdout pipe open, so `communicate()` then blocks until the orphan finishes on its own. The
  symptom is a timeout that reports success and still takes the full 30 seconds.
  `start_new_session=True` plus `os.killpg` is the fix.
- **`resolve()` anchors a relative path to the process cwd**, not to the agent's root. A
  confinement check that forgets this rejects `read_file("f.txt")` as an escape.

Both are in the tests now, which is the only reason they stay fixed.

## What was deliberately left out

Faithfulness has a cost, and these were judged not worth it for a learning port:

- **`parallel` / `serial` dispatch modes** — the async variants. Mechanical to add; teach nothing new.
- **Bundles, profiles-as-packages, `cordis.patch.yml` layering, HMR** — real dsh composes a
  plugin tree from ordered layers, each patchable by the ones above. That is config plumbing,
  not architectural insight. `profiles/default.py` is the one-layer version.
- **Context forking / isolate realms** — dsh scopes registrations per agent via `agent.ctx`.
- **Async.** dsh's adapters are `AsyncIterable` because Node is async; ours are sync generators.
  Identical streaming semantics, and it keeps `async def` out of the kernel and every plugin.
  The one real cost: tool calls in a batch run sequentially, where dsh runs them concurrently.
- **`replay_state`** — dsh's adapters carry provider-private metadata on `finish` for replaying
  a response. Ours don't, so history is rebuilt from the neutral vocabulary only.

### Cancellation is cooperative, and that is a real limit

`plugins/timeout.py` sets a deadline on `exec.token`; long-running tool bodies poll it with
`exec.token.check()`. A tool that never polls **cannot be interrupted** — Python has no safe
way to stop an arbitrary thread — and there is a test asserting exactly that rather than hiding
it. This replaced phase 5's daemon thread, which returned on time while the work carried on
invisibly. Failing visibly beats succeeding falsely.

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
