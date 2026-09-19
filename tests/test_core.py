"""Tests for the plugin micro-kernel.

One test per claim that makes this a plugin system rather than a pile of modules. If any of
these fail, "everything is a plugin" is marketing.
"""

from __future__ import annotations

import types

import pytest

from dshpy.core.context import Runtime, ServiceMissing
from dshpy.core.effects import EffectScope
from dshpy.core.events import EventBus


def plugin(name, inject=(), apply=None):
    """Build a plugin module on the fly. A plugin is just an object with name/inject/apply."""
    mod = types.SimpleNamespace()
    mod.name = name
    mod.inject = list(inject)
    mod.apply = apply or (lambda ctx, config=None: None)
    return mod


# --- idea 2: a context is a repository of services -----------------------------------------


def test_service_is_found_by_key_not_by_import():
    rt = Runtime()
    rt.mount(plugin("provider", apply=lambda ctx, cfg=None: ctx.provide("greeter", "hello")))

    seen = {}
    rt.mount(plugin("consumer", inject=["greeter"],
                    apply=lambda ctx, cfg=None: seen.update(value=ctx.greeter)))

    assert seen["value"] == "hello"


def test_missing_service_says_what_to_do_about_it():
    """The error a plugin author will actually hit, so it must name the likely cause."""
    with pytest.raises(ServiceMissing, match="inject"):
        Runtime().mount(plugin("lonely", apply=lambda ctx, cfg=None: ctx.nope))


def test_two_plugins_cannot_claim_the_same_key():
    rt = Runtime()
    rt.mount(plugin("a", apply=lambda ctx, cfg=None: ctx.provide("dup", 1)))
    with pytest.raises(RuntimeError, match="already provided"):
        rt.mount(plugin("b", apply=lambda ctx, cfg=None: ctx.provide("dup", 2)))


# --- idea 3: inject replaces boot ordering -------------------------------------------------


def test_plugin_with_unmet_inject_does_not_activate():
    rt = Runtime()
    entry = rt.mount(plugin("needs-llm", inject=["llm"]))
    assert entry.active is False


def test_it_activates_later_when_the_service_appears():
    rt = Runtime()
    entry = rt.mount(plugin("needs-llm", inject=["llm"]))
    assert entry.active is False

    rt.mount(plugin("llm-provider", apply=lambda ctx, cfg=None: ctx.provide("llm", object())))
    assert entry.active is True, "deferred activation never fired"


def test_mount_order_does_not_matter():
    """The point of inject: a profile is a set of plugins, not a sequence."""
    log = []

    def consumer(ctx, cfg=None):
        log.append(("consumer", ctx.svc))

    def provider(ctx, cfg=None):
        ctx.provide("svc", "value")
        log.append(("provider", None))

    # Consumer mounted FIRST, before the thing it depends on exists.
    rt = Runtime()
    rt.mount(plugin("consumer", inject=["svc"], apply=consumer))
    rt.mount(plugin("provider", apply=provider))

    assert [name for name, _ in log] == ["provider", "consumer"]


def test_transitive_dependencies_settle_to_a_fixed_point():
    """a -> provides x; b injects x, provides y; c injects y. Mounted in reverse order."""
    rt = Runtime()
    c = rt.mount(plugin("c", inject=["y"]))
    b = rt.mount(plugin("b", inject=["x"], apply=lambda ctx, cfg=None: ctx.provide("y", 2)))
    a = rt.mount(plugin("a", apply=lambda ctx, cfg=None: ctx.provide("x", 1)))

    assert (a.active, b.active, c.active) == (True, True, True)


def test_plugin_deactivates_when_a_service_it_injected_disappears():
    rt = Runtime()
    provider = rt.mount(plugin("p", apply=lambda ctx, cfg=None: ctx.provide("svc", 1)))
    consumer = rt.mount(plugin("c", inject=["svc"]))
    assert consumer.active is True

    rt.unmount(provider)
    assert consumer.active is False, "consumer kept a reference to a service that is gone"


# --- idea 5: registrations are reversible effects ------------------------------------------


def test_unmounting_removes_the_plugins_listeners():
    rt = Runtime()
    calls = []
    entry = rt.mount(plugin("listener",
                            apply=lambda ctx, cfg=None: ctx.on("ping", lambda: calls.append(1))))

    rt.events.emit("ping")
    assert len(calls) == 1

    rt.unmount(entry)
    rt.events.emit("ping")
    assert len(calls) == 1, "listener survived its plugin being unmounted"


def test_unmounting_frees_the_service_key_for_a_replacement():
    rt = Runtime()
    first = rt.mount(plugin("v1", apply=lambda ctx, cfg=None: ctx.provide("llm", "v1")))
    rt.unmount(first)
    rt.mount(plugin("v2", apply=lambda ctx, cfg=None: ctx.provide("llm", "v2")))
    assert rt.services["llm"] == "v2"


def test_effects_dispose_in_reverse_order():
    order = []
    scope = EffectScope("t")
    scope.add(lambda: order.append("first"))
    scope.add(lambda: order.append("second"))
    scope.dispose()
    assert order == ["second", "first"]


def test_a_failing_disposer_does_not_strand_the_others():
    order = []
    scope = EffectScope("t")
    scope.add(lambda: order.append("outer"))
    scope.add(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    scope.dispose()
    assert order == ["outer"]


def test_failed_activation_leaves_no_partial_registrations():
    rt = Runtime()

    def half_broken(ctx, cfg=None):
        ctx.on("ping", lambda: None)
        raise RuntimeError("second half failed")

    with pytest.raises(RuntimeError, match="second half"):
        rt.mount(plugin("broken", apply=half_broken))

    assert rt.events.listeners("ping") == [], "a failed activation leaked a listener"


# --- idea 4: dispatch modes ----------------------------------------------------------------


def test_waterfall_runs_listeners_in_order_and_reaches_the_final_behavior():
    bus = EventBus()
    order = []

    def outer(value, next):
        order.append("outer-before")
        result = next()
        order.append("outer-after")
        return result

    def inner(value, next):
        order.append("inner")
        return next()

    bus.on("evt", outer)
    bus.on("evt", inner)
    result = bus.waterfall("evt", "x", final=lambda v: f"final({v})")

    assert result == "final(x)"
    assert order == ["outer-before", "inner", "outer-after"]


def test_waterfall_listener_that_skips_next_short_circuits_everything_below():
    bus = EventBus()
    reached = []

    bus.on("evt", lambda v, next: "denied")           # no next() -> short circuit
    bus.on("evt", lambda v, next: reached.append("inner") or next())

    result = bus.waterfall("evt", "x", final=lambda v: reached.append("final"))

    assert result == "denied"
    assert reached == [], "short circuit did not stop the inner chain"


def test_waterfall_listener_can_transform_the_result_on_the_way_out():
    bus = EventBus()
    bus.on("evt", lambda v, next: next().upper())
    assert bus.waterfall("evt", "x", final=lambda v: "quiet") == "QUIET"


def test_prepend_puts_a_listener_ahead_of_existing_ones():
    bus = EventBus()
    order = []
    bus.on("evt", lambda next: order.append("normal") or next())
    bus.on("evt", lambda next: order.append("prepended") or next(), prepend=True)
    bus.waterfall("evt", final=lambda: None)
    assert order == ["prepended", "normal"]


def test_bail_returns_the_first_non_none_answer():
    bus = EventBus()
    bus.on("evt", lambda: None)
    bus.on("evt", lambda: "answer")
    bus.on("evt", lambda: "never reached")
    assert bus.bail("evt") == "answer"


def test_emit_isolates_a_raising_observer():
    bus = EventBus()
    seen = []
    bus.on("evt", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.on("evt", lambda: seen.append("still ran"))
    bus.emit("evt")
    assert seen == ["still ran"], "a broken observer broke the producer"


# --- introspection -------------------------------------------------------------------------


def test_dump_shows_what_is_waiting_and_why():
    rt = Runtime()
    rt.mount(plugin("waiting", inject=["absent"]))
    out = rt.dump()
    assert "WAITING on ['absent']" in out
