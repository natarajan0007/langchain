"""Reproducible demonstration of the fragmented execution context issue.

This script demonstrates that in the current LangChain architecture, execution
context is fragmented across multiple systems:

1. `RunnableConfig` (tags, metadata, run_id)
2. `CallbackManager` (run_id, parent_run_id, handlers)
3. `var_child_runnable_config` ContextVar (implicit propagation)
4. Tracer `run_map` (hierarchical run tree)

A component executing within a chain must understand all four systems to
reconstruct its full execution context. The proposed `ExecutionContext`
provides a single unified API.

Run with::

    cd libs/core
    uv run python docs/architecture/repro_context_issue.py
"""

from __future__ import annotations

import uuid

from langchain_core.callbacks.manager import CallbackManager
from langchain_core.execution.context import ExecutionContext
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    patch_config,
)
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run


class DemoTracer(BaseTracer):
    """A minimal tracer for demonstrating context fragmentation."""

    def __init__(self) -> None:
        """Initialize the demo tracer."""
        super().__init__()
        self.runs: list[Run] = []

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""
        self.runs.append(run)


def demonstrate_fragmented_context() -> None:
    """Show how context is fragmented across multiple systems."""
    print("=" * 70)
    print("BEFORE: Fragmented Context (Current Architecture)")
    print("=" * 70)

    # System 1: RunnableConfig
    config: RunnableConfig = {
        "tags": ["production", "v2"],
        "metadata": {"user_id": "u123", "session": "s456"},
        "run_id": uuid.uuid4(),
        "run_name": "my_agent",
    }
    resolved = ensure_config(config)

    print("\n1. RunnableConfig knows:")
    print(f"   tags={resolved.get('tags')}")
    print(f"   metadata={resolved.get('metadata')}")
    print(f"   run_id={resolved.get('run_id')}")
    print(f"   run_name={resolved.get('run_name')}")

    # System 2: CallbackManager
    tracer = DemoTracer()
    manager = CallbackManager(handlers=[tracer])
    run_manager = manager.on_chain_start(
        {"name": "my_agent"},
        {"input": "hello"},
        run_id=resolved.get("run_id"),
    )

    print("\n2. CallbackManager knows:")
    print(f"   run_id={run_manager.run_id}")
    print(f"   parent_run_id={run_manager.parent_run_id}")
    print(f"   handlers={[type(h).__name__ for h in run_manager.handlers]}")

    # System 3: Child config via patch_config
    child_config = patch_config(resolved, callbacks=run_manager.get_child())
    print("\n3. Child config (after patch_config) knows:")
    print(f"   callbacks type={type(child_config.get('callbacks')).__name__}")
    print(f"   tags={child_config.get('tags')}")

    # System 4: Tracer run_map
    print("\n4. Tracer run_map knows:")
    for rid, run in tracer.run_map.items():
        print(f"   run_id={rid}, name={run.name}, parent={run.parent_run_id}")

    run_manager.on_chain_end({"output": "done"})

    print("\n⚠ To reconstruct full context, a tool must query ALL FOUR systems.")
    print("  There is no single API to get: trace_id + parent + tags + metadata.")


def demonstrate_unified_context() -> None:
    """Show how the proposed ExecutionContext unifies all state."""
    print("\n")
    print("=" * 70)
    print("AFTER: Unified ExecutionContext (Proposed Architecture)")
    print("=" * 70)

    # Single context object carries everything
    root = ExecutionContext.create_root(
        name="my_agent",
        tags=["production", "v2"],
        metadata={"user_id": "u123", "session": "s456"},
    )

    print("\n1. Root context (single object):")
    print(f"   trace_id={root.trace_id}")
    print(f"   span_id={root.span_id}")
    print(f"   parent_span_id={root.parent_span_id}")
    print(f"   name={root.name}")
    print(f"   tags={root.tags}")
    print(f"   metadata={root.metadata}")

    # Child automatically inherits everything
    llm_ctx = root.create_child(name="llm_call", run_type="llm")
    print("\n2. Child context (LLM call):")
    print(f"   trace_id={llm_ctx.trace_id}  (same as root)")
    print(f"   span_id={llm_ctx.span_id}  (new)")
    print(f"   parent_span_id={llm_ctx.parent_span_id}  (== root.span_id)")
    print(f"   tags={llm_ctx.tags}  (inherited)")

    # Tool child with extra context
    tool_ctx = root.create_child(
        name="search_tool",
        run_type="tool",
        extra_tags=["search"],
        extra_metadata={"tool_version": "1.2"},
    )
    print("\n3. Child context (Tool call with extras):")
    print(f"   trace_id={tool_ctx.trace_id}  (same as root)")
    print(f"   span_id={tool_ctx.span_id}  (new)")
    print(f"   parent_span_id={tool_ctx.parent_span_id}  (== root.span_id)")
    print(f"   tags={tool_ctx.tags}  (inherited + extra)")
    print(f"   metadata={tool_ctx.metadata}  (inherited + extra)")

    # Bridge to existing RunnableConfig
    config = root.to_runnable_config()
    print("\n4. Bridge to RunnableConfig:")
    print(f"   config={config}")

    # Round-trip from RunnableConfig
    restored = ExecutionContext.from_runnable_config(config)
    print("\n5. Restored from RunnableConfig:")
    print(f"   trace_id={restored.trace_id}")
    print(f"   tags={restored.tags}")

    print("\n✅ Single API provides: trace_id + parent + tags + metadata.")
    print("   No need to query multiple systems.")


if __name__ == "__main__":
    demonstrate_fragmented_context()
    demonstrate_unified_context()
