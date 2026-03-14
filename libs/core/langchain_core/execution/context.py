"""Unified execution context for LangChain runtime.

Provides a single `ExecutionContext` object that carries all execution state
through the call stack. This replaces the fragmented context propagation across
`RunnableConfig`, `CallbackManager`, and `ContextVar` instances.

The `ExecutionContext` is designed to be:

- **Immutable**: The dataclass is frozen to prevent attribute mutation.
  Child contexts are new objects; parents are never modified.
- **Serializable**: All fields are JSON-safe primitives.
- **Bridge-compatible**: Can convert to/from `RunnableConfig` for migration.

Example:

    >>> ctx = ExecutionContext.create_root(name="my_chain", tags=["prod"])
    >>> child = ctx.create_child(name="llm_call", run_type="llm")
    >>> child.trace_id == ctx.trace_id  # Same trace
    True
    >>> child.parent_span_id == ctx.span_id  # Parent linkage
    True
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


def _new_span_id() -> str:
    """Generate a new span identifier."""
    return str(uuid.uuid4())


@dataclass(frozen=True)
class ExecutionContext:
    """Unified execution context carrying all state through the call stack.

    This provides a single source of truth for execution metadata, replacing
    the fragmented propagation across `RunnableConfig`, `CallbackManager`,
    and tracer `run_map` dictionaries.

    Attributes:
        trace_id: Unique identifier for the entire trace (root-level operation).
        span_id: Unique identifier for this specific span within the trace.
        parent_span_id: Span ID of the parent context, or `None` for root.
        run_type: Type of the current execution unit (e.g., `"chain"`,
            `"llm"`, `"tool"`, `"retriever"`).
        name: Human-readable name for the current span.
        tags: Tags inherited from parent and augmented locally.
        metadata: Metadata inherited from parent and augmented locally.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None
    run_type: str
    name: str
    tags: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_root(
        cls,
        *,
        name: str = "root",
        run_type: str = "chain",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> ExecutionContext:
        """Create a root execution context for a new trace.

        Args:
            name: Human-readable name for the root span.
            run_type: Type of the root execution unit.
            tags: Tags to attach to this trace.
            metadata: Metadata to attach to this trace.
            trace_id: Optional explicit trace ID. Generated if not provided.

        Returns:
            A new root `ExecutionContext`.
        """
        root_id = trace_id or _new_span_id()
        return cls(
            trace_id=root_id,
            span_id=root_id,
            parent_span_id=None,
            run_type=run_type,
            name=name,
            tags=tuple(tags or []),
            metadata=dict(metadata or {}),
        )

    def create_child(
        self,
        *,
        name: str,
        run_type: str = "chain",
        extra_tags: list[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ExecutionContext:
        """Create a child execution context inheriting from this context.

        The child shares the same `trace_id` and inherits tags and metadata,
        with optional additions.

        Args:
            name: Human-readable name for the child span.
            run_type: Type of the child execution unit.
            extra_tags: Additional tags to append to inherited tags.
            extra_metadata: Additional metadata to merge with inherited metadata.

        Returns:
            A new child `ExecutionContext`.
        """
        child_tags = self.tags
        if extra_tags:
            child_tags = self.tags + tuple(extra_tags)

        child_metadata = dict(self.metadata)
        if extra_metadata:
            child_metadata.update(extra_metadata)

        return ExecutionContext(
            trace_id=self.trace_id,
            span_id=_new_span_id(),
            parent_span_id=self.span_id,
            run_type=run_type,
            name=name,
            tags=child_tags,
            metadata=child_metadata,
        )

    @classmethod
    def from_runnable_config(cls, config: dict[str, Any]) -> ExecutionContext:
        """Create an `ExecutionContext` from a `RunnableConfig` dict.

        This is a bridge method for migration from the existing config-based
        context propagation system.

        Args:
            config: A `RunnableConfig` dictionary.

        Returns:
            An `ExecutionContext` populated from the config.
        """
        run_id = config.get("run_id")
        span_id = str(run_id) if run_id else _new_span_id()
        tags = list(config.get("tags") or [])
        metadata = dict(config.get("metadata") or {})
        name = config.get("run_name") or "unknown"

        return cls(
            trace_id=span_id,
            span_id=span_id,
            parent_span_id=None,
            run_type="chain",
            name=name,
            tags=tuple(tags),
            metadata=metadata,
        )

    def to_runnable_config(self) -> dict[str, Any]:
        """Convert this context to a `RunnableConfig`-compatible dict.

        This is a bridge method for migration from the existing config-based
        context propagation system.

        Returns:
            A dictionary compatible with `RunnableConfig`.
        """
        return {
            "run_id": uuid.UUID(self.span_id) if self._is_valid_uuid(self.span_id) else None,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "run_name": self.name,
        }

    @property
    def ancestry(self) -> list[str]:
        """Return the span ancestry as a list from root to current.

        Returns:
            List of span IDs representing the path from root to this span.
            For a root context, returns `[span_id]`.
            For a child context, returns `[parent_span_id, span_id]`.

        !!! note

            This only includes the immediate parent linkage stored in this
            context. For full ancestry reconstruction across the entire trace,
            use the tracer's `run_map`.
        """
        if self.parent_span_id is None:
            return [self.span_id]
        return [self.parent_span_id, self.span_id]

    @staticmethod
    def _is_valid_uuid(val: str) -> bool:
        """Check if a string is a valid UUID.

        Args:
            val: The string to check.

        Returns:
            `True` if the string is a valid UUID.
        """
        try:
            uuid.UUID(val)
        except (ValueError, AttributeError):
            return False
        return True


#: ContextVar for implicit propagation of execution context down the call stack.
#: This mirrors the pattern used by `var_child_runnable_config` in
#: `langchain_core.runnables.config`.
var_execution_context: ContextVar[ExecutionContext | None] = ContextVar(
    "execution_context", default=None
)


def get_current_context() -> ExecutionContext | None:
    """Get the current execution context from the ContextVar.

    Returns:
        The current `ExecutionContext`, or `None` if not set.
    """
    return var_execution_context.get()
