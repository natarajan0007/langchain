"""Instrumentation provider protocol for unified observability.

Defines a component-type-agnostic interface for instrumentation that follows
OpenTelemetry-compatible patterns. Providers receive span lifecycle events
and can record metrics and semantic events.

The key design principle is **separation of observation from control flow**:
instrumentation providers are read-only observers that do not affect execution.

Example:

    >>> from langchain_core.execution.context import ExecutionContext
    >>> provider = NoOpProvider()
    >>> ctx = ExecutionContext.create_root(name="test")
    >>> provider.on_span_start(ctx, inputs={"query": "hello"})
    >>> provider.on_span_end(ctx, outputs={"result": "world"})
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from langchain_core.execution.context import ExecutionContext

logger = logging.getLogger(__name__)


@runtime_checkable
class InstrumentationProvider(Protocol):
    """Protocol for instrumentation providers.

    Providers receive span lifecycle events (start, end, error) and can
    record metrics and semantic events. They are read-only observers that
    do not affect execution control flow.

    This interface is designed to be compatible with OpenTelemetry patterns,
    where spans represent units of work within a distributed trace.
    """

    def on_span_start(
        self,
        context: ExecutionContext,
        *,
        inputs: Any = None,
    ) -> None:
        """Called when a new span starts.

        Args:
            context: The execution context for this span.
            inputs: The inputs to the span (e.g., chain input, tool args).
        """
        ...

    def on_span_end(
        self,
        context: ExecutionContext,
        *,
        outputs: Any = None,
    ) -> None:
        """Called when a span completes successfully.

        Args:
            context: The execution context for this span.
            outputs: The outputs of the span (e.g., chain output, tool result).
        """
        ...

    def on_span_error(
        self,
        context: ExecutionContext,
        *,
        error: BaseException,
    ) -> None:
        """Called when a span fails with an error.

        Args:
            context: The execution context for this span.
            error: The exception that caused the failure.
        """
        ...

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        context: ExecutionContext,
        unit: str = "",
    ) -> None:
        """Record a numeric metric.

        Args:
            name: The metric name (e.g., ``"token_count"``, ``"latency_ms"``).
            value: The metric value.
            context: The execution context for attribution.
            unit: Optional unit of measurement.
        """
        ...

    def log_event(
        self,
        name: str,
        *,
        context: ExecutionContext,
        data: Any = None,
    ) -> None:
        """Log a semantic event.

        Args:
            name: The event name (e.g., ``"tool_selected"``, ``"retry_attempt"``).
            context: The execution context for attribution.
            data: Optional event payload.
        """
        ...


class NoOpProvider:
    """No-operation instrumentation provider.

    Silently discards all events. Useful as a default when no instrumentation
    is configured.
    """

    def on_span_start(
        self,
        context: ExecutionContext,
        *,
        inputs: Any = None,
    ) -> None:
        """No-op span start."""

    def on_span_end(
        self,
        context: ExecutionContext,
        *,
        outputs: Any = None,
    ) -> None:
        """No-op span end."""

    def on_span_error(
        self,
        context: ExecutionContext,
        *,
        error: BaseException,
    ) -> None:
        """No-op span error."""

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        context: ExecutionContext,
        unit: str = "",
    ) -> None:
        """No-op metric recording."""

    def log_event(
        self,
        name: str,
        *,
        context: ExecutionContext,
        data: Any = None,
    ) -> None:
        """No-op event logging."""


class CallbackBridgeProvider:
    """Bridge that wraps existing ``BaseCallbackHandler`` instances as an
    ``InstrumentationProvider``.

    This allows gradual migration from the callback-based instrumentation
    system to the new provider-based system.

    Example:

        >>> from langchain_core.callbacks.stdout import StdOutCallbackHandler
        >>> handler = StdOutCallbackHandler()
        >>> provider = CallbackBridgeProvider(handlers=[handler])
    """

    def __init__(self, handlers: list[Any] | None = None) -> None:
        """Initialize the bridge provider.

        Args:
            handlers: List of ``BaseCallbackHandler`` instances to delegate to.
        """
        self._handlers: list[Any] = list(handlers or [])

    def on_span_start(
        self,
        context: ExecutionContext,
        *,
        inputs: Any = None,
    ) -> None:
        """Translate span start to callback handler events.

        Maps to ``on_chain_start``, ``on_tool_start``, or ``on_llm_start``
        based on the context's ``run_type``.

        Args:
            context: The execution context for this span.
            inputs: The inputs to the span.
        """
        import uuid as uuid_mod  # noqa: PLC0415

        run_id = (
            uuid_mod.UUID(context.span_id)
            if ExecutionContext._is_valid_uuid(context.span_id)
            else uuid_mod.uuid4()
        )
        parent_run_id = (
            uuid_mod.UUID(context.parent_span_id)
            if context.parent_span_id
            and ExecutionContext._is_valid_uuid(context.parent_span_id)
            else None
        )

        event_name = {
            "chain": "on_chain_start",
            "llm": "on_llm_start",
            "tool": "on_tool_start",
            "retriever": "on_retriever_start",
        }.get(context.run_type, "on_chain_start")

        for handler in self._handlers:
            method = getattr(handler, event_name, None)
            if method is not None:
                try:
                    method(
                        {"name": context.name},
                        inputs,
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        tags=list(context.tags),
                        metadata=context.metadata,
                    )
                except Exception:
                    logger.warning(
                        "Error in callback handler %s.%s",
                        type(handler).__name__,
                        event_name,
                        exc_info=True,
                    )

    def on_span_end(
        self,
        context: ExecutionContext,
        *,
        outputs: Any = None,
    ) -> None:
        """Translate span end to callback handler events.

        Args:
            context: The execution context for this span.
            outputs: The outputs of the span.
        """
        import uuid as uuid_mod  # noqa: PLC0415

        run_id = (
            uuid_mod.UUID(context.span_id)
            if ExecutionContext._is_valid_uuid(context.span_id)
            else uuid_mod.uuid4()
        )

        event_name = {
            "chain": "on_chain_end",
            "llm": "on_llm_end",
            "tool": "on_tool_end",
            "retriever": "on_retriever_end",
        }.get(context.run_type, "on_chain_end")

        for handler in self._handlers:
            method = getattr(handler, event_name, None)
            if method is not None:
                try:
                    method(outputs, run_id=run_id)
                except Exception:
                    logger.warning(
                        "Error in callback handler %s.%s",
                        type(handler).__name__,
                        event_name,
                        exc_info=True,
                    )

    def on_span_error(
        self,
        context: ExecutionContext,
        *,
        error: BaseException,
    ) -> None:
        """Translate span error to callback handler events.

        Args:
            context: The execution context for this span.
            error: The exception that caused the failure.
        """
        import uuid as uuid_mod  # noqa: PLC0415

        run_id = (
            uuid_mod.UUID(context.span_id)
            if ExecutionContext._is_valid_uuid(context.span_id)
            else uuid_mod.uuid4()
        )

        event_name = {
            "chain": "on_chain_error",
            "llm": "on_llm_error",
            "tool": "on_tool_error",
            "retriever": "on_retriever_error",
        }.get(context.run_type, "on_chain_error")

        for handler in self._handlers:
            method = getattr(handler, event_name, None)
            if method is not None:
                try:
                    method(error, run_id=run_id)
                except Exception:
                    logger.warning(
                        "Error in callback handler %s.%s",
                        type(handler).__name__,
                        event_name,
                        exc_info=True,
                    )

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        context: ExecutionContext,
        unit: str = "",
    ) -> None:
        """No-op for callback bridge: callbacks have no metric concept.

        Args:
            name: The metric name.
            value: The metric value.
            context: The execution context.
            unit: Optional unit of measurement.
        """

    def log_event(
        self,
        name: str,
        *,
        context: ExecutionContext,
        data: Any = None,
    ) -> None:
        """Translate semantic events to ``on_text`` callback calls.

        Args:
            name: The event name.
            context: The execution context.
            data: Optional event payload.
        """
        import uuid as uuid_mod  # noqa: PLC0415

        run_id = (
            uuid_mod.UUID(context.span_id)
            if ExecutionContext._is_valid_uuid(context.span_id)
            else uuid_mod.uuid4()
        )

        text = f"[{name}] {data}" if data else f"[{name}]"
        for handler in self._handlers:
            method = getattr(handler, "on_text", None)
            if method is not None:
                try:
                    method(text, run_id=run_id)
                except Exception:
                    logger.warning(
                        "Error in callback handler %s.on_text",
                        type(handler).__name__,
                        exc_info=True,
                    )
