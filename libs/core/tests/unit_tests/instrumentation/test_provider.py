"""Tests for InstrumentationProvider and related classes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_core.execution.context import ExecutionContext
from langchain_core.instrumentation.provider import (
    CallbackBridgeProvider,
    InstrumentationProvider,
    NoOpProvider,
)


class TestNoOpProvider:
    """Tests for the NoOpProvider."""

    def test_noop_is_instrumentation_provider(self) -> None:
        """NoOpProvider should satisfy the InstrumentationProvider protocol."""
        provider = NoOpProvider()
        assert isinstance(provider, InstrumentationProvider)

    def test_noop_span_lifecycle(self) -> None:
        """NoOpProvider should accept all span lifecycle events silently."""
        provider = NoOpProvider()
        ctx = ExecutionContext.create_root(name="test")

        # Should not raise
        provider.on_span_start(ctx, inputs={"x": 1})
        provider.on_span_end(ctx, outputs={"y": 2})
        provider.on_span_error(ctx, error=RuntimeError("test"))

    def test_noop_metric(self) -> None:
        """NoOpProvider should accept metrics silently."""
        provider = NoOpProvider()
        ctx = ExecutionContext.create_root()
        provider.record_metric("latency", 100.0, context=ctx, unit="ms")

    def test_noop_event(self) -> None:
        """NoOpProvider should accept events silently."""
        provider = NoOpProvider()
        ctx = ExecutionContext.create_root()
        provider.log_event("tool_selected", context=ctx, data={"tool": "search"})


class RecordingProvider:
    """Test helper that records all instrumentation events."""

    def __init__(self) -> None:
        """Initialize with empty event lists."""
        self.spans_started: list[tuple[ExecutionContext, Any]] = []
        self.spans_ended: list[tuple[ExecutionContext, Any]] = []
        self.spans_errored: list[tuple[ExecutionContext, BaseException]] = []
        self.metrics: list[tuple[str, float, ExecutionContext]] = []
        self.events: list[tuple[str, ExecutionContext, Any]] = []

    def on_span_start(
        self,
        context: ExecutionContext,
        *,
        inputs: Any = None,
    ) -> None:
        """Record span start."""
        self.spans_started.append((context, inputs))

    def on_span_end(
        self,
        context: ExecutionContext,
        *,
        outputs: Any = None,
    ) -> None:
        """Record span end."""
        self.spans_ended.append((context, outputs))

    def on_span_error(
        self,
        context: ExecutionContext,
        *,
        error: BaseException,
    ) -> None:
        """Record span error."""
        self.spans_errored.append((context, error))

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        context: ExecutionContext,
        unit: str = "",
    ) -> None:
        """Record metric."""
        self.metrics.append((name, value, context))

    def log_event(
        self,
        name: str,
        *,
        context: ExecutionContext,
        data: Any = None,
    ) -> None:
        """Record event."""
        self.events.append((name, context, data))


class TestRecordingProvider:
    """Tests using the RecordingProvider to verify provider behavior."""

    def test_recording_is_instrumentation_provider(self) -> None:
        """RecordingProvider should satisfy the protocol."""
        provider = RecordingProvider()
        assert isinstance(provider, InstrumentationProvider)

    def test_span_lifecycle_recording(self) -> None:
        """Should record complete span lifecycle."""
        provider = RecordingProvider()
        ctx = ExecutionContext.create_root(name="test_chain")

        provider.on_span_start(ctx, inputs={"query": "hello"})
        provider.on_span_end(ctx, outputs={"result": "world"})

        assert len(provider.spans_started) == 1
        assert len(provider.spans_ended) == 1
        assert provider.spans_started[0] == (ctx, {"query": "hello"})
        assert provider.spans_ended[0] == (ctx, {"result": "world"})

    def test_span_error_recording(self) -> None:
        """Should record span errors."""
        provider = RecordingProvider()
        ctx = ExecutionContext.create_root()
        err = ValueError("test error")

        provider.on_span_start(ctx, inputs="x")
        provider.on_span_error(ctx, error=err)

        assert len(provider.spans_errored) == 1
        assert provider.spans_errored[0] == (ctx, err)

    def test_nested_span_recording(self) -> None:
        """Should correctly record nested span relationships."""
        provider = RecordingProvider()
        root = ExecutionContext.create_root(name="agent")
        child = root.create_child(name="tool", run_type="tool")

        provider.on_span_start(root, inputs="input")
        provider.on_span_start(child, inputs="tool_input")
        provider.on_span_end(child, outputs="tool_output")
        provider.on_span_end(root, outputs="final_output")

        assert len(provider.spans_started) == 2
        assert len(provider.spans_ended) == 2
        # Verify parent-child relationship
        assert provider.spans_started[1][0].parent_span_id == root.span_id

    def test_metric_recording(self) -> None:
        """Should record metrics with context."""
        provider = RecordingProvider()
        ctx = ExecutionContext.create_root()

        provider.record_metric("token_count", 150.0, context=ctx, unit="tokens")

        assert len(provider.metrics) == 1
        assert provider.metrics[0] == ("token_count", 150.0, ctx)

    def test_event_recording(self) -> None:
        """Should record semantic events."""
        provider = RecordingProvider()
        ctx = ExecutionContext.create_root()

        provider.log_event(
            "tool_selected", context=ctx, data={"tool": "calculator"}
        )

        assert len(provider.events) == 1
        assert provider.events[0] == ("tool_selected", ctx, {"tool": "calculator"})


class TestCallbackBridgeProvider:
    """Tests for the CallbackBridgeProvider migration adapter."""

    def test_bridge_is_instrumentation_provider(self) -> None:
        """CallbackBridgeProvider should satisfy the protocol."""
        bridge = CallbackBridgeProvider()
        assert isinstance(bridge, InstrumentationProvider)

    def test_bridge_on_span_start_chain(self) -> None:
        """Bridge should map chain span_start to on_chain_start."""
        handler = MagicMock()
        bridge = CallbackBridgeProvider(handlers=[handler])
        ctx = ExecutionContext.create_root(name="my_chain", run_type="chain")

        bridge.on_span_start(ctx, inputs={"query": "hello"})

        handler.on_chain_start.assert_called_once()
        call_args = handler.on_chain_start.call_args
        assert call_args[0][0] == {"name": "my_chain"}
        assert call_args[0][1] == {"query": "hello"}

    def test_bridge_on_span_start_tool(self) -> None:
        """Bridge should map tool span_start to on_tool_start."""
        handler = MagicMock()
        bridge = CallbackBridgeProvider(handlers=[handler])
        ctx = ExecutionContext.create_root(name="search", run_type="tool")

        bridge.on_span_start(ctx, inputs="search query")

        handler.on_tool_start.assert_called_once()

    def test_bridge_on_span_start_llm(self) -> None:
        """Bridge should map llm span_start to on_llm_start."""
        handler = MagicMock()
        bridge = CallbackBridgeProvider(handlers=[handler])
        ctx = ExecutionContext.create_root(name="gpt-4", run_type="llm")

        bridge.on_span_start(ctx, inputs="prompt")

        handler.on_llm_start.assert_called_once()

    def test_bridge_on_span_end_chain(self) -> None:
        """Bridge should map chain span_end to on_chain_end."""
        handler = MagicMock()
        bridge = CallbackBridgeProvider(handlers=[handler])
        ctx = ExecutionContext.create_root(name="chain", run_type="chain")

        bridge.on_span_end(ctx, outputs={"result": "done"})

        handler.on_chain_end.assert_called_once()

    def test_bridge_on_span_error(self) -> None:
        """Bridge should map span_error to on_chain_error."""
        handler = MagicMock()
        bridge = CallbackBridgeProvider(handlers=[handler])
        ctx = ExecutionContext.create_root(run_type="chain")
        err = RuntimeError("test")

        bridge.on_span_error(ctx, error=err)

        handler.on_chain_error.assert_called_once()

    def test_bridge_handler_error_is_caught(self) -> None:
        """Bridge should catch and log handler errors without re-raising."""
        handler = MagicMock()
        handler.on_chain_start.side_effect = RuntimeError("handler broke")
        bridge = CallbackBridgeProvider(handlers=[handler])
        ctx = ExecutionContext.create_root()

        # Should not raise
        bridge.on_span_start(ctx, inputs="x")

    def test_bridge_multiple_handlers(self) -> None:
        """Bridge should notify all handlers."""
        h1 = MagicMock()
        h2 = MagicMock()
        bridge = CallbackBridgeProvider(handlers=[h1, h2])
        ctx = ExecutionContext.create_root()

        bridge.on_span_start(ctx, inputs="x")

        h1.on_chain_start.assert_called_once()
        h2.on_chain_start.assert_called_once()

    def test_bridge_empty_handlers(self) -> None:
        """Bridge with no handlers should be a no-op."""
        bridge = CallbackBridgeProvider(handlers=[])
        ctx = ExecutionContext.create_root()
        # Should not raise
        bridge.on_span_start(ctx, inputs="x")
        bridge.on_span_end(ctx, outputs="y")
        bridge.on_span_error(ctx, error=RuntimeError("z"))

    def test_bridge_log_event(self) -> None:
        """Bridge should translate log_event to on_text."""
        handler = MagicMock()
        bridge = CallbackBridgeProvider(handlers=[handler])
        ctx = ExecutionContext.create_root()

        bridge.log_event("tool_selected", context=ctx, data={"tool": "calc"})

        handler.on_text.assert_called_once()
        call_args = handler.on_text.call_args
        assert "[tool_selected]" in call_args[0][0]

    def test_bridge_record_metric_is_noop(self) -> None:
        """Bridge record_metric should be a no-op (callbacks lack metrics)."""
        handler = MagicMock()
        bridge = CallbackBridgeProvider(handlers=[handler])
        ctx = ExecutionContext.create_root()

        bridge.record_metric("latency", 100.0, context=ctx)
        # No method should have been called on the handler for metrics
        handler.on_chain_start.assert_not_called()

    def test_bridge_parent_run_id_propagation(self) -> None:
        """Bridge should propagate parent_run_id from context."""
        handler = MagicMock()
        bridge = CallbackBridgeProvider(handlers=[handler])
        root = ExecutionContext.create_root()
        child = root.create_child(name="child", run_type="chain")

        bridge.on_span_start(child, inputs="x")

        call_kwargs = handler.on_chain_start.call_args[1]
        assert call_kwargs["parent_run_id"] is not None
