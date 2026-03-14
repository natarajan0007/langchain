"""Tests for ExecutionContext."""

from __future__ import annotations

import uuid

import pytest

from langchain_core.execution.context import (
    ExecutionContext,
    get_current_context,
    var_execution_context,
)


class TestExecutionContextCreateRoot:
    """Tests for ExecutionContext.create_root()."""

    def test_create_root_defaults(self) -> None:
        """Root context should have sensible defaults."""
        ctx = ExecutionContext.create_root()
        assert ctx.name == "root"
        assert ctx.run_type == "chain"
        assert ctx.parent_span_id is None
        assert ctx.tags == ()
        assert ctx.metadata == {}
        assert ctx.trace_id == ctx.span_id  # Root: trace_id == span_id

    def test_create_root_with_explicit_values(self) -> None:
        """Root context should accept explicit values."""
        ctx = ExecutionContext.create_root(
            name="my_agent",
            run_type="agent",
            tags=["prod", "v2"],
            metadata={"user": "alice"},
            trace_id="explicit-trace-id",
        )
        assert ctx.name == "my_agent"
        assert ctx.run_type == "agent"
        assert ctx.trace_id == "explicit-trace-id"
        assert ctx.span_id == "explicit-trace-id"
        assert ctx.parent_span_id is None
        assert ctx.tags == ("prod", "v2")
        assert ctx.metadata == {"user": "alice"}

    def test_create_root_generates_unique_ids(self) -> None:
        """Each root context should have a unique trace_id."""
        ctx1 = ExecutionContext.create_root()
        ctx2 = ExecutionContext.create_root()
        assert ctx1.trace_id != ctx2.trace_id

    def test_create_root_empty_tags_metadata(self) -> None:
        """Empty tags and metadata should be handled correctly."""
        ctx = ExecutionContext.create_root(tags=[], metadata={})
        assert ctx.tags == ()
        assert ctx.metadata == {}


class TestExecutionContextCreateChild:
    """Tests for ExecutionContext.create_child()."""

    def test_child_inherits_trace_id(self) -> None:
        """Child should share parent's trace_id."""
        root = ExecutionContext.create_root(trace_id="trace-abc")
        child = root.create_child(name="child_step")
        assert child.trace_id == "trace-abc"

    def test_child_has_new_span_id(self) -> None:
        """Child should have a different span_id from parent."""
        root = ExecutionContext.create_root()
        child = root.create_child(name="child_step")
        assert child.span_id != root.span_id

    def test_child_parent_linkage(self) -> None:
        """Child's parent_span_id should equal parent's span_id."""
        root = ExecutionContext.create_root()
        child = root.create_child(name="child_step")
        assert child.parent_span_id == root.span_id

    def test_child_inherits_tags(self) -> None:
        """Child should inherit parent's tags."""
        root = ExecutionContext.create_root(tags=["tag1", "tag2"])
        child = root.create_child(name="child")
        assert child.tags == ("tag1", "tag2")

    def test_child_extends_tags(self) -> None:
        """Child should extend parent's tags with extra_tags."""
        root = ExecutionContext.create_root(tags=["tag1"])
        child = root.create_child(name="child", extra_tags=["tag2", "tag3"])
        assert child.tags == ("tag1", "tag2", "tag3")

    def test_child_inherits_metadata(self) -> None:
        """Child should inherit parent's metadata."""
        root = ExecutionContext.create_root(metadata={"key": "value"})
        child = root.create_child(name="child")
        assert child.metadata == {"key": "value"}

    def test_child_extends_metadata(self) -> None:
        """Child should merge extra_metadata with inherited metadata."""
        root = ExecutionContext.create_root(metadata={"a": 1})
        child = root.create_child(name="child", extra_metadata={"b": 2})
        assert child.metadata == {"a": 1, "b": 2}

    def test_child_metadata_override(self) -> None:
        """Child extra_metadata should override parent values for same keys."""
        root = ExecutionContext.create_root(metadata={"a": 1, "b": 2})
        child = root.create_child(name="child", extra_metadata={"b": 99})
        assert child.metadata == {"a": 1, "b": 99}

    def test_child_does_not_mutate_parent(self) -> None:
        """Creating a child should not mutate the parent context."""
        root = ExecutionContext.create_root(
            tags=["tag1"], metadata={"key": "value"}
        )
        root.create_child(
            name="child", extra_tags=["tag2"], extra_metadata={"new": "data"}
        )
        assert root.tags == ("tag1",)
        assert root.metadata == {"key": "value"}

    def test_child_run_type(self) -> None:
        """Child should accept its own run_type."""
        root = ExecutionContext.create_root()
        llm_child = root.create_child(name="llm_call", run_type="llm")
        tool_child = root.create_child(name="tool_call", run_type="tool")
        assert llm_child.run_type == "llm"
        assert tool_child.run_type == "tool"

    def test_grandchild_context(self) -> None:
        """Grandchild should chain parent relationships correctly."""
        root = ExecutionContext.create_root(trace_id="trace-1")
        child = root.create_child(name="agent")
        grandchild = child.create_child(name="tool")
        assert grandchild.trace_id == "trace-1"
        assert grandchild.parent_span_id == child.span_id
        assert child.parent_span_id == root.span_id

    def test_no_extra_tags_or_metadata(self) -> None:
        """Child with no extras should just inherit parent values."""
        root = ExecutionContext.create_root(tags=["x"], metadata={"y": 1})
        child = root.create_child(name="child")
        assert child.tags == ("x",)
        assert child.metadata == {"y": 1}


class TestExecutionContextBridge:
    """Tests for RunnableConfig bridge methods."""

    def test_from_runnable_config_basic(self) -> None:
        """Should create context from a RunnableConfig dict."""
        config = {
            "tags": ["tag1"],
            "metadata": {"key": "val"},
            "run_name": "my_chain",
            "run_id": uuid.uuid4(),
        }
        ctx = ExecutionContext.from_runnable_config(config)
        assert ctx.name == "my_chain"
        assert ctx.tags == ("tag1",)
        assert ctx.metadata == {"key": "val"}
        assert ctx.span_id == str(config["run_id"])

    def test_from_runnable_config_missing_fields(self) -> None:
        """Should handle missing optional fields gracefully."""
        ctx = ExecutionContext.from_runnable_config({})
        assert ctx.name == "unknown"
        assert ctx.tags == ()
        assert ctx.metadata == {}
        assert ctx.parent_span_id is None

    def test_to_runnable_config_roundtrip(self) -> None:
        """to_runnable_config should produce a compatible dict."""
        root = ExecutionContext.create_root(
            name="test",
            tags=["t1"],
            metadata={"m": 1},
        )
        config = root.to_runnable_config()
        assert config["tags"] == ["t1"]
        assert config["metadata"] == {"m": 1}
        assert config["run_name"] == "test"

    def test_to_runnable_config_with_valid_uuid(self) -> None:
        """Should include run_id when span_id is a valid UUID."""
        trace_id = str(uuid.uuid4())
        root = ExecutionContext.create_root(trace_id=trace_id)
        config = root.to_runnable_config()
        assert config["run_id"] == uuid.UUID(trace_id)

    def test_to_runnable_config_with_non_uuid(self) -> None:
        """Should set run_id to None when span_id is not a valid UUID."""
        root = ExecutionContext(
            trace_id="not-a-uuid",
            span_id="not-a-uuid",
            parent_span_id=None,
            run_type="chain",
            name="test",
            tags=(),
        )
        config = root.to_runnable_config()
        assert config["run_id"] is None


class TestExecutionContextAncestry:
    """Tests for ancestry property."""

    def test_root_ancestry(self) -> None:
        """Root context ancestry should be just itself."""
        root = ExecutionContext.create_root(trace_id="r1")
        assert root.ancestry == ["r1"]

    def test_child_ancestry(self) -> None:
        """Child ancestry should include parent and self."""
        root = ExecutionContext.create_root()
        child = root.create_child(name="child")
        assert child.ancestry == [root.span_id, child.span_id]


class TestExecutionContextImmutability:
    """Tests verifying that ExecutionContext is frozen."""

    def test_frozen_dataclass(self) -> None:
        """Should not allow attribute mutation."""
        ctx = ExecutionContext.create_root()
        with pytest.raises(AttributeError):
            ctx.name = "mutated"  # type: ignore[misc]

    def test_frozen_span_id(self) -> None:
        """Should not allow span_id mutation."""
        ctx = ExecutionContext.create_root()
        with pytest.raises(AttributeError):
            ctx.span_id = "mutated"  # type: ignore[misc]


class TestExecutionContextVar:
    """Tests for ContextVar-based propagation."""

    def test_default_is_none(self) -> None:
        """Default context should be None."""
        assert get_current_context() is None

    def test_set_and_get(self) -> None:
        """Should be able to set and get context via ContextVar."""
        ctx = ExecutionContext.create_root(name="test_var")
        token = var_execution_context.set(ctx)
        try:
            current = get_current_context()
            assert current is not None
            assert current.name == "test_var"
        finally:
            var_execution_context.reset(token)

    def test_reset_restores_none(self) -> None:
        """Resetting the ContextVar should restore None."""
        ctx = ExecutionContext.create_root()
        token = var_execution_context.set(ctx)
        var_execution_context.reset(token)
        assert get_current_context() is None


class TestIsValidUuid:
    """Tests for the UUID validation helper."""

    def test_valid_uuid(self) -> None:
        """Should return True for valid UUIDs."""
        assert ExecutionContext._is_valid_uuid(str(uuid.uuid4())) is True

    def test_invalid_uuid(self) -> None:
        """Should return False for invalid UUIDs."""
        assert ExecutionContext._is_valid_uuid("not-a-uuid") is False

    def test_empty_string(self) -> None:
        """Should return False for empty strings."""
        assert ExecutionContext._is_valid_uuid("") is False
