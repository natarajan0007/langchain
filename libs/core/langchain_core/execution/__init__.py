"""Execution context module for unified execution state management.

!!! warning

    This module is part of an architectural improvement initiative.
    APIs may change in future versions.

Provides a single `ExecutionContext` object that carries all execution state
(trace IDs, span IDs, tags, metadata) through the call stack, replacing the
fragmented context propagation across `RunnableConfig`, `CallbackManager`,
and `ContextVar` instances.
"""

from langchain_core.execution.context import ExecutionContext

__all__ = ["ExecutionContext"]
