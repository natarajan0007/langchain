# LangChain Architecture Critique & Improvement Plan

**Author:** Architecture Review — AI-Assisted  
**Status:** Draft  
**Scope:** Agent orchestration, execution model, instrumentation surfaces  
**Timeline:** 3 weeks  
**Constraint:** ≤20% subsystem modification, no rewrites, no breaking changes

> **Disclaimer:** This document was produced with AI agent assistance as part of an
> architecture review process.

---

## Phase 1 — Repository Setup

### Environment Setup Commands

```bash
git clone https://github.com/langchain-ai/langchain.git
cd langchain/libs/core
pip install uv
uv sync --group test
uv run --group test pytest tests/unit_tests/ -x -q
```

### Key Directories

| Subsystem | Directory | Purpose |
|-----------|-----------|---------|
| Agent Orchestration | `libs/langchain/langchain_classic/agents/` | Agent types, executor, tool calling |
| Execution Runtime | `libs/core/langchain_core/runnables/` | Runnable ABC, composition, config |
| Instrumentation | `libs/core/langchain_core/callbacks/` | Callback handlers and managers |
| Tracing | `libs/core/langchain_core/tracers/` | BaseTracer, context, event streaming |
| Agent Schema | `libs/core/langchain_core/agents.py` | AgentAction, AgentFinish, AgentStep |

---

## Phase 2 — Codebase Architecture Map

### Repository Tree Summary

```
langchain/
├── libs/
│   ├── core/langchain_core/
│   │   ├── agents.py                    # AgentAction/AgentFinish schemas
│   │   ├── callbacks/
│   │   │   ├── base.py                  # BaseCallbackHandler, mixins (1,157 LOC)
│   │   │   └── manager.py              # CallbackManager hierarchy (2,697 LOC)
│   │   ├── runnables/
│   │   │   ├── base.py                  # Runnable ABC & composition (222KB)
│   │   │   ├── config.py               # RunnableConfig, context propagation
│   │   │   └── schema.py               # EventListener definitions
│   │   └── tracers/
│   │       ├── core.py                  # _TracerCore ABC (705 LOC)
│   │       ├── base.py                  # BaseTracer (937 LOC)
│   │       ├── context.py              # ContextVar-based tracer context
│   │       ├── event_stream.py         # Event streaming (1,100 LOC)
│   │       └── log_stream.py           # Log streaming (769 LOC)
│   └── langchain/langchain_classic/
│       └── agents/
│           ├── agent.py                 # AgentExecutor, BaseSingleActionAgent
│           ├── agent_iterator.py        # AgentExecutorIterator
│           └── tool_calling_agent/      # Modern tool-calling pattern
```

### Subsystem Details

#### 1. Agent Abstraction
- **Purpose:** Define agent loop (plan → act → observe → iterate)
- **Key modules:** `langchain_classic/agents/agent.py`, `langchain_core/agents.py`
- **Main classes:** `AgentExecutor`, `BaseSingleActionAgent`, `AgentAction`, `AgentFinish`
- **Dependencies:** Runnables (execution), Callbacks (instrumentation), Tools

#### 2. Execution Runtime
- **Purpose:** Composable execution primitives with sync/async/streaming support
- **Key modules:** `langchain_core/runnables/base.py`, `runnables/config.py`
- **Main classes:** `Runnable`, `RunnableSequence`, `RunnableParallel`, `RunnableConfig`
- **Dependencies:** Callbacks (via config propagation), Tracers (via callback hooks)

#### 3. Tool Calling System
- **Purpose:** Execute external tools with schema validation
- **Key modules:** `langchain_core/tools/`, `langchain_classic/agents/tool_calling_agent/`
- **Main classes:** `BaseTool`, `StructuredTool`, `ToolMessage`
- **Dependencies:** Callbacks (via `CallbackManagerForToolRun`)

#### 4. Callback / Instrumentation System
- **Purpose:** Event hooks for LLM, chain, tool, and retriever lifecycle
- **Key modules:** `langchain_core/callbacks/manager.py`, `callbacks/base.py`
- **Main classes:** `CallbackManager`, `BaseCallbackHandler`, `*RunManager` (8 variants)
- **Dependencies:** Tracers (as callback handler implementations)

#### 5. Memory / State Propagation
- **Purpose:** Propagate execution context (tags, metadata, run IDs) through call stack
- **Key modules:** `langchain_core/runnables/config.py`, `tracers/context.py`
- **Main classes:** `RunnableConfig`, `var_child_runnable_config` (ContextVar)
- **Dependencies:** Callbacks, Tracers

---

## Phase 3 — Architectural Problems

### Issue #1: No Unified Execution Context

**Symptom:** There is no single object representing the current execution context.
Context is spread across `RunnableConfig`, `CallbackManager`, `ContextVar`s, and
`run_map` dictionaries.

**Code locations:**
- `langchain_core/runnables/config.py:49-121` — `RunnableConfig` (TypedDict)
- `langchain_core/runnables/config.py:144-146` — `var_child_runnable_config` (ContextVar)
- `langchain_core/callbacks/manager.py:564-583` — `ParentRunManager.get_child()`
- `langchain_core/tracers/context.py:31-36` — separate ContextVars for tracers

**Root Cause:** Execution metadata (trace IDs, parent relationships, tags) is threaded
through multiple independent mechanisms with no single source of truth.

**Impact:** Makes it impossible to reliably inspect the full execution state at any
point. Custom instrumentation must understand 3+ context propagation systems.

**Example:** A tool executing within an agent cannot easily discover its full ancestry
(which agent, which chain, which user request) without traversing callback manager
parent pointers and tracer run maps independently.

### Issue #2: Callback Manager Type Explosion

**Symptom:** 16+ specialized `*RunManager` and `*CallbackManager` classes with
overlapping responsibilities.

**Code locations:**
- `langchain_core/callbacks/manager.py` — `CallbackManagerForLLMRun`,
  `CallbackManagerForChainRun`, `CallbackManagerForToolRun`,
  `CallbackManagerForRetrieverRun`, plus async variants (8 total run managers)
- `langchain_core/callbacks/base.py` — `LLMManagerMixin`, `ChainManagerMixin`,
  `ToolManagerMixin`, `RetrieverManagerMixin` (4 mixins)

**Root Cause:** Each component type (LLM, Chain, Tool, Retriever) gets its own
callback manager subclass instead of using a generic execution context with typed
events.

**Impact:** Adding a new component type requires creating 2+ new classes (sync + async
run manager) and updating the mixin hierarchy. The 2,697-line `manager.py` is
difficult to maintain.

### Issue #3: Dual Sync/Async Implementation Duplication

**Symptom:** Every callback manager and run manager has both sync and async variants
with nearly identical logic.

**Code locations:**
- `langchain_core/callbacks/manager.py:486-562` — `RunManager` (sync)
- `langchain_core/callbacks/manager.py:586-700` — `AsyncRunManager` (async, ~same code)
- `langchain_core/callbacks/manager.py:1367-1453` — `CallbackManager.on_chain_start`
- `langchain_core/callbacks/manager.py:1700-1800` — `AsyncCallbackManager.on_chain_start`

**Root Cause:** Python's sync/async divide forces separate implementations. No shared
abstraction or code generation strategy is used.

**Impact:** Bug fixes must be applied twice. New features require double implementation.
Risk of sync/async behavior divergence.

### Issue #4: Implicit Context Propagation Fragility

**Symptom:** `var_child_runnable_config` ContextVar silently drops context in
thread-pool and multi-process scenarios.

**Code locations:**
- `langchain_core/runnables/config.py:144-146` — `var_child_runnable_config`
- `langchain_core/runnables/config.py:225-275` — `ensure_config()` merges ContextVar
- `langchain_core/runnables/config.py:150-190` — `_set_config_context()` sets both
  config and LangSmith tracing context

**Root Cause:** ContextVar propagation is automatic within a single async context but
does not automatically propagate to `ThreadPoolExecutor` workers or subprocess
boundaries.

**Impact:** When `RunnableParallel` fans out work to threads, child runnables may lose
parent context unless `copy_context()` is explicitly called. This creates silent
tracing gaps.

### Issue #5: Tracer and Callback Handler Conflation

**Symptom:** `BaseTracer` is a `BaseCallbackHandler` subclass, mixing observation
(tracing) with control flow (callbacks).

**Code locations:**
- `langchain_core/tracers/base.py:1-20` — `BaseTracer(BaseCallbackHandler)`
- `langchain_core/tracers/core.py:40-75` — `_TracerCore` internal state
- `langchain_core/callbacks/base.py:455-507` — `BaseCallbackHandler` definition

**Root Cause:** The callback system was designed for both control flow hooks (e.g.,
streaming) and observation (tracing). Tracers were added as callback handler
implementations rather than as a separate system.

**Impact:** Tracers receive all callback events even when they only care about a
subset. A tracer bug can affect callback-driven control flow (streaming, etc.).

### Issue #6: `manager.py` God Module

**Symptom:** `callbacks/manager.py` is 2,697 lines with 16+ classes handling event
dispatch, context management, handler lifecycle, and error handling.

**Code locations:**
- `langchain_core/callbacks/manager.py` — full file

**Root Cause:** Organic growth without module boundaries. Each new component type
(retriever, tool, etc.) added more classes to the same file.

**Impact:** Difficult to navigate, test in isolation, or understand dependency flow.
High cognitive load for contributors.

### Issue #7: Run ID Generation Not Centralized

**Symptom:** Run IDs are generated at multiple entry points with different strategies.

**Code locations:**
- `langchain_core/callbacks/manager.py:1429-1430` — `uuid7()` in `on_chain_start`
- `langchain_core/callbacks/manager.py:1461` — `uuid7()` in `on_tool_start`
- `langchain_core/runnables/config.py:116-120` — optional `run_id` in config
- `langchain_core/runnables/base.py:2053` — `config.pop("run_id", None)` consumption

**Root Cause:** No central execution context factory. Each `on_*_start` method
independently generates IDs.

**Impact:** Difficult to ensure consistent ID generation strategy across all component
types. The `config.pop("run_id")` pattern is mutation-based and fragile.

---

## Phase 4 — Reproducible Issue: Missing Execution Context

See `docs/architecture/repro_context_issue.py` for a runnable script demonstrating
that tools executing within a chain cannot discover their full execution ancestry
through a single API.

### How to Run

```bash
cd libs/core
uv run python docs/architecture/repro_context_issue.py
```

### Expected Behavior
A tool should be able to discover its complete execution context (trace ID, parent
chain, tags, metadata) through a single, unified API.

### Actual Behavior
Context is fragmented across:
1. `RunnableConfig` (tags, metadata)
2. `CallbackManagerForToolRun` (run_id, parent_run_id, handlers)
3. `var_child_runnable_config` ContextVar (implicit propagation)
4. Tracer `run_map` (hierarchical run tree)

A tool must understand all four systems to reconstruct its execution context.

---

## Phase 5 — Target Architecture Proposal

### Design Principles

1. **Single execution context object** — One `ExecutionContext` carries all state
2. **Observation separate from control** — Instrumentation is a read-only layer
3. **Explicit over implicit** — Context is passed explicitly, with ContextVar as fallback
4. **Extension by composition** — New component types don't require new manager classes
5. **Dependency direction:** Runtime → Context → Instrumentation (never reverse)

### Proposed Codebase Layout (New Modules)

```
langchain_core/
├── execution/                          # NEW: Unified execution layer
│   ├── __init__.py
│   └── context.py                      # ExecutionContext definition
├── instrumentation/                    # NEW: First-class instrumentation
│   ├── __init__.py
│   └── provider.py                     # InstrumentationProvider protocol
├── callbacks/                          # EXISTING: Unchanged public API
│   ├── base.py
│   └── manager.py                      # Internal: delegates to execution/
├── runnables/                          # EXISTING: Uses ExecutionContext
│   ├── base.py
│   └── config.py                       # Bridge: RunnableConfig ↔ ExecutionContext
└── tracers/                            # EXISTING: Implements InstrumentationProvider
    ├── base.py
    └── core.py
```

### Ownership Boundaries

| Module | Owner | Stability |
|--------|-------|-----------|
| `execution/context.py` | Core team | Public stable API |
| `instrumentation/provider.py` | Core team | Public stable API |
| `callbacks/manager.py` | Core team | Public (legacy compat) |
| `runnables/config.py` | Core team | Public stable API |

---

## Phase 6 — Instrumentation Redesign

### Unified Observability Layer

The `InstrumentationProvider` protocol defines a component-type-agnostic interface
for observation. It follows OpenTelemetry-compatible patterns:

```
┌─────────────────────────────────────────────────┐
│              Execution Runtime                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Runnable  │  │  Agent   │  │   Tool   │      │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘      │
│        └──────────────┼──────────────┘           │
│                       │                          │
│              ┌────────▼────────┐                 │
│              │ExecutionContext  │                 │
│              │  trace_id       │                 │
│              │  span_id        │                 │
│              │  parent_span_id │                 │
│              │  tags, metadata │                 │
│              └────────┬────────┘                 │
│                       │                          │
│        ┌──────────────┼──────────────┐           │
│        ▼              ▼              ▼           │
│   ┌─────────┐  ┌──────────┐  ┌──────────┐      │
│   │ Tracing  │  │ Metrics  │  │  Events  │      │
│   │ Provider │  │ Provider │  │ Provider │      │
│   └─────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────┘
```

### Context Propagation Model

```
User Request
    │
    ├─► ExecutionContext.create_root(trace_id=T1)
    │       │
    │       ├─► Agent.invoke() → ctx.create_child(span_id=S1)
    │       │       │
    │       │       ├─► LLM.invoke() → ctx.create_child(span_id=S2)
    │       │       │       └─► provider.on_span_end(S2)
    │       │       │
    │       │       ├─► Tool.invoke() → ctx.create_child(span_id=S3)
    │       │       │       └─► provider.on_span_end(S3)
    │       │       │
    │       │       └─► provider.on_span_end(S1)
    │       │
    │       └─► provider.on_span_end(root)
```

---

## Phase 7 — Interface Definitions

See `langchain_core/execution/context.py` and
`langchain_core/instrumentation/provider.py` for full implementations.

---

## Phase 8 — Before vs After Execution Flow

### BEFORE Architecture

```
User Request
    │
    ▼
AgentExecutor.invoke(input, config={callbacks=[...]})
    │
    ├─► ensure_config(config)                    # Merge with ContextVar
    ├─► get_callback_manager_for_config(config)  # Create CallbackManager
    ├─► callback_manager.on_chain_start(...)     # Generate run_id, notify handlers
    │       │
    │       ├─► handle_event(handlers, "on_chain_start", ...)  # Fan out to N handlers
    │       └─► return CallbackManagerForChainRun(run_id, parent_run_id, handlers)
    │
    ├─► run_manager.get_child()                  # Create child CallbackManager
    ├─► patch_config(config, callbacks=child_mgr)# Mutate config with child manager
    ├─► set_config_context(child_config)         # Set ContextVar + LangSmith context
    │       │
    │       ├─► Tool.invoke(input, child_config)
    │       │       ├─► ensure_config(child_config)
    │       │       ├─► get_callback_manager_for_config(...)
    │       │       ├─► on_tool_start(...)       # Another run_id generated
    │       │       └─► on_tool_end(...)
    │       │
    │       └─► LLM.invoke(input, child_config)
    │               ├─► ensure_config(child_config)
    │               └─► on_llm_start(...)        # Yet another run_id generated
    │
    └─► run_manager.on_chain_end(output)         # Notify all handlers of completion
```

### AFTER Architecture (Proposed)

```
User Request
    │
    ▼
Runtime.invoke(input, config)
    │
    ├─► ExecutionContext.create_root(trace_id, tags, metadata)
    │       │
    │       ├─► instrumentation.on_span_start(ctx, inputs)
    │       │
    │       ├─► Agent step
    │       │       │
    │       │       ├─► child_ctx = ctx.create_child(name="llm_call")
    │       │       │       ├─► instrumentation.on_span_start(child_ctx, ...)
    │       │       │       └─► instrumentation.on_span_end(child_ctx, ...)
    │       │       │
    │       │       ├─► child_ctx = ctx.create_child(name="tool_call")
    │       │       │       ├─► instrumentation.on_span_start(child_ctx, ...)
    │       │       │       └─► instrumentation.on_span_end(child_ctx, ...)
    │       │       │
    │       │       └─► (iterate until AgentFinish)
    │       │
    │       └─► instrumentation.on_span_end(ctx, outputs)
    │
    └─► return output
```

---

## Phase 9 — 3-Week Implementation Plan

### Week 1: Central Execution Context

**Goal:** Ship `ExecutionContext` as a new module without modifying existing APIs.

**Files created:**
- `langchain_core/execution/__init__.py`
- `langchain_core/execution/context.py`

**Validation:**
- Unit tests in `tests/unit_tests/execution/test_context.py`
- Verify existing tests still pass

### Week 2: Instrumentation Provider Protocol

**Goal:** Ship `InstrumentationProvider` protocol and callback bridge adapter.

**Files created:**
- `langchain_core/instrumentation/__init__.py`
- `langchain_core/instrumentation/provider.py`

**Validation:**
- Unit tests in `tests/unit_tests/instrumentation/test_provider.py`
- Integration test: callback handler wrapped as InstrumentationProvider

### Week 3: Migration Compatibility Layer

**Goal:** Bridge adapters so existing `callbacks=[...]` usage works unchanged.

**Validation:**
- Existing callback-based tests pass unmodified
- New instrumentation tests pass

---

## Phase 10 — Migration Strategy

### Non-Breaking Rollout

**Old API (continues to work):**
```python
chain.invoke(input, config={"callbacks": [MyHandler()]})
```

**New API (opt-in):**
```python
from langchain_core.execution.context import ExecutionContext
from langchain_core.instrumentation.provider import InstrumentationProvider

ctx = ExecutionContext.create_root(tags=["production"])
chain.invoke(input, config=ctx.to_runnable_config())
```

### Compatibility Adapters

The `CallbackBridgeProvider` wraps existing `BaseCallbackHandler` instances as
`InstrumentationProvider` implementations, and vice versa.

---

## Phase 11 — Test Strategy

```bash
# Unit tests for execution context
pytest tests/unit_tests/execution/test_context.py

# Unit tests for instrumentation provider
pytest tests/unit_tests/instrumentation/test_provider.py

# Verify existing tests unaffected
pytest tests/unit_tests/tracers/ tests/unit_tests/runnables/ tests/unit_tests/callbacks/
```

---

## Phase 12 — What We Are Not Fixing

| Deferred Issue | Reason |
|---------------|--------|
| Sync/async duplication in manager.py | Requires major refactor; 3-week constraint |
| manager.py module splitting | High risk of breaking imports; needs careful deprecation |
| Runnable base.py (222KB) decomposition | Out of scope; separate initiative |
| Agent pattern consolidation | Owned by langchain package, not core |
| LangSmith-specific tracer coupling | Requires coordination with LangSmith team |

---

## Phase 13 — AI-Assisted Development Plan

### Safe Tasks for AI Agents

| Task | AI Role | Human Review |
|------|---------|--------------|
| Test generation | Generate unit tests from interface specs | Review coverage |
| Instrumentation injection | Add `on_span_start`/`on_span_end` calls | Review correctness |
| Refactoring assistance | Extract methods from god modules | Review boundaries |
| Static analysis | Run mypy/ruff on changed files | Review suppressions |
| Migration adapters | Generate bridge code from type signatures | Review semantics |
| Documentation | Generate docstrings from code analysis | Review accuracy |
