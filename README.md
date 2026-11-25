# 🤖 Multi-Agent AI Workflow System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/persistence-PostgreSQL-336791.svg)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/cache-Redis-red.svg)](https://redis.io/)
[![MCP](https://img.shields.io/badge/protocol-MCP-purple.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade multi-agent autonomous workflow system orchestrated with **LangGraph** and powered by **OpenAI GPT-4o**. Specialized agents collaborate to decompose complex goals, execute sandboxed code, conduct live web research, query and mutate relational databases, and dynamically evaluate outcomes with strict safety guardrails and human-in-the-loop checkpoints.

---

## 📐 Architecture

The workflow implements a stateful directed acyclic and cyclic graph orchestrated via LangGraph. An incoming request is parsed by a safety guard, decomposed into ordered subtasks by a Planner agent, and dispatched via a dynamic router to specialized worker agents equipped with Model Context Protocol (MCP) tool servers. An Evaluator agent validates outputs and dynamically triggers retries, replans, or completes the workflow.

```text
                           +------------------------+
                           |      User / Client     |
                           +------------------------+
                                       |
                                       v
                    +--------------------------------------+
                    |          FastAPI REST API            |
                    |  /run  |  /status  |  /approve       |
                    |  /reject  |  /trace  |  /stream (WS) |
                    +--------------------------------------+
                                       |
                                       v
                    +--------------------------------------+
                    |       LangGraph Workflow Engine      |
                    |                                      |
                    |               [ START ]              |
                    |                   |                  |
                    |                   v                  |
                    |             +-----------+            |
                    |             |   Guard   | (Step, Budget, Loop Checks)
                    |             +-----------+            |
                    |                   |                  |
                    |                   v                  |
                    |             +-----------+            |
                    |             |  Planner  |<-------+   |
                    |             +-----------+        |   |
                    |                   |              |   |
                    |                   v              |   |
                    |             +-----------+        |   |
                    |     +------>|  Router   |        |   |
                    |     |       +-----------+        |   |
                    |     |         |   |   |          |   |
                    |     |    +----+   |   +----+     |   |
                    |     |    |        |        |     |   |
                    |     |    v        v        v     |   |
                    |     | Researcher Coder DB Analyst|   |
                    |     |    |        |        |     |   |
                    |     |    v        v        v     |   |
                    |     | [Search] [Code]   [DB]     |   |
                    |     | [Tools ] [Tools]  [Tools*] |   |
                    |     |    |        |        |     |   |
                    |     |    +----+   |   +----+     |   |
                    |     |         |   |   |          |   |
                    |     |         v   v   v          |   |
                    |     |       +-----------+        |   |
                    |     |       | Evaluator |        |   |
                    |     |       +-----------+        |   |
                    |     |             |              |   |
                    |     |   (retry)   | (replan)     |   |
                    |     +-------------+--------------+   |
                    |                   |                  |
                    |              (complete)              |
                    |                   v                  |
                    |               [ __END__ ]            |
                    +--------------------------------------+
                         |                            |
                         v                            v
            +------------------------+   +------------------------+
            |  PostgreSQL Checkpoint |   |  Redis Cache & Telemetry|
            | - State Snapshots      |   | - Result Caching (TTL) |
            | - Thread Persistence   |   | - Token & Cost Budgets |
            | - Human-in-the-Loop    |   | - Redis Stream Traces  |
            +------------------------+   +------------------------+

  * Human approval checkpoint (interrupt_before) enforced for destructive write_db operations.
```

---

## ✨ Features

- **Specialized Multi-Agent Architecture**: Five distinct agent roles (Planner, Researcher, Coder, DB Analyst, Evaluator) with domain-specific system prompts and scoped tool access.
- **Dynamic LangGraph Orchestration**: Stateful graph execution with conditional routing, automated retries, dynamic replanning, and state reducers for append-only trace logging and result aggregation.
- **Model Context Protocol (MCP) Tool Servers**: Standardized FastMCP servers providing isolated Python code execution, Ruff-based static linting, Tavily web searching, URL scraping, and PostgreSQL querying.
- **Enterprise Safety Guardrails**: Multi-tier safeguards checking execution step limits, cumulative token consumption, dollar cost thresholds, and conversational loop detection.
- **Human-in-the-Loop (HITL) Checkpoints**: LangGraph state interrupts (`interrupt_before`) pause execution when destructive tools (such as database write operations) are scheduled, resuming only upon explicit human approval.
- **Durable State Persistence**: PostgreSQL checkpointer (`AsyncPostgresSaver`) records graph snapshots at every step, enabling fault recovery, workflow inspection, and resumption across server restarts.
- **Redis Caching & Real-Time Telemetry**: Redis-backed tool result caching with TTL, atomic token/cost budget counters, and Redis Streams (`xadd`) for distributed trace log auditing.
- **Full-Featured REST & WebSocket API**: Built with FastAPI, offering endpoints for workflow submission, status inspection, manual approval/rejection, trace retrieval, and real-time WebSocket state streaming.
- **Structured Error Handling**: All tools implement the standardized `ToolResult` pattern — tools never throw raw exceptions to the graph, returning structured JSON diagnostic outputs instead.

---

## 📋 Prerequisites

Before getting started, ensure you have the following installed:

- **Python 3.11+**: Required for modern typing, dataclasses, and pattern matching.
- **Docker & Docker Compose**: To launch containerized PostgreSQL and Redis services.
- **OpenAI API Key**: Required for GPT-4o LLM agent inference.
- **Tavily API Key**: Required for web search capabilities (obtainable at [tavily.com](https://tavily.com)).

---

## 🚀 Quick Start Guide

### 1. Clone & Install Dependencies

Clone the repository and install the project in editable mode with development dependencies:

```bash
git clone https://github.com/your-org/multi-agent.git
cd multi-agent

# Create and activate a virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install package with all dependencies
pip install -e ".[dev]"
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and populate your credentials:

```bash
# On Linux / macOS:
cp .env.example .env

# On Windows PowerShell:
Copy-Item .env.example .env
```

Ensure the following variables are configured in `.env`:

```env
# LLM Provider
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-4o

# Search Tools
TAVILY_API_KEY=tvly-your-tavily-api-key

# PostgreSQL Persistence
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/multi_agent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=multi_agent

# Redis Cache & Telemetry
REDIS_URL=redis://localhost:6379/0

# Safety Guardrail Limits
MAX_STEPS=50
MAX_TOKENS=100000
MAX_COST_USD=5.00

# API Server Host & Port
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Start Infrastructure Services

Spin up PostgreSQL (with persistent volumes and healthchecks) and Redis using Docker Compose:

```bash
docker-compose up -d postgres redis
```

Verify that the containers are healthy:

```bash
docker-compose ps
```

### 4. Run the API Server

Start the FastAPI application:

```bash
python -m src.api.main
```

Alternatively, launch using Uvicorn directly:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive OpenAPI documentation will be accessible at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 5. Test It: cURL Examples

#### A. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```
*Sample Response:*
```json
{
  "status": "healthy",
  "components": {
    "api": "up",
    "info": "Use /workflow/run to start a workflow"
  }
}
```

#### B. Start a New Workflow
```bash
curl -X POST "http://localhost:8000/workflow/run" \
     -H "Content-Type: application/json" \
     -d '{
       "goal": "Query the top sales items from the database, compute total revenue with python, and summarize the findings.",
       "max_steps": 50,
       "max_tokens": 100000,
       "max_cost_usd": 5.0
     }'
```
*Sample Response:*
```json
{
  "run_id": "8f3b7d12-6f23-4e89-a21c-99c0e5a8bc34",
  "status": "running",
  "message": "Workflow started for goal: Query the top sales items..."
}
```

#### C. Check Workflow Status
```bash
curl -X GET "http://localhost:8000/workflow/8f3b7d12-6f23-4e89-a21c-99c0e5a8bc34/status"
```
*Sample Response:*
```json
{
  "run_id": "8f3b7d12-6f23-4e89-a21c-99c0e5a8bc34",
  "goal": "Query the top sales items from the database, compute total revenue with python, and summarize the findings.",
  "status": "running",
  "current_task_index": 1,
  "total_tasks": 3,
  "step_count": 4,
  "budget": {
    "total_tokens": 1420,
    "prompt_tokens": 1150,
    "completion_tokens": 270,
    "total_cost_usd": 0.005575,
    "total_steps": 4,
    "tool_calls": 1
  },
  "error": null,
  "plan": [
    {
      "id": "task_1",
      "description": "Query sample_data table for sales records",
      "assigned_agent": "db_analyst",
      "requires_approval": false,
      "status": "completed"
    },
    {
      "id": "task_2",
      "description": "Calculate total revenue and average sale in Python",
      "assigned_agent": "coder",
      "requires_approval": false,
      "status": "in_progress"
    }
  ],
  "results": {
    "task_1": "Retrieved 5 sales records from database."
  }
}
```

#### D. Stream Live Workflow Updates (WebSocket)
Connect via `wscat` or any WebSocket client:
```bash
# Install wscat if not present: npm install -g wscat
wscat -c "ws://localhost:8000/workflow/8f3b7d12-6f23-4e89-a21c-99c0e5a8bc34/stream"
```
*Stream Messages:*
```json
{"run_id":"8f3b7d12-6f23-4e89-a21c-99c0e5a8bc34","status":"running","step_count":2,"current_task_index":0}
{"run_id":"8f3b7d12-6f23-4e89-a21c-99c0e5a8bc34","status":"running","step_count":5,"current_task_index":1}
{"run_id":"8f3b7d12-6f23-4e89-a21c-99c0e5a8bc34","status":"completed","step_count":8,"current_task_index":3}
```

#### E. View Audit Trace Log
```bash
curl -X GET "http://localhost:8000/workflow/8f3b7d12-6f23-4e89-a21c-99c0e5a8bc34/trace"
```
*Sample Response:*
```json
{
  "run_id": "8f3b7d12-6f23-4e89-a21c-99c0e5a8bc34",
  "total_entries": 6,
  "entries": [
    {
      "timestamp": "2026-10-19T16:00:01Z",
      "entry_type": "plan",
      "agent": "planner",
      "content": "Created plan with 3 subtasks",
      "metadata": {}
    },
    {
      "timestamp": "2026-10-19T16:00:03Z",
      "entry_type": "handoff",
      "agent": "router",
      "content": "Routing task 'Query sample_data' to db_analyst",
      "metadata": {"task_id": "task_1"}
    }
  ]
}
```

#### F. Approve a Human Checkpoint
When a task requests a destructive operation (e.g. `write_db`), the workflow pauses in `awaiting_approval`. Resume execution by issuing an approval:
```bash
curl -X POST "http://localhost:8000/workflow/8f3b7d12-6f23-4e89-a21c-99c0e5a8bc34/approve" \
     -H "Content-Type: application/json" \
     -d '{"feedback": "Approved. Proceed with table mutation."}'
```
*Sample Response:*
```json
{
  "status": "approved",
  "message": "Workflow resumed after approval"
}
```

#### G. Reject a Human Checkpoint
To cancel a pending destructive action and provide feedback to the agents:
```bash
curl -X POST "http://localhost:8000/workflow/8f3b7d12-6f23-4e89-a21c-99c0e5a8bc34/reject" \
     -H "Content-Type: application/json" \
     -d '{"feedback": "Rejected. Write operations to production schema are forbidden."}'
```
*Sample Response:*
```json
{
  "status": "rejected",
  "message": "Workflow updated with rejection feedback"
}
```

#### H. Resume a Paused Workflow
```bash
curl -X POST "http://localhost:8000/workflow/8f3b7d12-6f23-4e89-a21c-99c0e5a8bc34/resume"
```

---

## 📖 API Reference

| Method | Endpoint | Description | Request Body / Parameters | Response Status |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | API Information and status | None | `200 OK` |
| `GET` | `/health` | Healthcheck for API and components | None | `200 OK` |
| `POST` | `/workflow/run` | Start a new multi-agent workflow | `WorkflowRunRequest` (goal, max_steps, max_tokens, max_cost_usd) | `200 OK` (`WorkflowRunResponse`) |
| `GET` | `/workflow/{run_id}/status` | Retrieve current state, task progress, and budget | Path parameter: `run_id` | `200 OK` (`WorkflowStatusResponse`) |
| `GET` | `/workflow/{run_id}/trace` | Retrieve full chronological audit trace log | Path parameter: `run_id` | `200 OK` (`TraceResponse`) |
| `POST` | `/workflow/{run_id}/approve` | Approve pending human checkpoint and resume execution | Path parameter: `run_id`<br>`ApprovalActionRequest` (feedback) | `200 OK` |
| `POST` | `/workflow/{run_id}/reject` | Reject pending action with feedback and update state | Path parameter: `run_id`<br>`ApprovalActionRequest` (feedback) | `200 OK` |
| `POST` | `/workflow/{run_id}/resume` | Manually resume a paused workflow run | Path parameter: `run_id` | `200 OK` |
| `WS` | `/workflow/{run_id}/stream` | WebSocket endpoint streaming live workflow updates | Path parameter: `run_id` | `101 Switching Protocols` |

---

## 👥 Specialized Agents

The system decomposes goals across five distinct agents, each configured with dedicated system prompts and bound toolsets:

### 1. Planner Agent (`src/agents/planner.py`)
- **Role**: Strategic decomposition and task sequencing.
- **Responsibilities**:
  - Analyzes the high-level objective and generates an ordered list of `SubTask` records.
  - Assigns subtasks to the most qualified worker agent (`researcher`, `coder`, or `db_analyst`).
  - Flags destructive subtasks requiring human approval (`requires_approval=True`).
  - Supports dynamic replanning by taking into account prior failure contexts and partial results.

### 2. Researcher Agent (`src/agents/researcher.py`)
- **Role**: Open-web information gathering and synthesis.
- **Tools**: `web_search`, `fetch_url`.
- **Responsibilities**:
  - Executes targeted search queries to collect current real-world information.
  - Scrapes and parses webpage content from specific URLs.
  - Synthesizes findings into concise, factual summaries for downstream agents.

### 3. Coder Agent (`src/agents/coder.py`)
- **Role**: Python computation, algorithmic problem-solving, and data transformation.
- **Tools**: `execute_python`, `lint_code`.
- **Responsibilities**:
  - Generates type-annotated, modern Python 3.11+ code.
  - Lints code with Ruff before execution to identify syntax errors or anti-patterns.
  - Runs code inside an isolated subprocess and inspects stdout/stderr.
  - Employs an iterative self-correction loop (up to 3 attempts) upon syntax or runtime exceptions.

### 4. Database Analyst Agent (`src/agents/db_analyst.py`)
- **Role**: Relational database querying, analysis, and data mutations.
- **Tools**: `query_db`, `write_db`.
- **Responsibilities**:
  - Explores table schemas and executes read-only SQL queries (`SELECT`).
  - Performs aggregations, joins, and statistical calculations over database records.
  - Formulates mutating SQL statements (`INSERT`, `UPDATE`, `DELETE`) with required human authorization.

### 5. Evaluator Agent (`src/agents/evaluator.py`)
- **Role**: Output quality assurance and dynamic workflow steering.
- **Responsibilities**:
  - Reviews worker agent outputs against the original goal and subtask requirements.
  - Emits one of three structured verdicts:
    - `"complete"`: Subtask succeeded; advances to the next subtask or finalizes workflow.
    - `"retry"`: Output incomplete or erroneous; worker retries the subtask with feedback.
    - `"replan"`: Subtask failed fundamentally; routes back to Planner to recalculate execution strategy.

---

## 🛠️ Tool Ecosystem (MCP Servers)

Tools are exposed via FastMCP servers following the Model Context Protocol specification. All tools adhere to the `ToolResult` pattern, returning standardized JSON responses without throwing uncaught exceptions.

| Server | Tool | Type | Parameters | Description |
| :--- | :--- | :---: | :--- | :--- |
| **SearchServer** | `web_search` | Read | `query: str`<br>`max_results: int = 5` | Performs internet searches using Tavily API and returns structured snippets, titles, and URLs. |
| **SearchServer** | `fetch_url` | Read | `url: str`<br>`timeout_seconds: int = 10` | Fetches webpage text via HTTP client with automatic redirect following and 10,000-character truncation protection. |
| **CodeServer** | `execute_python` | Sandbox | `code: str`<br>`timeout_seconds: int = 30` | Executes Python code in a sandboxed subprocess with stdout/stderr capture, return code tracking, and automatic temporary file cleanup. |
| **CodeServer** | `lint_code` | Analysis | `code: str` | Performs static analysis using Ruff to detect syntax errors, unresolved imports, and stylistic violations. |
| **DatabaseServer**| `query_db` | Read | `sql: str`<br>`params: str = "[]"` | Executes read-only SQL statements. Rejects mutating queries (`INSERT`, `UPDATE`, `DELETE`, `DROP`, etc.) and enforces a 10s timeout. |
| **DatabaseServer**| `write_db` | Destructive | `sql: str`<br>`params: str = "[]"` | Executes mutating SQL statements. Tagged as destructive (`metadata={"is_destructive": True}`) to trigger human approval checkpoints. |

---

## 🛡️ Safety Features & Guardrails

Enterprise workflows require defense-in-depth safety controls to prevent infinite loops, resource depletion, and unauthorized mutations:

1. **Step Limit Guard (`check_step_limit`)**:
   - Monitored at every graph transition.
   - Compares the workflow `step_count` against `max_steps` (default 50).
   - Gracefully terminates execution if the step ceiling is reached.

2. **Budget Enforcement Guard (`check_budget`)**:
   - Integrates with `WorkflowCallbackHandler` to calculate real-time prompt tokens, completion tokens, and dollar expenditures.
   - Enforces configurable limits (`max_tokens=100,000`, `max_cost_usd=$5.00`).
   - Prevents unexpected LLM API consumption surges.

3. **Conversational Loop Detection (`check_loop_detection`)**:
   - Analyzes a sliding window of recent state messages (window size 6).
   - Detects repetitive or cyclic agent utterances.
   - Prevents ping-pong loops between worker agents and tool nodes.

4. **Human-in-the-Loop (HITL) Checkpoints**:
   - Configured via LangGraph `interrupt_before=["db_tools"]`.
   - Any proposed destructive action (e.g. `write_db`) automatically pauses execution state in PostgreSQL.
   - Operators can inspect the proposed query and parameters via `/workflow/{run_id}/status`, then invoke `/approve` to resume or `/reject` to abort.

5. **Tool Sandboxing & Query Isolation**:
   - Code execution runs in isolated subprocesses with timeout limits (30s) and non-root execution paths.
   - Database queries are protected with connection pooling, statement timeouts (10s), and read-only AST keyword checks.

---

## 📊 Running Evaluations

The repository includes a 20-scenario evaluation benchmark evaluating planning competence, tool execution accuracy, self-correction, safety guardrails, and HITL handling across various real-world tasks.

To execute the benchmark suite:

```bash
python -m src.evaluation.runner
```

### Benchmark Metric Tracking
The runner executes scenarios and logs results to the PostgreSQL `evaluation_results` table:
- **Task Success Rate**: Binary scoring based on target criteria fulfillment.
- **Execution Efficiency**: Total steps and wall-clock latency per scenario.
- **Resource Consumption**: Aggregate tokens consumed and exact dollar cost.
- **LLM Judge Evaluation**: Independent GPT-4o evaluation judging plan quality, tool choice, and final answer accuracy.

---

## 🧪 Running Tests

The test suite covers unit tests for core models and state reducers, mocked agent execution nodes, LangGraph workflow routing, persistence layers, and FastAPI routes.

Run the test suite with pytest:

```bash
pytest tests/ -v
```

Run tests with test coverage reporting:

```bash
pytest --cov=src --cov-report=term-missing tests/
```

### Test Suite Breakdown

- `tests/test_core.py`: Tests `WorkflowState` reducers, `SubTask`, `BudgetUsage`, `ToolResult`, and settings loading.
- `tests/test_agents.py`: Tests agent prompt templates, mock LLM parsing, JSON plan validation, and tool binding.
- `tests/test_graph.py`: Tests graph creation, safety guard execution, loop detection, and routing decisions.
- `tests/test_persistence.py`: Tests PostgreSQL checkpointer initialization, Redis caching, budget tracking, and database table creation.
- `tests/test_api.py`: Tests FastAPI routes, workflow submission, state queries, approval/rejection logic, and WebSocket endpoints.

---

## 📁 Project Structure

```text
multi-agent/
├── pyproject.toml              # Build config, dependencies, Ruff & Pytest settings
├── .env.example                # Example environment variable template
├── docker-compose.yml          # Container configuration for PostgreSQL 16 & Redis 7
├── Dockerfile                  # Production container definition for FastAPI app
├── README.md                   # System documentation & developer guide
├── src/
│   ├── __init__.py
│   ├── core/                   # State, data structures, and configuration
│   │   ├── __init__.py
│   │   ├── config.py           # Typed settings via pydantic & python-dotenv
│   │   ├── models.py           # Pydantic models (SubTask, ToolResult, TraceEntry, etc.)
│   │   └── state.py            # WorkflowState TypedDict & LangGraph reducers
│   ├── tools/                  # Model Context Protocol (MCP) tool servers
│   │   ├── __init__.py
│   │   ├── search_server.py    # FastMCP server: web_search, fetch_url
│   │   ├── code_server.py      # FastMCP server: execute_python, lint_code
│   │   ├── database_server.py  # FastMCP server: query_db, write_db (destructive)
│   │   └── tool_registry.py    # Tool registry, agent-tool mapping & approval policy
│   ├── agents/                 # Specialized agent node definitions
│   │   ├── __init__.py
│   │   ├── planner.py          # Goal decomposition & subtask generation
│   │   ├── researcher.py       # Web search & URL content synthesis
│   │   ├── coder.py            # Code generation, execution & self-debugging
│   │   ├── db_analyst.py       # SQL query execution & database management
│   │   └── evaluator.py        # Output evaluation, retries & replanning
│   ├── graph/                  # LangGraph workflow definition & safety guards
│   │   ├── __init__.py
│   │   ├── workflow.py         # StateGraph builder, router & compile logic
│   │   ├── guards.py           # Step limits, budget caps & loop detection
│   │   └── callbacks.py        # Token counting, cost tracking & trace logging
│   ├── persistence/            # Persistence & distributed state
│   │   ├── __init__.py
│   │   ├── checkpointer.py     # AsyncPostgresSaver LangGraph checkpointing
│   │   ├── redis_client.py     # Redis caching, budget counters & stream traces
│   │   └── database.py         # PostgreSQL schema initialization & sample data
│   ├── api/                    # FastAPI application & web layer
│   │   ├── __init__.py
│   │   ├── main.py             # App instantiation, CORS, and lifecycle events
│   │   ├── routes.py           # REST endpoints, WebSocket handler & HITL routes
│   │   └── schemas.py          # Request and response Pydantic schemas
│   └── evaluation/             # 20-scenario evaluation framework
│       ├── __init__.py
│       ├── scenarios.py        # Benchmark scenarios across diverse domains
│       ├── runner.py           # Evaluation execution harness & metrics logger
│       └── judge.py            # LLM-as-a-judge scoring & assessment logic
└── tests/                      # Automated test suite
    ├── __init__.py
    ├── conftest.py             # Pytest fixtures and mock environments
    ├── test_core.py            # Core state, model, and config tests
    ├── test_agents.py          # Agent prompt and logic tests
    ├── test_graph.py           # Graph routing and safety guard tests
    ├── test_persistence.py      # Database, checkpointer, and Redis tests
    └── test_api.py             # FastAPI endpoint and WebSocket tests
```

---

## 📄 License

This project is licensed under the terms of the **MIT License**. See [LICENSE](LICENSE) for details.
