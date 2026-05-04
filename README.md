# SROP — Stateful RAG Orchestration Pipeline

An async FastAPI backend that routes user queries to specialised AI agents, grounded in a ChromaDB vector store, with full session state persisted to SQLite across process restarts.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Chunking Strategy](#chunking-strategy)
- [Known Limitations](#known-limitations)
- [Time Spent](#time-spent)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)

---

## Quick Start

Get up and running in ≤ 5 minutes.

### Prerequisites

- **Python 3.11+**
- **A Gemini API key** ([Google AI Studio](https://aistudio.google.com))

> **Free-tier quota note:** The pipeline uses `gemini-2.0-flash`. A `429 RESOURCE_EXHAUSTED` error with `limit: 0` means your Google AI Studio account has exhausted its free quota. Quota is **account-level, not key-level** — generating multiple keys under the same account does not multiply it. Either enable billing on your Cloud project, or run in [Mock Mode](#mock-mode-no-api-key-required).

### Installation

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\Activate.ps1         # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Open .env and set GEMINI_API_KEY=your_key_here

# 4. Ingest documents into the vector store
mkdir docs
echo "# Deploy Keys\n\nTo rotate a deploy key, go to Settings > Deploy Keys." > docs/deploy-keys.md
python -m app.rag.ingest --path docs/

# 5. Start the server
uvicorn app.main:app --reload
```

**API live at** `http://localhost:8000`  
**Interactive docs at** `http://localhost:8000/docs`

### Verify

```bash
curl http://localhost:8000/healthz
# {"status":"ok"}
```

### Mock Mode (No API Key Required)

Set `MOCK_LLM=true` in your `.env` file. The orchestrator returns a deterministic stub — pipeline logic, state persistence, RAG retrieval, DB writes, and trace records all run exactly as in production. **Only the LLM call is replaced.**

Use this when billing is not enabled or free-tier quota is exhausted. All Phases 1–5 are fully exercisable without any external dependency.

---

## Architecture

### Component Map

| Layer | File(s) | Responsibility |
|-------|---------|-----------------|
| **HTTP** | `app/api/routes.py` | Request parsing, error mapping |
| **Pipeline** | `app/pipeline.py` | Orchestration, state I/O, DB writes |
| **Agents** | `app/agents/orchestrator.py` | ADK srop_root LlmAgent + routing |
| **Sub-agents** | `knowledge.py` / `account.py` | Domain-specific tools |
| **RAG** | `app/rag/store.py` | Async ChromaDB search, cosine scoring |
| **Ingest** | `app/rag/ingest.py` | CLI chunker → embedder → upsert |
| **DB** | `models.py` + `session.py` | SQLAlchemy 2.x async ORM |
| **Config** | `app/config.py` | pydantic-settings, .env loading |

### Request Flow

```
Client
  └─ POST /v1/chat/{id}
       └─ pipeline.py
            ├─ 1. Load session (SELECT state_json)
            ├─ 2. run_turn() via asyncio.wait_for [30 s timeout]
            │       └─ srop_root (gemini-2.0-flash)
            │             ├─ knowledge_agent → search_docs() → ChromaDB
            │             └─ account_agent  → get_recent_builds()
            │                                  get_account_status()
            └─ 3. Persist (UPDATE state_json, INSERT trace + messages)
                   └─ ChatResponse → Client
```

---

## Design Decisions

### State Persistence — DB-backed state_json

**Chosen pattern:** A `state_json` JSON column on the `sessions` table (SQLite).

```json
{
  "user_id":        "string",
  "plan_tier":      "free | pro | enterprise",
  "turn_count":     3,
  "last_agent":     "knowledge_agent",
  "recent_results": ["[turn via knowledge_agent] Q: ...", "..."]
}
```

#### Why This Over Alternatives?

| Option | Decision | Reason |
|--------|----------|--------|
| SQLite JSON column | ✅ chosen | Zero deps, atomic writes, survives restarts |
| Redis session cache | ❌ rejected | External dep; lost on flush |
| In-memory dict | ❌ rejected | Dies on restart; breaks horizontal scale |
| Separate session_state table | ❌ rejected | No benefit over JSON col for ~1 KB blob; more JOINs |

#### Key Properties

- **Survives restarts** by definition — SQLite is on disk.
- **Zero external dependencies** — no Redis, no Memcached.
- **Atomic per turn** — single UPDATE inside the same SQLAlchemy transaction as messages and agent_traces inserts.
- **Acceptable throughput** — a support concierge handles tens of concurrent sessions, well within SQLite's write capacity.
- **Plan tier source of truth** — `sessions.plan_tier` column is authoritative; `state_json.plan_tier` is a convenience copy synced on every load.

---

## Chunking Strategy

**Approach:** Paragraph-boundary sliding window — **400 tokens per chunk, 80-token overlap.**

```
document text
     │
     ▼
Split on \n{2,}  ──── paragraph list
     │
     ▼  for each paragraph:
     │
     ├── len(para_tokens) > 400? ──YES──► token-level sliding window
     │                                    (fallback for large code blocks)
     └── NO ──► greedy pack into buffer
                     │
                     └── buffer + para > 400?
                              │
                              YES ──► emit chunk
                                      seed next buffer with buf[-80:]
                              NO  ──► extend buffer
```

### Justification

| Decision | Justification |
|----------|--------------|
| **Paragraph boundaries first** | A paragraph is a natural semantic unit in Markdown — a tutorial step, config option, warning block. Keeping it intact prevents a question and its answer from being split across chunks. |
| **400-token window** | Balances retrieval precision against context completeness. `all-MiniLM-L6-v2` has a 256-subword limit — a 400-word proxy comfortably fits after subword expansion of typical English prose. |
| **80-token overlap** | Prevents answers spanning a chunk boundary from being silently lost. 80 tokens ≈ 2–3 sentences, enough to re-establish context. |
| **Token-level fallback** | Large code blocks or tables that exceed 400 words in a single paragraph are handled by a pure token sliding window. |
| **Deterministic chunk IDs** | `sha256(file_path + str(chunk_index))[:16]` — re-ingesting the same file is idempotent via ChromaDB's upsert. |

---

## Known Limitations

Honest current limitations, not unfinished work.

### 🔴 Critical — Production Blockers

| Limitation | Detail | Mitigation Path |
|-----------|--------|-----------------|
| **Mock account data** | `get_recent_builds` and `get_account_status` return static fixtures. | Replace with real API calls to your CI/CD and account services. |
| **No authentication** | All three endpoints are unauthenticated. | Add OAuth2/JWT middleware; scope traces to the authenticated user. |
| **Single-process SQLite** | Single-writer lock. WAL mode must be enabled explicitly under multiple uvicorn workers. | Swap `DATABASE_URL` to `postgresql+asyncpg://...` for multi-replica deploys. ORM layer requires zero changes. |
| **Gemini quota exhaustion** | Quota is account-level, not key-level. Multiple keys on the same Google account share the same limit. 429 with `limit: 0` = account fully blocked at free tier. | Enable billing on the linked Google Cloud project, or use `MOCK_LLM=true` for local development. |

### 🟡 Operational — Lower-Priority

| Limitation | Detail | Mitigation Path |
|-----------|--------|-----------------|
| **Local embedding model** | `all-MiniLM-L6-v2` runs on CPU. First load ~2 s; inference ~10 ms/chunk. | Swap `MODEL_NAME` in `store.py` / `ingest.py` to `text-embedding-004` or `text-embedding-ada-002`. |
| **In-memory ADK session service** | `InMemorySessionService` is process-local. ADK conversation history is lost on restart. SROP state is unaffected — it comes from SQLite. | Use ADK's `DatabaseSessionService` or a custom SQLite-backed implementation. |
| **No streaming** | `/v1/chat` returns only after the full ADK event stream is consumed. | Wrap `runner.run_async` in a `StreamingResponse` and yield events as SSE. |
| **No hot-reload of vector store** | New docs require re-running ingest and a process restart to invalidate the VectorStore singleton. | Add `POST /v1/admin/reindex` to reset the singleton and trigger ingest. |

---

## Time Spent

| Phase | Description | Time |
|-------|-------------|------|
| **Phase 1** | Scaffold — Directory structure, ORM models, async engine, pydantic-settings, lifespan | ~45 min |
| **Phase 2** | API endpoints — Schemas (Pydantic v2, UUID, Literal), routes with structured error codes | ~30 min |
| **Phase 3** | RAG pipeline — `ingest.py` (chunker, frontmatter, deterministic IDs), `store.py` (async ChromaDB, cosine normalisation) | ~60 min |
| **Phase 4** | ADK agents — AccountAgent, KnowledgeAgent, RootOrchestrator with AgentTool, ADK Runner + InMemorySessionService | ~75 min |
| **Phase 5** | Pipeline — SessionState dataclass, 10-step run_pipeline, explicit UPDATE for state persistence, PipelineResult | ~45 min |
| **Phase 6** | Tests — `conftest.py`, `test_integration.py` (mock boundary, state persistence), `test_rag.py` (skip if empty) | ~45 min |
| **Phase 7** | Documentation — README, design decisions, limitations | ~20 min |
| **TOTAL** | | **~6 hours** |

---

## API Reference

**Base URL:** `http://localhost:8000`

All error responses carry a `"code"` field:
```json
{"detail": {"code": "SESSION_NOT_FOUND"}}
```

### Health Check

```http
GET /healthz
```

Health check. Returns immediately, no DB query.

**Example:**
```bash
curl http://localhost:8000/healthz
# {"status":"ok"}
```

---

### Create Session

```http
POST /v1/sessions
```

Create a new stateful session for a user.

**Request Body:**

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `user_id` | string | Yes | Any non-empty string |
| `plan_tier` | string | Yes | `"free"` \| `"pro"` \| `"enterprise"` |

**Example:**
```bash
curl -s -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice@example.com",
    "plan_tier": "pro"
  }' | jq .
```

**Response 201:**
```json
{"session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
```

**Errors:**

| Status | Code | Cause |
|--------|------|-------|
| 422 | — | Invalid `plan_tier` value |

---

### Chat Endpoint

```http
POST /v1/chat/{session_id}
```

Send a message and receive an AI-generated reply. Runs the ADK agent, persists state, and writes a trace row.

**Example — Knowledge Query:**
```bash
SESSION_ID="3fa85f64-5717-4562-b3fc-2c963f66afa6"

curl -s -X POST "http://localhost:8000/v1/chat/${SESSION_ID}" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I rotate a deploy key?"}' | jq .
```

**Example — Account Query:**
```bash
curl -s -X POST "http://localhost:8000/v1/chat/${SESSION_ID}" \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me my last 3 failed builds"}' | jq .
```

**Response 200:**
```json
{
  "reply": "To rotate a deploy key, navigate to Settings → Deploy Keys...",
  "routed_to": "knowledge_agent",
  "trace_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7"
}
```

**Errors:**

| Status | Code | Cause |
|--------|------|-------|
| 404 | `SESSION_NOT_FOUND` | Session ID not in DB |
| 503 | `STORE_NOT_READY` | Vector store empty — run ingest first |
| 504 | `UPSTREAM_TIMEOUT` | ADK agent exceeded 30-second wall-clock limit |

---

### Fetch Trace

```http
GET /v1/traces/{trace_id}
```

Fetch the full agent trace for a completed turn. Use `trace_id` from the `/v1/chat` response.

**Example:**
```bash
TRACE_ID="7c9e6679-7425-40de-944b-e07fc1f90ae7"
curl -s "http://localhost:8000/v1/traces/${TRACE_ID}" | jq .
```

**Response 200:**
```json
{
  "trace_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "routed_to": "knowledge_agent",
  "tool_calls": [
    {"tool": "search_docs", "args": {"query": "rotate deploy key", "k": 5}}
  ],
  "retrieved_chunk_ids": ["a1b2c3d4e5f6a7b8", "d4e5f6a7b8c9d0e1"],
  "latency_ms": 1847,
  "created_at": "2025-05-04T02:14:00Z"
}
```

**Errors:**

| Status | Code | Cause |
|--------|------|-------|
| 404 | `TRACE_NOT_FOUND` | Trace ID not in DB |

---

### Full Demo Script

```bash
#!/usr/bin/env bash
BASE="http://localhost:8000"

echo "==> Create session"
RESP=$(curl -s -X POST "$BASE/v1/sessions" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo@helix.io","plan_tier":"pro"}')
echo "$RESP" | jq .
SESSION_ID=$(echo "$RESP" | jq -r .session_id)

echo "==> Turn 1: knowledge query"
RESP=$(curl -s -X POST "$BASE/v1/chat/$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"message":"How do I rotate a deploy key?"}')
echo "$RESP" | jq .
TRACE_ID=$(echo "$RESP" | jq -r .trace_id)

echo "==> Turn 2: account query"
curl -s -X POST "$BASE/v1/chat/$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"message":"Show me my last 3 failed builds"}' | jq .

echo "==> Fetch trace from turn 1"
curl -s "$BASE/v1/traces/$TRACE_ID" | jq .
```

---

## Running Tests

```bash
# All tests (no ingest required)
pytest -v

# RAG unit tests (requires ingest)
python -m app.rag.ingest --path docs/
pytest tests/test_rag.py -v

# With coverage
pip install pytest-cov
pytest --cov=app --cov-report=term-missing
```

---

## Project Structure

```
app/
├── app/
│   ├── main.py              # FastAPI factory, lifespan, /healthz
│   ├── config.py            # pydantic-settings (.env)
│   ├── pipeline.py          # SROP orchestration (SessionState, run_pipeline)
│   ├── api/
│   │   ├── routes.py        # 3 endpoints + error mapping
│   │   └── schemas.py       # Pydantic v2 request/response models
│   ├── db/
│   │   ├── models.py        # sessions, messages, agent_traces (SQLAlchemy 2.x)
│   │   └── session.py       # async engine + get_db dependency
│   ├── rag/
│   │   ├── ingest.py        # CLI: python -m app.rag.ingest --path docs/
│   │   └── store.py         # async ChromaDB wrapper, VectorStoreEmptyError
│   └── agents/
│       ├── account.py       # AccountAgent (get_recent_builds, get_account_status)
│       ├── knowledge.py     # KnowledgeAgent (search_docs)
│       └── orchestrator.py  # srop_root LlmAgent, AgentTool routing, run_turn
├── tests/
│   ├── test_integration.py
│   └── test_rag.py
├── conftest.py
├── pytest.ini
├── requirements.txt
└── .env.example
```

---

## License

This project is provided as-is for reference and educational purposes.

---

**Built with:** FastAPI · ChromaDB · SQLAlchemy · Google Gemini API · ADK

**Last Updated:** May 4, 2025
