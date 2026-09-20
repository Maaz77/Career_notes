# Conversation Summary — FastAPI / REST / Async-Concurrency Deep Dive

## Original goal / context
No single project goal — this was a rolling Q&A session (progressive learning style) covering REST API design in FastAPI, then branching into Python concurrency internals (async, threads, processes) triggered by a file-download feature being discussed.

---

## 1. REST API design concepts covered

- **PATCH vs PUT vs POST**
  - POST = create, PUT = full replace, PATCH = partial update
  - PATCH is NOT guaranteed idempotent (e.g. incrementing a counter)
  - Mnemonic: Post = birth, PUT = rebirth, PATCH = surgery

- **Swagger vs FastAPI**
  - Swagger = tool suite (Swagger UI/Editor/Codegen) around the OpenAPI Spec
  - FastAPI auto-generates the OpenAPI spec and serves Swagger UI at `/docs`, ReDoc at `/redoc`

- **Route vs Endpoint vs API**
  - API = whole service; Endpoint = specific URL+method; Route = code mapping endpoint→handler

- **Swagger docs don't auto-show all possible status codes.** Must declare explicitly via `responses={...}` on the route decorator — FastAPI can't statically infer `raise HTTPException(...)` branches.

- **Documenting mixed return types (json/text/file) in Swagger:**
```python
  @app.get(
      "/data/{id}",
      response_model=None,  # mixed return types, skip auto schema
      responses={
          200: {
              "description": "Successful response",
              "content": {
                  "application/json": {"example": {"id": 1, "name": "foo"}},
                  "text/plain": {"example": "plain text result"},
                  "application/octet-stream": {"example": "binary file content"},
              },
          },
          404: {"description": "Not found", "content": {"application/json": {"example": {"detail": "Not found"}}}},
      },
  )
  def get_data(id: int): ...
```

---

## 2. Request parameters — POST body vs query, Pydantic, Depends()

- Simple types (`str`, `int`) as route params → FastAPI treats as **query params**
- Pydantic model as route param → FastAPI treats as **request body**

- **Pydantic validates key names AND types.**
  - Unknown/extra keys → **silently ignored** by default
  - Missing required keys → **422 Unprocessable Entity** (`"field required"`)
  - To reject unknown keys explicitly: `model_config = ConfigDict(extra="forbid")`
  - Same behavior applies to **query param models**: `?naame=John` where `name` is required → 422. If `name` is `Optional[str] = None`, no 422 — just silently ignored/defaults to None.

- **`Depends()` is NOT for Pydantic models.** It's for reusable injected logic (DB sessions, auth, common params):
```python
  def get_db():
      db = SessionLocal()
      try:
          yield db
      finally:
          db.close()

  @app.post("/users")
  def create_user(user: User, db: Session = Depends(get_db)):
      ...
```
  - `Depends()` with **no argument** → invalid, raises `TypeError` at startup (needs a callable).
  - `user: User = Depends(User)` → **valid but changes semantics**: turns model fields into **query params**, not body — rarely what's wanted for POST.

- **Optional nested Pydantic field:**
```python
  class User(BaseModel):
      name: str
      age: int
      address: Optional[Address] = None
```

- **Optional whole request body:**
```python
  from fastapi import Body
  from typing import Optional

  @app.post("/users")
  def create_user(user: Optional[User] = Body(None)):
      if user is None:
          return {"message": "no user provided"}
      return user
```

- **Testing via curl:**
```bash
  curl -X POST "http://localhost:8000/users?name=John&age=30"
  curl -X POST "http://localhost:8000/users" -H "Content-Type: application/json" -d '{"name": "John", "age": 30}'
  curl -v ...   # verbose, shows status/headers
```

---

## 3. httpx vs requests

- `requests` = sync only
- `httpx` = sync **and** async (`httpx.AsyncClient`), supports HTTP/2, used by FastAPI's `TestClient`
- Rule: async/FastAPI context → `httpx`; simple scripts → `requests` fine

---

## 4. File download endpoint (Excel bytes) — full discussion

- **Filename delivery**: via `Content-Disposition` response header — NOT inside the blob (blob = raw bytes only, no metadata).
```python
  from fastapi.responses import Response

  @app.get("/export")
  def export_excel():
      excel_bytes = generate_excel()
      return Response(
          content=excel_bytes,
          media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          headers={"Content-Disposition": "attachment; filename=report.xlsx"}
      )
```

- **TypeScript client-side download:**
```typescript
  const response = await fetch("/export");
  const blob = await response.blob();

  const disposition = response.headers.get("Content-Disposition");
  const filename = disposition?.split("filename=")[1] ?? "download.xlsx";

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();

  URL.revokeObjectURL(url); // cleanup
```

- **`Content-Disposition` returning `null` in JS** → CORS issue. Browser hides non-default headers from JS unless server explicitly exposes them:
```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_methods=["*"],
      allow_headers=["*"],
      expose_headers=["Content-Disposition"],  # required
  )
```

- **`Response` vs `FileResponse`**:
  - `Response` = bytes in memory
  - `FileResponse` = reads from disk by path
  - `FileResponse` from in-memory bytes IS possible but requires writing to a temp file first — extra overhead, not recommended unless you already have a file on disk.
```python
  with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
      tmp.write(excel_bytes)
      tmp_path = tmp.name
  return FileResponse(path=tmp_path, filename="report.xlsx", media_type="...")
```
  - `delete=False` required because `FileResponse` is **lazy** — actual file read happens *after* the route function returns, so the file must still exist on disk at that point. Putting the return inside the `with` block does NOT allow `delete=True`, for the same lazy-read reason.

- **Temp file cleanup** — use `BackgroundTasks` (runs after response is fully sent):
```python
  from fastapi import BackgroundTasks

  def cleanup(path: str):
      os.remove(path)

  @app.get("/export")
  def export_excel(background_tasks: BackgroundTasks):
      ...
      background_tasks.add_task(cleanup, tmp_path)
      return FileResponse(path=tmp_path, filename="report.xlsx", media_type="...")
```

- **Risks of custom headers in middleware**: leaking server internals, overriding route-specific headers, overly permissive CORS (`allow_origins=["*"]` risk for authenticated APIs), negligible perf cost, unintended cross-origin exposure.

- **"Works locally, may break elsewhere" — real risks**:
  - Reverse proxies / gateways (Nginx, API Gateway, CloudFront) can **strip custom headers** unless explicitly passed through (most common real-world culprit)
  - CORS enforced more strictly in production (specific origins vs `*`)
  - CDN/load balancer caching responses without the header
  - Older browser inconsistencies (rare in 2024+)

---

## 5. Concurrency deep dive (async / threads / processes / event loop)

- **Sync HTTP call in a spawned thread**: blocks **that thread only** — other threads unaffected. GIL is released during I/O, so other threads' Python code can run freely while one thread waits.

- **How async matches incoming responses to the right pending request**: each request opens its own socket (unique file descriptor). Event loop keeps an internal map `{fd → Task}`. OS notifies loop which fd has data ready (via epoll/kqueue/IOCP); loop resumes the matching Task. Correlation is structural (per-connection), not by inspecting response content.

- **Coroutine vs Thread**:
  | | Thread | Coroutine |
  |---|---|---|
  | Managed by | OS | Event loop |
  | Scheduling | Preemptive | Cooperative (`await` points only) |
  | Memory | Heavy (~MBs) | Light (~KBs) |
  | Parallelism | Yes (multi-core, GIL-limited for Python code) | No, single-threaded turn-taking |
  | Blocking risk | Only that thread stalls | A blocking call inside async freezes the **whole event loop** |

- **Is there one thread running the event loop?** By default yes — one thread runs both the event loop scheduling logic AND your coroutine code, taking turns. Exceptions: threads you spawn explicitly, or libraries using `run_in_executor` to offload blocking calls to a hidden thread pool.

- **Event loop explained (plain-language, step by step)**: a single loop that maintains a list of paused tasks + readiness info (timers, socket data), and repeatedly does: "is anything ready to resume? → run it until next `await` → repeat." Never runs two things at the exact same instant — just switches fast during idle/waiting time instead of blocking.

- **Async loop over objects sending POST requests — efficiency**:
  - ❌ Sync loop: fully sequential, N × latency
  - ⚠️ `for obj: await client.post(...)` inside async — STILL sequential (common mistake), same N × latency
  - ✅ Concurrent version:
```python
    async def send_all(objects):
        async with httpx.AsyncClient() as client:
            tasks = [client.post(url, json=obj) for obj in objects]
            responses = await asyncio.gather(*tasks)
        return responses
```
  - With concurrency limiting via `asyncio.Semaphore`:
```python
    sem = asyncio.Semaphore(10)

    async def send_one(client, obj):
        async with sem:
            return await client.post(url, json=obj)

    async def send_all(objects):
        async with httpx.AsyncClient() as client:
            tasks = [send_one(client, obj) for obj in objects]
            return await asyncio.gather(*tasks)
```
  - Key point: `async`/`await` alone ≠ concurrency; need `gather`/`create_task` to actually run things simultaneously.

- **Thread vs Process for background jobs**:
  - I/O-bound → thread (or better, async) — GIL released during I/O
  - CPU-bound → process — GIL blocks true parallel Python execution across threads; separate processes get separate interpreters/GILs → real multi-core use
  - Celery-style workers are typically separate processes for isolation + scaling; internally can still use async/threads for I/O-bound sub-tasks.

- **Process spawn limits**:
  - Hard OS ceiling: `pid_max` (`/proc/sys/kernel/pid_max`), system-wide, rarely the actual bottleneck
  - Real bottleneck: **memory** (~20–50MB+ baseline per process)
  - CPU-bound work: diminishing returns beyond `os.cpu_count()` processes
  - File descriptor limits (`ulimit -n`) also apply

- **FastAPI built-in background tasks**:
  - Only `BackgroundTasks` — runs in-process, after response sent
  - Limitations: lost on crash/restart, no retries, no persistence, no scheduling, not distributed — good only for lightweight fire-and-forget work
  - For real job needs: Celery, RQ, arq (async-native), Dramatiq, APScheduler (cron-like)

- **Process lifecycle (`multiprocessing`)**:
  - Natural end: target function returns or raises (must check `p.exitcode` — exceptions inside child are NOT auto-raised to parent)
  - `.join()` — does **NOT** kill; just blocks the **calling thread** until the process finishes (or timeout elapses, after which process may still be alive — must check `is_alive()` then `.terminate()`)
  - `.terminate()` — SIGTERM, abrupt, no cleanup code runs inside child
  - `.kill()` — SIGKILL, even more forceful, uncatchable
  - `daemon=True` → child dies automatically when parent exits; `daemon=False` (default) → child becomes orphaned, keeps running
  - Zombie processes: occur if child finishes but parent never calls `.join()` to reap it — always eventually call `.join()`
  - **Important clarification**: `.join()` blocks the **calling thread**, not the event loop — `multiprocessing` is plain sync/blocking, unrelated to asyncio unless deliberately bridged via `asyncio.subprocess` (`await proc.wait()`, which does NOT block the event loop).

---

## 6. Background job design pattern (status-polling with timeout) — main applied design discussion

**Scenario**: background process talks to an external server, updates a `status` field in a DB table on completion; UI polls that status and blocks while `processing`. Need: if job exceeds a timeout, mark it `failed` so UI can react.

**Recommended pattern — two-layer timeout + atomic status transitions:**

1. **Schema** — include `started_at` timestamp (essential for detecting staleness even if worker dies):
```sql
   CREATE TABLE jobs (
       id UUID PRIMARY KEY,
       status VARCHAR(20) DEFAULT 'processing',  -- processing | success | failed | timeout
       started_at TIMESTAMP DEFAULT now(),
       updated_at TIMESTAMP DEFAULT now(),
       result JSONB
   );
```

2. **Layer 1 — in-task timeout** (handles common case: external call hangs):
```python
   async def process_job(job_id: str):
       try:
           result = await asyncio.wait_for(call_external_server(), timeout=30)
           await db.execute(
               "UPDATE jobs SET status='success', result=:r, updated_at=now() "
               "WHERE id=:id AND status='processing'",
               {"r": result, "id": job_id}
           )
       except asyncio.TimeoutError:
           await db.execute(
               "UPDATE jobs SET status='timeout', updated_at=now() "
               "WHERE id=:id AND status='processing'",
               {"id": job_id}
           )
       except Exception:
           await db.execute(
               "UPDATE jobs SET status='failed', updated_at=now() "
               "WHERE id=:id AND status='processing'",
               {"id": job_id}
           )
```

3. **Layer 2 — watchdog/reaper** (handles worker crash / OOM-kill / server reboot — layer 1 never runs in that case):
```python
   # run periodically via APScheduler / Celery beat / cron
   async def reap_stale_jobs():
       await db.execute(
           "UPDATE jobs SET status='timeout', updated_at=now() "
           "WHERE status='processing' AND started_at < now() - interval '60 seconds'"
       )
```

4. **Atomic guard clause is critical**: every update uses `WHERE id=:id AND status='processing'` to avoid race conditions between layer 1 and layer 2 (e.g. success arriving at the same moment the reaper sweeps) — first writer wins, second becomes a safe no-op.

5. **Client-side defense in depth** — UI has its own independent polling ceiling regardless of server-side guarantees:
```javascript
   const POLL_INTERVAL = 2000;
   const MAX_POLL_TIME = 60000;
   const start = Date.now();

   const poll = async () => {
     const res = await fetch(`/jobs/${id}`);
     const { status } = await res.json();
     if (status === "processing") {
       if (Date.now() - start > MAX_POLL_TIME) {
         showError("Taking too long, please retry.");
         return;
       }
       setTimeout(poll, POLL_INTERVAL);
     } else {
       handleResult(status);
     }
   };
```

**Core principle stated**: never rely on a single point of timeout enforcement — the worker responsible for catching its own timeout might itself be dead. Pair in-task timeout + external watchdog + client-side ceiling.

---

## Open questions / not yet resolved
- No concrete implementation was written for the user's actual job table / worker — pattern given is general-purpose, not yet adapted to their specific external-server integration.
- Choice of watchdog scheduling mechanism (APScheduler vs Celery beat vs plain cron) not decided — just listed as options.
- Not discussed: how the "background process" itself is currently implemented (raw `multiprocessing`, `asyncio` task, or an external queue like Celery/arq) — this affects how layer 1's timeout should actually be wired in.
- User had asked about `multiprocessing` process lifecycle in a general sense; not yet confirmed whether their actual background job uses `multiprocessing`, threads, or async tasks.

## Current state
Conceptual/design-pattern stage — no code has been written yet against the user's real job table or external server integration. Last concrete deliverable was the general async-timeout + watchdog + atomic-update pattern; last clarification was about `.join()` blocking the calling thread (not the event loop) in `multiprocessing`.