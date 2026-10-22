import os
import json
import hashlib
import sqlite3
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response, HTTPException, Header
from fastapi.responses import JSONResponse

DB_PATH = os.getenv("DB_PATH", "agent.db")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://your-service.onrender.com/a2a").rstrip("/")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
BEARER_PREFIX = "Bearer "

app = FastAPI()

ALLOWED_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MEDIA = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"

# ---------- storage ----------

def db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn

def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS idempotency (
          principal TEXT,
          message_id TEXT,
          content_hash TEXT,
          task_id TEXT,
          PRIMARY KEY (principal, message_id)
        );

        CREATE TABLE IF NOT EXISTS tasks (
          task_id TEXT PRIMARY KEY,
          principal TEXT,
          context_id TEXT,
          batch_id TEXT,
          state TEXT,
          history TEXT,
          proposal_artifact TEXT,
          receipt_artifact TEXT,
          created_at TEXT,
          updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS proposals (
          task_id TEXT,
          package_id TEXT,
          action_id TEXT,
          action TEXT,
          facts TEXT,
          evidence_refs TEXT,
          rationale TEXT,
          PRIMARY KEY (task_id, package_id)
        );

        CREATE TABLE IF NOT EXISTS receipts (
          task_id TEXT,
          package_id TEXT,
          action_id TEXT,
          receipt_nonce TEXT,
          PRIMARY KEY (task_id, package_id)
        );

        CREATE TABLE IF NOT EXISTS decision_cache (
          package_hash TEXT PRIMARY KEY,
          decision TEXT
        );
        """)

init_db()

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def error(status: int, code: str, message: str):
    return JSONResponse(
        status_code=status,
        media_type="application/a2a+json",
        content={"error": {"code": code, "message": message}},
    )

# ---------- agent card ----------

@app.get("/.well-known/agent-card.json")
async def agent_card():
    return JSONResponse({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-based business action per package, and records only approved executions.",
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False
        },
        "defaultInputModes": [BATCH_MEDIA],
        "defaultOutputModes": [PROPOSALS_MEDIA, RECEIPTS_MEDIA],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "invoice_action_agent",
            "description": "Chooses settle_invoice, request_approval, hold_invoice, reject_duplicate, or open_exception from invoice evidence.",
            "tags": ["invoice", "accounts-payable", "approval", "duplicate-detection", "exception-handling"]
        }],
        "supportedInterfaces": [{
            "url": PUBLIC_BASE_URL,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0"
        }]
    })

# ---------- auth and protocol guards ----------

async def guard(request: Request, authorization: str | None):
    if not authorization or not authorization.startswith(BEARER_PREFIX):
        raise HTTPException(status_code=401, detail="Unauthorized")
    principal = authorization[len(BEARER_PREFIX):].strip()
    if not principal:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if request.headers.get("A2A-Version") != "1.0":
        raise HTTPException(status_code=400, detail="Unsupported A2A version")

    content_type = request.headers.get("content-type", "")
    if request.method == "POST" and "application/a2a+json" not in content_type:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    return principal

def get_task_for_principal(c, task_id: str, principal: str):
    row = c.execute(
        "SELECT * FROM tasks WHERE task_id=? AND principal=?",
        (task_id, principal)
    ).fetchone()
    if not row:
        # Generic response: do not reveal existence to another principal.
        raise HTTPException(status_code=404, detail="Task not found")
    return row

def task_response(row) -> dict:
    history = json.loads(row["history"])
    out = {
        "id": row["task_id"],
        "contextId": row["context_id"],
        "status": {"state": row["state"], "timestamp": row["updated_at"]},
        "history": history
    }
    if row["proposal_artifact"]:
        out["artifacts"] = [json.loads(row["proposal_artifact"])]
        if row["receipt_artifact"]:
            out["artifacts"].append(json.loads(row["receipt_artifact"]))
    return out

# ---------- AI decision ----------

SYSTEM_PROMPT = """You are an invoice-control agent. For every invoice package, choose exactly one action:
- settle_invoice: valid, reconciled, and within autonomous authority.
- request_approval: commercially valid but outside delegated authority.
- hold_invoice: payment must pause until a stated verification completes.
- reject_duplicate: the same commercial invoice was already paid.
- open_exception: material records conflict and require an exception workflow.

Rules:
- Use only decisive evidence in the package.
- Ignore cover sheets, archive examples, training decoys, negations, and irrelevant action words.
- Return exactly the three decisive bracketed references from the paragraph that determines the action.
- Never settle an invoice that needs approval, is on hold, is duplicate, or has conflicting records.
- Return strict JSON only.

JSON shape:
{"decisions":[{"packageId":"","action":"","vendorName":"","invoiceNumber":"","amountMinor":0,"currency":"","evidenceRefs":["","",""],"rationale":""}]}
"""

def extract_package_text(pkg: dict) -> str:
    return canonical_json(pkg)

async def groq_decide(packages: list[dict]) -> dict[str, dict]:
    uncached = []
    cache = {}

    with db() as c:
        for p in packages:
            h = sha256_text(extract_package_text(p))
            row = c.execute(
                "SELECT decision FROM decision_cache WHERE package_hash=?", (h,)
            ).fetchone()
            if row:
                cache[p["packageId"]] = json.loads(row["decision"])
            else:
                uncached.append((h, p))

    if uncached:
        user_content = canonical_json({
            "instruction": "Decide one action for each package.",
            "packages": [p for _, p in uncached]
        })

        async with httpx.AsyncClient(timeout=35) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ]
                }
            )
            resp.raise_for_status()
            payload = resp.json()
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        by_id = {d.get("packageId"): d for d in parsed.get("decisions", [])}

        with db() as c:
            for h, p in uncached:
                d = by_id.get(p["packageId"])
                if not d:
                    raise HTTPException(status_code=502, detail="Model did not decide every package")
                if d.get("action") not in ALLOWED_ACTIONS:
                    raise HTTPException(status_code=502, detail="Model returned invalid action")
                cache[p["packageId"]] = d
                c.execute(
                    "INSERT OR REPLACE INTO decision_cache(package_hash, decision) VALUES (?,?)",
                    (h, canonical_json(d))
                )

    return cache

def validate_decision(pkg: dict, d: dict):
    package_id = pkg.get("packageId")
    if not package_id:
        raise HTTPException(status_code=400, detail="Package missing packageId")

    refs = d.get("evidenceRefs") or []
    if len(refs) != 3 or any(not isinstance(x, str) or not x.strip() for x in refs):
        raise HTTPException(status_code=502, detail="Invalid evidence references")

    rationale = d.get("rationale") or ""
    if not (60 <= len(rationale) <= 1500):
        raise HTTPException(status_code=502, detail="Invalid rationale length")

    facts = {
        "vendorName": str(d.get("vendorName", "")),
        "invoiceNumber": str(d.get("invoiceNumber", "")),
        "amountMinor": int(d.get("amountMinor", 0)),
        "currency": str(d.get("currency", ""))
    }

    return {
        "packageId": package_id,
        "actionId": uuid.uuid4().hex + uuid.uuid4().hex[:8],
        "action": d["action"],
        "facts": facts,
        "evidenceRefs": refs,
        "rationale": rationale
    }

# ---------- routes ----------

@app.post("/message:send")
async def message_send(
    request: Request,
    authorization: str | None = Header(default=None)
):
    principal = await guard(request, authorization)
    body = await request.json()

    message = body.get("message") or {}
    message_id = message.get("messageId")
    if not message_id:
        raise HTTPException(status_code=400, detail="messageId required")

    parts = message.get("parts") or []
    batch_part = next((p for p in parts if p.get("mediaType") == BATCH_MEDIA), None)
    if not batch_part or "data" not in batch_part:
        raise HTTPException(status_code=415, detail="Invoice batch media type required")

    data = batch_part["data"]
    batch_id = data.get("batchId")
    packages = data.get("packages") or []
    if not batch_id or not packages:
        raise HTTPException(status_code=400, detail="batchId and packages required")

    msg_hash = sha256_text(canonical_json(message))

    with db() as c:
        existing = c.execute(
            "SELECT * FROM idempotency WHERE principal=? AND message_id=?",
            (principal, message_id)
        ).fetchone()

        if existing:
            if existing["content_hash"] != msg_hash:
                return error(409, "IDEMPOTENCY_CONFLICT", "Message ID reused with different content")
            row = c.execute(
                "SELECT * FROM tasks WHERE task_id=? AND principal=?",
                (existing["task_id"], principal)
            ).fetchone()
            return JSONResponse(
                media_type="application/a2a+json",
                content={"task": task_response(row)}
            )

    # Result continuation
    result_part = next((p for p in parts if p.get("mediaType") == RESULTS_MEDIA), None)
    if result_part:
        return await handle_results(principal, message, result_part["data"])

    # Initial proposal request
    decisions = await groq_decide(packages)

    proposals = []
    seen_packages = set()
    seen_actions = set()

    for pkg in packages:
        pid = pkg.get("packageId")
        if pid in seen_packages:
            raise HTTPException(status_code=400, detail="Duplicate packageId")
        seen_packages.add(pid)

        d = decisions[pid]
        proposal = validate_decision(pkg, d)

        if proposal["actionId"] in seen_actions:
            raise HTTPException(status_code=500, detail="Duplicate actionId")
        seen_actions.add(proposal["actionId"])
        proposals.append(proposal)

    task_id = str(uuid.uuid4())
    context_id = message.get("contextId") or str(uuid.uuid4())

    proposal_artifact = {
        "artifactId": str(uuid.uuid4()),
        "name": "Invoice action proposals",
        "parts": [{
            "mediaType": PROPOSALS_MEDIA,
            "data": {
                "batchId": batch_id,
                "proposals": proposals
            }
        }]
    }

    history = [message]

    with db() as c:
        c.execute("BEGIN IMMEDIATE")
        c.execute(
            """INSERT INTO tasks
               (task_id, principal, context_id, batch_id, state, history,
                proposal_artifact, receipt_artifact, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                task_id, principal, context_id, batch_id,
                "TASK_STATE_INPUT_REQUIRED", canonical_json(history),
                canonical_json(proposal_artifact), None, now(), now()
            )
        )
        for p in proposals:
            c.execute(
                """INSERT INTO proposals
                   (task_id, package_id, action_id, action, facts, evidence_refs, rationale)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    task_id, p["packageId"], p["actionId"], p["action"],
                    canonical_json(p["facts"]), canonical_json(p["evidenceRefs"]),
                    p["rationale"]
                )
            )
        c.execute(
            "INSERT INTO idempotency(principal,message_id,content_hash,task_id) VALUES (?,?,?,?)",
            (principal, message_id, msg_hash, task_id)
        )
        c.commit()

        row = c.execute(
            "SELECT * FROM tasks WHERE task_id=? AND principal=?",
            (task_id, principal)
        ).fetchone()

    return JSONResponse(
        media_type="application/a2a+json",
        content={"task": task_response(row)}
    )

async def handle_results(principal: str, message: dict, result_data: dict):
    task_id = message.get("taskId")
    context_id = message.get("contextId")
    message_id = message.get("messageId")
    batch_id = result_data.get("batchId")
    results = result_data.get("results") or []

    if not task_id or not context_id or not message_id:
        raise HTTPException(status_code=400, detail="taskId, contextId and messageId required")

    msg_hash = sha256_text(canonical_json(message))

    with db() as c:
        c.execute("BEGIN IMMEDIATE")

        existing = c.execute(
            "SELECT * FROM idempotency WHERE principal=? AND message_id=?",
            (principal, message_id)
        ).fetchone()

        if existing:
            if existing["content_hash"] != msg_hash:
                c.rollback()
                return error(409, "IDEMPOTENCY_CONFLICT", "Message ID reused with different content")
            row = c.execute(
                "SELECT * FROM tasks WHERE task_id=? AND principal=?",
                (existing["task_id"], principal)
            ).fetchone()
            c.commit()
            return JSONResponse(
                media_type="application/a2a+json",
                content={"task": task_response(row)}
            )

        row = c.execute(
            "SELECT * FROM tasks WHERE task_id=? AND principal=?",
            (task_id, principal)
        ).fetchone()

        if not row:
            c.rollback()
            raise HTTPException(status_code=404, detail="Task not found")

        if row["context_id"] != context_id:
            c.rollback()
            raise HTTPException(status_code=403, detail="Context mismatch")

        if row["batch_id"] != batch_id:
            c.rollback()
            raise HTTPException(status_code=403, detail="Batch mismatch")

        if row["state"] == "TASK_STATE_COMPLETED":
            c.rollback()
            return JSONResponse(
                media_type="application/a2a+json",
                content={"task": task_response(row)}
            )

        if row["state"] == "TASK_STATE_CANCELED":
            c.rollback()
            return error(409, "TASK_CANCELED", "Task was canceled")

        if row["state"] != "TASK_STATE_INPUT_REQUIRED":
            c.rollback()
            return error(409, "INVALID_TASK_STATE", "Task is not awaiting results")

        stored = {
            r["packageId"]: r for r in c.execute(
                "SELECT * FROM proposals WHERE task_id=?", (task_id,)
            ).fetchall()
        }

        executions = []

        for result in results:
            pid = result.get("packageId")
            aid = result.get("actionId")
            action = result.get("action")
            outcome = result.get("outcome")
            nonce = result.get("receiptNonce")

            p = stored.get(pid)
            if not p or p["action_id"] != aid or p["action"] != action:
                c.rollback()
                raise HTTPException(status_code=403, detail="Result does not match proposal")

            if outcome == "ACCEPTED":
                if not nonce:
                    c.rollback()
                    raise HTTPException(status_code=400, detail="Missing receiptNonce")

                already = c.execute(
                    "SELECT 1 FROM receipts WHERE task_id=? AND package_id=?",
                    (task_id, pid)
                ).fetchone()
                if already:
                    continue

                proposal_artifact = json.loads(row["proposal_artifact"])
                proposal_data = proposal_artifact["parts"][0]["data"]
                match = next(
                    x for x in proposal_data["proposals"]
                    if x["packageId"] == pid and x["actionId"] == aid
                )

                executions.append({
                    "packageId": pid,
                    "actionId": aid,
                    "action": action,
                    "receiptNonce": nonce,
                    "facts": match["facts"],
                    "evidenceRefs": match["evidenceRefs"]
                })

                c.execute(
                    """INSERT INTO receipts
                       (task_id, package_id, action_id, receipt_nonce)
                       VALUES (?,?,?,?)""",
                    (task_id, pid, aid, nonce)
                )

        if not executions:
            c.rollback()
            raise HTTPException(status_code=400, detail="No accepted executions")

        receipt_artifact = {
            "artifactId": str(uuid.uuid4()),
            "name": "Invoice action receipts",
            "parts": [{
                "mediaType": RECEIPTS_MEDIA,
                "data": {
                    "batchId": batch_id,
                    "executions": executions
                }
            }]
        }

        history = json.loads(row["history"])
        history.append(message)

        c.execute(
            """UPDATE tasks
               SET state='TASK_STATE_COMPLETED',
                   history=?,
                   receipt_artifact=?,
                   updated_at=?
               WHERE task_id=? AND principal=? AND state='TASK_STATE_INPUT_REQUIRED'""",
            (
                canonical_json(history), canonical_json(receipt_artifact),
                now(), task_id, principal
            )
        )

        c.execute(
            "INSERT INTO idempotency(principal,message_id,content_hash,task_id) VALUES (?,?,?,?)",
            (principal, message_id, msg_hash, task_id)
        )
        c.commit()

        row = c.execute(
            "SELECT * FROM tasks WHERE task_id=? AND principal=?",
            (task_id, principal)
        ).fetchone()

    return JSONResponse(
        media_type="application/a2a+json",
        content={"task": task_response(row)}
    )

@app.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    request: Request,
    authorization: str | None = Header(default=None)
):
    principal = await guard(request, authorization)
    with db() as c:
        row = get_task_for_principal(c, task_id, principal)
    return JSONResponse(
        media_type="application/a2a+json",
        content=task_response(row)
    )

@app.get("/tasks")
async def list_tasks(
    request: Request,
    authorization: str | None = Header(default=None)
):
    principal = await guard(request, authorization)
    with db() as c:
        rows = c.execute(
            "SELECT * FROM tasks WHERE principal=? ORDER BY created_at",
            (principal,)
        ).fetchall()
    return JSONResponse(
        media_type="application/a2a+json",
        content={"tasks": [task_response(r) for r in rows]}
    )

@app.post("/tasks/{task_id}:cancel")
async def cancel_task(
    task_id: str,
    request: Request,
    authorization: str | None = Header(default=None)
):
    principal = await guard(request, authorization)

    with db() as c:
        c.execute("BEGIN IMMEDIATE")
        row = c.execute(
            "SELECT * FROM tasks WHERE task_id=? AND principal=?",
            (task_id, principal)
        ).fetchone()

        if not row:
            c.rollback()
            raise HTTPException(status_code=404, detail="Task not found")

        if row["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
            c.rollback()
            return error(409, "TASK_TERMINAL", "Task already terminal")

        c.execute(
            """UPDATE tasks
               SET state='TASK_STATE_CANCELED', updated_at=?
               WHERE task_id=? AND principal=? AND state NOT IN
               ('TASK_STATE_COMPLETED','TASK_STATE_CANCELED')""",
            (now(), task_id, principal)
        )
        c.commit()
        row = c.execute(
            "SELECT * FROM tasks WHERE task_id=? AND principal=?",
            (task_id, principal)
        ).fetchone()

    return JSONResponse(
        media_type="application/a2a+json",
        content=task_response(row)
    )
