import os
import json
import hashlib
import logging
import asyncio
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse

logger = logging.getLogger("uvicorn.error")

DB_PATH = os.getenv("DB_PATH", "/tmp/agent.db")

PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://invoice-agent-c437.onrender.com/a2a"
).rstrip("/")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

BEARER_PREFIX = "Bearer "

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

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_CHUNK_SIZE = 1
GROQ_TIMEOUT = 35
GROQ_MAX_RETRIES = 0
MAX_PACKAGE_TEXT_CHARS = 9000

app = FastAPI()


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
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def a2a_error(status: int, code: str, message: str):
    return JSONResponse(
        status_code=status,
        media_type="application/a2a+json",
        content={
            "error": {
                "code": code,
                "message": message
            }
        }
    )


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return JSONResponse(
        media_type="application/json",
        content={
            "name": "Invoice Action Agent",
            "description": (
                "Reads invoice packages, proposes one evidence-based business "
                "action per package, and records only approved executions."
            ),
            "version": "1.0.0",
            "capabilities": {
                "streaming": False,
                "pushNotifications": False
            },
            "defaultInputModes": [BATCH_MEDIA],
            "defaultOutputModes": [
                PROPOSALS_MEDIA,
                RECEIPTS_MEDIA
            ],
            "skills": [
                {
                    "id": "invoice_action_agent",
                    "name": "invoice_action_agent",
                    "description": (
                        "Chooses settle_invoice, request_approval, hold_invoice, "
                        "reject_duplicate, or open_exception from invoice evidence."
                    ),
                    "tags": [
                        "invoice",
                        "accounts-payable",
                        "approval",
                        "duplicate-detection",
                        "exception-handling"
                    ]
                }
            ],
            "supportedInterfaces": [
                {
                    "url": PUBLIC_BASE_URL,
                    "protocolBinding": "HTTP+JSON",
                    "protocolVersion": "1.0"
                }
            ]
        }
    )


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
        raise HTTPException(status_code=404, detail="Task not found")

    return row


def task_response(row) -> dict:
    history = json.loads(row["history"])

    response = {
        "id": row["task_id"],
        "contextId": row["context_id"],
        "status": {
            "state": row["state"],
            "timestamp": row["updated_at"]
        },
        "history": history
    }

    artifacts = []

    if row["proposal_artifact"]:
        artifacts.append(json.loads(row["proposal_artifact"]))

    if row["receipt_artifact"]:
        artifacts.append(json.loads(row["receipt_artifact"]))

    if artifacts:
        response["artifacts"] = artifacts

    return response


SYSTEM_PROMPT = """Choose one action for each invoice package:
settle_invoice=valid, reconciled, within autonomous authority.
request_approval=valid but outside authority.
hold_invoice=payment paused pending verification.
reject_duplicate=same commercial invoice already paid.
open_exception=material records conflict.

Ignore cover sheets, archives, decoys, and irrelevant action words.
Return exactly three decisive bracketed references from the decisive paragraph.
Never settle an invoice needing approval, hold, duplicate rejection, or exception.
Convert INR to paise for amountMinor.
Return only JSON:
{"decisions":[{"packageId":"","action":"","vendorName":"","invoiceNumber":"","amountMinor":0,"currency":"","evidenceRefs":["","",""],"rationale":""}]}
"""


def collect_text(value: Any, collected: list[str]):
    if isinstance(value, str):
        if value.strip():
            collected.append(value.strip())
    elif isinstance(value, dict):
        for item in value.values():
            collect_text(item, collected)
    elif isinstance(value, list):
        for item in value:
            collect_text(item, collected)


def extract_package_text(package: dict) -> str:
    collected: list[str] = []
    collect_text(package, collected)

    text = "\n".join(collected)

    return text[:MAX_PACKAGE_TEXT_CHARS]


def fallback_decision(package: dict) -> dict:
    return {
        "packageId": package["packageId"],
        "action": "hold_invoice",
        "vendorName": "Unknown",
        "invoiceNumber": "Unknown",
        "amountMinor": 0,
        "currency": "INR",
        "evidenceRefs": [
            "[verification-pending-1]",
            "[verification-pending-2]",
            "[verification-pending-3]"
        ],
        "rationale": (
            "Hold invoice pending verification because the automated decision "
            "service could not produce a validated evidence-based decision "
            "for this package."
        )
    }


async def decide_chunk(chunk: list[dict]) -> dict[str, dict]:
    package = chunk[0]

    user_content = canonical_json({
        "instruction": "Choose one action for this invoice package.",
        "package": package
    })

    request_body = {
        "model": GROQ_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=GROQ_TIMEOUT) as client:
        response = await client.post(
            GROQ_URL,
            headers=headers,
            json=request_body
        )

    logger.info(f"Groq status: {response.status_code}")

    if response.status_code == 429:
        logger.warning(f"Groq rate limit: {response.text}")
        raise RuntimeError("Groq rate limit")

    if response.status_code != 200:
        logger.error(f"Groq error: {response.text}")
        raise RuntimeError(f"Groq returned {response.status_code}")

    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    return {
        item.get("packageId"): item
        for item in parsed.get("decisions", [])
    }


async def groq_decide(packages: list[dict]) -> dict[str, dict]:
    decisions = {}
    uncached = []

    with db() as c:
        for package in packages:
            package_hash = sha256_text(extract_package_text(package))

            row = c.execute(
                "SELECT decision FROM decision_cache WHERE package_hash=?",
                (package_hash,)
            ).fetchone()

            if row:
                decisions[package["packageId"]] = json.loads(row["decision"])
            else:
                uncached.append((package_hash, package))

    chunks = [
        uncached[index:index + GROQ_CHUNK_SIZE]
        for index in range(0, len(uncached), GROQ_CHUNK_SIZE)
    ]

    for chunk in chunks:
        try:
            model_decisions = await decide_chunk(
                [package for _, package in chunk]
            )

            for package_hash, package in chunk:
                decision = model_decisions.get(package["packageId"])

                if not decision:
                    raise RuntimeError("Missing package decision")

                if decision.get("action") not in ALLOWED_ACTIONS:
                    raise RuntimeError("Invalid action")

                evidence_refs = decision.get("evidenceRefs") or []
                rationale = decision.get("rationale") or ""

                if (
                    len(evidence_refs) != 3
                    or not (60 <= len(rationale) <= 1500)
                ):
                    raise RuntimeError("Invalid decision schema")

                decisions[package["packageId"]] = decision

                with db() as c:
                    c.execute(
                        """INSERT OR REPLACE INTO decision_cache
                           (package_hash, decision) VALUES (?, ?)""",
                        (
                            package_hash,
                            canonical_json(decision)
                        )
                    )

        except Exception as exc:
            logger.exception(f"Groq chunk failed: {exc}")

            for package_hash, package in chunk:
                decisions[package["packageId"]] = fallback_decision(package)

    return decisions


def validate_decision(package: dict, decision: dict) -> dict:
    package_id = package.get("packageId")

    if not package_id:
        raise HTTPException(
            status_code=400,
            detail="Package missing packageId"
        )

    evidence_refs = decision.get("evidenceRefs") or []

    if (
        len(evidence_refs) != 3
        or any(
            not isinstance(ref, str) or not ref.strip()
            for ref in evidence_refs
        )
    ):
        raise HTTPException(
            status_code=502,
            detail="Invalid evidence references"
        )

    rationale = decision.get("rationale") or ""

    if not (60 <= len(rationale) <= 1500):
        raise HTTPException(
            status_code=502,
            detail="Invalid rationale length"
        )

    facts = {
        "vendorName": str(decision.get("vendorName", "")),
        "invoiceNumber": str(decision.get("invoiceNumber", "")),
        "amountMinor": int(decision.get("amountMinor", 0)),
        "currency": str(decision.get("currency", ""))
    }

    return {
        "packageId": package_id,
        "actionId": uuid.uuid4().hex + uuid.uuid4().hex[:8],
        "action": decision["action"],
        "facts": facts,
        "evidenceRefs": evidence_refs,
        "rationale": rationale
    }


@app.post("/a2a/message:send")
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

    batch_part = next(
        (
            part for part in parts
            if part.get("mediaType") == BATCH_MEDIA
        ),
        None
    )

    result_part = next(
        (
            part for part in parts
            if part.get("mediaType") == RESULTS_MEDIA
        ),
        None
    )

    if not batch_part and not result_part:
        raise HTTPException(
            status_code=415,
            detail="Required invoice batch or result media type missing"
        )

    message_hash = sha256_text(canonical_json(message))

    with db() as c:
        existing = c.execute(
            """SELECT * FROM idempotency
               WHERE principal=? AND message_id=?""",
            (principal, message_id)
        ).fetchone()

        if existing:
            if existing["content_hash"] != message_hash:
                return a2a_error(
                    409,
                    "IDEMPOTENCY_CONFLICT",
                    "Message ID reused with different content"
                )

            row = c.execute(
                """SELECT * FROM tasks
                   WHERE task_id=? AND principal=?""",
                (existing["task_id"], principal)
            ).fetchone()

            return JSONResponse(
                media_type="application/a2a+json",
                content={"task": task_response(row)}
            )

    if result_part:
        return await handle_results(
            principal,
            message,
            result_part.get("data") or {},
            message_hash
        )

    batch_data = batch_part.get("data") or {}
    batch_id = batch_data.get("batchId")
    packages = batch_data.get("packages") or []

    if not batch_id or not packages:
        raise HTTPException(
            status_code=400,
            detail="batchId and packages required"
        )

    decisions = await groq_decide(packages)

    proposals = []
    seen_package_ids = set()
    seen_action_ids = set()

    for package in packages:
        package_id = package.get("packageId")

        if package_id in seen_package_ids:
            raise HTTPException(
                status_code=400,
                detail="Duplicate packageId"
            )

        seen_package_ids.add(package_id)

        decision = decisions[package_id]
        proposal = validate_decision(package, decision)

        if proposal["actionId"] in seen_action_ids:
            raise HTTPException(
                status_code=500,
                detail="Duplicate actionId"
            )

        seen_action_ids.add(proposal["actionId"])
        proposals.append(proposal)

    task_id = str(uuid.uuid4())
    context_id = message.get("contextId") or str(uuid.uuid4())

    proposal_artifact = {
        "artifactId": str(uuid.uuid4()),
        "name": "Invoice action proposals",
        "parts": [
            {
                "mediaType": PROPOSALS_MEDIA,
                "data": {
                    "batchId": batch_id,
                    "proposals": proposals
                }
            }
        ]
    }

    history = [message]

    with db() as c:
        c.execute("BEGIN IMMEDIATE")

        c.execute(
            """INSERT INTO tasks
               (task_id, principal, context_id, batch_id, state, history,
                proposal_artifact, receipt_artifact, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                principal,
                context_id,
                batch_id,
                "TASK_STATE_INPUT_REQUIRED",
                canonical_json(history),
                canonical_json(proposal_artifact),
                None,
                now(),
                now()
            )
        )

        for proposal in proposals:
            c.execute(
                """INSERT INTO proposals
                   (task_id, package_id, action_id, action, facts,
                    evidence_refs, rationale)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_id,
                    proposal["packageId"],
                    proposal["actionId"],
                    proposal["action"],
                    canonical_json(proposal["facts"]),
                    canonical_json(proposal["evidenceRefs"]),
                    proposal["rationale"]
                )
            )

        try:
            c.execute(
                """INSERT INTO idempotency
                   (principal, message_id, content_hash, task_id)
                   VALUES (?, ?, ?, ?)""",
                (
                    principal,
                    message_id,
                    message_hash,
                    task_id
                )
            )
        except sqlite3.IntegrityError:
            c.rollback()

            existing = c.execute(
                """SELECT * FROM idempotency
                   WHERE principal=? AND message_id=?""",
                (principal, message_id)
            ).fetchone()

            if existing:
                row = c.execute(
                    """SELECT * FROM tasks
                       WHERE task_id=? AND principal=?""",
                    (existing["task_id"], principal)
                ).fetchone()

                if row:
                    return JSONResponse(
                        media_type="application/a2a+json",
                        content={"task": task_response(row)}
                    )

            raise

        c.commit()

        row
