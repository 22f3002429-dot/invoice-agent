import os
import json
import hashlib
import logging
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
GROQ_CHUNK_SIZE = 3
GROQ_TIMEOUT = 40

app = FastAPI()


# ---------------- storage ----------------

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


# ---------------- agent card ----------------

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


# ---------------- protocol guards ----------------

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


# ---------------- AI decisions ----------------

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
- Convert money to the smallest currency unit for amountMinor. For INR, rupees become paise.
- Return strict JSON only.

JSON shape:
{"decisions":[{"packageId":"","action":"","vendorName":"","invoiceNumber":"","amountMinor":0,"currency":"","evidenceRefs":["","",""],"rationale":""}]}
"""


def extract_package_text(package: dict) -> str:
    return canonical_json(package)


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
    user_content = canonical_json({
        "instruction": "Decide one action for each package.",
        "packages": chunk
    })

    async with httpx.AsyncClient(timeout=GROQ_TIMEOUT) as client:
        response = await client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
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
        )

    logger.info(f"Groq status: {response.status_code}")

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


def
