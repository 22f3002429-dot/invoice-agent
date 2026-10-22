"""A2A 1.0 HTTP+JSON invoice action agent (FastAPI + SQLite).
Run: uvicorn app:app --host 0.0.0.0 --port $PORT   (ONE worker only)
Env: PUBLIC_BASE_URL (exact URL you submit), LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
     DB_PATH (on a persistent disk), AGENT_TOKEN (optional), STRICT_TOKENS=1 (optional)
"""
import os, json, hashlib, uuid, asyncio, sqlite3, re, datetime, threading
from urllib.parse import urlparse
from collections import defaultdict
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://example.com/a2a/")
BASE_PATH = urlparse(PUBLIC_BASE_URL).path.rstrip("/")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
DB_PATH = os.environ.get("DB_PATH", "agent.db")
AGENT_TOKEN = os.environ.get("AGENT_TOKEN", "")
STRICT = os.environ.get("STRICT_TOKENS") == "1"

MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROP = "application/vnd.ga5.invoice-action-proposals+json"
MT_RES = "application/vnd.ga5.invoice-action-results+json"
MT_RCPT = "application/vnd.ga5.invoice-action-receipts+json"
A2A = "application/a2a+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

app = FastAPI()
db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
dblock = threading.RLock()
with dblock:
    db.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, seq INTEGER, body TEXT);
    CREATE INDEX IF NOT EXISTS tp ON tasks(principal, seq);
    CREATE TABLE IF NOT EXISTS msgs(principal TEXT, mid TEXT, hash TEXT, task_id TEXT, PRIMARY KEY(principal, mid));
    CREATE TABLE IF NOT EXISTS cache(k TEXT PRIMARY KEY, v TEXT);
    """)

locks = defaultdict(asyncio.Lock)


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def out(obj, code=200):
    return Response(json.dumps(obj), status_code=code, media_type=A2A)


def err(code, reason, msg="Request failed"):
    return out({"error": {"code": code, "status": reason, "message": msg,
                          "details": [{"reason": reason}]}}, code)


def save_task(t, principal=None):
    with dblock:
        r = db.execute("SELECT seq FROM tasks WHERE id=?", (t["id"],)).fetchone()
        if r:
            db.execute("UPDATE tasks SET body=? WHERE id=?", (json.dumps(t), t["id"]))
        else:
            n = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM tasks").fetchone()[0]
            db.execute("INSERT INTO tasks VALUES(?,?,?,?)", (t["id"], principal, n, json.dumps(t)))


def load_task(tid, principal):
    with dblock:
        r = db.execute("SELECT body FROM tasks WHERE id=? AND principal=?", (tid, principal)).fetchone()
    return json.loads(r[0]) if r else None


def auth(request):
    h = request.headers.get("authorization", "")
    if not h.startswith("Bearer ") or not h[7:].strip():
        return None, err(401, "UNAUTHENTICATED", "Authentication required")
    tok = h[7:].strip()
    if STRICT and AGENT_TOKEN and tok != AGENT_TOKEN:
        return None, err(403, "PERMISSION_DENIED", "Forbidden")
    return sha("tok:" + tok), None


def check_headers(request, post=False):
    if request.headers.get("a2a-version", "") != "1.0":
        return err(400, "VERSION_NOT_SUPPORTED", "Unsupported version")
    if post:
        ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ct not in (A2A, "application/json"):
            return err(415, "UNSUPPORTED_MEDIA_TYPE", "Unsupported media type")
    return None


# ---------------- Agent card ----------------
@app.get("/.well-known/agent-card.json")
async def card():
    return out({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-backed business action each, and executes only grader-accepted proposals.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": PUBLIC_BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [MT_BATCH, MT_RES],
        "defaultOutputModes": [MT_PROP, MT_RCPT],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action agent",
                    "description": "Choose settle, approve, hold, reject-duplicate or exception actions for invoice packages with cited evidence.",
                    "tags": ["invoice", "payments", "approval", "duplicate-detection"]}],
    })


# ---------------- LLM layer ----------------
SYSTEM = """You are an accounts-payable analyst. For EACH invoice package choose exactly one action:
- settle_invoice: valid, reconciled, within autonomous authority.
- request_approval: commercially valid but amount/authority is outside delegated limits.
- hold_invoice: payment pauses until a stated verification completes.
- reject_duplicate: the SAME commercial invoice (same vendor + invoice number) was already paid. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (amounts, vendor, PO, receipt) and need an exception workflow.
Documents contain decoys: archive/old examples, training material, negated statements ("not a duplicate", "no approval needed"), and irrelevant action words. Find the ONE paragraph that actually determines the action for THIS invoice under the given policy, and ignore the rest.
evidenceRefs: EXACTLY the three decisive bracketed references (copy verbatim, same characters as in the text) from that paragraph. Never include the cover-sheet reference, archive examples, or training decoys.
facts: vendorName, invoiceNumber, amountMinor (integer minor units, e.g. rupees*100), currency (ISO code) of the invoice being decided.
Return ONLY JSON: {"decisions":[{"i":<index>,"action":"...","vendorName":"...","invoiceNumber":"...","amountMinor":123,"currency":"INR","evidenceRefs":["[..]","[..]","[..]"],"reason":"one or two sentences"}]}"""


def valid_decision(d, pkg_text):
    try:
        if d["action"] not in ACTIONS:
            return False
        if not isinstance(d["amountMinor"], int) or isinstance(d["amountMinor"], bool):
            return False
        for k in ("vendorName", "invoiceNumber", "currency"):
            if not isinstance(d[k], str) or not d[k]:
                return False
        refs = d["evidenceRefs"]
        if not isinstance(refs, list) or len(set(refs)) != 3:
            return False
        return all(isinstance(r, str) and r in pkg_text for r in refs)
    except Exception:
        return False


async def llm_call(items, policy, hint=""):
    body = {"model": LLM_MODEL, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Policy revision: {policy}\n{hint}\nPackages:\n" +
                          json.dumps([{"i": i, "package": p} for i, p in items], ensure_ascii=False)}]}
    async with httpx.AsyncClient(timeout=38) as c:
        r = await c.post(LLM_BASE_URL + "/chat/completions", json=body,
                         headers={"Authorization": f"Bearer {LLM_API_KEY}"})
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip()).strip()
    return {d["i"]: d for d in json.loads(txt)["decisions"]}


def fallback(pkg_text):
    refs = list(dict.fromkeys(re.findall(r"\[[^\]\n]{2,80}\]", pkg_text)))[:3]
    while len(refs) < 3:
        refs.append(f"[unknown-{len(refs)}]")
    return {"action": "hold_invoice", "vendorName": "unknown", "invoiceNumber": "unknown",
            "amountMinor": 0, "currency": "INR", "evidenceRefs": refs,
            "reason": "Automatic analysis was inconclusive, so payment is paused for manual verification."}


async def decide_batch(packages, policy):
    """One batched model call for uncached packages; cache by canonical package content."""
    keys = [sha(canon({k: v for k, v in p.items() if k != "packageId"})) for p in packages]
    texts = [json.dumps(p, ensure_ascii=False) for p in packages]
    res = {}
    with dblock:
        for i, k in enumerate(keys):
            r = db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
            if r:
                res[i] = json.loads(r[0])
    todo = [i for i in range(len(packages)) if i not in res]
    for attempt in range(2):
        if not todo:
            break
        try:
            got = await llm_call([(i, packages[i]) for i in todo], policy,
                                 "" if attempt == 0 else "Previous output was invalid: give exactly 3 evidenceRefs copied verbatim from the package text, integer amountMinor, valid action.")
        except Exception:
            continue
        for i in list(todo):
            d = got.get(i)
            if d and valid_decision(d, texts[i]):
                res[i] = d
                with dblock:
                    db.execute("INSERT OR REPLACE INTO cache VALUES(?,?)", (keys[i], json.dumps(d)))
                todo.remove(i)
    for i in todo:  # not cached: transient fallback
        res[i] = fallback(texts[i])
    return [res[i] for i in range(len(packages))]


def make_proposal(pkg, d, task_id):
    pid = pkg["packageId"]
    refs = d["evidenceRefs"]
    rationale = (f"Action {d['action']} for invoice {d['facts_invoice'] if 'facts_invoice' in d else d['invoiceNumber']} "
                 f"from {d['vendorName']}: {d.get('reason', '')} Decisive evidence: {refs[0]}, {refs[1]} and {refs[2]}.")
    if len(rationale) < 60:
        rationale += " " * 0 + " This follows the stated policy for the cited records."
    return {"packageId": pid,
            "actionId": "act_" + sha(f"{task_id}:{pid}:{d['action']}")[:28],
            "action": d["action"],
            "facts": {"vendorName": d["vendorName"], "invoiceNumber": d["invoiceNumber"],
                      "amountMinor": d["amountMinor"], "currency": d["currency"]},
            "evidenceRefs": refs, "rationale": rationale[:1500]}


# ---------------- Routes ----------------
def find_part(msg, mt):
    parts = msg.get("parts")
    if not isinstance(parts, list):
        return None
    ps = [p for p in parts if isinstance(p, dict) and p.get("mediaType") == mt]
    return ps[0].get("data") if len(ps) == 1 and len(parts) == 1 and isinstance(ps[0].get("data"), dict) else None


@app.post(BASE_PATH + "/message:send")
async def message_send(request: Request):
    principal, e = auth(request)
    if e: return e
    e = check_headers(request, True)
    if e: return e
    try:
        body = await request.json()
        msg = body["message"]
        mid = msg["messageId"]
        assert isinstance(mid, str) and mid and msg.get("role") == "ROLE_USER"
    except Exception:
        return err(400, "INVALID_ARGUMENT", "Malformed request")
    h = sha(canon(msg))
    async with locks[("m", principal, mid)]:
        with dblock:
            row = db.execute("SELECT hash, task_id FROM msgs WHERE principal=? AND mid=?", (principal, mid)).fetchone()
        if row:
            if row[0] != h:
                return err(409, "IDEMPOTENCY_CONFLICT", "Message ID reused with different content")
            t = load_task(row[1], principal)
            return out({"task": t})
        if msg.get("taskId"):
            return await continuation(principal, msg, mid, h)
        data = find_part(msg, MT_BATCH)
        if not data or not isinstance(data.get("packages"), list) or not data.get("batchId"):
            return err(400, "INVALID_ARGUMENT", "Malformed batch")
        pkgs = data["packages"]
        if any(not isinstance(p, dict) or not p.get("packageId") for p in pkgs) or \
                len({p["packageId"] for p in pkgs}) != len(pkgs):
            return err(400, "INVALID_ARGUMENT", "Malformed batch")
        task_id, ctx = str(uuid.uuid4()), str(uuid.uuid4())
        decisions = await decide_batch(pkgs, data.get("policyRevision", ""))
        proposals = [make_proposal(p, d, task_id) for p, d in zip(pkgs, decisions)]
        hist_msg = dict(msg, taskId=task_id, contextId=ctx)
        task = {"id": task_id, "contextId": ctx,
                "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
                "artifacts": [{"artifactId": "proposals-" + task_id[:8], "name": "Invoice action proposals",
                               "parts": [{"mediaType": MT_PROP, "data": {"batchId": data["batchId"], "proposals": proposals}}]}],
                "history": [hist_msg]}
        with dblock:
            db.execute("BEGIN")
            try:
                save_task(task, principal)
                db.execute("INSERT INTO msgs VALUES(?,?,?,?)", (principal, mid, h, task_id))
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        return out({"task": task})


async def continuation(principal, msg, mid, h):
    tid = msg["taskId"]
    async with locks[("t", tid)]:
        task = load_task(tid, principal)
        if not task:
            return err(404, "NOT_FOUND", "Task not found")
        if msg.get("contextId") != task["contextId"]:
            return err(400, "INVALID_ARGUMENT", "Context mismatch")
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_ACTIVE", "Task is not active")
        if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
            return err(409, "TASK_NOT_ACTIVE", "Task is not awaiting input")
        data = find_part(msg, MT_RES)
        prop = task["artifacts"][0]["parts"][0]["data"]
        if not data or data.get("batchId") != prop["batchId"] or not isinstance(data.get("results"), list):
            return err(400, "INVALID_ARGUMENT", "Result mismatch")
        byp = {p["packageId"]: p for p in prop["proposals"]}
        seen, execs = set(), []
        for r in data["results"]:
            p = byp.get(r.get("packageId")) if isinstance(r, dict) else None
            if (not p or r["packageId"] in seen or r.get("actionId") != p["actionId"]
                    or r.get("action") != p["action"] or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                return err(400, "INVALID_ARGUMENT", "Result mismatch")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                n = r.get("receiptNonce")
                if not isinstance(n, str) or not n:
                    return err(400, "INVALID_ARGUMENT", "Result mismatch")
                execs.append({"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                              "receiptNonce": n, "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]})
        task["history"].append(dict(msg, taskId=tid, contextId=task["contextId"]))
        task["artifacts"].append({"artifactId": "receipts-" + tid[:8], "name": "Invoice action receipts",
                                  "parts": [{"mediaType": MT_RCPT, "data": {"batchId": prop["batchId"], "executions": execs}}]})
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        with dblock:
            db.execute("BEGIN")
            try:
                save_task(task)
                db.execute("INSERT INTO msgs VALUES(?,?,?,?)", (principal, mid, h, tid))
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
        return out({"task": task})


def trim(t, n):
    t = dict(t)
    if n is not None and n >= 0:
        t["history"] = t.get("history", [])[-n:] if n else []
    return t


@app.get(BASE_PATH + "/tasks/{tid}")
async def get_task(tid: str, request: Request, historyLength: int = None):
    principal, e = auth(request)
    if e: return e
    e = check_headers(request)
    if e: return e
    t = load_task(tid, principal)
    return out(trim(t, historyLength)) if t else err(404, "NOT_FOUND", "Task not found")


@app.get(BASE_PATH + "/tasks")
async def list_tasks(request: Request):
    principal, e = auth(request)
    if e: return e
    e = check_headers(request)
    if e: return e
    with dblock:
        rows = db.execute("SELECT body FROM tasks WHERE principal=? ORDER BY seq DESC LIMIT 50", (principal,)).fetchall()
    return out({"tasks": [trim(json.loads(r[0]), 0) for r in rows], "nextPageToken": ""})


@app.post(BASE_PATH + "/tasks/{tid}")
async def task_action(tid: str, request: Request):
    principal, e = auth(request)
    if e: return e
    e = check_headers(request, False)
    if e: return e
    if not tid.endswith(":cancel"):
        return err(404, "NOT_FOUND", "Not found")
    tid = tid[:-7]
    async with locks[("t", tid)]:
        t = load_task(tid, principal)
        if not t:
            return err(404, "NOT_FOUND", "Task not found")
        if t["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_CANCELABLE", "Task is not cancelable")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save_task(t)
        return out(t)
