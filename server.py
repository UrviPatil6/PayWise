"""
Minimal web server for the browser demo UI (web/index.html). This is a
thin wrapper around Agent, not a new interface - the actual grading
interface is still `from agent import Agent; agent.next(...)` per the
assignment spec. This just gives that same call an HTTP front door so it
can be driven from a browser instead of the terminal (cli.py).

One Agent instance per browser session, kept in memory only - same
lifetime/scope tradeoff as the CLI (see README "Known limitations"), just
addressable over HTTP. Not meant to run as a real multi-user production
service: no auth, no persistence, no session expiry - it's a demo surface.

In production (e.g. Render - see render.yaml), this module is imported by
a real WSGI server (`gunicorn server:app`), which never executes the
`if __name__ == "__main__"` block below - that block is only for local
`python server.py` runs. `sessions` being an in-process dict means a
platform restart/redeploy/free-tier spin-down clears all active
conversations, same as the CLI losing state when the process exits.
"""

import os
import threading
import time
import uuid

from flask import Flask, jsonify, request, send_from_directory

from agent import Agent
from config import get_logger
from state import ConversationState

logger = get_logger(__name__)
app = Flask(__name__, static_folder="web", static_url_path="")

# session_id -> Agent. In-memory only, cleared on server restart - matches
# the assignment's own model where an Agent() is one conversation's worth
# of state and nothing more.
sessions: dict[str, Agent] = {}

# Retry limits (MAX_ACCOUNT_LOOKUP_ATTEMPTS, MAX_VERIFICATION_ATTEMPTS) live
# on the Agent/Session, so they reset for free every time a new session is
# created - unlimited POST /api/session calls would make those limits
# meaningless. A per-IP sliding window is a coarse defense (request.
# remote_addr is the proxy's IP on Render unless ProxyFix is configured -
# a known limitation, not a real per-client guarantee), but it's enough to
# stop trivial scripted brute-forcing without adding a dependency.
_SESSION_CREATION_WINDOW_SECONDS = 300
_SESSION_CREATION_LIMIT = 10
_session_creation_log: dict[str, list] = {}
_session_creation_lock = threading.Lock()


def _session_creation_allowed(client_ip: str) -> bool:
    now = time.time()
    with _session_creation_lock:
        recent = [t for t in _session_creation_log.get(client_ip, []) if now - t < _SESSION_CREATION_WINDOW_SECONDS]
        if len(recent) >= _SESSION_CREATION_LIMIT:
            _session_creation_log[client_ip] = recent
            return False
        recent.append(now)
        _session_creation_log[client_ip] = recent
        return True

# session_id -> [{"sender": "user"|"bot", "text": str}, ...]. Separate from
# Agent/Session on purpose: Session holds business state (what's been
# verified, collected, etc.), not display history - the chat transcript is
# purely a UI concern, needed only so a page refresh can rebuild what was
# on screen (see GET /api/session/<id>), same in-memory/no-persistence
# scope as `sessions` above.
transcripts: dict[str, list] = {}


def _progress_steps(agent: Agent) -> list:
    """
    A UI-only projection of session state into the 5 steps the demo's
    sidebar visualizes. This is presentation logic, not business logic -
    it reads Session fields and formats them for display, it never
    decides anything. Deliberately only exposes what's already safe to
    show the user (account_id, balance, verified true/false, amount,
    transaction_id) - the same fields the chat replies themselves would
    reveal, never DOB/Aadhaar/pincode/card data.

    Each step gets a status: "done", "active" (in progress right now),
    "failed" (conversation closed without completing it), or "pending".
    """
    s = agent.session
    closed = s.state == ConversationState.CLOSED
    steps = []

    account_done = s.account is not None
    steps.append({
        "key": "account",
        "title": "Account Lookup",
        "status": "done" if account_done else ("failed" if closed else "active"),
        "detail": (
            f"Account found ({s.account.account_id})" if account_done
            else "Account not found" if closed
            else "Waiting for account ID"
        ),
    })

    identity_active = account_done and not s.verified and not closed
    steps.append({
        "key": "identity",
        "title": "Identity Verification",
        "status": "done" if s.verified else ("failed" if closed and account_done else ("active" if identity_active else "pending")),
        "detail": (
            "Identity verified" if s.verified
            else "Verification failed" if closed and account_done
            else f"{s.verification_attempts} failed attempt(s)" if s.verification_attempts and identity_active
            else "Waiting for name + DOB/Aadhaar/pincode" if identity_active
            else "Not started"
        ),
    })

    steps.append({
        "key": "balance",
        "title": "Balance",
        "status": "done" if s.verified else "pending",
        "detail": f"₹{s.account.balance:,.2f} outstanding" if s.verified else "Available after verification",
    })

    payment_done = s.transaction_id is not None
    payment_active = s.verified and not payment_done and not closed
    steps.append({
        "key": "payment",
        "title": "Payment Details",
        "status": "done" if payment_done else ("failed" if closed and s.verified else ("active" if payment_active else "pending")),
        "detail": (
            f"₹{s.payment_amount:,.2f} charged" if payment_done
            else "Payment not completed" if closed and s.verified
            else "Awaiting payment details (amount & card info)" if payment_active
            else "Not started"
        ),
    })

    steps.append({
        "key": "confirmation",
        "title": "Confirmation",
        "status": "done" if payment_done else ("failed" if closed and s.verified else "pending"),
        "detail": f"Transaction {s.transaction_id}" if payment_done else "Payment confirmation and receipt",
    })

    return steps


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/session")
def new_session():
    if not _session_creation_allowed(request.remote_addr or "unknown"):
        return jsonify({"error": "Too many new conversations from this connection - please wait a few minutes and try again."}), 429
    session_id = uuid.uuid4().hex
    agent = Agent()
    sessions[session_id] = agent
    transcripts[session_id] = []
    return jsonify({"session_id": session_id, "progress": _progress_steps(agent)})


@app.get("/api/session/<session_id>")
def get_session(session_id):
    """Lets the client resume a session across a page refresh (sessionId
    cached client-side in sessionStorage - see web/index.html) instead of
    always starting a new Agent(). Progress is computed fresh from the
    real Agent every time, never cached, so a session that no longer
    exists here (server restarted, Render free-tier spin-down) correctly
    404s instead of a stale client-side guess pretending it's still alive.
    The transcript is what lets the client rebuild the actual chat
    bubbles on resume, not just the sidebar - it's plain display text
    already shown to this same user once, nothing new is exposed.
    """
    agent = sessions.get(session_id)
    if agent is None:
        return jsonify({"error": "unknown session_id"}), 404
    return jsonify({
        "session_id": session_id,
        "progress": _progress_steps(agent),
        "transcript": transcripts.get(session_id, []),
    })


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    user_input = body.get("message", "")

    agent = sessions.get(session_id)
    if agent is None:
        return jsonify({"error": "unknown session_id - call /api/session first"}), 400

    transcript = transcripts.setdefault(session_id, [])
    if user_input.strip():
        # AWAIT_CARD_DETAILS messages can contain a real card number and
        # CVV typed as plain chat text - never persist that verbatim, even
        # in this display-only transcript. Same rule agent._extract already
        # applies to keep raw card data out of the LLM path, extended here
        # to keep it out of this second in-memory store too (see
        # FINDINGS.md's card-data-clearing findings for why this class of
        # data gets this treatment and DOB/Aadhaar/pincode don't).
        is_card_details_turn = agent.session.state == ConversationState.AWAIT_CARD_DETAILS
        display_text = "[card details submitted]" if is_card_details_turn else user_input
        transcript.append({"sender": "user", "text": display_text})

    result = agent.next(user_input)
    transcript.append({"sender": "bot", "text": result["message"]})

    result["progress"] = _progress_steps(agent)
    return jsonify(result)


if __name__ == "__main__":
    # 0.0.0.0 + $PORT: what Render (and most PaaS platforms) require for
    # local `python server.py` to also work unmodified in that environment.
    # Real deployments should still run behind gunicorn (see render.yaml),
    # not this dev server.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
