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
        "title": "Balance Retrieved",
        "status": "done" if s.verified else "pending",
        "detail": f"Outstanding balance ₹{s.account.balance:,.2f}" if s.verified else "Awaiting identity verification",
    })

    payment_done = s.transaction_id is not None
    payment_active = s.verified and not payment_done and not closed
    steps.append({
        "key": "payment",
        "title": "Payment",
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
    session_id = uuid.uuid4().hex
    agent = Agent()
    sessions[session_id] = agent
    return jsonify({"session_id": session_id, "progress": _progress_steps(agent)})


@app.get("/api/session/<session_id>")
def get_session(session_id):
    """Lets the client resume a session across a page refresh (sessionId
    cached client-side in sessionStorage - see web/index.html) instead of
    always starting a new Agent(). Only ever returns the same
    non-sensitive progress projection /api/chat already exposes, computed
    fresh from the real Agent - never cached, so a session that no longer
    exists here (server restarted, Render free-tier spin-down) correctly
    404s instead of a stale client-side guess pretending it's still alive.
    """
    agent = sessions.get(session_id)
    if agent is None:
        return jsonify({"error": "unknown session_id"}), 404
    return jsonify({"session_id": session_id, "progress": _progress_steps(agent)})


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    user_input = body.get("message", "")

    agent = sessions.get(session_id)
    if agent is None:
        return jsonify({"error": "unknown session_id - call /api/session first"}), 400

    result = agent.next(user_input)
    result["progress"] = _progress_steps(agent)
    return jsonify(result)


if __name__ == "__main__":
    # 0.0.0.0 + $PORT: what Render (and most PaaS platforms) require for
    # local `python server.py` to also work unmodified in that environment.
    # Real deployments should still run behind gunicorn (see render.yaml),
    # not this dev server.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
