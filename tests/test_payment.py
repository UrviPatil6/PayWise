"""
Payment-flow specific tests: partial/full payments, every documented
process-payment error code, the payment retry limit, and that raw card
data doesn't linger in session state after an attempt.
"""

from decimal import Decimal

from agent import Agent
from state import ConversationState
from tools import PaymentOutcome


def _verified_agent_ready_for_amount(mock_lookup) -> Agent:
    agent = Agent()
    agent.next("ACC1001")
    agent.next("Nithin Jain")
    agent.next("DOB is 1990-05-14")
    return agent


def _give_card_details(agent: Agent):
    return agent.next("card number 4532 0151 1283 0366, expiry 12/27, cvv 123")


def test_partial_payment_allowed(mock_lookup, monkeypatch):
    monkeypatch.setattr(
        "agent.process_payment",
        lambda *a, **k: PaymentOutcome(success=True, transaction_id="txn_partial", error_code=None),
    )
    agent = _verified_agent_ready_for_amount(mock_lookup)
    agent.next("can I do 500 for now?")
    r = _give_card_details(agent)
    assert "successful" in r["message"].lower()


def test_full_balance_payment_phrase(mock_lookup, monkeypatch):
    captured_amount = {}

    def fake_process_payment(account_id, amount, card):
        captured_amount["amount"] = amount
        return PaymentOutcome(success=True, transaction_id="txn_full", error_code=None)

    monkeypatch.setattr("agent.process_payment", fake_process_payment)
    agent = _verified_agent_ready_for_amount(mock_lookup)
    agent.next("just clear the full amount")
    _give_card_details(agent)
    assert captured_amount["amount"] == Decimal("1250.75")


def test_amount_exceeding_balance_rejected_before_any_api_call(mock_lookup, monkeypatch):
    calls = []
    monkeypatch.setattr("agent.process_payment", lambda *a, **k: calls.append(1))
    agent = _verified_agent_ready_for_amount(mock_lookup)
    r = agent.next("I want to pay 5000")
    assert "balance" in r["message"].lower()
    assert calls == []
    assert agent.session.state == ConversationState.AWAIT_AMOUNT


def test_insufficient_balance_from_api_routes_back_to_amount(mock_lookup, monkeypatch):
    monkeypatch.setattr(
        "agent.process_payment",
        lambda *a, **k: PaymentOutcome(success=False, transaction_id=None, error_code="insufficient_balance"),
    )
    agent = _verified_agent_ready_for_amount(mock_lookup)
    agent.next("500")
    r = _give_card_details(agent)
    assert "balance" in r["message"].lower()
    assert agent.session.state == ConversationState.AWAIT_AMOUNT


def test_invalid_card_from_api_routes_back_to_card_details(mock_lookup, monkeypatch):
    monkeypatch.setattr(
        "agent.process_payment",
        lambda *a, **k: PaymentOutcome(success=False, transaction_id=None, error_code="invalid_card"),
    )
    agent = _verified_agent_ready_for_amount(mock_lookup)
    agent.next("500")
    r = _give_card_details(agent)
    assert "card" in r["message"].lower()
    assert agent.session.state == ConversationState.AWAIT_CARD_DETAILS


def test_invalid_cvv_from_api(mock_lookup, monkeypatch):
    monkeypatch.setattr(
        "agent.process_payment",
        lambda *a, **k: PaymentOutcome(success=False, transaction_id=None, error_code="invalid_cvv"),
    )
    agent = _verified_agent_ready_for_amount(mock_lookup)
    agent.next("500")
    r = _give_card_details(agent)
    assert "cvv" in r["message"].lower()


def test_expired_card_caught_locally_before_api_call(mock_lookup, monkeypatch):
    calls = []
    monkeypatch.setattr("agent.process_payment", lambda *a, **k: calls.append(1))
    agent = _verified_agent_ready_for_amount(mock_lookup)
    agent.next("500")
    r = agent.next("card number 4532 0151 1283 0366, expiry 01/2020, cvv 123")
    assert "expired" in r["message"].lower()
    assert calls == []  # never reached the API - local validation caught it


def test_payment_retry_limit_exhausted_closes_conversation(mock_lookup, monkeypatch):
    monkeypatch.setattr(
        "agent.process_payment",
        lambda *a, **k: PaymentOutcome(success=False, transaction_id=None, error_code="invalid_card"),
    )
    agent = _verified_agent_ready_for_amount(mock_lookup)
    agent.next("500")
    for _ in range(3):
        r = _give_card_details(agent)
    assert agent.session.state == ConversationState.CLOSED
    assert "retry limit" in r["message"].lower() or "end" in r["message"].lower()


def test_raw_card_data_cleared_from_session_after_attempt(mock_lookup, monkeypatch):
    monkeypatch.setattr(
        "agent.process_payment",
        lambda *a, **k: PaymentOutcome(success=True, transaction_id="txn_x", error_code=None),
    )
    agent = _verified_agent_ready_for_amount(mock_lookup)
    agent.next("500")
    _give_card_details(agent)
    assert agent.session.card_number is None
    assert agent.session.cvv is None


def test_invalid_card_number_caught_immediately_even_before_cvv_is_given(mock_lookup, monkeypatch):
    # Regression test: an invalid card number used to sit unvalidated for
    # as long as some other field (e.g. CVV) was still missing - the agent
    # would just keep asking for the missing piece instead of flagging the
    # bad number right away. It must be caught the moment it's given,
    # regardless of what else is still missing. Uses a Luhn-invalid (but
    # correct-length) number so this is exercised through the regex
    # extractor alone - a too-short number never becomes a candidate in
    # the first place (regex requires 13-19 digits), which would make the
    # test pass without actually touching the fixed code path.
    calls = []
    monkeypatch.setattr("agent.process_payment", lambda *a, **k: calls.append(1))
    agent = _verified_agent_ready_for_amount(mock_lookup)
    agent.next("500")
    r = agent.next("card number 4532015112830367")  # fails Luhn - no expiry/cvv given yet
    assert "card" in r["message"].lower()
    assert "cvv" not in r["message"].lower()  # must not ask for CVV before flagging the bad number
    assert agent.session.card_number is None  # rejected, not held onto
    assert calls == []


def test_card_details_prompt_acknowledges_what_was_already_captured(mock_lookup, monkeypatch):
    # A prompt that always lists the same "please share X, Y, Z" regardless
    # of what's already been given reads as not having listened. Once a
    # field is captured, later prompts must say so - not just silently
    # shrink the "missing" list.
    monkeypatch.setattr(
        "agent.process_payment",
        lambda *a, **k: PaymentOutcome(success=True, transaction_id="txn_ack_test", error_code=None),
    )
    agent = _verified_agent_ready_for_amount(mock_lookup)
    agent.next("500")
    r = agent.next("expiry 05/28")
    assert "got your" in r["message"].lower()
    assert "05/28" in r["message"] or "expiry" in r["message"].lower()

    r = agent.next("cvv 123")
    assert "got your" in r["message"].lower()
    assert "cvv" in r["message"].lower()

    # Giving the card number now completes all three fields and triggers
    # the actual payment - confirm the success message never echoes the
    # full card number, only ever the last 4 digits (already covered
    # elsewhere for logging; this checks the user-facing message too).
    r = agent.next("card number 4532 0151 1283 0366")
    assert "4532015112830366" not in r["message"]
    assert "4532" not in r["message"]
    assert "successful" in r["message"].lower()
