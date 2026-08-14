"""
Edge cases called out explicitly in the assignment: the ACC1004 leap-year
DOB, retry-limit enforcement, network failures, and graceful recovery from
input the extractor can't parse at all.
"""

from state import ConversationState
from tools import ApiUnavailableError, PaymentOutcome


def test_leap_year_correct_date_verifies(mock_lookup):
    from agent import Agent

    agent = Agent()
    agent.next("ACC1004")
    agent.next("Rahul Mehta")
    r = agent.next("DOB is 1988-02-29")
    assert "verified" in r["message"].lower()
    assert agent.session.verified is True


def test_leap_year_nearby_invalid_date_does_not_crash_or_burn_an_attempt(mock_lookup):
    from agent import Agent

    agent = Agent()
    agent.next("ACC1004")
    agent.next("Rahul Mehta")
    r = agent.next("DOB is 1990-02-29")  # 1990 isn't a leap year - not a real date
    assert "valid" in r["message"].lower()
    assert agent.session.verification_attempts == 0  # a format error, not a wrong guess
    assert agent.session.state == ConversationState.AWAIT_IDENTITY


def test_leap_year_wrong_but_valid_date_does_count_as_an_attempt(mock_lookup):
    from agent import Agent

    agent = Agent()
    agent.next("ACC1004")
    agent.next("Rahul Mehta")
    agent.next("DOB is 1988-03-01")  # a real date, just the wrong one
    assert agent.session.verification_attempts == 1


def test_account_not_found_retry_limit_closes_conversation(mock_lookup):
    from agent import Agent

    agent = Agent()
    for _ in range(3):
        r = agent.next("ACC9999")
    assert agent.session.state == ConversationState.CLOSED
    assert "account id" in r["message"].lower()


def test_verification_retry_limit_closes_and_never_reveals_correct_field(mock_lookup):
    from agent import Agent

    agent = Agent()
    agent.next("ACC1001")
    for _ in range(3):
        agent.next("Nithin Jain")
        r = agent.next("DOB is 1999-01-01")  # always wrong
    assert agent.session.state == ConversationState.CLOSED
    # Must never leak the real DOB/Aadhaar/pincode values into the response.
    assert "1990-05-14" not in r["message"]
    assert "4321" not in r["message"]
    assert "400001" not in r["message"]


def test_closed_conversation_repeats_same_message(mock_lookup):
    from agent import Agent

    agent = Agent()
    for _ in range(3):
        agent.next("ACC9999")
    closing = agent.next("ACC9999")["message"]
    again = agent.next("hello?")["message"]
    assert closing == again


def test_lookup_network_failure_gives_retryable_message_not_a_crash(monkeypatch):
    from agent import Agent

    def raise_unavailable(_account_id):
        raise ApiUnavailableError("simulated network failure")

    monkeypatch.setattr("agent.lookup_account", raise_unavailable)
    agent = Agent()
    r = agent.next("ACC1001")
    assert "trouble" in r["message"].lower() or "try again" in r["message"].lower()
    assert agent.session.state == ConversationState.AWAIT_ACCOUNT_ID  # unchanged, so a retry is possible


def test_payment_network_failure_gives_retryable_message(mock_lookup, monkeypatch):
    from agent import Agent

    def raise_unavailable(*a, **k):
        raise ApiUnavailableError("simulated network failure")

    monkeypatch.setattr("agent.process_payment", raise_unavailable)
    agent = Agent()
    agent.next("ACC1001")
    agent.next("Nithin Jain")
    agent.next("DOB is 1990-05-14")
    agent.next("500")
    r = agent.next("card number 4532 0151 1283 0366, expiry 12/27, cvv 123")
    assert "trouble" in r["message"].lower() or "try again" in r["message"].lower()
    assert agent.session.state == ConversationState.AWAIT_CARD_DETAILS


def test_unparseable_input_does_not_crash_and_reasks(mock_lookup):
    from agent import Agent

    agent = Agent()
    r = agent.next("asdkjhaskjdh ??? 😀😀😀 !!!")
    assert "account ID" in r["message"]


def test_empty_message_handled_gracefully(mock_lookup):
    from agent import Agent

    agent = Agent()
    r = agent.next("")
    assert "account ID" in r["message"]


def test_bare_numeric_account_id_resolves_via_fallback(mock_lookup):
    # Regression test: a user typing "1001" with no "ACC" prefix at all
    # used to loop forever on the generic "please share your account ID"
    # prompt, since extraction found nothing and no counter ever engaged.
    from agent import Agent

    agent = Agent()
    r = agent.next("1001")
    assert "full name" in r["message"].lower()  # ACC1001 was found and looked up
    assert agent.session.account is not None


def test_repeated_unparseable_account_id_input_eventually_closes(mock_lookup):
    # Regression test: gibberish/short numbers that never resolve to a
    # real account ID must still hit a retry limit and close cleanly,
    # rather than repeating the same prompt indefinitely.
    from agent import Agent

    agent = Agent()
    for _ in range(3):
        r = agent.next("65")
    assert agent.session.state == ConversationState.CLOSED
    assert "account id" in r["message"].lower()


def test_greeting_alone_does_not_count_as_a_failed_attempt(mock_lookup):
    from agent import Agent

    agent = Agent()
    agent.next("Hi")
    agent.next("hii")
    agent.next("hello there")
    assert agent.session.account_lookup_attempts == 0


def test_agent_never_discloses_dob_aadhaar_pincode_at_any_point(mock_lookup, monkeypatch):
    from agent import Agent

    monkeypatch.setattr(
        "agent.process_payment",
        lambda *a, **k: PaymentOutcome(success=True, transaction_id="txn_privacy", error_code=None),
    )
    agent = Agent()
    transcript = []
    for msg in ["Hi", "ACC1001", "Nithin Jain", "DOB is 1990-05-14", "500",
                "card 4532 0151 1283 0366 expiry 12/27 cvv 123"]:
        transcript.append(agent.next(msg)["message"])
    full_text = " ".join(transcript)
    assert "4321" not in full_text  # aadhaar_last4
    assert "400001" not in full_text  # pincode
    # The agent's own messages never restate the DOB back, even though the
    # user supplied it themselves - it just confirms "verified".
    assert "1990-05-14" not in full_text
