"""
Shared test fixtures.

Every agent-level test uses a fake lookup_account instead of hitting the
real network - that keeps the unit/integration suite fast and fully
offline-deterministic. The evals/ suite (a separate, explicit choice - see
evals/evaluation.py) exercises the real API end to end instead.
"""

from decimal import Decimal

import pytest

from models import Account
from tools import LookupOutcome

SAMPLE_ACCOUNTS = {
    "ACC1001": Account(
        account_id="ACC1001", full_name="Nithin Jain", dob="1990-05-14",
        aadhaar_last4="4321", pincode="400001", balance=Decimal("1250.75"),
    ),
    "ACC1002": Account(
        account_id="ACC1002", full_name="Rajarajeswari Balasubramaniam", dob="1985-11-23",
        aadhaar_last4="9876", pincode="400002", balance=Decimal("540.00"),
    ),
    "ACC1003": Account(
        account_id="ACC1003", full_name="Priya Agarwal", dob="1992-08-10",
        aadhaar_last4="2468", pincode="400003", balance=Decimal("0.00"),
    ),
    "ACC1004": Account(
        account_id="ACC1004", full_name="Rahul Mehta", dob="1988-02-29",
        aadhaar_last4="1357", pincode="400004", balance=Decimal("3200.50"),
    ),
}


def fake_lookup_account(account_id: str) -> LookupOutcome:
    account = SAMPLE_ACCOUNTS.get(account_id)
    if account is None:
        return LookupOutcome(found=False, account=None, error_code="account_not_found")
    return LookupOutcome(found=True, account=account, error_code=None)


@pytest.fixture
def mock_lookup(monkeypatch):
    """Patches agent.lookup_account (patched where it's *used*, not where
    it's defined - agent.py imported the name directly)."""
    monkeypatch.setattr("agent.lookup_account", fake_lookup_account)


@pytest.fixture(autouse=True)
def disable_llm_by_default(monkeypatch):
    """Forces the regex extraction path for the whole suite by default,
    regardless of whether a GROQ_API_KEY/LLM_API_KEY happens to be set in
    whatever environment the tests run in. This is what makes the suite
    hermetic and deterministic: it should test the same behavior whether
    or not a real key is present. Tests that specifically want to exercise
    the LLM path (tests/test_llm_extractor.py) override this explicitly.
    """
    monkeypatch.setattr("agent.LLM_ENABLED", False)
