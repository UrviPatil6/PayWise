"""
Verification is the assignment's strictest requirement: deterministic,
exact-match, no fuzzy/case-insensitive logic. These tests exercise
verify_identity() directly with no agent/state/API involved.
"""

from decimal import Decimal

from models import Account
from verification import verify_identity

ACCOUNT = Account(
    account_id="ACC1001", full_name="Nithin Jain", dob="1990-05-14",
    aadhaar_last4="4321", pincode="400001", balance=Decimal("1250.75"),
)


def test_name_plus_correct_dob_verifies():
    assert verify_identity(ACCOUNT, "Nithin Jain", claimed_dob="1990-05-14") is True


def test_name_plus_correct_aadhaar_verifies():
    assert verify_identity(ACCOUNT, "Nithin Jain", claimed_aadhaar_last4="4321") is True


def test_name_plus_correct_pincode_verifies():
    assert verify_identity(ACCOUNT, "Nithin Jain", claimed_pincode="400001") is True


def test_name_plus_all_three_correct_verifies():
    assert verify_identity(ACCOUNT, "Nithin Jain", "1990-05-14", "4321", "400001") is True


def test_wrong_name_never_verifies_even_with_correct_secondary():
    assert verify_identity(ACCOUNT, "Someone Else", claimed_dob="1990-05-14") is False


def test_correct_name_wrong_dob_fails():
    assert verify_identity(ACCOUNT, "Nithin Jain", claimed_dob="1990-05-15") is False


def test_correct_name_no_secondary_factor_fails():
    # This is the "not enough info yet" case, not a wrong guess - the agent
    # layer is responsible for asking again rather than treating it as a
    # failed attempt.
    assert verify_identity(ACCOUNT, "Nithin Jain") is False


def test_name_case_mismatch_is_strict_no_fuzzy_match():
    assert verify_identity(ACCOUNT, "nithin jain", claimed_dob="1990-05-14") is False
    assert verify_identity(ACCOUNT, "NITHIN JAIN", claimed_dob="1990-05-14") is False


def test_name_with_extra_whitespace_is_strict():
    assert verify_identity(ACCOUNT, "Nithin  Jain", claimed_dob="1990-05-14") is False


def test_one_correct_and_two_wrong_secondary_factors_still_verifies():
    # Only one matching secondary factor is required - wrong values in the
    # other two fields shouldn't block verification if the name and at
    # least one factor are exactly right.
    assert verify_identity(ACCOUNT, "Nithin Jain", "1999-01-01", "0000", "400001") is True


def test_all_secondary_factors_wrong_fails():
    assert verify_identity(ACCOUNT, "Nithin Jain", "1999-01-01", "0000", "999999") is False


def test_none_full_name_fails():
    assert verify_identity(ACCOUNT, None, claimed_dob="1990-05-14") is False
