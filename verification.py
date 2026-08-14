"""
Identity verification - the one piece of business logic the assignment is
strictest about: it must be exact-match Python, never an LLM judgment call.

This is a single pure function rather than a "Verifier" class because there
is no state to carry between calls and only one comparison rule to express -
a class would just be ceremony around one `if`.
"""

from typing import Optional

from models import Account


def verify_identity(
    account: Account,
    claimed_full_name: Optional[str],
    claimed_dob: Optional[str] = None,
    claimed_aadhaar_last4: Optional[str] = None,
    claimed_pincode: Optional[str] = None,
) -> bool:
    """
    Verified only if the full name is an exact match AND at least one of
    dob / aadhaar_last4 / pincode is also an exact match. Fields the user
    hasn't provided are simply not compared (None != mismatch) - this is
    what lets verification be attempted the moment name + any one secondary
    factor are known, without penalizing the user for not supplying all
    three secondary fields.

    Strict equality only: no .lower(), no whitespace fuzzing, no partial
    matches. The assignment explicitly forbids fuzzy/case-insensitive
    matching, and account data (name spelling, DOB format) comes from a
    trusted API response, so exact comparison is the correct, safe default.
    """
    if claimed_full_name is None or claimed_full_name != account.full_name:
        return False

    if claimed_dob is not None and claimed_dob == account.dob:
        return True
    if claimed_aadhaar_last4 is not None and claimed_aadhaar_last4 == account.aadhaar_last4:
        return True
    if claimed_pincode is not None and claimed_pincode == account.pincode:
        return True

    return False
