"""
Deterministic input validation - pure functions, no I/O, no LLM.

The assignment is explicit that validation must happen before any API call
and must not depend on the LLM. These functions are the enforcement point:
extractor.py turns messy text into candidate values, and every candidate
passes through here before it's used to build an API request or accepted
into conversation state.

Each function returns (is_valid, error_message_or_None) so callers in
agent.py can show the user a specific, actionable reason instead of a
generic "that didn't work."
"""

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple

ACCOUNT_ID_PATTERN = re.compile(r"^ACC\d{3,6}$")
AADHAAR_LAST4_PATTERN = re.compile(r"^\d{4}$")
PINCODE_PATTERN = re.compile(r"^\d{6}$")


def validate_account_id(account_id: str) -> Tuple[bool, Optional[str]]:
    if not account_id or not ACCOUNT_ID_PATTERN.match(account_id):
        return False, "Account IDs look like ACC followed by digits, e.g. ACC1001."
    return True, None


def validate_dob(dob: str) -> Tuple[bool, Optional[str]]:
    """dob must already be normalized to YYYY-MM-DD by the extractor.

    Using date(y, m, d) to validate (rather than a regex) is what correctly
    rejects the ACC1004 edge case family for free: 1988-02-29 is a real
    leap-year date and passes, but 1989-02-29 or 1990-02-29 raise
    ValueError and are correctly rejected - no special-casing needed.
    """
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", dob or "")
    if not match:
        return False, "Please provide your date of birth as YYYY-MM-DD, e.g. 1990-05-14."
    year, month, day = (int(g) for g in match.groups())
    try:
        date(year, month, day)
    except ValueError:
        return False, f"{dob} isn't a valid calendar date."
    if date(year, month, day) > date.today():
        return False, "That date of birth is in the future."
    return True, None


def validate_aadhaar_last4(value: str) -> Tuple[bool, Optional[str]]:
    if not value or not AADHAAR_LAST4_PATTERN.match(value):
        return False, "The last 4 digits of Aadhaar should be exactly 4 digits."
    return True, None


def validate_pincode(value: str) -> Tuple[bool, Optional[str]]:
    if not value or not PINCODE_PATTERN.match(value):
        return False, "Pincode should be exactly 6 digits."
    return True, None


def validate_amount(amount: Optional[Decimal], balance: Decimal) -> Tuple[bool, Optional[str]]:
    if amount is None:
        return False, "I couldn't understand that amount."
    if amount <= 0:
        return False, "The payment amount must be greater than zero."
    # exponent > -2 means more than 2 digits after the decimal point,
    # e.g. Decimal("10.999") has exponent -3.
    if amount.as_tuple().exponent < -2:
        return False, "The amount can have at most 2 decimal places."
    if amount > balance:
        return False, f"That's more than your outstanding balance of ₹{balance:,.2f}."
    return True, None


def normalize_card_number(raw: str) -> str:
    return re.sub(r"[\s-]", "", raw or "")


def luhn_check(card_number: str) -> bool:
    digits = [int(d) for d in card_number]
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def validate_card_number(raw: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Returns (is_valid, error_message, normalized_number)."""
    normalized = normalize_card_number(raw)
    if not normalized.isdigit() or not (13 <= len(normalized) <= 19):
        return False, "That doesn't look like a valid card number.", None
    if not luhn_check(normalized):
        return False, "That card number failed validation. Please double-check it.", None
    return True, None, normalized


def is_amex(card_number: str) -> bool:
    return card_number[:2] in ("34", "37")


def validate_cvv(cvv: str, card_number: str) -> Tuple[bool, Optional[str]]:
    expected_length = 4 if is_amex(card_number) else 3
    if not cvv or not cvv.isdigit() or len(cvv) != expected_length:
        return False, f"CVV should be {expected_length} digits for this card."
    return True, None


def validate_expiry(month: Optional[int], year: Optional[int]) -> Tuple[bool, Optional[str]]:
    if month is None or year is None:
        return False, "I need the card's expiry month and year."
    if not (1 <= month <= 12):
        return False, "Expiry month should be between 1 and 12."
    if year < 100:  # defensive; extractor should already expand 2-digit years
        year += 2000
    today = date.today()
    # Cards are valid through the end of their expiry month.
    if (year, month) < (today.year, today.month):
        return False, "That card has expired."
    if year > today.year + 30:
        return False, "That expiry year doesn't look right."
    return True, None


def parse_decimal(raw: str) -> Optional[Decimal]:
    try:
        return Decimal(raw)
    except (InvalidOperation, ValueError):
        return None


def mask_card_number(card_number: str) -> str:
    """Card number for display/logging only - last 4 digits visible."""
    digits = normalize_card_number(card_number)
    return f"**** **** **** {digits[-4:]}" if len(digits) >= 4 else "****"
