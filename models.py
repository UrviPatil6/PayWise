"""
Pydantic schemas for data that crosses a trust boundary: the account API
response, and the structured fields pulled out of a user's free-text
message. Both are places where "the data might not be what we expect"
(malformed API response, extractor finds nothing) so validation earns its
keep here. Everything purely internal (session state, API call results)
uses plain dataclasses instead - see state.py and tools.py.
"""

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, field_validator


class Account(BaseModel):
    """Validated shape of a successful /api/lookup-account response."""

    account_id: str
    full_name: str
    dob: str
    aadhaar_last4: str
    pincode: str
    balance: Decimal

    @field_validator("balance", mode="before")
    @classmethod
    def _balance_from_float(cls, v):
        # The API returns balance as a JSON float (e.g. 1250.75). Converting
        # float -> Decimal directly picks up binary floating-point noise
        # (Decimal(1250.75) != Decimal("1250.75")), which would break exact
        # "amount <= balance" comparisons later. Going through str() first
        # gives us the decimal value the API actually intended.
        if isinstance(v, float):
            return str(v)
        return v


class ExtractedFields(BaseModel):
    """
    Structured data pulled from a single user message. Every field is
    optional and independently nullable - a message may mention one field,
    several, or none. This is deliberately the same shape the assignment's
    own example shows: the extractor only ever fills in what was actually
    present in the text, never guesses at what's missing.
    """

    account_id: Optional[str] = None
    full_name: Optional[str] = None
    dob: Optional[str] = None
    aadhaar_last4: Optional[str] = None
    pincode: Optional[str] = None

    payment_amount: Optional[Decimal] = None
    full_balance_requested: bool = False

    card_number: Optional[str] = None
    cvv: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None
    cardholder_name: Optional[str] = None

    # A small, cross-cutting classification of the message that isn't tied
    # to any one conversation step - e.g. "cancel" can be said at almost
    # any point. Deliberately just a few intents (not a general chatbot
    # classifier): "cancel" (end the conversation), "small_talk" (a
    # greeting/pleasantry/thanks - acknowledge, don't treat as data),
    # "balance_query" (asking about their balance directly). Anything not
    # recognized is treated as None - agent.py never trusts an
    # unrecognized value, only acts on these three exact strings.
    intent: Optional[str] = None
