"""
Conversation state: what turn of the flow we're in, and everything
collected so far. This is the "memory" that lets Agent.next() be a pure
function of (current session, new user message) - which is what makes the
multi-turn interface in the assignment work without any manual reset
between calls.

ConversationState only has the states that matter at a turn boundary (i.e.
the states the agent can be "waiting" in between calls to next()). Account
lookup and payment processing are not separate states here even though the
assignment's suggested list includes them: both API calls happen
synchronously inside the single next() call that triggers them, so there's
never a turn boundary where the agent is "in" a lookup or "in" a payment.
Which actions were actually taken is recorded in `action_log` instead
(see below) - that's what the eval suite checks tool-call sequences against.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from models import Account


class ConversationState(str, Enum):
    AWAIT_ACCOUNT_ID = "AWAIT_ACCOUNT_ID"
    AWAIT_IDENTITY = "AWAIT_IDENTITY"
    AWAIT_AMOUNT = "AWAIT_AMOUNT"
    AWAIT_CARD_DETAILS = "AWAIT_CARD_DETAILS"
    CLOSED = "CLOSED"


@dataclass
class Session:
    state: ConversationState = ConversationState.AWAIT_ACCOUNT_ID
    greeted: bool = False

    # Set once /api/lookup-account succeeds. Holds the sensitive fields
    # (dob, aadhaar_last4, pincode) used only for the in-memory verification
    # comparison in verification.py - never read back out to the user.
    account: Optional[Account] = None
    account_lookup_attempts: int = 0

    # The most recent account ID the user gave, successful or not - kept
    # only to detect a *correction* ("actually it's ACC1001") so the reply
    # can be phrased as "updated to X" instead of a generic re-ask. Purely
    # a phrasing aid (see agent._process_turn's tone_hints) - never read by
    # any verification or lookup logic.
    last_account_id_attempted: Optional[str] = None

    # Identity claims the user has made so far, merged in as extracted
    # across turns so information given out of order or early isn't lost.
    claimed_full_name: Optional[str] = None
    claimed_dob: Optional[str] = None
    claimed_aadhaar_last4: Optional[str] = None
    claimed_pincode: Optional[str] = None
    verified: bool = False
    verification_attempts: int = 0

    # Set the first time a single-word name claim (e.g. "urvi") prompts a
    # one-time "could you share your full name?" nudge, rather than letting
    # it silently proceed to a verification attempt that's certain to fail
    # against a two-word account name. Only nudges once, so a user whose
    # real name genuinely is one word isn't stuck forever - see
    # agent._handle_identity.
    name_clarification_asked: bool = False

    payment_amount: Optional[Decimal] = None

    # The most recent amount the user gave, whether or not it was accepted
    # (unlike payment_amount, which gets cleared back to None the moment a
    # given amount is rejected as invalid - see agent._handle_amount) -
    # kept only so a follow-up correction ("actually make that 500") can
    # still be recognized and acknowledged as an update even after the
    # first attempt was cleared. Purely a phrasing aid (see
    # agent._process_turn's tone_hints), never read by validation logic.
    last_payment_amount_attempted: Optional[Decimal] = None

    card_number: Optional[str] = None
    cvv: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None
    cardholder_name: Optional[str] = None
    payment_attempts: int = 0

    transaction_id: Optional[str] = None

    # Set once the conversation reaches CLOSED, so every subsequent call to
    # next() repeats the same closing message instead of recomputing it -
    # required for the "consistent and deterministic across repeated runs"
    # rule once the conversation is already over.
    closing_message: Optional[str] = None

    # Every lookup/verify/payment action taken, e.g. "lookup_account:success",
    # "verify_identity:failed", "process_payment:success". Purely an
    # observability trail - the eval suite asserts against this to check the
    # agent called the right API at the right time, not just that the final
    # message reads well.
    action_log: List[str] = field(default_factory=list)

    def has_secondary_identity_factor(self) -> bool:
        return any(
            [self.claimed_dob, self.claimed_aadhaar_last4, self.claimed_pincode]
        )

    def clear_card_details(self) -> None:
        """Called right after a payment attempt, success or failure.

        The assignment requires not persisting raw card data beyond what's
        necessary for the API call - once that call has been made, there's
        no further use for the raw number/CVV, so they're dropped from
        memory immediately rather than lingering in session state.
        """
        self.card_number = None
        self.cvv = None
        self.expiry_month = None
        self.expiry_year = None
