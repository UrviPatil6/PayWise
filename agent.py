"""
Agent.next() - the required public interface, and the orchestration layer.

  user_input -> _extract() -> ExtractedFields -> _merge_extracted() -> Session
             -> handler chain (account ID -> identity -> amount -> card details)
             -> deterministic message -> naturalize() -> {"message": ...}

All verification, validation, retry limits, and API calls happen in plain
Python here - see DESIGN.md and FINDINGS.md.
"""

from decimal import Decimal
from typing import Optional

from config import (
    LLM_ENABLED,
    MAX_ACCOUNT_LOOKUP_ATTEMPTS,
    MAX_PAYMENT_ATTEMPTS,
    MAX_VERIFICATION_ATTEMPTS,
    get_logger,
)
from extractor import extract_fields
from llm_extractor import extract_fields_llm
from models import ExtractedFields
from responder import naturalize
from state import ConversationState, Session
from tools import ApiUnavailableError, lookup_account, process_payment
from validators import (
    validate_account_id,
    validate_aadhaar_last4,
    validate_amount,
    validate_card_number,
    validate_cvv,
    validate_dob,
    validate_expiry,
    validate_pincode,
)
from verification import verify_identity

logger = get_logger(__name__)

# error_code -> (message, what to do next): retry_card / retry_amount / terminal.
PAYMENT_ERROR_ACTIONS = {
    "invalid_card": ("That card couldn't be processed - please double-check the card number.", "retry_card"),
    "invalid_cvv": ("That CVV wasn't accepted - please double-check it.", "retry_card"),
    "invalid_expiry": ("That expiry date wasn't accepted - please double-check it.", "retry_card"),
    "invalid_request": ("There was a problem with the payment details provided - please double-check them.", "retry_card"),
    "invalid_amount": ("That amount wasn't accepted by the payment processor.", "retry_amount"),
    "insufficient_balance": ("That amount exceeds your outstanding balance.", "retry_amount"),
    "account_not_found": ("We hit an unexpected issue locating your account for payment.", "terminal"),
}


def _money(value: Decimal) -> str:
    return f"₹{value:,.2f}"


class Agent:
    def __init__(self):
        self.session = Session()

    def next(self, user_input: str) -> dict:
        first_turn = not self.session.greeted
        self.session.greeted = True
        was_already_closed = self.session.state == ConversationState.CLOSED

        try:
            message, tone_hint = self._process_turn(user_input or "")
            crashed = False
        except Exception:
            logger.exception("unexpected error while processing a turn")
            message = "Sorry, something went wrong on my end. Could you try that again?"
            tone_hint = None
            crashed = True

        if first_turn:
            message = f"Hello! {message}"
            tone_hint = (
                f"{tone_hint}; " if tone_hint else ""
            ) + "This message already opens with \"Hello!\" - vary that greeting naturally in your rewording, don't add a second one on top of it."

        # Retry counts and closing messages aren't guarded by _facts_preserved - see FINDINGS.md §2, finding 12.
        security_critical = self.session.state == ConversationState.CLOSED or (
            self.session.action_log and self.session.action_log[-1] == "verify_identity:failed"
        )

        if LLM_ENABLED and not crashed and not was_already_closed:
            if security_critical:
                self.session.action_log.append("respond:security_critical_skip")
            else:
                rephrased = naturalize(message, self.session.state, tone_hint, self._response_checklist())
                if rephrased is not None:
                    self.session.action_log.append("respond:naturalized")
                    message = rephrased
                else:
                    self.session.action_log.append("respond:template_fallback")

        if not was_already_closed and self.session.state == ConversationState.CLOSED:
            self.session.closing_message = message

        return {"message": message}

    # ------------------------------------------------------------------
    # Turn processing
    # ------------------------------------------------------------------

    def _process_turn(self, user_input: str) -> tuple:
        """Returns (message, tone_hint). tone_hint only ever nudges
        *phrasing* (see responder.py) and is built entirely below from
        deterministic checks - never from anything the LLM decided."""
        session = self.session

        if session.state == ConversationState.CLOSED:
            return session.closing_message or (
                "This conversation has ended. Feel free to start a new one if you need anything else."
            ), None

        extracted = self._extract(user_input, session.state)

        if extracted.intent == "cancel":
            return self._handle_cancel(), None

        tone_hints = []
        if extracted.intent == "small_talk":
            tone_hints.append(
                "The customer's message included a greeting, pleasantry, or thanks - briefly "
                "and warmly acknowledge that before/alongside the rest of your reply."
            )

        # Gated on AWAIT_ACCOUNT_ID: once an account is found there's no path to switch it (§3 finding 11).
        # Format-validated before use: extracted.account_id hasn't been through
        # validate_account_id() yet at this point in the turn, and this value gets
        # interpolated straight into the tone_hint string the LLM sees - an
        # unvalidated value here would be a narrow prompt-injection surface via
        # the LLM extraction path (the regex path can't produce anything but a
        # well-formed ACCxxxx string, but the schema doesn't constrain the LLM
        # path the same way).
        account_id_is_valid = bool(extracted.account_id) and validate_account_id(extracted.account_id)[0]
        if (
            session.state == ConversationState.AWAIT_ACCOUNT_ID
            and account_id_is_valid
            and session.last_account_id_attempted
            and extracted.account_id != session.last_account_id_attempted
        ):
            tone_hints.append(
                f"The customer just corrected the account ID from {session.last_account_id_attempted} to "
                f"{extracted.account_id} - briefly acknowledge the update (e.g. \"Got it, I'll use "
                f"{extracted.account_id}.\") rather than treating it as unrelated new information."
            )
        if account_id_is_valid:
            session.last_account_id_attempted = extracted.account_id

        if (
            extracted.payment_amount is not None
            and session.last_payment_amount_attempted is not None
            and extracted.payment_amount != session.last_payment_amount_attempted
        ):
            tone_hints.append(
                f"The customer just changed the payment amount from {_money(session.last_payment_amount_attempted)} "
                f"to {_money(extracted.payment_amount)} - briefly acknowledge the update rather than treating "
                "it as unrelated new information."
            )
        if extracted.payment_amount is not None:
            session.last_payment_amount_attempted = extracted.payment_amount

        self._merge_extracted(extracted)

        message = None
        if session.state == ConversationState.AWAIT_ACCOUNT_ID:
            message = self._handle_account_id(extracted, user_input)

        if message is None and session.state == ConversationState.AWAIT_IDENTITY:
            message = self._handle_identity()
            if message is not None and session.action_log and session.action_log[-1] == "verify_identity:failed":
                tone_hints.append(
                    "This is a verification failure - be empathetic and reassuring in tone, without "
                    "revealing which specific detail didn't match."
                )

        if message is None and session.state == ConversationState.AWAIT_AMOUNT:
            message = self._handle_amount()

        if message is None and session.state == ConversationState.AWAIT_CARD_DETAILS:
            message = self._handle_card_details()

        if message is None:
            message = "Could you tell me more about what you'd like to do?"  # unreachable in practice

        if extracted.intent == "balance_query" and session.account is not None:
            if session.verified:
                balance_line = f"Your outstanding balance is {_money(session.account.balance)}."
                if balance_line not in message:
                    message = f"{balance_line} {message}"
            else:
                note = "Once your identity is verified, I can share your balance."
                if note not in message:
                    message = f"{message} {note}"

        return message, ("; ".join(tone_hints) if tone_hints else None)

    def _handle_cancel(self) -> str:
        session = self.session
        session.state = ConversationState.CLOSED
        session.action_log.append("cancelled_by_user")
        session.closing_message = (
            "No problem - I've ended this session. Feel free to start a new conversation anytime "
            "if you'd like to make a payment."
        )
        return session.closing_message

    def _extract(self, user_input: str, state: ConversationState) -> ExtractedFields:
        """LLM-first; extractor.py's regex parser is the deterministic fallback."""
        if LLM_ENABLED:
            llm_result = extract_fields_llm(user_input, state, self._llm_session_context())
            if llm_result is not None:
                self.session.action_log.append("extract:llm")
                return self._cross_check_with_regex(llm_result, user_input, state)
            self.session.action_log.append("extract:llm_failed_fallback_to_rules")

        return extract_fields(user_input, state)

    def _cross_check_with_regex(
        self, extracted: ExtractedFields, user_input: str, state: ConversationState
    ) -> ExtractedFields:
        """The LLM can return a partial extraction from a compound message
        and still count as success - fill any field it left null from
        regex (never override one it did return). full_name is the one
        exception: it's *replaced*, not just filled, if it doesn't appear
        verbatim in the user's message (the model reliably title-cases a
        lowercase name otherwise, which would make strict, case-sensitive
        verification silently case-insensitive - see FINDINGS.md §2, finding 5)."""
        regex_result = extract_fields(user_input, state)

        if extracted.full_name and extracted.full_name not in user_input:
            logger.warning("LLM-extracted full_name did not appear verbatim in the user's message - using regex's value for this field")
            extracted.full_name = regex_result.full_name
        elif extracted.full_name is None:
            extracted.full_name = regex_result.full_name

        for field in (
            "account_id", "dob", "aadhaar_last4", "pincode", "payment_amount",
            "card_number", "cvv", "expiry_month", "expiry_year", "cardholder_name",
        ):
            if getattr(extracted, field) is None:
                setattr(extracted, field, getattr(regex_result, field))
        if not extracted.full_balance_requested:
            extracted.full_balance_requested = regex_result.full_balance_requested
        if extracted.intent is None:
            extracted.intent = regex_result.intent

        return extracted

    def _llm_session_context(self) -> dict:
        """Small, explicitly-filtered view of session state for the LLM
        extractor - never DOB, Aadhaar, pincode, or raw card values, only
        what's needed to resolve a short/ambiguous reply."""
        session = self.session
        return {
            "account_id": session.account.account_id if session.account else None,
            "claimed_full_name": session.claimed_full_name,
            "identity_verified": session.verified,
            "payment_amount": str(session.payment_amount) if session.payment_amount is not None else None,
            "card_number_provided": session.card_number is not None,
            "cvv_provided": session.cvv is not None,
            "expiry_provided": session.expiry_month is not None and session.expiry_year is not None,
            "cardholder_name_provided": session.cardholder_name is not None,
        }

    def _response_checklist(self) -> list:
        """Plain-English "what we know so far" facts for responder.py, same
        safety scope as _llm_session_context() - lets a rewording say
        "I have the expiry and CVV, just need your card number" instead of
        a generic re-ask, without the phrasing step deciding anything."""
        session = self.session
        state = session.state
        lines = []

        if state == ConversationState.AWAIT_IDENTITY:
            lines.append("Account found." if session.account else "Account not yet found.")
            lines.append("Full name provided." if session.claimed_full_name else "Full name still needed.")
            lines.append(
                "A secondary identity factor (DOB, Aadhaar last 4, or pincode) provided."
                if session.has_secondary_identity_factor()
                else "Secondary identity factor (DOB, Aadhaar last 4, or pincode) still needed."
            )
        elif state == ConversationState.AWAIT_AMOUNT:
            lines.append("Identity verified.")
        elif state == ConversationState.AWAIT_CARD_DETAILS:
            if session.payment_amount is not None:
                lines.append(f"Payment amount confirmed: {_money(session.payment_amount)}.")
            lines.append("Card number provided." if session.card_number is not None else "Card number still needed.")
            lines.append(
                "Expiry provided." if (session.expiry_month is not None and session.expiry_year is not None) else "Expiry still needed."
            )
            lines.append("CVV provided." if session.cvv is not None else "CVV still needed.")
            lines.append("Cardholder name provided." if session.cardholder_name is not None else "Cardholder name still needed.")

        return lines

    def _merge_extracted(self, extracted: ExtractedFields) -> None:
        session = self.session

        # Locked in once verified - later messages are about payment, not identity.
        if not session.verified:
            if extracted.full_name:
                session.claimed_full_name = extracted.full_name
            if extracted.dob:
                session.claimed_dob = extracted.dob
            if extracted.aadhaar_last4:
                session.claimed_aadhaar_last4 = extracted.aadhaar_last4
            if extracted.pincode:
                session.claimed_pincode = extracted.pincode

        # Captured whenever mentioned, even before verification - but never
        # used until the handler chain actually reaches that step.
        if extracted.card_number:
            session.card_number = extracted.card_number
        if extracted.cvv:
            session.cvv = extracted.cvv
        if extracted.expiry_month:
            session.expiry_month = extracted.expiry_month
        if extracted.expiry_year:
            session.expiry_year = extracted.expiry_year
        if extracted.cardholder_name:
            session.cardholder_name = extracted.cardholder_name

        if extracted.payment_amount is not None:
            session.payment_amount = extracted.payment_amount
        if extracted.full_balance_requested and session.account is not None:
            session.payment_amount = session.account.balance

    # ------------------------------------------------------------------
    # Step 1: account ID -> lookup
    # ------------------------------------------------------------------

    def _handle_account_id(self, extracted: ExtractedFields, user_input: str) -> Optional[str]:
        session = self.session

        def close_out_of_attempts() -> str:
            session.state = ConversationState.CLOSED
            session.closing_message = (
                "I'm still not able to get a valid account ID after several tries, so I'll end "
                "this session here. Please start a new conversation and share your account ID "
                "in the format ACC followed by digits, e.g. ACC1001."
            )
            return session.closing_message

        if not extracted.account_id:
            # No digits at all (e.g. "Hi") doesn't cost a retry attempt.
            if not any(ch.isdigit() for ch in user_input):
                return "Please share your account ID to get started."
            session.account_lookup_attempts += 1
            if session.account_lookup_attempts >= MAX_ACCOUNT_LOOKUP_ATTEMPTS:
                return close_out_of_attempts()
            return "I didn't quite catch that. Account IDs look like ACC followed by digits, e.g. ACC1001 - could you share it again?"

        is_valid, error = validate_account_id(extracted.account_id)
        if not is_valid:
            session.account_lookup_attempts += 1
            if session.account_lookup_attempts >= MAX_ACCOUNT_LOOKUP_ATTEMPTS:
                return close_out_of_attempts()
            return f"{error} Could you share it again?"

        try:
            outcome = lookup_account(extracted.account_id)
        except ApiUnavailableError:
            return "I'm having trouble reaching our systems right now. Please try again in a moment."

        if not outcome.found:
            session.account_lookup_attempts += 1
            session.action_log.append("lookup_account:not_found")
            if session.account_lookup_attempts >= MAX_ACCOUNT_LOOKUP_ATTEMPTS:
                return close_out_of_attempts()
            return f"I couldn't find an account with ID {extracted.account_id}. Could you double-check and share it again?"

        session.account = outcome.account
        session.action_log.append("lookup_account:success")
        session.state = ConversationState.AWAIT_IDENTITY
        return None  # let _handle_identity run in this same turn

    # ------------------------------------------------------------------
    # Step 2: identity verification (deterministic - no LLM involved)
    # ------------------------------------------------------------------

    def _handle_identity(self) -> Optional[str]:
        session = self.session

        if session.claimed_full_name is None:
            return "Got it. Could you please confirm your full name?"

        # A single-word claim would just burn a retry against a two-word
        # account name - nudge once for the full name, then accept
        # whatever's given next (even another single word) as the answer.
        if " " not in session.claimed_full_name.strip() and not session.name_clarification_asked:
            session.name_clarification_asked = True
            session.claimed_full_name = None
            return "Thanks! Could you share your full name (first and last), exactly as it appears on the account?"

        # A structurally invalid secondary factor is a typo, not a wrong
        # guess - drop it and re-ask instead of burning a verification attempt.
        if session.claimed_dob is not None:
            ok, error = validate_dob(session.claimed_dob)
            if not ok:
                session.claimed_dob = None
                return f"{error} Could you provide your date of birth again (YYYY-MM-DD)?"
        if session.claimed_aadhaar_last4 is not None:
            ok, error = validate_aadhaar_last4(session.claimed_aadhaar_last4)
            if not ok:
                session.claimed_aadhaar_last4 = None
                return f"{error} Could you provide it again?"
        if session.claimed_pincode is not None:
            ok, error = validate_pincode(session.claimed_pincode)
            if not ok:
                session.claimed_pincode = None
                return f"{error} Could you provide it again?"

        if not session.has_secondary_identity_factor():
            return "Thanks. Could you also verify your date of birth, Aadhaar last 4 digits, or pincode?"

        verified = verify_identity(
            session.account,
            session.claimed_full_name,
            session.claimed_dob,
            session.claimed_aadhaar_last4,
            session.claimed_pincode,
        )

        if verified:
            session.verified = True
            session.action_log.append("verify_identity:success")
            session.state = ConversationState.AWAIT_AMOUNT
            return None  # let _handle_amount run in this same turn

        session.verification_attempts += 1
        session.action_log.append("verify_identity:failed")
        session.claimed_full_name = None
        session.claimed_dob = None
        session.claimed_aadhaar_last4 = None
        session.claimed_pincode = None

        if session.verification_attempts >= MAX_VERIFICATION_ATTEMPTS:
            session.state = ConversationState.CLOSED
            session.closing_message = (
                "I'm unable to verify your identity after multiple attempts, so I have to "
                "end this session here for security reasons. Please contact support if you need help."
            )
            return session.closing_message

        remaining = MAX_VERIFICATION_ATTEMPTS - session.verification_attempts
        # Deliberately generic - never states *which* field was wrong.
        return (
            f"Those details don't match our records. You have {remaining} attempt(s) left - "
            "could you confirm your full name, and your date of birth, Aadhaar last 4 digits, or pincode?"
        )

    # ------------------------------------------------------------------
    # Step 3: payment amount
    # ------------------------------------------------------------------

    def _handle_amount(self) -> Optional[str]:
        session = self.session
        balance = session.account.balance

        if session.payment_amount is None:
            return f"Identity verified. Your outstanding balance is {_money(balance)}. How much would you like to pay today?"

        is_valid, error = validate_amount(session.payment_amount, balance)
        if not is_valid:
            session.payment_amount = None
            return f"{error} How much would you like to pay?"

        session.action_log.append(f"amount_accepted:{session.payment_amount}")
        session.state = ConversationState.AWAIT_CARD_DETAILS
        return None  # let _handle_card_details run in this same turn

    # ------------------------------------------------------------------
    # Step 4: card details -> process payment
    # ------------------------------------------------------------------

    def _handle_card_details(self) -> Optional[str]:
        session = self.session

        # _merge_extracted() applies a new amount unconditionally regardless
        # of state (so a mid-flow correction is captured at all) - so a
        # later message can replace an already-validated amount with one
        # that was never checked against the balance. Re-validate here.
        if session.payment_amount is not None:
            is_valid, error = validate_amount(session.payment_amount, session.account.balance)
            if not is_valid:
                session.payment_amount = None
                session.state = ConversationState.AWAIT_AMOUNT
                return f"{error} How much would you like to pay? Your outstanding balance is {_money(session.account.balance)}."

        if session.cardholder_name is None:
            session.cardholder_name = session.claimed_full_name

        # Validate each field the moment it's available, not only once all
        # three are present - otherwise a bad card number sits unvalidated
        # while the agent just keeps asking for the still-missing fields.
        if session.card_number is not None:
            is_valid, error, normalized_card = validate_card_number(session.card_number)
            if not is_valid:
                session.card_number = None
                return f"{error} Could you re-enter your card number?"
            session.card_number = normalized_card

        if session.expiry_month is not None and session.expiry_year is not None:
            is_valid, error = validate_expiry(session.expiry_month, session.expiry_year)
            if not is_valid:
                session.expiry_month = None
                session.expiry_year = None
                return f"{error} Could you re-enter the expiry (MM/YY)?"

        # CVV length depends on card network, so needs a validated card number first.
        if session.cvv is not None and session.card_number is not None:
            is_valid, error = validate_cvv(session.cvv, session.card_number)
            if not is_valid:
                session.cvv = None
                return f"{error} Could you re-enter your CVV?"

        missing = []
        if session.card_number is None:
            missing.append("card number")
        if session.expiry_month is None or session.expiry_year is None:
            missing.append("expiry (MM/YY)")
        if session.cvv is None:
            missing.append("CVV")
        if missing:
            # Acknowledge what's already captured, not just what's still
            # needed. Card number is only ever referenced masked.
            captured = []
            if session.card_number is not None:
                captured.append(f"card number ending {session.card_number[-4:]}")
            if session.expiry_month is not None and session.expiry_year is not None:
                captured.append(f"expiry {session.expiry_month:02d}/{session.expiry_year % 100:02d}")
            if session.cvv is not None:
                captured.append("CVV")
            if captured:
                return f"Got your {', '.join(captured)}. To charge {_money(session.payment_amount)}, please also share your {', '.join(missing)}."
            return f"To charge {_money(session.payment_amount)}, please share your {', '.join(missing)}."

        card_payload = {
            "cardholder_name": session.cardholder_name,
            "card_number": session.card_number,
            "cvv": session.cvv,
            "expiry_month": session.expiry_month,
            "expiry_year": session.expiry_year,
        }

        try:
            outcome = process_payment(session.account.account_id, session.payment_amount, card_payload)
        except ApiUnavailableError:
            session.clear_card_details()
            session.payment_attempts += 1
            session.action_log.append("process_payment:api_unavailable")
            if session.payment_attempts >= MAX_PAYMENT_ATTEMPTS:
                session.state = ConversationState.CLOSED
                session.closing_message = (
                    "I'm still unable to reach the payment system after multiple attempts. "
                    "Please try again later."
                )
                return session.closing_message
            return "I'm having trouble reaching the payment system right now. Please try again in a moment."

        # Raw card number/CVV have done their job - drop them regardless of outcome.
        session.clear_card_details()

        if outcome.success:
            session.transaction_id = outcome.transaction_id
            session.action_log.append("process_payment:success")
            session.state = ConversationState.CLOSED
            session.closing_message = (
                f"Payment successful! {_money(session.payment_amount)} was charged "
                f"(transaction ID {outcome.transaction_id}). Thanks, {session.account.full_name} - "
                "have a great day!"
            )
            return session.closing_message

        session.action_log.append(f"process_payment:failed:{outcome.error_code}")
        session.payment_attempts += 1
        return self._handle_payment_failure(outcome.error_code)

    def _handle_payment_failure(self, error_code: Optional[str]) -> str:
        session = self.session
        message, action = PAYMENT_ERROR_ACTIONS.get(
            error_code, ("The payment couldn't be completed.", "retry_card")
        )

        if action == "terminal" or session.payment_attempts >= MAX_PAYMENT_ATTEMPTS:
            session.state = ConversationState.CLOSED
            session.closing_message = (
                f"{message} We've reached the retry limit for this session, so I'll have to "
                "end it here. Please try again later or contact support."
            )
            return session.closing_message

        if action == "retry_amount":
            session.state = ConversationState.AWAIT_AMOUNT
            session.payment_amount = None
            return f"{message} How much would you like to pay? Your outstanding balance is {_money(session.account.balance)}."

        return f"{message} Please re-enter your card details to try again."
