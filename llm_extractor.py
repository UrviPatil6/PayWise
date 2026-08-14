"""LLM structured extraction, with a deterministic (regex) fallback. See DESIGN.md."""

import json
from typing import Optional

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_ENABLED,
    LLM_FALLBACK_API_KEY,
    LLM_FALLBACK_BASE_URL,
    LLM_FALLBACK_ENABLED,
    LLM_FALLBACK_MODEL,
    LLM_FALLBACK_SUPPORTS_FORCED_TOOL_CHOICE,
    LLM_FALLBACK_USES_REASONING_EFFORT,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    LLM_USES_REASONING_EFFORT,
    get_logger,
)
from models import ExtractedFields
from state import ConversationState

logger = get_logger(__name__)

_STATE_HINTS = {
    ConversationState.AWAIT_ACCOUNT_ID: "waiting for the customer's account ID (format: ACC followed by digits, e.g. ACC1001).",
    ConversationState.AWAIT_IDENTITY: "waiting for the customer's full name, and/or a secondary identity factor: date of birth, last 4 digits of Aadhaar, or pincode.",
    ConversationState.AWAIT_AMOUNT: "waiting for how much the customer wants to pay - a specific amount, or a request to pay the full outstanding balance.",
    ConversationState.AWAIT_CARD_DETAILS: "waiting for card payment details: card number, cardholder name, CVV, and expiry month/year.",
    ConversationState.CLOSED: "the conversation has ended.",
}

_FIELDS_SCHEMA = {
    "type": "object",
    "properties": {
        "account_id": {"type": ["string", "null"], "description": "Account ID, normalized like ACC1001 (uppercase, no spaces)."},
        "full_name": {"type": ["string", "null"], "description": "The customer's full name, exactly as they said it - preserve spelling and casing."},
        "dob": {"type": ["string", "null"], "description": "Date of birth normalized to YYYY-MM-DD."},
        "aadhaar_last4": {"type": ["string", "null"], "description": "Last 4 digits of Aadhaar, digits only."},
        "pincode": {"type": ["string", "null"], "description": "6-digit pincode, digits only."},
        "payment_amount": {"type": ["number", "null"], "description": "A specific payment amount mentioned, as a plain number."},
        "full_balance_requested": {"type": "boolean", "description": "True if the customer asked to pay their entire/full outstanding balance rather than a specific amount."},
        "card_number": {"type": ["string", "null"], "description": "Card number, digits only, spaces removed."},
        "cvv": {"type": ["string", "null"], "description": "CVV, digits only."},
        "expiry_month": {"type": ["integer", "null"], "description": "Card expiry month, 1-12."},
        "expiry_year": {"type": ["integer", "null"], "description": "Card expiry year, always 4 digits - expand a 2-digit year into the 2000s (27 -> 2027)."},
        "cardholder_name": {"type": ["string", "null"], "description": "Name on the card. If a short reply is clearly answering this field, extract it even without an explicit label."},
        "intent": {
            "type": ["string", "null"],
            "enum": ["cancel", "small_talk", "balance_query"],
            "description": (
                "'cancel' - wants to stop/end (e.g. 'never mind'). 'small_talk' - a greeting, "
                "pleasantry, or thanks with no other content to act on. 'balance_query' - directly "
                "asking what they owe. Leave null for anything else, including messages that also "
                "contain real data to extract."
            ),
        },
    },
    "required": [],
}

_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_fields",
        "description": "Extract structured fields explicitly present in the customer's message. Only fill in a field if it is actually stated - never guess or infer a value that wasn't said.",
        "parameters": _FIELDS_SCHEMA,
    },
}

SYSTEM_PROMPT = """Extract structured information from the customer's latest message in a payment-collection conversation.

Extract only information explicitly stated or clearly provided in context. Never invent, guess, or infer a value that is not present.

Rules:
- If two names appear (a nickname and a formal name), extract only the fuller/formal one.
- Preserve the customer's exact name spelling and casing - never "correct" it.
- If a short reply clearly answers the field requested by the current step, extract it even without an explicit label. Do not infer a value when the reply could refer to multiple fields.
- Extract every field present, even from a single compound message.
- Use intent only for cancel, small_talk, or balance_query - leave it null otherwise.

Current step: {state_hint}

Already collected this conversation (use only to resolve what a short reply is answering - never as a reason to invent a value the message doesn't contain):
{session_context}"""


def _call_provider(
    openai_module, text: str, state: ConversationState, session_context: dict, api_key: str, base_url: str, model: str,
    forced_tool_choice: bool, uses_reasoning_effort: bool,
) -> Optional[ExtractedFields]:
    tool_choice = {"type": "function", "function": {"name": "extract_fields"}} if forced_tool_choice else "auto"
    system_content = SYSTEM_PROMPT.format(
        state_hint=_STATE_HINTS.get(state, ""),
        session_context=json.dumps(session_context),
    )
    create_kwargs = {
        "model": model,
        "temperature": 0,
        "tools": [_TOOL],
        "tool_choice": tool_choice,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text},
        ],
    }
    if uses_reasoning_effort:
        create_kwargs["reasoning_effort"] = "low"

    try:
        # We handle provider fallback ourselves - no need for the SDK's own retries.
        client = openai_module.OpenAI(api_key=api_key, base_url=base_url, timeout=LLM_TIMEOUT_SECONDS, max_retries=0)
        response = client.chat.completions.create(**create_kwargs)
    except Exception as exc:
        logger.warning("LLM extraction call to %s failed (%s)", base_url, type(exc).__name__)
        return None

    tool_calls = response.choices[0].message.tool_calls if response.choices else None
    if not tool_calls:
        logger.warning("provider at %s did not call the extraction tool", base_url)
        return None

    try:
        arguments = json.loads(tool_calls[0].function.arguments)
        return ExtractedFields(**arguments)
    except Exception:
        logger.warning("provider at %s returned a response that didn't match the expected field schema", base_url)
        return None


def extract_fields_llm(text: str, state: ConversationState, session_context: Optional[dict] = None) -> Optional[ExtractedFields]:
    if not LLM_ENABLED:
        return None
    session_context = session_context or {}

    try:
        import openai  # lazy import: only needed when the feature is actually enabled
    except ImportError:
        logger.warning("An LLM API key is set but the openai package isn't installed - skipping LLM extraction")
        return None

    result = _call_provider(
        openai, text, state, session_context, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
        forced_tool_choice=True, uses_reasoning_effort=LLM_USES_REASONING_EFFORT,
    )
    if result is not None:
        return result

    if LLM_FALLBACK_ENABLED:
        logger.warning("primary LLM extraction failed - trying fallback provider")
        result = _call_provider(
            openai, text, state, session_context, LLM_FALLBACK_API_KEY, LLM_FALLBACK_BASE_URL, LLM_FALLBACK_MODEL,
            forced_tool_choice=LLM_FALLBACK_SUPPORTS_FORCED_TOOL_CHOICE,
            uses_reasoning_effort=LLM_FALLBACK_USES_REASONING_EFFORT,
        )
        if result is not None:
            return result

    return None
