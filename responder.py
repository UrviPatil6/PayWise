"""LLM response phrasing (rewording, never generation), with a deterministic template fallback. See DESIGN.md."""

import re
from typing import Optional

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_FALLBACK_API_KEY,
    LLM_FALLBACK_BASE_URL,
    LLM_FALLBACK_ENABLED,
    LLM_FALLBACK_MODEL,
    LLM_FALLBACK_USES_REASONING_EFFORT,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    LLM_USES_REASONING_EFFORT,
    get_logger,
)
from state import ConversationState

logger = get_logger(__name__)

_STATE_DESCRIPTIONS = {
    ConversationState.AWAIT_ACCOUNT_ID: "waiting for the customer's account ID",
    ConversationState.AWAIT_IDENTITY: "verifying the customer's identity (full name plus one of DOB / Aadhaar last 4 / pincode)",
    ConversationState.AWAIT_AMOUNT: "identity already verified; waiting for a payment amount",
    ConversationState.AWAIT_CARD_DETAILS: "collecting card payment details (number, expiry, CVV)",
    ConversationState.CLOSED: "the conversation has ended",
}

# Facts that must survive rephrasing verbatim: currency amounts, account
# IDs, transaction IDs. A rephrasing that drops or alters one is discarded.
_FACT_PATTERN = re.compile(r"₹[\d,]+\.\d{2}|ACC\d{3,6}|txn_\S+")

SYSTEM_PROMPT = """You are the conversational layer of a payment collection agent.

The deterministic system has already decided what must be communicated. Rewrite that message so it sounds natural, friendly, concise, and human.

Rules:
- Preserve the meaning and every fact in the message exactly.
- Never change amounts, account IDs, transaction IDs, retry counts, or instructions.
- Never claim an action that is not stated.
- Never add new information, even if a "what we know" checklist is provided for context - use it only to shape phrasing.
- Do not repeat information the customer has already provided.
- Ask only for information that is still required.
- If the customer corrected something, acknowledge it naturally.
- Avoid repetitive phrases such as "Could you please..."
- Avoid excessive enthusiasm and emojis.
- Use 1-2 short sentences.
- Output only the final response."""


def _facts_preserved(template_message: str, rephrased: str) -> bool:
    required = _FACT_PATTERN.findall(template_message)
    return all(fact in rephrased for fact in required)


def _build_user_content(template_message: str, state: ConversationState, checklist: Optional[list]) -> str:
    checklist_block = "\n".join(f"- {line}" for line in checklist) if checklist else "- Nothing yet - this is early in the conversation."
    # Delimiters + "rewrite this" framing (not a bare user turn) keeps the
    # model from answering the message instead of rewording it.
    return (
        f"CURRENT STEP: {_STATE_DESCRIPTIONS.get(state, '')}\n\n"
        f"WHAT WE KNOW SO FAR:\n{checklist_block}\n\n"
        f"DETERMINISTIC MESSAGE (reword only the text inside <<<...>>>):\n"
        f"<<<{template_message}>>>\n\n"
        "TASK: Rewrite the text inside <<<...>>> so it sounds natural and warm."
    )


def _call_provider(
    openai_module, template_message: str, state: ConversationState, tone_hint: Optional[str], checklist: Optional[list],
    api_key: str, base_url: str, model: str, uses_reasoning_effort: bool,
) -> Optional[str]:
    system_content = SYSTEM_PROMPT
    if tone_hint:
        system_content += f"\n\nTone note (phrasing only, never a new fact): {tone_hint}"

    create_kwargs = {
        "model": model,
        "temperature": 0.4,
        "max_tokens": 150,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": _build_user_content(template_message, state, checklist)},
        ],
    }
    if uses_reasoning_effort:
        create_kwargs["reasoning_effort"] = "low"

    try:
        # We handle provider fallback ourselves - no need for the SDK's own retries.
        client = openai_module.OpenAI(api_key=api_key, base_url=base_url, timeout=LLM_TIMEOUT_SECONDS, max_retries=0)
        response = client.chat.completions.create(**create_kwargs)
    except Exception as exc:
        logger.warning("response rephrasing call to %s failed (%s)", base_url, type(exc).__name__)
        return None

    rephrased = (response.choices[0].message.content or "").strip()
    if not rephrased:
        return None

    if not _facts_preserved(template_message, rephrased):
        logger.warning("rephrasing from %s dropped a required fact", base_url)
        return None

    return rephrased


def naturalize(
    template_message: str, state: ConversationState, tone_hint: Optional[str] = None, checklist: Optional[list] = None
) -> Optional[str]:
    """Rephrased version of template_message, or None if rephrasing isn't
    possible/safe - callers should use template_message as-is in that case.
    tone_hint and checklist are optional, deterministically-computed
    phrasing aids (see agent._process_turn / agent._response_checklist) -
    neither can smuggle in a new fact, since _facts_preserved still checks
    every candidate against the template alone.
    """
    try:
        import openai
    except ImportError:
        return None

    result = _call_provider(
        openai, template_message, state, tone_hint, checklist, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
        uses_reasoning_effort=LLM_USES_REASONING_EFFORT,
    )
    if result is not None:
        return result

    if LLM_FALLBACK_ENABLED:
        logger.warning("primary response rephrasing failed - trying fallback provider")
        result = _call_provider(
            openai, template_message, state, tone_hint, checklist, LLM_FALLBACK_API_KEY, LLM_FALLBACK_BASE_URL, LLM_FALLBACK_MODEL,
            uses_reasoning_effort=LLM_FALLBACK_USES_REASONING_EFFORT,
        )
        if result is not None:
            return result

    return None
