"""
Regex/rule-based field extraction (models.ExtractedFields) - the
deterministic fallback used when no LLM is configured or llm_extractor.py
fails. Every field is independently optional. `state` is only consulted
for genuinely ambiguous bare replies (e.g. a lone "400001" is a pincode
only while waiting on identity info) - every keyword-anchored pattern
works regardless of state, so volunteered-early info is still picked up.
"""

import re
from decimal import Decimal
from typing import Optional, Tuple

from models import ExtractedFields
from state import ConversationState

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

DIGIT_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

# Conversational filler that must never be read as a single-word name
# attempt (see _extract_full_name's bare-reply fallback).
_NAME_STOPWORDS = {
    "yes", "no", "ok", "okay", "sure", "yeah", "yep", "nope", "please",
    "thanks", "thank you", "hi", "hello", "hey", "what", "why", "how",
    "wait", "hmm", "um", "uh", "idk", "maybe", "got it", "gotcha", "alright",
    "cool", "great thanks",
}

# Strips a directly-appended date ("Nithin Jain 1990-05-14") so the name
# portion can still be recognized - see _extract_full_name.
_DATE_SPAN_PATTERN = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b")

# Catches "DOB is 1990-05-14" -> (date stripped) "DOB is", which is
# letters-only and would otherwise pass the bare-name shape check.
_FIELD_LABEL_WORDS = {
    "dob", "date", "birth", "born", "aadhaar", "pincode", "pin", "code",
    "name", "cvv", "expiry", "card", "amount", "pay", "balance",
}


def _looks_like_a_name(candidate: str) -> bool:
    return (
        bool(re.fullmatch(r"[A-Za-z][A-Za-z .'\-]*[A-Za-z]", candidate))
        and candidate.lower() not in _NAME_STOPWORDS
        and not (set(candidate.lower().split()) & _FIELD_LABEL_WORDS)
        # "never mind"/"how are you" pass the letters-only shape but aren't names.
        and _detect_intent(candidate) is None
    )

_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

FULL_BALANCE_PHRASES = re.compile(
    r"\b(full amount|full balance|entire balance|whole balance|the whole thing|"
    r"clear (it all|the full amount|everything)|pay (it all|everything|in full)|"
    r"everything (i owe|owed)?)\b",
    re.IGNORECASE,
)

# Deterministic fallback for intent detection - deliberately conservative
# (e.g. bare "cancel"/"stop" excluded) since it has no negation handling.
_CANCEL_PHRASES = re.compile(
    r"\b(never\s?mind|forget it|i(?:'ve| have)? changed my mind|not interested (any\s?more|anymore)?|let'?s stop (here|this)?)\b",
    re.IGNORECASE,
)
_BALANCE_QUERY_PHRASES = re.compile(
    r"\b(how much (do|did) i owe|what'?s my balance|what is my balance|"
    r"how much is my balance|check my balance|what do i owe)\b",
    re.IGNORECASE,
)
_SMALL_TALK_PHRASES = re.compile(
    r"\b(how are you|thanks|thank you|thx|no problem|sounds good|"
    r"good morning|good afternoon|good evening|great|awesome)\b",
    re.IGNORECASE,
)


def _detect_intent(text: str) -> Optional[str]:
    if _CANCEL_PHRASES.search(text):
        return "cancel"
    if _BALANCE_QUERY_PHRASES.search(text):
        return "balance_query"
    # Trailing punctuation ("Hi!", "Okay.", "Got it...") shouldn't defeat
    # the exact-phrase stopword check below - real greetings/acks are
    # punctuated far more often than the bare word.
    stripped_lower = text.strip().lower().rstrip("!.?,;: ")
    if stripped_lower in _NAME_STOPWORDS or _SMALL_TALK_PHRASES.search(text):
        return "small_talk"
    return None


def _words_to_amount(text: str) -> Optional[Decimal]:
    """Parses simple spelled-out amounts like 'a thousand', 'five hundred',
    'twelve hundred and fifty'. Covers common conversational amounts, not a
    general English-number parser - good enough for a payment collection
    bot where amounts are spoken, not essays.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    total = 0
    current = 0
    found_any = False
    for word in words:
        if word == "a" or word == "an":
            current = current or 1
            continue
        if word in _ONES:
            current += _ONES[word]
            found_any = True
        elif word in _TENS:
            current += _TENS[word]
            found_any = True
        elif word == "hundred":
            current = (current or 1) * 100
            found_any = True
        elif word == "thousand":
            total += (current or 1) * 1000
            current = 0
            found_any = True
        elif word in ("and",):
            continue
    total += current
    return Decimal(total) if found_any and total > 0 else None


def _extract_account_id(text: str, state: ConversationState) -> Optional[str]:
    match = re.search(r"\bACC\s*-?\s*(\d{3,6})\b", text, re.IGNORECASE)
    if match:
        return f"ACC{match.group(1)}"

    # No "ACC" prefix at all - only trusted in this state, so a bare
    # number elsewhere (an amount, a CVV) is never read as an account ID.
    if state == ConversationState.AWAIT_ACCOUNT_ID:
        stripped = text.strip()
        if re.fullmatch(r"\d{3,6}", stripped):
            return f"ACC{stripped}"

        # A number named near the word "account" in an otherwise wordy
        # reply ("my account is 1001", "the account should be 1001") -
        # anchored on the keyword (like Aadhaar/pincode below) so it
        # can't fire on an unrelated number elsewhere in the message.
        for digits in _digit_groups_near(text, r"account", window=20):
            if 3 <= len(digits) <= 6:
                return f"ACC{digits}"

    return None


def _extract_full_name(text: str, state: ConversationState) -> Optional[str]:
    # Highest priority: explicit "full name is X" - handles the "you can
    # call me Raja but my full name is Rajarajeswari Balasubramaniam" case,
    # where the nickname earlier in the sentence must NOT be picked up.
    for pattern in (
        r"full name is\s+([A-Za-z][A-Za-z .'\-]*)",
        r"(?:my name is|name is|i am|i'm)\s+([A-Za-z][A-Za-z .'\-]*)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _clean_name(match.group(1))

    # "it's Nithin, Nithin Jain" - nickname then full name after a comma.
    match = re.search(r"it'?s\s+[A-Za-z]+,\s*([A-Za-z][A-Za-z .'\-]*)", text, re.IGNORECASE)
    if match:
        return _clean_name(match.group(1))

    # A Title-Case, comma-separated segment with no keyword at all, e.g.
    # "ACC1001, Nithin Jain, DOB 1990-05-14" - a common way to list several
    # fields in one message without ever saying "my name is". Segments
    # like "ACC1001" (has digits) or "DOB" (all caps) don't match this
    # shape, which keeps it from firing on the other fields in the list.
    for segment in re.split(r",| and ", text):
        segment = segment.strip()
        if re.fullmatch(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3}", segment):
            return segment

    # Bare reply while waiting on identity info, e.g. "Nithin Jain" or just
    # "Nithin" - even a single word lets the conversation move forward
    # (correctly failing verification and counting a retry) rather than
    # repeating "confirm your full name" forever.
    if state == ConversationState.AWAIT_IDENTITY:
        stripped = text.strip()
        if _looks_like_a_name(stripped):
            return _clean_name(stripped)

        # Name directly followed by a date with no separator, e.g.
        # "Nithin Jain 1990-05-14" - strip the date and see if a name remains.
        remainder = _DATE_SPAN_PATTERN.sub("", text).strip(" ,.-")
        if remainder and remainder != stripped and _looks_like_a_name(remainder):
            return _clean_name(remainder)

    return None


def _clean_name(raw: str) -> str:
    # Trim at the first word that signals the name has ended, e.g.
    # "Nithin Jain and my dob is..." -> "Nithin Jain".
    stop_words = (" and ", " dob", " born", " aadhaar", " pincode", " date of birth")
    name = raw.strip().rstrip(".,")
    lowered = name.lower()
    cut = len(name)
    for stop in stop_words:
        idx = lowered.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return name[:cut].strip().rstrip(",")


def _expand_two_digit_year(yy: int) -> int:
    # Same rule Python's own %y strptime directive uses: 69-99 -> 1900s,
    # 00-68 -> 2000s. Correct for the sample accounts (all born 1985-1992).
    return 1900 + yy if yy >= 69 else 2000 + yy


def _extract_dob(text: str) -> Optional[str]:
    # Already ISO: 1990-05-14
    match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if match:
        y, m, d = match.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    # Numeric day-month-year: 14-05-1990 or 14/05/1990 (Indian convention).
    match = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", text)
    if match:
        d, m, y = (int(g) for g in match.groups())
        return f"{y:04d}-{m:02d}-{d:02d}"

    # "14th May 1990" / "14 May 90"
    match = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4}|\d{2})\b", text
    )
    if match:
        d, month_name, y = match.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            year = int(y) if len(y) == 4 else _expand_two_digit_year(int(y))
            return f"{year:04d}-{month:02d}-{int(d):02d}"

    # "May 14, 1990" / "May 14, 90"
    match = re.search(
        r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4}|\d{2})\b", text
    )
    if match:
        month_name, d, y = match.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            year = int(y) if len(y) == 4 else _expand_two_digit_year(int(y))
            return f"{year:04d}-{month:02d}-{int(d):02d}"

    return None


def _digit_groups_near(text: str, keyword_pattern: str, window: int = 25):
    """Yields digit-only strings (spaces stripped) found within `window`
    characters after each occurrence of keyword_pattern. Used for Aadhaar/
    pincode: scanning near the keyword instead of the whole message avoids
    picking up unrelated digits elsewhere in a multi-field message.
    """
    for keyword_match in re.finditer(keyword_pattern, text, re.IGNORECASE):
        chunk = text[keyword_match.end(): keyword_match.end() + window]
        for digit_match in re.finditer(r"\d(?:[\d ]*\d)?", chunk):
            yield re.sub(r"\s", "", digit_match.group(0))


def _extract_aadhaar_last4(text: str, state: ConversationState) -> Optional[str]:
    # Require at least 2 digits so a filler number right after the keyword
    # - e.g. the "4" in "Aadhaar last 4 is 9876" - doesn't get mistaken for
    # the value itself; the real value is always the 4+ digit group.
    for digits in _digit_groups_near(text, r"aadhaar"):
        if len(digits) >= 2:
            return digits[-4:]

    if state == ConversationState.AWAIT_IDENTITY:
        stripped = re.sub(r"\s", "", text.strip())
        if re.fullmatch(r"\d{4}", stripped):
            return stripped

    return None


def _extract_pincode(text: str, state: ConversationState) -> Optional[str]:
    for digits in _digit_groups_near(text, r"pin\s?code"):
        if len(digits) == 6:
            return digits

    if state == ConversationState.AWAIT_IDENTITY:
        stripped = re.sub(r"\s", "", text.strip())
        if re.fullmatch(r"\d{6}", stripped):
            return stripped

    return None


_MONEY_KEYWORD = re.compile(r"₹|rs\.?|inr|rupees|\bpay\b|\bamount\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d{1,2})?")


def _extract_amount(text: str, state: ConversationState) -> Tuple[Optional[Decimal], bool]:
    if FULL_BALANCE_PHRASES.search(text):
        return None, True

    keyword_match = _MONEY_KEYWORD.search(text)
    if keyword_match:
        # Look for the number right after the keyword ("pay 300", "₹500"),
        # then right before it ("500 rupees"). Anchoring the search to the
        # keyword's neighborhood - rather than taking the first number
        # anywhere in the message - matters once a message also contains
        # unrelated digits, e.g. an account ID: "ACC1001, ... pay 300"
        # would otherwise latch onto the "1001".
        after = text[keyword_match.end(): keyword_match.end() + 20]
        number_match = _NUMBER.search(after)
        if number_match:
            return Decimal(number_match.group(0).replace(",", "")), False

        words_amount = _words_to_amount(after)
        if words_amount is not None:
            return words_amount, False

        before = text[max(0, keyword_match.start() - 20): keyword_match.start()]
        number_match = re.search(r"\d[\d,]*(?:\.\d{1,2})?\s*$", before)
        if number_match:
            return Decimal(number_match.group(0).strip().replace(",", "")), False

    elif state == ConversationState.AWAIT_AMOUNT:
        # No currency/payment keyword at all - only trust a bare number or
        # spelled-out number when we specifically asked for an amount.
        number_match = _NUMBER.search(text)
        if number_match:
            return Decimal(number_match.group(0).replace(",", "")), False
        words_amount = _words_to_amount(text)
        if words_amount is not None:
            return words_amount, False

    return None, False


def _extract_card_number(text: str) -> Optional[str]:
    # Grouped form first, e.g. "4532 0151 1283 0366" - spaces are only
    # allowed *within* a proper 4-digit grouping, not just anywhere. A
    # looser "digits and spaces mixed together" pattern would greedily eat
    # into an adjacent, differently-formatted number (e.g. the "12" of a
    # following "12/27" expiry) when there's no punctuation between them.
    match = re.search(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,7}\b", text)
    if match:
        return re.sub(r"[ -]", "", match.group(0))

    # Fallback: one contiguous run of digits, no internal spaces.
    match = re.search(r"\b\d{13,19}\b", text)
    if match:
        return match.group(0)

    return None


_CVV_LABEL = r"(?:cvv|security code|security number|card verification (?:value|code|number))"


def _extract_cvv(text: str) -> Optional[str]:
    """Requires an explicit label ("CVV", "security code", ...) - never a
    bare or embedded number. A lone 3-4 digit number elsewhere in a
    message is exactly as likely to be an amount correction, an expiry
    year, or unrelated small talk as it is to be a CVV; regex has no way
    to tell those apart from shape alone, and a wrong guess here means
    submitting a payment with a CVV the user never actually gave. Regex
    only handles clearly-labelled cases - resolving a genuinely unlabeled
    short reply from context (e.g. "it's 123" when CVV is the one field
    left) is exactly what the LLM path's session-context awareness is
    for; see llm_extractor.py's SYSTEM_PROMPT."""
    match = re.search(rf"{_CVV_LABEL}\D{{0,15}}(\d{{3,4}})", text, re.IGNORECASE)
    if match:
        return match.group(1)

    # "CVV is one two three"
    match = re.search(
        rf"{_CVV_LABEL}\D{{0,10}}((?:(?:zero|one|two|three|four|five|six|seven|eight|nine)[\s,]*){{3,4}})",
        text, re.IGNORECASE,
    )
    if match:
        words = re.findall(r"[a-zA-Z]+", match.group(1))
        digits = "".join(DIGIT_WORDS[w.lower()] for w in words if w.lower() in DIGIT_WORDS)
        if len(digits) in (3, 4):
            return digits

    return None


def _extract_expiry(text: str) -> Tuple[Optional[int], Optional[int]]:
    # MM/YY or MM/YYYY, e.g. "12/27" or "12/2027"
    match = re.search(r"\b(\d{1,2})\s*/\s*(\d{2,4})\b", text)
    if match:
        month, year = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            if year < 100:
                year += 2000
            return month, year

    # "expires December 2027" - a bare month-name + year with no leading
    # day number, which is what tells this apart from a DOB like "14th May
    # 1990" (DOB always has a day digit before the month name).
    match = re.search(r"\b([A-Za-z]+)\s+(\d{4})\b", text)
    if match:
        month = MONTHS.get(match.group(1).lower())
        preceding = text[: match.start()]
        if month and not re.search(r"\d\s*$", preceding.rstrip("stndrh")):
            return month, int(match.group(2))

    return None, None


def _extract_cardholder_name(text: str) -> Optional[str]:
    match = re.search(
        r"(?:cardholder name is|name on (?:the )?card is|card ?holder(?: name)? is)\s+([A-Za-z][A-Za-z .'\-]*)",
        text,
        re.IGNORECASE,
    )
    return _clean_name(match.group(1)) if match else None


def extract_fields(text: str, state: ConversationState) -> ExtractedFields:
    amount, full_balance = _extract_amount(text, state)
    month, year = _extract_expiry(text)
    return ExtractedFields(
        account_id=_extract_account_id(text, state),
        full_name=_extract_full_name(text, state),
        dob=_extract_dob(text),
        aadhaar_last4=_extract_aadhaar_last4(text, state),
        pincode=_extract_pincode(text, state),
        payment_amount=amount,
        full_balance_requested=full_balance,
        card_number=_extract_card_number(text),
        cvv=_extract_cvv(text),
        expiry_month=month,
        expiry_year=year,
        cardholder_name=_extract_cardholder_name(text),
        intent=_detect_intent(text),
    )
