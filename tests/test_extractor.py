"""
These use the exact "what real users sound like" examples from the
assignment brief, so a reviewer can check extractor coverage directly
against the spec.
"""

from decimal import Decimal

from extractor import extract_fields
from state import ConversationState

ACC_ID = ConversationState.AWAIT_ACCOUNT_ID
IDENTITY = ConversationState.AWAIT_IDENTITY
AMOUNT = ConversationState.AWAIT_AMOUNT
CARD = ConversationState.AWAIT_CARD_DETAILS


def test_account_id_messy_variants():
    assert extract_fields("yeah my account number is ACC1001 I think", ACC_ID).account_id == "ACC1001"
    assert extract_fields("it's ACC 1001", ACC_ID).account_id == "ACC1001"
    assert extract_fields("account id: acc1001", ACC_ID).account_id == "ACC1001"


def test_account_id_bare_digits_with_no_prefix():
    # A user who just types the digits with no "ACC" text at all, e.g.
    # replying "1001" to "please share your account ID" - only trusted in
    # this state, since a bare number elsewhere means something else.
    assert extract_fields("1001", ACC_ID).account_id == "ACC1001"
    assert extract_fields("48556", ACC_ID).account_id == "ACC48556"
    assert extract_fields("1001", AMOUNT).account_id is None
    assert extract_fields("65", ACC_ID).account_id is None  # too short to be plausible


def test_account_id_number_near_the_word_account():
    # "my account is 1001" has no "ACC" prefix and isn't a bare-digits-only
    # reply either - anchored on the word "account" (like Aadhaar/pincode's
    # keyword anchoring) rather than "the one digit group in the message",
    # so it can't repeat the cross-field CVV bug (see test_extractor's CVV
    # tests) where an unrelated number elsewhere gets misread.
    assert extract_fields("my account is 1001", ACC_ID).account_id == "ACC1001"
    assert extract_fields("yeah I think the account should be 1001", ACC_ID).account_id == "ACC1001"
    # Only trusted in this state, and never for a message about something else.
    assert extract_fields("my account is 1001", IDENTITY).account_id is None
    assert extract_fields("actually pay 700", CARD).account_id is None


def test_full_name_variants():
    assert extract_fields("my name is Nithin Jain", IDENTITY).full_name == "Nithin Jain"
    assert extract_fields("it's Nithin, Nithin Jain", IDENTITY).full_name == "Nithin Jain"
    assert (
        extract_fields(
            "you can call me Raja but my full name is Rajarajeswari Balasubramaniam", IDENTITY
        ).full_name
        == "Rajarajeswari Balasubramaniam"
    )
    # Bare reply with no keywords at all.
    assert extract_fields("Nithin Jain", IDENTITY).full_name == "Nithin Jain"


def test_full_name_its_prefix_strips_the_filler_word():
    # Regression test for a real bug: "it's"/"its" wasn't recognized as a
    # name-introduction phrase outside the comma-separated nickname
    # pattern ("it's Nithin, Nithin Jain"), so a bare "its Nithin" fell
    # through to the AWAIT_IDENTITY bare-reply fallback and was extracted
    # literally, filler word included ("its Nithin") - which could never
    # match a real account name. The comma pattern must still take
    # priority so "it's Nithin, Nithin Jain" keeps extracting the fuller
    # name, not just "Nithin".
    assert extract_fields("its Nithin", IDENTITY).full_name == "Nithin"
    assert extract_fields("it's Nithin", IDENTITY).full_name == "Nithin"
    assert extract_fields("it's Nithin, Nithin Jain", IDENTITY).full_name == "Nithin Jain"


def test_full_name_single_word_reply():
    # Regression test: a single-word reply ("urvi") used to be silently
    # dropped (the fallback required 2+ words), which meant the agent
    # repeated the same "confirm your full name" prompt forever with no
    # explanation. It's extracted now so the conversation can move
    # forward - it will correctly fail verification later since it won't
    # match a real two-word account name, which is the right place for
    # that rejection to happen.
    assert extract_fields("urvi", IDENTITY).full_name == "urvi"
    assert extract_fields("nithin", IDENTITY).full_name == "nithin"


def test_full_name_stopwords_not_treated_as_names():
    # Conversational filler must not be misread as a name attempt.
    for word in ("ok", "sure", "hi", "hello", "yes", "no", "thanks"):
        assert extract_fields(word, IDENTITY).full_name is None


def test_full_name_and_dob_concatenated_with_no_separator():
    # Regression test for a real gap found in live use: a name directly
    # followed by a date with nothing joining them ("Nithin Jain
    # 1990-05-14") used to only extract the DOB, silently dropping the
    # name - the agent would ask for the name again on the very next turn
    # even though it was right there. Casing is still preserved verbatim
    # (security-critical - see verification requirements), so the
    # lowercase variant must come back lowercase, not "corrected".
    result = extract_fields("Nithin Jain 1990-05-14", IDENTITY)
    assert result.full_name == "Nithin Jain"
    assert result.dob == "1990-05-14"

    result = extract_fields("nithin jain 14-05-1990", IDENTITY)
    assert result.full_name == "nithin jain"
    assert result.dob == "1990-05-14"


def test_field_label_words_not_misread_as_a_name_after_date_stripping():
    # Regression test for a bug introduced (and caught before shipping) by
    # the fix above: stripping the date out of "DOB is 1990-05-14" leaves
    # "DOB is", which is letters-only and would otherwise pass the bare-name
    # shape check. Field-label words must never be extracted as a name.
    result = extract_fields("DOB is 1990-05-14", IDENTITY)
    assert result.full_name is None
    assert result.dob == "1990-05-14"

    assert extract_fields("my pincode is 400001", IDENTITY).full_name is None
    assert extract_fields("aadhaar 4321", IDENTITY).full_name is None


def test_dob_variants():
    assert extract_fields("I was born on 14th May 1990", IDENTITY).dob == "1990-05-14"
    assert extract_fields("DOB is May 14, 90", IDENTITY).dob == "1990-05-14"
    assert extract_fields("14-05-1990", IDENTITY).dob == "1990-05-14"
    assert extract_fields("1990-05-14", IDENTITY).dob == "1990-05-14"


def test_dob_shaped_dates_are_never_misread_as_an_expiry():
    # Regression test for a real bug: _extract_expiry's guard against a
    # DOB like "14th May 1990" being misread as an expiry only worked when
    # there was no ordinal suffix ("14 May 1990") - the guard stripped the
    # "th"/"st"/"nd"/"rd" suffix *after* checking for trailing whitespace,
    # so it never actually reached the suffix and the guard silently did
    # nothing. A plain DOB with an ordinal suffix and zero card context
    # anywhere in the message or conversation must never set an expiry.
    for phrase in ("14th May 1990", "1st January 1990", "2nd March 1990", "3rd April 1990"):
        result = extract_fields(phrase, IDENTITY)
        assert result.expiry_month is None and result.expiry_year is None, phrase
        assert result.dob is not None, phrase  # the DOB itself must still extract correctly

    # A genuine expiry statement (no leading day digit) must still work.
    month, year = extract_fields("expires December 2027", CARD).expiry_month, extract_fields("expires December 2027", CARD).expiry_year
    assert (month, year) == (12, 2027)


def test_aadhaar_and_pincode_variants():
    assert extract_fields("last four of my Aadhaar is 4321", IDENTITY).aadhaar_last4 == "4321"
    assert extract_fields("pincode? it's 4 0 0 0 0 1", IDENTITY).pincode == "400001"
    result = extract_fields("Aadhaar ends with 9876, shall I give pincode instead?", IDENTITY)
    assert result.aadhaar_last4 == "9876"
    assert result.pincode is None  # no digits were actually given for pincode


def test_payment_amount_variants():
    assert extract_fields("I want to pay a thousand rupees", AMOUNT).payment_amount == Decimal("1000")
    assert extract_fields("just clear the full amount", AMOUNT).full_balance_requested is True
    assert extract_fields("can I do 500 for now?", AMOUNT).payment_amount == Decimal("500")
    assert extract_fields("I'll pay the whole thing", AMOUNT).full_balance_requested is True


def test_card_detail_variants():
    assert extract_fields("the card number is 4532 0151 1283 0366", CARD).card_number == "4532015112830366"
    month, year = (
        extract_fields("expires December 2027", CARD).expiry_month,
        extract_fields("expires December 2027", CARD).expiry_year,
    )
    assert (month, year) == (12, 2027)
    result = extract_fields("12/27", CARD)
    assert (result.expiry_month, result.expiry_year) == (12, 2027)
    assert extract_fields("CVV is one two three", CARD).cvv == "123"


def test_cvv_requires_an_explicit_label():
    # Regression test for a real, confirmed bug: an earlier version trusted
    # any lone 3-4 digit number in AWAIT_CARD_DETAILS as the CVV, including
    # in messages that were about something else entirely - "actually pay
    # 700" (an amount correction) got its "700" read as a CVV and reached
    # process_payment with a value the user never gave as a CVV. Regex must
    # only ever trust an explicitly labelled CVV; an unlabeled short reply
    # like "it's 123" is left to the LLM path's session-context awareness
    # (see llm_extractor.SYSTEM_PROMPT) rather than guessed at by regex.
    assert extract_fields("actually pay 700", CARD).cvv is None
    assert extract_fields("my card expires in May 2028", CARD).cvv is None
    assert extract_fields("call me back at 234", CARD).cvv is None
    assert extract_fields("111", CARD).cvv is None  # bare, unlabeled - no longer trusted
    assert extract_fields("it's 123", CARD).cvv is None  # unlabeled - regex no longer guesses

    # Labelled cases (including a couple of extra ways to say "CVV") still work.
    assert extract_fields("the cvv, 4321", CARD).cvv == "4321"
    assert extract_fields("security code 123", CARD).cvv == "123"
    assert extract_fields("security number 4321", CARD).cvv == "4321"
    assert extract_fields("123 or 456?", CARD).cvv is None


def test_multiple_fields_in_one_message():
    result = extract_fields(
        "My account ID is ACC1001, my name is Nithin Jain, DOB 1990-05-14", ACC_ID
    )
    assert result.account_id == "ACC1001"
    assert result.full_name == "Nithin Jain"
    assert result.dob == "1990-05-14"


def test_does_not_hallucinate_unset_fields():
    result = extract_fields("Hi there", ACC_ID)
    assert result.account_id is None
    assert result.full_name is None
    assert result.payment_amount is None
    assert result.card_number is None


def test_bare_number_is_state_dependent():
    # A lone "400001" only means pincode when we're actually collecting
    # identity info - in other states it shouldn't be guessed at all.
    assert extract_fields("400001", IDENTITY).pincode == "400001"
    assert extract_fields("400001", AMOUNT).pincode is None
