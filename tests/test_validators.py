from decimal import Decimal

import validators as v


def test_account_id_format():
    assert v.validate_account_id("ACC1001")[0] is True
    assert v.validate_account_id("1001")[0] is False
    assert v.validate_account_id("")[0] is False


def test_dob_valid():
    assert v.validate_dob("1990-05-14")[0] is True


def test_dob_leap_year_valid_case_acc1004():
    # 1988 is a leap year - Feb 29 is a real calendar date.
    assert v.validate_dob("1988-02-29")[0] is True


def test_dob_leap_year_invalid_nearby_dates():
    # Both are "nearby but incorrect" per the assignment's ACC1004 note:
    # neither 1989 nor 1990 is a leap year, so Feb 29 doesn't exist in them.
    assert v.validate_dob("1989-02-29")[0] is False
    assert v.validate_dob("1990-02-29")[0] is False


def test_dob_garbage_format():
    ok, msg = v.validate_dob("not a date")
    assert ok is False
    assert msg


def test_dob_future_rejected():
    assert v.validate_dob("2999-01-01")[0] is False


def test_aadhaar_last4():
    assert v.validate_aadhaar_last4("4321")[0] is True
    assert v.validate_aadhaar_last4("432")[0] is False
    assert v.validate_aadhaar_last4("43219")[0] is False


def test_pincode():
    assert v.validate_pincode("400001")[0] is True
    assert v.validate_pincode("40001")[0] is False


def test_amount_positive_and_within_balance():
    ok, _ = v.validate_amount(Decimal("500"), Decimal("1250.75"))
    assert ok is True


def test_amount_zero_or_negative_rejected():
    assert v.validate_amount(Decimal("0"), Decimal("100"))[0] is False
    assert v.validate_amount(Decimal("-5"), Decimal("100"))[0] is False


def test_amount_too_many_decimals_rejected():
    assert v.validate_amount(Decimal("10.999"), Decimal("100"))[0] is False


def test_amount_exceeds_balance_rejected():
    ok, msg = v.validate_amount(Decimal("2000"), Decimal("1250.75"))
    assert ok is False
    assert "balance" in msg.lower()


def test_amount_none_rejected():
    assert v.validate_amount(None, Decimal("100"))[0] is False


def test_luhn_check_known_valid_card():
    assert v.luhn_check("4532015112830366") is True


def test_luhn_check_known_invalid_card():
    assert v.luhn_check("4532015112830367") is False


def test_validate_card_number_normalizes_spaces():
    ok, err, normalized = v.validate_card_number("4532 0151 1283 0366")
    assert ok is True
    assert normalized == "4532015112830366"


def test_validate_card_number_wrong_length():
    ok, err, normalized = v.validate_card_number("1234567890")
    assert ok is False


def test_cvv_standard_card():
    assert v.validate_cvv("123", "4532015112830366")[0] is True
    assert v.validate_cvv("12", "4532015112830366")[0] is False
    assert v.validate_cvv("1234", "4532015112830366")[0] is False


def test_cvv_amex_requires_4_digits():
    amex_number = "371449635398431"
    assert v.validate_cvv("1234", amex_number)[0] is True
    assert v.validate_cvv("123", amex_number)[0] is False


def test_expiry_valid_future_date():
    assert v.validate_expiry(12, 2027)[0] is True


def test_expiry_expired_rejected():
    assert v.validate_expiry(1, 2020)[0] is False


def test_expiry_invalid_month_rejected():
    assert v.validate_expiry(13, 2027)[0] is False
    assert v.validate_expiry(0, 2027)[0] is False


def test_expiry_missing_fields_rejected():
    assert v.validate_expiry(None, 2027)[0] is False
    assert v.validate_expiry(12, None)[0] is False


def test_mask_card_number_hides_all_but_last_four():
    masked = v.mask_card_number("4532015112830366")
    assert masked == "**** **** **** 0366"
    assert "4532" not in masked
