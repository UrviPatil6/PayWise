"""
Thin HTTP client for the two provided APIs. This is the only file that
knows about HTTP status codes, JSON shapes, and network failure modes -
agent.py just calls lookup_account()/process_payment() and gets back a
typed outcome or a raised ApiUnavailableError.

Result objects (LookupOutcome/PaymentOutcome) are plain dataclasses, not
Pydantic models: they're built by this module, not parsed from an external
source, so there's nothing to validate - a dataclass is the simpler choice
per "prefer a function/plain structure over a class when validation adds
no value."

Logging discipline: every log line in this file only includes whitelisted
fields (account_id, status code, API error_code, masked card suffix).
Nothing here ever logs the request body, response body, raw card number,
or CVV - that's what "do not log raw card data" actually requires in code,
not just in a comment.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import requests

from config import API_BASE_URL, HTTP_MAX_RETRIES, HTTP_TIMEOUT_SECONDS, get_logger
from models import Account
from validators import mask_card_number

logger = get_logger(__name__)


class ApiUnavailableError(Exception):
    """Raised for anything that isn't a well-formed business response:
    network failure, timeout, unexpected status code, or malformed JSON.
    agent.py treats this uniformly as a RETRYABLE failure.
    """


@dataclass
class LookupOutcome:
    found: bool
    account: Optional[Account]
    error_code: Optional[str]


@dataclass
class PaymentOutcome:
    success: bool
    transaction_id: Optional[str]
    error_code: Optional[str]


def _post_with_retry(path: str, payload: dict) -> requests.Response:
    url = f"{API_BASE_URL}{path}"
    last_error = None
    for attempt in range(HTTP_MAX_RETRIES + 1):
        try:
            return requests.post(url, json=payload, timeout=HTTP_TIMEOUT_SECONDS)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            logger.warning("request to %s failed (attempt %d): %s", path, attempt + 1, type(exc).__name__)
    raise ApiUnavailableError(f"{path} unreachable after {HTTP_MAX_RETRIES + 1} attempts") from last_error


def _parse_json(response: requests.Response) -> dict:
    try:
        return response.json()
    except ValueError as exc:
        raise ApiUnavailableError("API returned a non-JSON response") from exc


def lookup_account(account_id: str) -> LookupOutcome:
    response = _post_with_retry("/api/lookup-account", {"account_id": account_id})

    if response.status_code == 200:
        body = _parse_json(response)
        try:
            account = Account(**body)
        except Exception as exc:  # pydantic ValidationError, missing/wrong-typed fields
            raise ApiUnavailableError("lookup-account response missing expected fields") from exc
        logger.info("lookup_account account_id=%s status=200", account_id)
        return LookupOutcome(found=True, account=account, error_code=None)

    if response.status_code == 404:
        body = _parse_json(response)
        error_code = body.get("error_code", "account_not_found")
        logger.info("lookup_account account_id=%s status=404 error_code=%s", account_id, error_code)
        return LookupOutcome(found=False, account=None, error_code=error_code)

    raise ApiUnavailableError(f"lookup-account returned unexpected status {response.status_code}")


def process_payment(account_id: str, amount: Decimal, card: dict) -> PaymentOutcome:
    """`card` has keys: cardholder_name, card_number, cvv, expiry_month, expiry_year."""
    payload = {
        "account_id": account_id,
        "amount": float(amount),
        "payment_method": {"type": "card", "card": card},
    }
    response = _post_with_retry("/api/process-payment", payload)
    masked = mask_card_number(card["card_number"])

    if response.status_code == 200:
        body = _parse_json(response)
        txn_id = body.get("transaction_id")
        if not body.get("success") or not txn_id:
            raise ApiUnavailableError("process-payment 200 response missing transaction_id")
        logger.info("process_payment account_id=%s card=%s status=200 result=success", account_id, masked)
        return PaymentOutcome(success=True, transaction_id=txn_id, error_code=None)

    if response.status_code in (400, 422):
        body = _parse_json(response)
        # The live API returns 400 {"error_code": "invalid_args"} for some
        # malformed requests (e.g. a bad expiry) instead of the documented
        # per-field error codes. Our own validators.py should catch these
        # before we ever get here; "invalid_request" is a fallback label
        # for the rare case something still slips through.
        error_code = body.get("error_code") or "invalid_request"
        if error_code == "invalid_args":
            error_code = "invalid_request"
        logger.info(
            "process_payment account_id=%s card=%s status=%d error_code=%s",
            account_id, masked, response.status_code, error_code,
        )
        return PaymentOutcome(success=False, transaction_id=None, error_code=error_code)

    raise ApiUnavailableError(f"process-payment returned unexpected status {response.status_code}")
