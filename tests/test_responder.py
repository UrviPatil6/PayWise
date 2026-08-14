"""
responder.py in isolation - same faked-OpenAI-client approach as
test_llm_extractor.py. Covers: a clean rephrase passing through, and the
two ways a rephrase gets rejected (API failure, dropped fact) - both must
degrade to None so agent.py falls back to the deterministic template.
"""

import responder


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        return self._response


class _FakeChat:
    def __init__(self, response):
        self.completions = _FakeCompletions(response)


class _FakeOpenAIClient:
    def __init__(self, response):
        self.chat = _FakeChat(response)


def _install_fake_openai(monkeypatch, content=None, raise_error=None):
    import openai

    def fake_constructor(*args, **kwargs):
        if raise_error:
            raise raise_error
        return _FakeOpenAIClient(_FakeResponse(content))

    monkeypatch.setattr(openai, "OpenAI", fake_constructor)


def test_valid_rephrase_preserving_facts_is_used(monkeypatch):
    _install_fake_openai(monkeypatch, content="Hey there! Your balance is ₹1,250.75 - how much would you like to pay?")
    result = responder.naturalize(
        "Identity verified. Your outstanding balance is ₹1,250.75. How much would you like to pay today?",
        None,
    )
    assert result == "Hey there! Your balance is ₹1,250.75 - how much would you like to pay?"


def test_rephrase_that_drops_a_required_amount_is_rejected(monkeypatch):
    # The rephrase changes the amount - must not be trusted.
    _install_fake_openai(monkeypatch, content="Your balance is ₹999.00 - how much would you like to pay?")
    result = responder.naturalize(
        "Identity verified. Your outstanding balance is ₹1,250.75. How much would you like to pay today?",
        None,
    )
    assert result is None


def test_rephrase_that_drops_a_transaction_id_is_rejected(monkeypatch):
    _install_fake_openai(monkeypatch, content="All set, your payment went through!")
    result = responder.naturalize(
        "Payment successful! ₹500.00 was charged (transaction ID txn_abc123). Thanks!",
        None,
    )
    assert result is None


def test_api_failure_degrades_to_none(monkeypatch):
    _install_fake_openai(monkeypatch, raise_error=ConnectionError("simulated failure"))
    result = responder.naturalize("Please share your account ID to get started.", None)
    assert result is None


def test_empty_response_degrades_to_none(monkeypatch):
    _install_fake_openai(monkeypatch, content="")
    result = responder.naturalize("Please share your account ID to get started.", None)
    assert result is None


def test_falls_back_to_second_provider_when_primary_fails(monkeypatch):
    monkeypatch.setattr("responder.LLM_BASE_URL", "https://primary.example/v1")
    monkeypatch.setattr("responder.LLM_FALLBACK_ENABLED", True)
    monkeypatch.setattr("responder.LLM_FALLBACK_API_KEY", "fake-fallback-key")
    monkeypatch.setattr("responder.LLM_FALLBACK_BASE_URL", "https://fallback.example/v1")
    monkeypatch.setattr("responder.LLM_FALLBACK_MODEL", "fallback-model")

    def fake_constructor(*args, base_url=None, **kwargs):
        if base_url == "https://primary.example/v1":
            raise ConnectionError("primary provider unreachable")
        return _FakeOpenAIClient(_FakeResponse("Hi there! Could you share your account ID?"))

    import openai
    monkeypatch.setattr(openai, "OpenAI", fake_constructor)

    result = responder.naturalize("Please share your account ID to get started.", None)
    assert result == "Hi there! Could you share your account ID?"


def test_returns_none_when_both_providers_fail(monkeypatch):
    monkeypatch.setattr("responder.LLM_FALLBACK_ENABLED", True)
    monkeypatch.setattr("responder.LLM_FALLBACK_API_KEY", "fake-fallback-key")
    _install_fake_openai(monkeypatch, raise_error=ConnectionError("both providers down"))
    result = responder.naturalize("Please share your account ID to get started.", None)
    assert result is None
