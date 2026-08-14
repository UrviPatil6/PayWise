"""
llm_extractor.py in isolation - the OpenAI-compatible client (used for
Groq) is faked out so these run offline. Covers: disabled-by-default
behavior, a valid structured response, and every failure mode (network
error, malformed tool arguments, model not calling the tool at all)
degrading to None instead of raising - that's the "fail gracefully
without an LLM" contract agent.py relies on.
"""

import json

import llm_extractor
from state import ConversationState


class _FakeFunction:
    def __init__(self, arguments: dict):
        self.arguments = json.dumps(arguments)


class _FakeToolCall:
    def __init__(self, arguments: dict):
        self.function = _FakeFunction(arguments)


class _FakeMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, choices):
        self.choices = choices


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


def _install_fake_openai(monkeypatch, response=None, raise_error=None):
    import openai

    def fake_constructor(*args, **kwargs):
        if raise_error:
            raise raise_error
        return _FakeOpenAIClient(response)

    monkeypatch.setattr(openai, "OpenAI", fake_constructor)


def test_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr("llm_extractor.LLM_ENABLED", False)
    result = llm_extractor.extract_fields_llm("my name is Nithin Jain", ConversationState.AWAIT_IDENTITY)
    assert result is None


def test_parses_valid_tool_call_response(monkeypatch):
    monkeypatch.setattr("llm_extractor.LLM_ENABLED", True)
    monkeypatch.setattr("llm_extractor.LLM_API_KEY", "fake-key-for-test")
    response = _FakeResponse([_FakeChoice(_FakeMessage([
        _FakeToolCall({"full_name": "Nithin Jain", "dob": "1990-05-14"})
    ]))])
    _install_fake_openai(monkeypatch, response=response)

    result = llm_extractor.extract_fields_llm(
        "it's me, Nithin Jain, born 14th May 1990", ConversationState.AWAIT_IDENTITY
    )
    assert result is not None
    assert result.full_name == "Nithin Jain"
    assert result.dob == "1990-05-14"
    assert result.account_id is None  # fields not returned by the model stay null


def test_degrades_to_none_on_api_error(monkeypatch):
    monkeypatch.setattr("llm_extractor.LLM_ENABLED", True)
    monkeypatch.setattr("llm_extractor.LLM_API_KEY", "fake-key-for-test")
    _install_fake_openai(monkeypatch, raise_error=ConnectionError("simulated network failure"))

    result = llm_extractor.extract_fields_llm("ACC1001", ConversationState.AWAIT_ACCOUNT_ID)
    assert result is None


def test_degrades_to_none_on_malformed_tool_arguments(monkeypatch):
    monkeypatch.setattr("llm_extractor.LLM_ENABLED", True)
    monkeypatch.setattr("llm_extractor.LLM_API_KEY", "fake-key-for-test")
    # expiry_month should be an int/null - a schema mismatch like this must
    # not raise up into the caller.
    response = _FakeResponse([_FakeChoice(_FakeMessage([
        _FakeToolCall({"expiry_month": ["not", "a", "number"]})
    ]))])
    _install_fake_openai(monkeypatch, response=response)

    result = llm_extractor.extract_fields_llm("expires sometime", ConversationState.AWAIT_CARD_DETAILS)
    assert result is None


def test_degrades_to_none_when_model_does_not_call_the_tool(monkeypatch):
    monkeypatch.setattr("llm_extractor.LLM_ENABLED", True)
    monkeypatch.setattr("llm_extractor.LLM_API_KEY", "fake-key-for-test")
    _install_fake_openai(monkeypatch, response=_FakeResponse([_FakeChoice(_FakeMessage(None))]))

    result = llm_extractor.extract_fields_llm("hello", ConversationState.AWAIT_ACCOUNT_ID)
    assert result is None


def _enable_fallback(monkeypatch, forced_tool_choice=True):
    monkeypatch.setattr("llm_extractor.LLM_FALLBACK_ENABLED", True)
    monkeypatch.setattr("llm_extractor.LLM_FALLBACK_API_KEY", "fake-fallback-key")
    monkeypatch.setattr("llm_extractor.LLM_FALLBACK_BASE_URL", "https://fallback.example/v1")
    monkeypatch.setattr("llm_extractor.LLM_FALLBACK_MODEL", "fallback-model")
    monkeypatch.setattr("llm_extractor.LLM_FALLBACK_SUPPORTS_FORCED_TOOL_CHOICE", forced_tool_choice)


def test_falls_back_to_second_provider_when_primary_fails(monkeypatch):
    # The primary (Groq) failing shouldn't mean giving up on the LLM path
    # entirely when a second provider is configured - it should get one
    # real attempt there before falling back to the regex parser.
    monkeypatch.setattr("llm_extractor.LLM_ENABLED", True)
    monkeypatch.setattr("llm_extractor.LLM_API_KEY", "fake-primary-key")
    monkeypatch.setattr("llm_extractor.LLM_BASE_URL", "https://primary.example/v1")
    _enable_fallback(monkeypatch)

    success = _FakeResponse([_FakeChoice(_FakeMessage([_FakeToolCall({"full_name": "Nithin Jain"})]))])

    def fake_constructor(*args, base_url=None, **kwargs):
        if base_url == "https://primary.example/v1":
            raise ConnectionError("primary provider unreachable")
        return _FakeOpenAIClient(success)

    import openai
    monkeypatch.setattr(openai, "OpenAI", fake_constructor)

    result = llm_extractor.extract_fields_llm("my name is Nithin Jain", ConversationState.AWAIT_IDENTITY)
    assert result is not None
    assert result.full_name == "Nithin Jain"


def test_returns_none_when_both_providers_fail(monkeypatch):
    monkeypatch.setattr("llm_extractor.LLM_ENABLED", True)
    monkeypatch.setattr("llm_extractor.LLM_API_KEY", "fake-primary-key")
    _enable_fallback(monkeypatch)
    _install_fake_openai(monkeypatch, raise_error=ConnectionError("both providers down"))

    result = llm_extractor.extract_fields_llm("my name is Nithin Jain", ConversationState.AWAIT_IDENTITY)
    assert result is None


def test_fallback_not_attempted_when_not_configured(monkeypatch):
    # No OPENROUTER_API_KEY-equivalent set - LLM_FALLBACK_ENABLED stays
    # False, so a primary failure goes straight to None (regex fallback),
    # matching pre-fallback-feature behavior exactly.
    monkeypatch.setattr("llm_extractor.LLM_ENABLED", True)
    monkeypatch.setattr("llm_extractor.LLM_API_KEY", "fake-primary-key")
    monkeypatch.setattr("llm_extractor.LLM_FALLBACK_ENABLED", False)
    _install_fake_openai(monkeypatch, raise_error=ConnectionError("primary down"))

    result = llm_extractor.extract_fields_llm("my name is Nithin Jain", ConversationState.AWAIT_IDENTITY)
    assert result is None
