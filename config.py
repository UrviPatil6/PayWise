"""Env-driven settings and logging setup - no business logic here."""

import logging
import os


def _read_dotenv(path: str = ".env") -> None:
    """Minimal KEY=VALUE .env loader (not worth a dependency for this)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_read_dotenv()

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://se-payment-verification-api.service.external.usea2.aws.prodigaltech.com",
).rstrip("/")

# HTTP-level retry (network blips), distinct from the business retry limits below.
HTTP_TIMEOUT_SECONDS = float(os.environ.get("HTTP_TIMEOUT_SECONDS", "10"))
HTTP_MAX_RETRIES = int(os.environ.get("HTTP_MAX_RETRIES", "1"))

# How many wrong attempts before the agent ends the conversation.
MAX_VERIFICATION_ATTEMPTS = int(os.environ.get("MAX_VERIFICATION_ATTEMPTS", "3"))
MAX_ACCOUNT_LOOKUP_ATTEMPTS = int(os.environ.get("MAX_ACCOUNT_LOOKUP_ATTEMPTS", "3"))
MAX_PAYMENT_ATTEMPTS = int(os.environ.get("MAX_PAYMENT_ATTEMPTS", "3"))

# LLM understanding + phrasing (llm_extractor.py, responder.py). Any
# OpenAI-compatible chat-completions endpoint works - only base URL, key,
# and model change per provider. Off unless a key is configured; the agent
# runs fully on the regex/template path either way.
LLM_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
LLM_ENABLED = bool(LLM_API_KEY)
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "8"))
# `reasoning_effort` is rejected outright by standard (non-reasoning) models
# like gpt-4o-mini - set false when LLM_MODEL points at one of those.
LLM_USES_REASONING_EFFORT = os.environ.get("LLM_USES_REASONING_EFFORT", "true").lower() == "true"

# Second provider, tried only if the primary call itself fails, before
# falling back to the deterministic path. Defaults to OpenRouter.
LLM_FALLBACK_API_KEY = (
    os.environ.get("LLM_FALLBACK_API_KEY")
    or os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENROUTER_API_KEY_1")
)
LLM_FALLBACK_BASE_URL = os.environ.get("LLM_FALLBACK_BASE_URL", "https://openrouter.ai/api/v1")
LLM_FALLBACK_MODEL = os.environ.get("LLM_FALLBACK_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
# Not every model honors a *forced* tool choice - set false if yours doesn't.
LLM_FALLBACK_SUPPORTS_FORCED_TOOL_CHOICE = (
    os.environ.get("LLM_FALLBACK_SUPPORTS_FORCED_TOOL_CHOICE", "true").lower() == "true"
)
LLM_FALLBACK_USES_REASONING_EFFORT = (
    os.environ.get("LLM_FALLBACK_USES_REASONING_EFFORT", "true").lower() == "true"
)
LLM_FALLBACK_ENABLED = bool(LLM_FALLBACK_API_KEY)


def get_logger(name: str) -> logging.Logger:
    """Configured logger. Callers must only pass whitelisted, non-sensitive
    fields (state names, account IDs, error codes, masked card suffixes) -
    never raw user input or request/response bodies (see tools.py)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    return logger
