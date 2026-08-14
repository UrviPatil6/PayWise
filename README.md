# Payment Collection Agent

A conversational agent that looks up an account, verifies the caller's identity, shares their outstanding balance, and collects a card payment — built against the provided lookup-account / process-payment API.

See [DESIGN.md](DESIGN.md) for the architecture writeup and design rationale (kept to the assignment's requested 1-2 pages), and [FINDINGS.md](FINDINGS.md) for the detailed bug-by-bug history and extended implementation notes. This file is setup, usage, and reference.

## Project overview

The agent runs a fixed conversation flow — account ID → identity verification → amount → card details → payment — but accepts the messy, conversational way real users actually type ("yeah my account number is ACC1001 I think", "CVV is one two three", "just clear the full amount"), and replies in natural, varied phrasing rather than fixed template sentences. It never re-asks for information already given. Two LLM calls bracket every turn — understanding the incoming message, and phrasing the outgoing one — but every security-sensitive decision in between (identity verification, payment authorization, retry limits, what fact gets communicated) is plain, deterministic Python that never asks a model's opinion.

## Architecture

```
user_input
   │
   ▼
agent._extract()             ← LLM first (llm_extractor.py: system prompt +
   │                             forced structured tool-use, temp 0), regex
   │                             (extractor.py) as the automatic fallback
   │                             when no key is set or the call fails.
   │                             Neither path decides anything - only
   │                             produces candidate fields.
   ▼
agent._merge_extracted()     ← merges into session state (state.py), out-of-order safe
   │
   ▼
agent._handle_*() chain      ← the business state machine:
   │                             AWAIT_ACCOUNT_ID → AWAIT_IDENTITY → AWAIT_AMOUNT
   │                             → AWAIT_CARD_DETAILS → CLOSED
   │                           each step: validate (validators.py) → verify
   │                           (verification.py) / call API (tools.py) → advance
   │                           → a deterministic template message: the exact
   │                             fact(s) and instruction the user needs
   ▼
agent.naturalize() call      ← LLM rephrases that template to sound natural
   │                             (responder.py) - never generates from
   │                             scratch, only reworded; discarded (falls
   │                             back to the template) if a required fact
   │                             goes missing under rephrasing.
   ▼
{"message": "..."}
```

| File | Responsibility |
|---|---|
| `agent.py` | `Agent.next()` — the public interface; turn orchestration and the state-transition chain |
| `state.py` | `ConversationState` enum + `Session` dataclass (all per-conversation memory) |
| `llm_extractor.py` | Primary NL *understanding*: system prompt + tool-call to an OpenAI-compatible LLM (Groq by default), schema-validated, temp 0 |
| `extractor.py` | Deterministic fallback extraction — regex only, used when no LLM is configured or a call fails |
| `responder.py` | Primary *response phrasing*: rewords the deterministic template message, never generates one from scratch |
| `validators.py` | Format/Luhn/date/amount validation — pure functions |
| `verification.py` | The one deterministic identity-match rule |
| `tools.py` | HTTP client for the two APIs: timeouts, retries, typed outcomes |
| `models.py` | Pydantic schemas for the account API response and extracted fields |
| `config.py` | Env-driven settings and logging setup |
| `cli.py` | Interactive terminal demo |
| `server.py` / `web/index.html` | Optional browser chat demo — thin Flask wrapper over `Agent`, static single-file UI |

**The LLM boundary is intentionally narrow: understand the input, phrase the output — never decide anything in between.** `llm_extractor.py` turns one message into a structured `ExtractedFields` object; `responder.py` turns a deterministic message into a rewording of itself. Neither ever sees raw account data or `Session` directly, and neither has any way to call `tools.py` — each is given, at most, a small, explicitly-filtered summary `agent.py` builds specifically to hand over (see the next paragraph), never the real thing. Extracted fields still have to pass `validators.py` and `verification.py` before they can affect anything; rephrased responses still have to pass a fact-preservation check (every amount/account ID/transaction ID in the original must survive verbatim) before they're shown to the user, falling back to the exact deterministic text otherwise. Messages with a security-relevant free-form fact that check doesn't cover - a verification-failure retry count, any conversation-closing message - skip the LLM step entirely (`agent.next()`'s `security_critical` check) rather than widen the check to catch it; see FINDINGS.md §2, finding 12. Identity verification and payment authorization are never delegated to the model, per the assignment's explicit "do not let the LLM decide verification/payment" constraint — see FINDINGS.md §2 for the full reasoning, including the real bugs this design caught and fixed during live testing and an external audit (a reasoning-model token-budget truncation, a prompt-framing issue where the model tried to *answer* the message instead of rewording it, an LLM silently title-casing a lowercase name in a way that could have made strict verification effectively case-insensitive, the `openai` SDK's own internal retries turning a fast fallback into a multi-second hang under rate limiting, a retry count that the original fact-check didn't guard, and - the most serious of all - a stray digit in an unrelated message that could reach a real payment as a fabricated CVV).

**Two LLM providers, not one.** Both `llm_extractor.py` and `responder.py` try a primary provider (Groq) first; if that call itself fails - a real, recurring situation once a free-tier key has taken heavy use, confirmed repeatedly during this project's own testing - a second provider (OpenRouter, configured separately) gets one real attempt before falling all the way back to the deterministic path. This isn't a generic multi-provider abstraction; it's a deliberately narrow "primary, then one fallback" pair, because that's what the actual failure mode (a rate-limited key) called for. Each provider is genuinely tested, not assumed to work the same way: OpenRouter's free routing for one model rejected a *forced* tool choice outright, and a different free model that accepted forced tool choice turned out unreliable with `tool_choice="auto"` on the same input run twice - the model and setting actually shipped (`nvidia/nemotron-3-nano-30b-a3b:free`, forced tool choice) were chosen because they held up under repeated live testing, not because they were the first thing tried.

**A small, cross-cutting intent classification, plus session context and phrasing tone — still never a decision the LLM makes.** Alongside the usual fields, both extractors can report one `intent`: `"cancel"`, `"small_talk"`, or `"balance_query"` (`models.ExtractedFields.intent`). Only `"cancel"` is acted on directly, and even then via a single unconditional Python check (`agent._handle_cancel()`) — the model proposes a classification, Python decides whether to end the conversation, exactly the same split as every other field. `llm_extractor.py` is also given a small, explicitly-filtered view of the current session (`agent._llm_session_context()`: account ID, claimed name, verified flag, claimed amount, and boolean *presence* of card fields — never DOB, Aadhaar, pincode, or raw card values) so it can resolve a short or ambiguous reply against what's already known. `responder.py` gets the phrasing-side equivalent (`agent._response_checklist()`): a short, plain-English "what we know so far" list (e.g. "Expiry provided.", "Card number still needed.") in the same non-sensitive scope, so a rewording can say "Thanks — I have the expiry and CVV, I just need your card number" instead of a generic re-ask, without ever being told to *decide* what's missing (that decision already happened in the handler chain). It also accepts an optional `tone_hint` — a short, deterministically-computed note (a correction just happened, small talk should be acknowledged, this is a verification failure) that only ever nudges *phrasing*, still checked by the same fact-preservation rule as every other rephrasing. See FINDINGS.md §2, findings 10-11 — including a real bug findings 11 caught: a correction `tone_hint` that could describe an account switch that never actually happened, fixed by scoping the check to when the switch is still possible.

## Setup instructions

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

No API key or account setup is required to run the agent — the payment-verification API is unauthenticated, and without `GROQ_API_KEY` the agent runs on the regex extractor for every turn (fully functional, just narrower phrasing coverage than with the LLM enabled).

## Environment variables

All optional; the agent works out of the box with defaults. See `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `API_BASE_URL` | the provided service URL | override for testing against a different environment |
| `HTTP_TIMEOUT_SECONDS` | `10` | per-request timeout |
| `HTTP_MAX_RETRIES` | `1` | automatic retries on connection/timeout errors |
| `MAX_VERIFICATION_ATTEMPTS` | `3` | identity verification retry limit |
| `MAX_ACCOUNT_LOOKUP_ATTEMPTS` | `3` | account-not-found retry limit |
| `MAX_PAYMENT_ATTEMPTS` | `3` | payment failure retry limit |
| `LOG_LEVEL` | `INFO` | logging verbosity |
| `GROQ_API_KEY` | unset | primary LLM provider for both `llm_extractor.py` (understanding) and `responder.py` (phrasing); unset means deterministic-only (regex extraction + fixed template responses) |
| `LLM_MODEL` | `openai/gpt-oss-20b` | model used for the primary provider's calls |
| `LLM_BASE_URL` | Groq's endpoint | override to point the primary provider at a different OpenAI-compatible endpoint; use with `LLM_API_KEY` instead of `GROQ_API_KEY` |
| `OPENROUTER_API_KEY` | unset | optional **second** provider, tried only if the primary call itself fails (rate limit, network error, timeout) - before giving up on the LLM layer entirely |
| `LLM_FALLBACK_MODEL` | `nvidia/nemotron-3-nano-30b-a3b:free` | model used for the fallback provider's calls |
| `LLM_FALLBACK_SUPPORTS_FORCED_TOOL_CHOICE` | `true` | whether the fallback model/provider can be forced to call the extraction tool; set `false` if yours rejects a forced tool choice (tested per-model, not assumed - see FINDINGS.md §2, finding 8) |
| `LLM_API_KEY` | unset | primary provider key, alternative to `GROQ_API_KEY` - use with `LLM_BASE_URL`/`LLM_MODEL` to point at a different OpenAI-compatible provider (e.g. OpenAI itself) |
| `LLM_TIMEOUT_SECONDS` | `8` | per-call timeout for both LLM calls |
| `LLM_USES_REASONING_EFFORT` | `true` | whether to send `reasoning_effort="low"` to the primary model; set `false` for a standard (non-reasoning) model, which rejects this parameter outright |
| `LLM_FALLBACK_API_KEY` | unset | fallback provider key, alternative to `OPENROUTER_API_KEY`/`OPENROUTER_API_KEY_1` |
| `LLM_FALLBACK_BASE_URL` | OpenRouter's endpoint | override to point the fallback provider at a different OpenAI-compatible endpoint |
| `LLM_FALLBACK_USES_REASONING_EFFORT` | `true` | same as `LLM_USES_REASONING_EFFORT`, for the fallback provider - kept separate since primary and fallback can be different kinds of model |

## How to run

Interactively, from the terminal:

```bash
python cli.py
```

Or programmatically, via the required interface:

```python
from agent import Agent

agent = Agent()
print(agent.next("Hi")["message"])
print(agent.next("My account ID is ACC1001")["message"])
```

### Web UI (optional demo)

A minimal browser chat UI is included — `server.py` (Flask) wraps `Agent` behind two endpoints (`POST /api/session` creates one, `POST /api/chat` drives it) and serves the static page in `web/`:

```bash
python server.py
```

Then open `http://127.0.0.1:5000`. This is a thin HTTP front door onto the exact same `Agent.next()` call the CLI and the grading harness use — not a separate interface. One `Agent` per browser session, in-memory only, same lifetime/scope caveats as the CLI (see "Known limitations"). No auth, no persistence — it's a demo surface, not a production deployment.

## How to run tests

Requires `pip install -r requirements.txt` first (see Setup instructions above) - `tests/test_llm_extractor.py` and `tests/test_responder.py` import the `openai` package to fake its client, even though no real LLM call is ever made.

```bash
python -m pytest tests/ -v
```

126 tests, fully offline — the two APIs are mocked (see `tests/conftest.py`), and both LLM paths (extraction, response phrasing) are force-disabled by an autouse fixture so the suite behaves identically whether or not `GROQ_API_KEY` happens to be set in the environment it runs in. `llm_extractor.py` and `responder.py` are each tested separately with a faked OpenAI-compatible client (`tests/test_llm_extractor.py`, `tests/test_responder.py`) covering a valid response and every failure mode degrading to `None` — for the responder specifically, that includes a rephrase that drops a required amount or transaction ID, which must be rejected. Covers validators, verification, extraction, response phrasing, and full multi-turn agent flows.

## How to run evaluations

```bash
python evals/evaluation.py
```

This runs 38 scripted conversations **against the real live payment API** (not mocked) and checks not just the final message but the actual sequence of actions taken (`session.action_log`), final conversation state, and specific session field values. Prints a report and writes `evals/results.json`. See "Evaluation approach" in DESIGN.md for what each metric means, and FINDINGS.md §7 for the detailed historical results.

Extraction and response phrasing within the eval run through whatever `agent._extract()` / `agent.next()`'s naturalization step would normally use — regex + fixed templates if `GROQ_API_KEY` isn't set, LLM-first (with fallback) for both if it is. All 38 scenarios pass against the deterministic path (confirmed with no key configured); re-run with a real key before relying on the LLM-path claim, since the newest scenarios (added alongside a security fix - see FINDINGS.md §2, finding 13) haven't yet been re-verified against a live model in this environment. `session.action_log` entries like `extract:llm`, `extract:llm_failed_fallback_to_rules`, `respond:naturalized`, `respond:template_fallback` show which path handled each turn when you do.

## Example conversations

Card numbers are replaced with `[card number]` below — see "Security considerations". Full transcripts (with the test card number that was actually used) are reproducible by running the commands above; nothing here is a real card.

These use the exact deterministic template wording (no `GROQ_API_KEY` set) since that's the guaranteed, reproducible baseline. With a key configured, the same flow gets naturally-phrased responses instead - see "With response phrasing" below for a real captured example of the same flow.

### Successful payment

```
User: Hi
Agent: Hello! Please share your account ID to get started.
User: My account ID is ACC1001
Agent: Got it. Could you please confirm your full name?
User: Nithin Jain
Agent: Thanks. Could you also verify your date of birth, Aadhaar last 4 digits, or pincode?
User: DOB is 1990-05-14
Agent: Identity verified. Your outstanding balance is ₹1,250.75. How much would you like to pay today?
User: I want to pay 500
Agent: To charge ₹500.00, please share your card number, expiry (MM/YY), CVV.
User: card number [card number], expiry 12/27, cvv [cvv]
Agent: Payment successful! ₹500.00 was charged (transaction ID txn_1786627228482_qj9xmde). Thanks, Nithin Jain - have a great day!
```

### Verification failure → retry exhaustion

```
User: ACC1001
Agent: Hello! Got it. Could you please confirm your full name?
User: Nithin Jain
Agent: Thanks. Could you also verify your date of birth, Aadhaar last 4 digits, or pincode?
User: DOB is 1999-01-01
Agent: Those details don't match our records. You have 2 attempt(s) left - could you confirm your full name, and your date of birth, Aadhaar last 4 digits, or pincode?
User: Nithin Jain
Agent: Thanks. Could you also verify your date of birth, Aadhaar last 4 digits, or pincode?
User: DOB is 1999-01-02
Agent: Those details don't match our records. You have 1 attempt(s) left - could you confirm your full name, and your date of birth, Aadhaar last 4 digits, or pincode?
User: Nithin Jain
Agent: Thanks. Could you also verify your date of birth, Aadhaar last 4 digits, or pincode?
User: DOB is 1999-01-03
Agent: I'm unable to verify your identity after multiple attempts, so I have to end this session here for security reasons. Please contact support if you need help.
```

Notice the response never states which field was wrong or what the correct value is — only that "those details don't match."

### Payment failure → recovery

```
User: ACC1001
Agent: Hello! Got it. Could you please confirm your full name?
User: Nithin Jain
Agent: Thanks. Could you also verify your date of birth, Aadhaar last 4 digits, or pincode?
User: DOB is 1990-05-14
Agent: Identity verified. Your outstanding balance is ₹1,250.75. How much would you like to pay today?
User: pay 200
Agent: To charge ₹200.00, please share your card number, expiry (MM/YY), CVV.
User: card number [invalid card number], expiry 12/27, cvv [cvv]
Agent: That card number failed validation. Please double-check it. Could you re-enter your card number?
User: card number [card number], expiry 12/27, cvv [cvv]
Agent: Payment successful! ₹200.00 was charged (transaction ID txn_1786627231882_tb2knfi). Thanks, Nithin Jain - have a great day!
```

The invalid card number failed a local Luhn check, so it was rejected before any API call was made (see `session.action_log` — `process_payment` only appears once, after the corrected card).

### Edge case: leap-year DOB (ACC1004)

```
User: ACC1004
Agent: Hello! Got it. Could you please confirm your full name?
User: Rahul Mehta
Agent: Thanks. Could you also verify your date of birth, Aadhaar last 4 digits, or pincode?
User: DOB is 1990-02-29
Agent: 1990-02-29 isn't a valid calendar date. Could you provide your date of birth again (YYYY-MM-DD)?
User: DOB is 1988-02-29
Agent: Identity verified. Your outstanding balance is ₹3,200.50. How much would you like to pay today?
```

ACC1004's real DOB, `1988-02-29`, is a valid leap-year date. `1990-02-29` isn't a leap year, so that date doesn't exist on the calendar at all — the agent catches this with `datetime.date()` and gives a specific "not a valid date" response, **without** counting it as a failed verification attempt (it's a format problem, not a wrong guess). A date that's simply wrong but still real, e.g. `1988-03-01`, is handled differently: it fails the identity match and does count as an attempt.

### With response phrasing (`GROQ_API_KEY` set)

A real captured run of the same happy-path flow, with the response-phrasing layer active. The underlying decisions - what's asked for, when verification happens, what gets charged - are identical to the deterministic version above; only the wording differs, and it varies between runs since it's genuinely generated, not picked from a list:

```
User: Hi
Agent: Hi there! Could you please share your account ID so we can get started?
User: yeah my account number is ACC1001 I think
Agent: Sure thing. Could you confirm your full name?
User: urvi
Agent: Thanks! Please provide your full name (first and last) exactly as it appears on the account.
User: Nithin Jain
Agent: Thanks. Could you also verify your date of birth, Aadhaar last 4 digits, or pincode?
User: 14 May 1990
Agent: Identity verified. Your outstanding balance is ₹1,250.75. How much would you like to pay today?
User: can I do 500 for now?
Agent: To process the ₹500.00 charge, could you please provide your card number, expiry (MM/YY), and CVV?
User: card number [card number], expiry 12/27, cvv [cvv]
Agent: Payment successful! ₹500.00 was charged (transaction ID txn_1786653671091_90mg3l0). Thanks, Nithin Jain – have a great day!
```

Two things worth noting in this transcript: `urvi` is a single word, so the agent explicitly asks for the full name rather than silently attempting (and failing) verification with just a first name - a deterministic workflow decision (`agent.py:_handle_identity`), not something the LLM decided. And every number that matters (₹1,250.75, ₹500.00, the transaction ID) is identical to what the deterministic template would have said - that's the fact-preservation check in `responder.py` at work, not luck.

## Security considerations

- **Card number and CVV are never logged.** `tools.py` only ever logs whitelisted fields (account ID, HTTP status, API error code, and the card number masked to its last 4 digits via `validators.mask_card_number`) — never the request/response body or raw user input.
- **Raw card data is cleared from memory** immediately after each payment attempt (`Session.clear_card_details()`), win or lose — it's only needed for the single API call that used it.
- **Account PII (DOB, Aadhaar, pincode) is never sent back to the user.** It's used only for the in-memory `verify_identity()` comparison in `verification.py`. A failed verification attempt gets a generic "those details don't match" — never which field was wrong — so the response can't be used to narrow down a guess.
- **Verification is exact-match Python, not an LLM judgment call.** No case-folding, no fuzzy matching. `llm_extractor.py` can propose that a message *contains* the string "Nithin Jain" as a `full_name` candidate; it never gets a vote on whether that candidate constitutes a verified identity. See DESIGN.md.
- **The only real secret in this project is the LLM API key (`GROQ_API_KEY`)**, and it's optional — the agent is fully functional without it. It's read from the environment, never hardcoded, and `.env.example` ships with it commented out.
- **Card data never appears in this README** or in committed test fixtures — examples above use `[card number]` / `[cvv]` placeholders.

## Known limitations

- **Without a `GROQ_API_KEY`, extraction falls back to `extractor.py`'s regex parser**, which covers every phrasing pattern in the assignment brief (and the extra ones the eval suite and an external audit turned up) but isn't a general-purpose parser — a sufficiently unusual phrasing could fail to extract in this mode. With a key configured, `llm_extractor.py` handles the general case and the regex path only serves as the fallback for when a call fails. See FINDINGS.md §2 for the full bug-by-bug history (13 real bugs found and fixed via live testing and an external audit, not assumed correct) - most notably finding 13: a regex fallback that could let an unrelated message's stray digits reach a real payment as a fabricated CVV, the most serious bug found in this project.
- **The live API's `insufficient_balance` and `invalid_card`/`invalid_cvv`/`invalid_expiry` error codes are effectively unreachable in normal operation**, because local validation (Luhn check, expiry check, `amount <= balance`) always catches these cases first and never calls the API with bad data. This is intentional defense-in-depth, but it means the eval suite can only exercise the API's own error-code handling for these paths via a mocked unit test (`tests/test_payment.py`), not the live eval.
- **No persistence.** Session state lives only in the `Agent` instance's memory for the life of the process — there's no database, so a new `Agent()` starts a fresh conversation. The assignment's interface doesn't ask for cross-session persistence.
- **One payment per conversation.** Once a conversation reaches `CLOSED` (success or terminal failure), it doesn't offer to start a new payment — it just repeats the closing message. Starting over means creating a new `Agent()`.
- **Cardholder name defaults to the verified account name** if the user doesn't explicitly state a different one during the card step, rather than asking a redundant question, since the API doesn't validate this field against the account anyway.

## Design decisions

Summarized here; full reasoning in DESIGN.md.

1. LLM understanding and LLM phrasing are both primary-when-configured, with fully-capable deterministic fallbacks — not either/or, for either. This keeps the agent deterministic and gradeable without a key, while still using real model calls with real system prompts for both NL-understanding and response-generation.
2. Verification and payment authorization are 100% Python, never inferred by a model, regardless of which extractor produced the fields or how the response ends up phrased.
3. Response phrasing is *rewording*, not *generation from scratch* — the LLM is given the exact deterministic message and told to reword it, then the reworded text is checked to still contain every required fact before it's trusted. This is a smaller, safer task than "decide what to say," and catches a rephrase that silently drops a number.
4. Card/amount data volunteered early is captured but never acted on until the state machine actually reaches that step — this is what makes "bypass verification" attempts structurally impossible rather than merely discouraged.
5. Each handler in the state chain returns `None` (state advanced, let the next step in the same turn run too) or a message string (stop here) — this is how one message with several fields at once can cascade through multiple states in a single `next()` call without needing a separate multi-message reply.
6. A single-word name claim (e.g. "urvi") prompts once for the full name rather than silently being used as a verification attempt certain to fail — a deterministic workflow decision, not something left to the LLM to notice or not.

## Future improvements

- A "start a new payment" flow after `CLOSED`, instead of requiring a fresh `Agent()`.
- Configurable/localized currency formatting (currently hardcoded to ₹).
- Structured (JSON) logging instead of plain text, for easier log aggregation in a real deployment.
- Broader adversarial testing of the response-phrasing prompt - two real bugs (reasoning-token truncation, the model trying to answer a template instead of rewording it) were found by live testing a handful of messages; a wider adversarial sweep would likely find more edge cases in less common message types.
