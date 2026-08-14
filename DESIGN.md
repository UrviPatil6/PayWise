# Design Document

*The detailed bug-by-bug history, live-testing findings, and extended architecture/evaluation
notes live in [FINDINGS.md](FINDINGS.md). This document is the 1-2 page summary the assignment
asks for.*

## Architecture overview

```
user_input → agent._extract() → agent._merge_extracted() → state-chain handlers → agent.naturalize() → {"message": ...}
              (LLM, regex fallback)   (into Session)         (validate/verify/call     (LLM rewords the
                                                               API/advance)              template)
```

`agent.py` (orchestration + state machine) is the only file that makes a decision. `state.py` holds session memory. Understanding is split across `llm_extractor.py` (LLM, primary) and `extractor.py` (regex, fallback); phrasing across `responder.py` (LLM, primary) and the deterministic template text (fallback). `validators.py`, `verification.py`, and `tools.py` are pure/IO-isolated support modules. No agent framework, no ORM — a single in-process `Agent` holding one `Session` is the entire runtime.

A turn runs through a plain sequential handler chain: `_handle_account_id → _handle_identity → _handle_amount → _handle_card_details`. Each handler returns `None` if it advanced the state (letting the next handler run in the same turn) or a message if it needs to stop and ask something — this is what lets one compound message ("ACC1001, Nithin Jain, DOB 1990-05-14, pay 300") walk through lookup, verification, and amount acceptance in a single `next()` call.

## Why LLM vs. deterministic

```
LLM #1 - Understanding          Python - Decision                    LLM #2 - Communication
user message                    state, validation, verification,     deterministic decision
  -> structured fields             retry limits, payment auth,          -> natural wording
                                    API calls, security
```

The LLM never sees the middle column and never produces it — it only proposes structured fields on the way in, and rewords an already-decided message on the way out. This is enforced structurally, not by convention: `ExtractedFields` has no `verified` field at all, so no extraction call can set verification status, however it was prompted. Every validation call, retry counter, API call, and the one identity-match comparison are plain Python with zero model involvement.

Both LLM calls degrade gracefully — no key configured, or a call fails, and the agent falls back to the regex parser / deterministic template respectively, with identical business behavior and narrower phrasing. Response phrasing works by *rewording* a message Python already decided, not generating one from an abstract state description; every rewording is checked to still contain every amount/ID/transaction-ID from the original (`_facts_preserved`) or it's discarded. Messages carrying a security-relevant fact that check can't cover (a verification-failure retry count, any closing message) skip the LLM step entirely rather than trusting a wider check to catch it.

## Key decisions

- **Verification is one pure function**, strict `==` only — required by the assignment, and held even when the LLM extraction path tries to "clean up" a lowercase name (`agent._cross_check_with_regex` requires it to appear verbatim in the raw message or falls back to regex).
- **Card/amount data volunteered early is captured but never acted on** until the state machine actually reaches that step — this is what makes "skip verification" attempts structurally impossible, not just discouraged.
- **A cascading `if`-chain, not a state-machine framework** — reads top-to-bottom like the actual conversation; cancellation fit in as a one-line short-circuit.
- **Ambiguous numeric input is never trusted as sensitive card data.** CVV extraction requires an explicit label ("cvv 123", "security code 123") — never a bare or embedded number. This is the direct fix for the most serious bug found in this project (a stray digit in an unrelated message getting submitted as a real CVV) — see FINDINGS.md §2, finding 13.

## Tradeoffs accepted

- Two extraction paths and two phrasing paths to maintain, in exchange for real LLM usage that never makes "deterministic across repeated runs" depend on a network call succeeding.
- In-memory-only session, no persistence — matches the assignment's `Agent()`-per-conversation interface; no crash recovery.
- Cardholder name defaults to the verified account name unless stated otherwise (the API doesn't check it against the account anyway).
- Design rationale — especially bug-fix history — lives in FINDINGS.md, not source comments; source comments note only what's non-obvious, in one line.

## Security

DOB/Aadhaar/pincode are read only for the in-memory `verify_identity()` comparison and never appear in any user-facing message. Card number and CVV are never logged (only a masked suffix); raw card fields are cleared from `Session` immediately after every payment attempt, success or fail. Verification and payment authorization cannot be bypassed by the LLM by construction — no `verified` field exists to hallucinate, and payment handlers are only reachable through a state graph gated by an actual `verify_identity()` return. Both LLM calls receive only a small, explicitly-filtered context built by `agent.py` — never raw `Session`/`Account`.

## Evaluation approach

Two layers: `tests/` (126 pytest unit/integration tests, both APIs mocked, LLM paths force-disabled for determinism) and `evals/` (38 scripted multi-turn conversations against the **real** live payment API, checking final outcome, the `action_log` tool-call sequence, and specific session field values — not just the final message text). `evals/evaluation.py` aggregates these into the metrics the assignment asks for (success rate, verification accuracy, tool-call correctness, invalid-API-call rate, failure-recovery rate, extraction accuracy) plus a per-category breakdown.

Current result: 126/126 and 38/38 passing against the deterministic path (this repo ships with no API key configured, so that's the guaranteed baseline). Re-run with a real LLM key before relying on the LLM-path claim for the newest scenarios — see FINDINGS.md §7 for what's been live-verified historically and why that history predates the current scenario set.

**Where the agent still struggles**: a sufficiently unusual phrasing can fail to extract under the regex-only fallback (the LLM path handles the general case when configured); and there is no fuzz/property-based test coverage searching for *new* instances of finding 13's failure shape (an unrelated message misread during a specific collection state) beyond the two now-fixed and regression-tested cases.

## Future improvements

- Property-based/fuzz testing of `extractor.py` to search for more field-disambiguation bugs of finding 13's shape, rather than relying on enumerated examples.
- Broaden adversarial live-LLM testing — most real bugs found so far came from live use or external audit, not the automated suite.
- A "start a new payment" flow after `CLOSED`, instead of requiring a fresh `Agent()`.
- Structured logging / correlation IDs if this ran as a real service instead of an in-process object.

Full reasoning behind every item above, plus the complete findings log, is in [FINDINGS.md](FINDINGS.md).
