"""
Automated evaluation suite. Unlike tests/, which mocks the two APIs for
fast, offline, deterministic unit/integration testing, this runs every
scenario against the real live API - it's the "does this actually work
end to end" check.

Each scenario in test_cases.json is a scripted conversation plus a small
set of declarative expectations (final outcome, expected action-log
subsequence, session field values, required/forbidden substrings in the
agent's replies). This runner plays the script through a real Agent
instance and checks the expectations, then prints a report with the
metrics the assignment asks for.

Run: python evals/evaluation.py
"""

import json
import os
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools  # noqa: E402
from agent import Agent  # noqa: E402
from state import ConversationState  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"


def _get_dotted(obj, dotted_path):
    for part in dotted_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _classify_outcome(session) -> str:
    if session.state != ConversationState.CLOSED:
        return "still_open"
    if session.transaction_id:
        return "payment_success"
    return "closed_locked_out"


def _is_ordered_subsequence(expected, actual) -> bool:
    it = iter(actual)
    return all(item in it for item in expected)


def run_case(case: dict) -> dict:
    """Runs one scripted conversation and checks all declared expectations.
    Returns a result dict with pass/fail per check plus the raw transcript,
    so a failure is easy to diagnose without re-running anything.
    """
    original_base_url = tools.API_BASE_URL
    if case.get("simulate_network_failure"):
        # Point at a reserved, unroutable address so the real requests
        # library hits a genuine connection failure - this exercises
        # tools.py's actual retry/timeout handling rather than a mocked one.
        tools.API_BASE_URL = "http://10.255.255.1"

    checks = {}
    transcript = []
    try:
        agent = Agent()
        for turn in case["turns"]:
            transcript.append(agent.next(turn)["message"])
        session = agent.session

        if "expect_outcome" in case:
            actual = _classify_outcome(session)
            checks["outcome"] = (actual == case["expect_outcome"], f"expected {case['expect_outcome']}, got {actual}")

        if "expect_action_log_subsequence" in case:
            ok = _is_ordered_subsequence(case["expect_action_log_subsequence"], session.action_log)
            checks["action_log_subsequence"] = (ok, f"expected subsequence {case['expect_action_log_subsequence']} in {session.action_log}")

        if "expect_no_actions_of_type" in case:
            forbidden = case["expect_no_actions_of_type"]
            violations = [a for a in session.action_log if any(a.startswith(f"{t}:") for t in forbidden)]
            checks["no_forbidden_actions"] = (not violations, f"unexpected actions logged: {violations}")

        if "expect_session_fields" in case:
            for field, expected_value in case["expect_session_fields"].items():
                actual_value = _get_dotted(session, field)
                if isinstance(expected_value, str) and isinstance(actual_value, Decimal):
                    ok = str(actual_value) == expected_value
                else:
                    ok = actual_value == expected_value
                checks[f"field:{field}"] = (ok, f"expected {field}={expected_value!r}, got {actual_value!r}")

        final_message = transcript[-1] if transcript else ""
        if "required_substring_final" in case:
            needle = case["required_substring_final"].lower()
            ok = needle in final_message.lower()
            checks["required_substring_final"] = (ok, f"expected {needle!r} in final message: {final_message!r}")

        if "forbidden_substrings" in case:
            full_text = " ".join(transcript)
            leaked = [s for s in case["forbidden_substrings"] if s in full_text]
            checks["no_forbidden_substrings"] = (not leaked, f"leaked sensitive values: {leaked}")

        error = None
    except Exception as exc:  # a scenario crashing the agent is itself a finding worth reporting
        error = f"{type(exc).__name__}: {exc}"

    finally:
        tools.API_BASE_URL = original_base_url

    passed = error is None and all(ok for ok, _ in checks.values())
    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "category": case.get("category", []),
        "passed": passed,
        "error": error,
        "checks": checks,
        "transcript": transcript,
    }


def compute_metrics(results: list) -> dict:
    total = len(results)
    end_to_end_success_rate = sum(r["passed"] for r in results) / total

    def category_pass_rate(cat):
        in_cat = [r for r in results if cat in r["category"]]
        return sum(r["passed"] for r in in_cat) / len(in_cat) if in_cat else None

    verification_results = [r for r in results if "verification" in r["category"]]
    verification_accuracy = (
        sum(1 for r in verification_results if all(ok for k, (ok, _) in r["checks"].items() if k.startswith("field:verified") or k == "outcome"))
        / len(verification_results)
        if verification_results
        else None
    )

    tool_call_results = [r for r in results if "action_log_subsequence" in r["checks"]]
    correct_tool_call_rate = (
        sum(1 for r in tool_call_results if r["checks"]["action_log_subsequence"][0]) / len(tool_call_results)
        if tool_call_results
        else None
    )

    forbidden_action_results = [r for r in results if "no_forbidden_actions" in r["checks"]]
    invalid_api_call_rate = (
        sum(1 for r in forbidden_action_results if not r["checks"]["no_forbidden_actions"][0]) / len(forbidden_action_results)
        if forbidden_action_results
        else None
    )

    outcome_results = [r for r in results if "outcome" in r["checks"]]
    state_transition_correctness = (
        sum(1 for r in outcome_results if r["checks"]["outcome"][0]) / len(outcome_results) if outcome_results else None
    )

    failure_results = [r for r in results if "failure_handling" in r["category"]]
    failure_recovery_rate = sum(r["passed"] for r in failure_results) / len(failure_results) if failure_results else None

    field_checks = [(ok, name) for r in results for name, (ok, _) in r["checks"].items() if name.startswith("field:")]
    required_field_extraction_accuracy = (
        sum(1 for ok, _ in field_checks if ok) / len(field_checks) if field_checks else None
    )

    return {
        "end_to_end_success_rate": end_to_end_success_rate,
        "verification_accuracy": verification_accuracy,
        "correct_tool_call_rate": correct_tool_call_rate,
        "invalid_api_call_rate": invalid_api_call_rate,
        "state_transition_correctness": state_transition_correctness,
        "failure_recovery_rate": failure_recovery_rate,
        "required_field_extraction_accuracy": required_field_extraction_accuracy,
        "category_breakdown": {
            cat: category_pass_rate(cat)
            for cat in sorted({c for r in results for c in r["category"]})
        },
    }


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results = [run_case(case) for case in cases]
    metrics = compute_metrics(results)

    print(f"\n{'=' * 60}\nEVALUATION RESULTS ({len(results)} scenarios)\n{'=' * 60}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']}")
        if not r["passed"]:
            if r["error"]:
                print(f"        error: {r['error']}")
            for name, (ok, detail) in r["checks"].items():
                if not ok:
                    print(f"        {name}: {detail}")

    print(f"\n{'-' * 60}\nMETRICS\n{'-' * 60}")

    def fmt(value):
        return "n/a" if value is None else f"{value:.0%}"

    print(f"End-to-end success rate:              {fmt(metrics['end_to_end_success_rate'])}")
    print(f"Verification accuracy:                {fmt(metrics['verification_accuracy'])}")
    print(f"Correct tool/API call sequence rate:  {fmt(metrics['correct_tool_call_rate'])}")
    print(f"Invalid API call rate:                {fmt(metrics['invalid_api_call_rate'])}")
    print(f"State transition correctness:         {fmt(metrics['state_transition_correctness'])}")
    print(f"Failure recovery rate:                {fmt(metrics['failure_recovery_rate'])}")
    print(f"Required-field extraction accuracy:   {fmt(metrics['required_field_extraction_accuracy'])}")
    print("\nBy category:")
    for cat, rate in metrics["category_breakdown"].items():
        print(f"  {cat:20s} {fmt(rate)}")

    output_path = Path(__file__).resolve().parent / "results.json"
    output_path.write_text(json.dumps({"results": results, "metrics": metrics}, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
