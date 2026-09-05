"""
Guardrail regression tests.

These do not test that the agent makes money. They test that it cannot
misbehave: no discount above the cap, no contact with someone who opted
out, no unsafe message text reaching a customer, and no ground-truth
leakage into the policy.

Run:
    pytest -v
"""
import ast
import pathlib

import pytest

from agent.compose import BANNED, OPT_OUT, validate
from agent.policy import INCENTIVE_CAP, MAX_TOUCHES, decide
from data.taxonomy import CAUSES

REPO = pathlib.Path(__file__).resolve().parent.parent


def make_event(amount_paise=249900, opted_out=False, lang="hi-IN", hour=14):
    """A minimal event. Deliberately carries NO true_cause key -- see
    test_policy_never_reads_true_cause."""
    return {
        "event_id": "evt_test",
        "amount_paise": amount_paise,
        "timestamp": f"2026-01-01T{hour:02d}:30:00+05:30",
        "bank_healthy_after_min": 25,
        "customer": {"lang_signal": lang, "opted_out": opted_out,
                     "prior_recoveries": 0},
    }


# ---------------------------------------------------------------------------
# 1. Incentives are capped, always.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cause", CAUSES)
@pytest.mark.parametrize("amount_paise", [4999, 5000, 49900, 249900, 99999900])
@pytest.mark.parametrize("touch", range(1, MAX_TOUCHES + 1))
def test_incentive_never_exceeds_cap(cause, amount_paise, touch):
    d = decide(make_event(amount_paise=amount_paise), cause, touch_no=touch)
    assert 0.0 <= d["incentive_pct"] <= INCENTIVE_CAP, (
        f"{cause} @ Rs {amount_paise/100} touch {touch} "
        f"returned incentive {d['incentive_pct']}")


def test_unknown_cause_gets_no_incentive_and_refuses():
    d = decide(make_event(), "some_cause_we_never_trained_on")
    assert d["intervention"] == "none"
    assert d["incentive_pct"] == 0.0
    assert d["rule"] == "unknown_cause:refuse"


# ---------------------------------------------------------------------------
# 2. Opted-out customers are never contacted.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cause", CAUSES)
@pytest.mark.parametrize("touch", range(1, MAX_TOUCHES + 1))
def test_opted_out_customer_is_never_contacted(cause, touch):
    d = decide(make_event(opted_out=True), cause, touch_no=touch)
    assert d["intervention"] == "none", f"{cause} contacted an opted-out customer"
    assert d["rule"] == "stop:customer_opted_out"
    assert d["incentive_pct"] == 0.0
    assert d["msg_lang"] is None


@pytest.mark.parametrize("cause", CAUSES)
def test_touch_cap_is_enforced(cause):
    d = decide(make_event(), cause, touch_no=MAX_TOUCHES + 1)
    assert d["intervention"] == "none"
    assert d["rule"] == "stop:touch_cap_reached"


@pytest.mark.parametrize("cause", CAUSES)
def test_tiny_orders_are_not_worth_contacting(cause):
    d = decide(make_event(amount_paise=4999), cause)
    assert d["intervention"] == "none"
    assert d["rule"] == "stop:below_economic_threshold"


# ---------------------------------------------------------------------------
# 3. validate() rejects each banned category.
# ---------------------------------------------------------------------------
NO_INCENTIVE = {"incentive_pct": 0.0}


@pytest.mark.parametrize("word", BANNED)
def test_validate_rejects_every_coercive_word(word):
    msg = f"Aapka payment {word} hai. {OPT_OUT}"
    ok, why = validate(msg, NO_INCENTIVE)
    assert not ok and why.startswith("coercive_language:"), f"{word!r} slipped through"


@pytest.mark.parametrize("msg", [
    "Get 20% off today! " + OPT_OUT,
    "Flat 10 % discount for you. " + OPT_OUT,
    "Yeh raha aapka discount. " + OPT_OUT,
])
def test_validate_rejects_unauthorised_discount(msg):
    ok, why = validate(msg, NO_INCENTIVE)
    assert not ok and why == "discount_not_authorised"


def test_validate_rejects_missing_opt_out():
    ok, why = validate("Please pay soon.", NO_INCENTIVE)
    assert not ok and why == "missing_opt_out"


def test_validate_rejects_wrong_incentive_amount():
    """Policy authorised 5%; the LLM offered something else."""
    ok, why = validate(f"Yahan 20% off milega. {OPT_OUT}", {"incentive_pct": 0.05})
    assert not ok and why == "incentive_mismatch"


def test_validate_rejects_overlong_message():
    ok, why = validate("a" * 400 + OPT_OUT, NO_INCENTIVE)
    assert not ok and why == "too_long"


def test_validate_accepts_a_clean_message():
    ok, why = validate(
        f"Namaste! Aapka payment complete nahi ho paaya. {OPT_OUT}", NO_INCENTIVE)
    assert ok and why == "ok"


# ---------------------------------------------------------------------------
# 4. The policy never receives or reads true_cause.
# ---------------------------------------------------------------------------
def test_policy_works_on_events_with_no_true_cause_key():
    """make_event() omits true_cause entirely. If decide() read it, this
    would KeyError for every cause."""
    ev = make_event()
    assert "true_cause" not in ev
    for cause in CAUSES:
        d = decide(ev, cause)
        assert d["rule"], f"{cause} produced no rule"


def test_policy_source_contains_no_true_cause_reference():
    """Docstrings may mention it; executable code may not."""
    src = (REPO / "agent" / "policy.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            continue          # string literals / docstrings are fine
        if isinstance(node, ast.Attribute) and node.attr == "true_cause":
            pytest.fail("policy.py reads .true_cause")
        if isinstance(node, ast.Name) and node.id == "true_cause":
            pytest.fail("policy.py references a true_cause variable")


def test_experiment_passes_predicted_cause_not_truth():
    """The agent arm must call decide() with the classifier's prediction."""
    src = (REPO / "eval" / "run_experiment.py").read_text()
    assert 'decide(e, preds[e["event_id"]], touch)' in src, (
        "the agent arm no longer runs on predicted causes")
    assert 'decide(e, e["true_cause"]' not in src


def test_identical_customers_get_identical_offers():
    """The determinism claim the whole design rests on."""
    for cause in CAUSES:
        a = decide(make_event(), cause)
        b = decide(make_event(), cause)
        assert a == b, f"{cause} produced non-deterministic decisions"
