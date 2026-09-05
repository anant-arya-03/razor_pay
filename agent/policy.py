"""
Policy engine + baseline agent.

THE RULE THAT MATTERS: money decisions are deterministic and auditable.
The LLM is not in this file and never will be. It only writes message text
in agent/compose.py. If an LLM picked who gets a discount, two identical
customers could get different offers, and a payments company will treat
that as a hard fail.

Incentives are capped, contact is capped, and every decision returns the
rule that produced it so the audit log can explain itself.
"""

MAX_TOUCHES = 3
BASELINE_TOUCHES = 3          # control gets the SAME budget. fair fight.
INCENTIVE_CAP = 0.05          # never exceed 5% of order value
INCENTIVE_MIN_ORDER = 500.0   # no discount on tiny orders
QUIET_START, QUIET_END = 21, 9   # IST: no contact 21:00 -> 09:00

LANG_MAP = {"hi-IN": "hinglish", "hi-en": "hinglish", "en-IN": "en"}


def _hour(event):
    return int(event["timestamp"][11:13])


def stopping_check(event, touch_no):
    """Returns (allowed, reason). Guardrails run BEFORE the policy."""
    if event["customer"]["opted_out"]:
        return False, "customer_opted_out"
    if touch_no > MAX_TOUCHES:
        return False, "touch_cap_reached"
    if event["amount_paise"] < 5000:            # < Rs 50, not worth contacting
        return False, "below_economic_threshold"
    return True, "ok"


def decide(event, predicted_cause, touch_no=1):
    """
    Deterministic mapping: predicted cause -> bounded action.
    Never reads true_cause. Returns an auditable decision record.
    """
    allowed, why = stopping_check(event, touch_no)
    if not allowed:
        return {"intervention": "none", "rule": f"stop:{why}",
                "timing_correct": False, "msg_lang": None,
                "incentive_pct": 0.0, "touch_no": touch_no,
                "delay_min": 0, "escalate": False}

    amount = event["amount_paise"] / 100
    lang = LANG_MAP.get(event["customer"]["lang_signal"], "en")
    in_quiet = _hour(event) >= QUIET_START or _hour(event) < QUIET_END

    d = {"msg_lang": lang, "incentive_pct": 0.0, "touch_no": touch_no,
         "escalate": False}

    if predicted_cause == "bank_downtime":
        wait = event.get("bank_healthy_after_min") or 30
        d |= {"intervention": "silent_retry", "rule": "downtime:wait_for_bank",
              "delay_min": wait + 10, "timing_correct": True}

    elif predicted_cause == "insufficient_funds":
        # wait for the salary cycle -- this is the timing bet
        d |= {"intervention": "nudge_delayed", "rule": "low_balance:salary_cycle",
              "delay_min": 60 * 24 * 2, "timing_correct": True}

    elif predicted_cause == "auth_drop_otp":
        # intent was high and is decaying -- go now unless it is quiet hours
        d |= {"intervention": "nudge_now" if not in_quiet else "nudge_delayed",
              "rule": "otp_drop:strike_while_warm",
              "delay_min": 0 if not in_quiet else 60 * 10,
              "timing_correct": True}

    elif predicted_cause == "hard_decline":
        d |= {"intervention": "new_instrument_link",
              "rule": "hard_decline:request_new_method",
              "delay_min": 15, "timing_correct": True}

    elif predicted_cause == "checkout_abandoned":
        if amount >= INCENTIVE_MIN_ORDER:
            d |= {"intervention": "incentive_link",
                  "rule": "abandoned:capped_incentive",
                  "incentive_pct": INCENTIVE_CAP, "delay_min": 90,
                  "timing_correct": True}
        else:
            d |= {"intervention": "nudge_delayed",
                  "rule": "abandoned:nudge_no_incentive",
                  "delay_min": 90, "timing_correct": True}

    elif predicted_cause == "mandate_failure":
        d |= {"intervention": "new_instrument_link",
              "rule": "mandate:re_authorize",
              "delay_min": 30, "timing_correct": True,
              "escalate": touch_no >= 2}

    else:
        d |= {"intervention": "none", "rule": "unknown_cause:refuse",
              "delay_min": 0, "timing_correct": False}

    assert d["incentive_pct"] <= INCENTIVE_CAP, "incentive cap breached"
    return d


def baseline_decide(event, touch_no=1):
    """
    The control arm. NOT a strawman -- it is competent current practice:
    three generic English reminders at 0h / 24h / 72h, same for every
    customer, ignoring why the payment actually failed.

    Give the control the SAME touch budget as the agent. If you hand the
    agent 3 messages and the baseline 1, your lift is just message count
    and a panel will say so in the first minute.
    """
    if touch_no > BASELINE_TOUCHES or event["customer"]["opted_out"]:
        return {"intervention": "none", "rule": "baseline:single_touch",
                "timing_correct": False, "msg_lang": None,
                "incentive_pct": 0.0, "touch_no": touch_no,
                "delay_min": 0, "escalate": False}
    return {"intervention": "nudge_now", "rule": "baseline:generic_retry",
            "timing_correct": False, "msg_lang": "en", "incentive_pct": 0.0,
            "touch_no": touch_no, "delay_min": 0, "escalate": False}
