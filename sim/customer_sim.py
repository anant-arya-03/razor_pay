"""
Customer response simulator.

READ THIS BEFORE YOU DEFEND THE PROJECT.

This file is a MODEL, not measured reality. Every number below is an
assumption. The honest claim you are allowed to make is RELATIVE:

    "Under this stated response model, the agent recovers X% more than the
     baseline, and that gap holds across a sweep of the model's parameters."

You are NOT allowed to claim absolute rupees are what Razorpay would see.
Say this in the README, in the video, and in the form. Volunteering the
limit of your own evidence is the single cheapest credibility win available
to you, and hiding it is the fastest way to fail the panel.

Assumptions and their sources:
  - base_recoverability: how much of each cause is winnable at all.
    Directionally from public payment-failure literature: bank downtime is
    mostly transient and highly recoverable; hard declines need a new
    instrument and mostly are not.
  - intervention_fit: how well each action suits each cause. Derived from
    Razorpay's own published "next steps" per error reason.
  - lang_lift: the project hypothesis -- language-matched messaging lifts
    response. This is the parameter you MUST sweep, because your whole
    thesis rests on it.
"""
import random

# Calibrated so the CONTROL arm lands in the 12-20% band that published
# payment-retry recovery rates sit in. If your baseline recovers 2% or 60%,
# your simulator is broken and every downstream number is worthless.
BASE_RECOVERABILITY = {
    "insufficient_funds": 0.20,
    "bank_downtime": 0.38,
    "auth_drop_otp": 0.20,
    "hard_decline": 0.09,
    "checkout_abandoned": 0.11,
    "mandate_failure": 0.15,
}

# multiplier in [0,1]: how well the intervention fits the TRUE cause
INTERVENTION_FIT = {
    "insufficient_funds": {"none": 0.0, "silent_retry": 0.35, "nudge_now": 0.55,
                           "nudge_delayed": 1.00, "new_instrument_link": 0.50,
                           "incentive_link": 0.85},
    "bank_downtime":      {"none": 0.0, "silent_retry": 1.00, "nudge_now": 0.40,
                           "nudge_delayed": 0.85, "new_instrument_link": 0.30,
                           "incentive_link": 0.45},
    "auth_drop_otp":      {"none": 0.0, "silent_retry": 0.15, "nudge_now": 1.00,
                           "nudge_delayed": 0.70, "new_instrument_link": 0.45,
                           "incentive_link": 0.80},
    "hard_decline":       {"none": 0.0, "silent_retry": 0.02, "nudge_now": 0.25,
                           "nudge_delayed": 0.30, "new_instrument_link": 1.00,
                           "incentive_link": 0.80},
    "checkout_abandoned": {"none": 0.0, "silent_retry": 0.05, "nudge_now": 0.70,
                           "nudge_delayed": 0.75, "new_instrument_link": 0.40,
                           "incentive_link": 1.00},
    "mandate_failure":    {"none": 0.0, "silent_retry": 0.20, "nudge_now": 0.50,
                           "nudge_delayed": 0.70, "new_instrument_link": 1.00,
                           "incentive_link": 0.60},
}

DEFAULTS = {
    "lang_lift": 0.12,        # relative lift when message language matches
    "fatigue_decay": 0.65,    # each extra touch multiplies p by this
    "optout_base": 0.010,     # per contactful touch
    "optout_wrong_lang": 0.008,
    "msg_cost_rupees": 0.35,
    "silent_retry_cost": 0.02,
}

CONTACTFUL = {"nudge_now", "nudge_delayed", "new_instrument_link", "incentive_link"}


class CustomerSim:
    def __init__(self, seed=7, **overrides):
        self.rng = random.Random(seed)
        self.p = {**DEFAULTS, **overrides}

    def respond(self, event, action):
        """
        action = {intervention, timing_correct, msg_lang, incentive_pct, touch_no}
        returns dict(recovered, opted_out, cost_rupees, revenue_rupees)
        """
        cause = event["true_cause"]
        iv = action["intervention"]
        amount = event["amount_paise"] / 100

        if event["customer"]["opted_out"] or iv == "none":
            return {"recovered": False, "opted_out": False,
                    "cost_rupees": 0.0, "revenue_rupees": 0.0}

        p = BASE_RECOVERABILITY[cause] * INTERVENTION_FIT[cause][iv]

        # timing: right action at the wrong moment loses most of its value
        if not action.get("timing_correct", False):
            p *= 0.70

        # language match -- the project hypothesis
        if iv in CONTACTFUL:
            want = event["customer"]["lang_signal"]
            matched = (action.get("msg_lang") == want) or (
                want == "hi-en" and action.get("msg_lang") == "hinglish")
            if matched:
                p *= (1 + self.p["lang_lift"])

        # message fatigue
        p *= self.p["fatigue_decay"] ** max(0, action.get("touch_no", 1) - 1)

        # returning customers convert better
        p *= 1 + 0.05 * event["customer"]["prior_recoveries"]

        p = max(0.0, min(0.85, p))
        recovered = self.rng.random() < p

        # cost
        cost = (self.p["msg_cost_rupees"] if iv in CONTACTFUL
                else self.p["silent_retry_cost"])
        inc = action.get("incentive_pct", 0.0)
        if recovered and inc > 0:
            cost += amount * inc

        # opt-out risk on contactful touches, worse in the wrong language
        opted = False
        if iv in CONTACTFUL and not recovered:
            q = self.p["optout_base"]
            if action.get("msg_lang") != event["customer"]["lang_signal"]:
                q += self.p["optout_wrong_lang"]
            opted = self.rng.random() < q

        return {"recovered": recovered, "opted_out": opted,
                "cost_rupees": round(cost, 2),
                "revenue_rupees": round(amount if recovered else 0.0, 2)}
