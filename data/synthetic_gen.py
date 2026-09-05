"""
Generate a labelled corpus of payment-failure events.

Every event carries `ground_truth_cause`, which is what makes both the
classifier metrics and the money experiment measurable.

Noise we inject on purpose (a clean generator would make the task fake):
  - paraphrase + typos + truncation in error_description
  - 8% of events get a *generic* description carrying no cause signal
  - 5% label noise (mislabelled at source, like real ops data)
  - overlapping error_codes across causes

Usage:
    python -m data.synthetic_gen --n 3000 --out artifacts/events.jsonl --seed 42
"""
import argparse
import json
import random
from datetime import datetime, timedelta, timezone

from data.taxonomy import CAUSES, CAUSE_PROFILE

IST = timezone(timedelta(hours=5, minutes=30))

# Anchor the corpus to a FIXED instant so --seed alone fully determines the
# output. This previously used datetime.now(), which made each event's
# hour-of-day depend on wall-clock time. That flows into the quiet-hours
# branch in agent/policy.py (auth_drop_otp -> nudge_now vs nudge_delayed),
# so the agent arm produced different money numbers on every run while the
# baseline arm -- which never reads the hour -- stayed identical. Chosen
# once, not searched for a favourable result.
EPOCH = datetime(2026, 1, 1, 12, 0, tzinfo=IST)

BANKS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "PNB", "BOB", "YESB", "IDFC"]

# Prevalence in a realistic failure mix. Not uniform -- class imbalance is
# part of the problem, and macro-F1 is what we report because of it.
PREVALENCE = {
    "insufficient_funds": 0.26,
    "bank_downtime": 0.20,
    "auth_drop_otp": 0.22,
    "hard_decline": 0.12,
    "checkout_abandoned": 0.14,
    "mandate_failure": 0.06,
}

GENERIC_DESCRIPTIONS = [
    "Payment failed.",
    "The transaction could not be completed.",
    "Payment unsuccessful. Please try again.",
    "Transaction declined.",
    "An error occurred while processing the payment.",
]

LANG_SIGNALS = [("hi-IN", 0.46), ("en-IN", 0.34), ("hi-en", 0.20)]


def _pick(weighted, rng):
    r, acc = rng.random(), 0.0
    for value, w in weighted:
        acc += w
        if r <= acc:
            return value
    return weighted[-1][0]


def _corrupt(text, rng):
    """Typos, casing drift, truncation -- makes the text messy like real logs."""
    if rng.random() < 0.18:
        text = text.upper() if rng.random() < 0.5 else text.lower()
    if rng.random() < 0.20 and len(text) > 40:
        text = text[: rng.randint(25, len(text) - 5)].rstrip() + "..."
    if rng.random() < 0.22:
        i = rng.randrange(len(text))
        text = text[:i] + text[i + 1:]          # dropped char
    if rng.random() < 0.12:
        text = text.replace(" ", "  ", 1)        # double space
    return text


def make_event(idx, rng, now):
    cause = _pick(list(PREVALENCE.items()), rng)
    p = CAUSE_PROFILE[cause]

    if rng.random() < 0.08:
        desc = rng.choice(GENERIC_DESCRIPTIONS)   # no cause signal at all
    else:
        desc = _corrupt(rng.choice(p["descriptions"]), rng)

    ts = now - timedelta(minutes=rng.randint(0, 60 * 24 * 14))
    amount = int(rng.choice([199, 349, 499, 799, 1249, 1999, 2499, 4999, 9999])
                 * rng.uniform(0.9, 1.15)) * 100  # paise

    observed = cause
    if rng.random() < 0.05:                       # 5% label noise
        observed = rng.choice([c for c in CAUSES if c != cause])

    return {
        "event_id": f"evt_{idx:06d}",
        "type": ("checkout_abandoned" if cause == "checkout_abandoned"
                 else "mandate_failed" if cause == "mandate_failure"
                 else "payment_failed"),
        "amount_paise": amount,
        "error_code": _pick(p["codes"], rng),
        "error_source": _pick(p["sources"], rng),
        "error_step": _pick(p["steps"], rng),
        "error_reason": rng.choice(p["reasons"]),
        "error_description": desc,
        "method": _pick(p["methods"], rng),
        "bank": rng.choice(BANKS),
        "attempt_number": rng.choices([1, 2, 3], weights=[0.72, 0.21, 0.07])[0],
        "customer": {
            "lang_signal": _pick(LANG_SIGNALS, rng),
            "prior_recoveries": rng.choices([0, 1, 2], weights=[0.75, 0.2, 0.05])[0],
            "opted_out": rng.random() < 0.03,
            "salary_day": rng.choice([1, 1, 1, 5, 7, 10]),
        },
        "timestamp": ts.isoformat(),
        "bank_healthy_after_min": (rng.randint(5, 90)
                                   if cause == "bank_downtime" else None),
        "ground_truth_cause": observed,
        "true_cause": cause,          # ONLY for simulator; never fed to the model
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="artifacts/events.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    now = EPOCH
    events = [make_event(i, rng, now) for i in range(a.n)]

    with open(a.out, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    counts = {}
    for e in events:
        counts[e["true_cause"]] = counts.get(e["true_cause"], 0) + 1
    total = sum(e["amount_paise"] for e in events) / 100

    print(f"wrote {len(events)} events -> {a.out}")
    print(f"total value at risk: Rs {total:,.0f}")
    for c in CAUSES:
        print(f"  {c:22s} {counts.get(c,0):5d}  ({counts.get(c,0)/len(events):.1%})")


if __name__ == "__main__":
    main()
