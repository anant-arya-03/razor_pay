# Design Decisions

## 1. The LLM never decides money

`agent/policy.py` contains no LLM. Cause -> intervention, timing, channel and
incentive cap are deterministic rules, and every decision returns the rule
name that produced it (`downtime:wait_for_bank`, `abandoned:capped_incentive`).

The LLM runs in exactly one place — `agent/compose.py` — where it writes
message text. Even there its output is untrusted: `validate()` rejects
coercive language, messages over 320 chars, missing opt-out lines, and any
discount the policy did not authorise. A rejected message falls back to a
deterministic template and the rejection reason is logged.

Rationale: two identical customers must receive identical offers. An LLM
choosing discounts cannot guarantee that, and for a payments company
inconsistent offers are a compliance problem, not a UX quirk.

## 2. Why a trained classifier and not a lookup table

The obvious objection is that `error_code` already tells you the cause. It
does not. `BAD_REQUEST_ERROR` covers four of our six causes, so a code-only
lookup reaches 39.9% accuracy / 0.205 macro-F1 on held out data. The
distinguishing signal sits in `error_description`, which is free text whose
wording varies by bank and acquirer. A TF-IDF (word + character n-gram)
model over description plus context reaches 93.3% / 0.919.

We ship the lookup baseline in `models/train_classifier.py` so this claim is
reproducible rather than asserted. Character n-grams are included because
real gateway logs contain typos and truncation.

## 3. The control arm gets the same touch budget

First run of the experiment produced +52pp. That was wrong: the baseline had
one touch and the agent had three, so the lift was measuring message volume,
not intelligence. The baseline is now competent current practice — three
generic English reminders at 0h / 24h / 72h — and the lift is +17.6pp.

A strawman control is the fastest way to make a real result unbelievable.

## 4. Synthetic data, stated as such

No public labelled dataset of payment failures exists. Events are generated
from Razorpay's published error schema (`error_code`, `error_source`,
`error_step`, `error_reason`, `error_description`) and documented reason
list. The generator deliberately injects typos, truncation, 8% signal-free
generic descriptions, 5% label noise, and error codes that overlap across
causes. Clean data would have produced a fake-looking 99%.

## 5. The simulator is a model, and we sweep it

`sim/customer_sim.py` encodes response probabilities we chose. So the claim
is relative, not absolute: under a stated response model, the agent beats
the baseline. `--sweep` re-runs the whole experiment across 24 settings of
the two most load-bearing parameters. The agent wins 24/24, range +13.4pp to
+20.1pp.

Notable honest result: with `lang_lift = 0.00` — assuming language matching
does nothing at all — the agent still gains +14.5 to +19.0pp. Cause-aware
timing and intervention selection drive the outcome; Hinglish is a secondary
contributor. We report this because it contradicts our own initial hypothesis.

## 6. Templates fall back when Ollama is absent

`compose.py` uses Ollama/gemma2:9b when reachable and deterministic Hinglish
templates otherwise. A reviewer cloning this repo will not have our model
server running, and a demo that dies on their machine is worth nothing.

## What we did not build, and why

- **Voice recovery.** Named in the track brief, but roughly a week of work
  adding no evidence the text pipeline does not already provide.
- **Fine-tuned transformer.** TF-IDF already reaches the practical ceiling
  set by 5% label noise. A transformer would add training time, not signal.
- **Auth, dashboard, multi-tenancy.** Not what the track is measuring.

## 7. The corpus is anchored to a fixed epoch, so the run is reproducible

`data/synthetic_gen.py` originally timestamped events relative to
`datetime.now()`. Despite `--seed 42`, that made every event's hour-of-day
depend on when you ran it -- which feeds the quiet-hours branch in
`agent/policy.py`, flipping `auth_drop_otp` between `nudge_now` and
`nudge_delayed`. The agent arm therefore produced different money numbers on
every run, while the baseline arm -- which never reads the hour -- stayed
identical to the rupee. That asymmetry is exactly how we caught it.

The corpus is now anchored to a fixed `EPOCH`, and `requirements.txt` pins
scikit-learn and pandas. A reviewer cloning the repo reproduces the README
table exactly. Docs that drift from the code are worse than no docs.
