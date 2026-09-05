# Cause-Aware Payment Recovery Agent

Razorpay AI Buildathon 2026 — Track 3: AI Revenue Recovery

Most merchants respond to a failed payment with one generic English retry SMS,
regardless of *why* it failed. A bank outage, an exhausted balance, an
abandoned OTP screen and an expired card all get the same message. This agent
classifies the cause from the gateway's free-text error, picks a bounded
intervention matched to that cause, and reports how much money it actually
recovered against a fair control arm.

## Results

### 1. Root-cause classifier (held-out, n=750)

| system | input | accuracy | macro-F1 |
|---|---|---|---|
| `rule_lookup` | `error_code` only | 39.9% | 0.205 |
| `tfidf_lr` | free-text description + context | **93.3%** | **0.919** |

The lookup baseline exists to answer "why not just a dictionary?".
`BAD_REQUEST_ERROR` covers four of the six causes, so code-only lookup cannot
separate them. Labels carry 5% injected noise, so ~0.95 macro-F1 is the ceiling.

### 2. Money recovered (same 750 held-out events)

Control arm = competent current practice: three generic English reminders at
0h / 24h / 72h. **Same touch budget as the agent** — otherwise the lift would
just be message count.

| metric | baseline | agent | delta |
|---|---|---|---|
| value at risk | Rs 19,47,342 | Rs 19,47,342 | — |
| value recovered | Rs 3,17,507 | Rs 6,61,035 | **+Rs 3,43,528** |
| value recovery rate | 16.30% | **33.95%** | **+17.6pp** |
| messages sent | 2,000 | 1,779 | **-221** |
| opt-out rate | 5.33% | **3.73%** | **-1.60pp** |
| cost per recovery | Rs 6.19 | Rs 15.33 | +Rs 9.14 |

The agent recovers roughly twice the value while sending *fewer* messages and
annoying *fewer* customers. Cost per recovery is higher because it uses capped
incentives on abandoned checkouts; net contribution is still Rs 3.40L higher.

### 3. Does it survive our assumptions being wrong?

The customer response model in `sim/customer_sim.py` is a stated model, not
measured reality. Swept across 24 configurations of its two most load-bearing
parameters:

**Agent wins 24/24. Lift range +13.4pp to +20.1pp. Worst case +13.4pp.**

Honest finding worth stating: at `lang_lift = 0.00` — i.e. assuming
language-matched messaging has *no* effect at all — the agent still gains
+14.5 to +19.0pp. **Cause-aware timing and intervention choice drive the
result; Hinglish is a secondary contributor, not the mechanism.**

## What is and is not claimed

- **Claimed:** relative lift under a stated, swept response model.
- **Not claimed:** these absolute rupee figures are what Razorpay would see.
- Events are synthetic, generated from Razorpay's published error schema
  (`error_code` / `error_source` / `error_step` / `error_reason` /
  `error_description`) and its documented error-reason list.

## Design decision: the LLM does not touch money

`agent/policy.py` is deterministic. Cause -> intervention, timing, channel and
incentive cap are rules, and every decision returns the rule that produced it.
The LLM writes message *text* only. Two identical customers must get identical
offers; an LLM choosing discounts cannot guarantee that.

## Run it

```bash
pip install -r requirements.txt
make demo        # generate -> train -> experiment -> sweep
```

The run is deterministic: `--seed 42` plus a fixed corpus epoch fully
determine every number above, and `requirements.txt` pins the library
versions they were produced under. A clean clone reproduces this table
exactly.

Outputs land in `artifacts/`: `classifier_metrics.json`,
`experiment_results.json`, and `audit_log.jsonl` (one line per decision, with
predicted cause, rule fired, and rupee outcome).

# razor_pay
