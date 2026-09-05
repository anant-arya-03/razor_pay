# Architecture

```
 failed payment event
 (Razorpay webhook / API)
          |
          v
 +----------------------+
 | models/              |   TF-IDF (word 1-2 + char 3-5) -> LogisticRegression
 | train_classifier.py  |   input : error_description + code/source/step/method
 |                      |   output: 1 of 6 causes        93.3% acc / 0.919 F1
 +----------------------+   baseline: code-only lookup   39.9% acc / 0.205 F1
          |
          v  predicted cause (never ground truth)
 +----------------------+
 | agent/policy.py      |   DETERMINISTIC. No LLM.
 |                      |   guardrails run FIRST:
 |                      |     opted out / touch cap 3 / <Rs 50 / quiet hours
 |                      |   then cause -> action:
 |                      |     bank_downtime      -> silent_retry, no message
 |                      |     insufficient_funds -> delayed nudge (salary cycle)
 |                      |     auth_drop_otp      -> immediate nudge
 |                      |     hard_decline       -> new instrument link
 |                      |     checkout_abandoned -> capped 5% incentive link
 |                      |     mandate_failure    -> re-auth link, escalate at 2
 +----------------------+   returns the rule name with every decision
          |
          v  action + language + incentive
 +----------------------+
 | agent/compose.py     |   THE ONLY LLM IN THE SYSTEM
 |                      |   Ollama / gemma2:9b, template fallback
 |                      |   validate(): rejects coercive language, missing
 |                      |   opt-out, unauthorised discounts, >320 chars
 +----------------------+   rejected -> safe template, reason logged
          |
          v
 +----------------------+
 | sim/customer_sim.py  |   stated response model, swept in eval
 +----------------------+
          |
          v
 +----------------------+
 | eval/run_experiment  |   baseline arm vs agent arm, same events,
 |                      |   same 3-touch budget
 |                      |   -> artifacts/audit_log.jsonl  (1 line / decision)
 +----------------------+   -> artifacts/experiment_results.json
```

## Audit record

Every decision emits:

```json
{"event_id":"evt_000042","arm":"agent","touch":1,
 "predicted_cause":"auth_drop_otp","true_cause":"auth_drop_otp",
 "intervention":"nudge_now","rule":"otp_drop:strike_while_warm",
 "delay_min":0,"msg_lang":"hinglish","incentive_pct":0.0,
 "recovered":true,"cost_rupees":0.35,"revenue_rupees":799.0}
```

Predicted cause, the rule that fired, the action, and the rupee outcome.
Enough to reconstruct why any customer was contacted.
