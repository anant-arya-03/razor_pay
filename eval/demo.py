"""
End-to-end demo. THIS IS WHAT YOU SCREEN-RECORD FOR THE VIDEO.

Runs real held-out events through the full pipeline and prints every step:
    raw gateway error -> model prediction -> policy rule -> message -> outcome

Deliberately includes the cases that make the project credible:
    - a bank downtime event where the agent sends NO message at all
    - an event the agent REFUSES to act on
    - the headline baseline-vs-agent comparison

Usage:
    python -m eval.demo
    python -m eval.demo --llm      # if Ollama is running
"""
import argparse
import json

from agent.compose import compose
from agent.policy import decide
from sim.customer_sim import CustomerSim

W = 74


def rule(ch="-"):
    print(ch * W)


def show(event, pred, use_llm):
    sim = CustomerSim(seed=11)
    print()
    rule("=")
    print(f"  EVENT {event['event_id']}   Rs {event['amount_paise']/100:,.0f}"
          f"   {event['method'].upper()}   {event['bank']}")
    rule("=")

    print("\n  [1] WHAT THE GATEWAY GAVE US")
    print(f"      code   : {event['error_code']}")
    print(f"      source : {event['error_source']}   step: {event['error_step']}")
    print(f"      text   : \"{event['error_description'][:80]}\"")

    print("\n  [2] MODEL DIAGNOSIS")
    hit = "CORRECT" if pred == event["true_cause"] else "WRONG"
    print(f"      predicted : {pred}")
    print(f"      actual    : {event['true_cause']}   [{hit}]")

    d = decide(event, pred, touch_no=1)
    d["predicted_cause"] = pred

    print("\n  [3] POLICY DECISION  (deterministic, no LLM)")
    print(f"      rule      : {d['rule']}")
    print(f"      action    : {d['intervention']}")
    print(f"      wait      : {d['delay_min']} min")
    print(f"      language  : {d['msg_lang']}")
    print(f"      incentive : {d['incentive_pct']:.0%}  (cap 5%)")

    print("\n  [4] MESSAGE")
    if d["intervention"] == "none":
        print(f"      NO CONTACT -- guardrail fired: {d['rule']}")
    elif d["intervention"] == "silent_retry":
        print("      NO MESSAGE SENT.")
        print("      Bank was down. Retrying silently costs Rs 0.02 and")
        print("      annoys nobody. A generic system would have messaged here.")
    else:
        m = compose(event, d, use_llm=use_llm)
        print(f"      source: {m['source']}   guardrail: {m['validation_reason']}")
        print(f"      \"{m['text']}\"")

    r = sim.respond(event, d)
    print("\n  [5] OUTCOME")
    print(f"      recovered : {r['recovered']}")
    print(f"      revenue   : Rs {r['revenue_rupees']:,.0f}")
    print(f"      cost      : Rs {r['cost_rupees']:,.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="use Ollama if running")
    a = ap.parse_args()

    preds = json.load(open("artifacts/test_predictions.json"))
    events = [json.loads(l) for l in open("artifacts/events.jsonl")
              if json.loads(l)["event_id"] in preds]

    print("\n" + "=" * W)
    print("  CAUSE-AWARE PAYMENT RECOVERY AGENT")
    print("  Razorpay AI Buildathon 2026 - Track 3")
    print("=" * W)
    print(f"  Held-out events (model never saw these): {len(events)}")
    print(f"  Message generation: {'Ollama / gemma2:9b' if a.llm else 'templates'}")

    # one event per cause, plus a refusal
    want = ["auth_drop_otp", "bank_downtime", "checkout_abandoned", "hard_decline"]
    seen = set()
    for e in events:
        c = preds[e["event_id"]]
        hindi = e["customer"]["lang_signal"] in ("hi-IN", "hi-en")
        if c in want and c not in seen and not e["customer"]["opted_out"] and hindi:
            seen.add(c)
            show(e, c, a.llm)
        if len(seen) == len(want):
            break

    refused = next((e for e in events if e["customer"]["opted_out"]), None)
    if refused:
        show(refused, preds[refused["event_id"]], a.llm)

    res = json.load(open("artifacts/experiment_results.json"))
    b, g = res["baseline"], res["agent"]
    print()
    rule("=")
    print(f"  RESULT ACROSS ALL {b['n_events']} HELD-OUT EVENTS")
    rule("=")
    print(f"\n  {'':<24}{'baseline':>14}{'agent':>14}")
    rule()
    for label, k, f in [("value recovered", "revenue", "Rs {:,.0f}"),
                        ("recovery rate", "value_recovery_rate", "{:.2%}"),
                        ("messages sent", "touches", "{:,.0f}"),
                        ("opt-out rate", "optout_rate", "{:.2%}")]:
        print(f"  {label:<24}{f.format(b[k]):>14}{f.format(g[k]):>14}")
    lift = (g["value_recovery_rate"] - b["value_recovery_rate"]) * 100
    print(f"\n  +{lift:.1f}pp value recovered, "
          f"{b['touches']-g['touches']} FEWER messages, "
          f"{(b['optout_rate']-g['optout_rate'])*100:.2f}pp fewer opt-outs")
    print("\n  Same events. Same 3-touch budget. Only the decisions differ.")
    try:
        sw = json.load(open("artifacts/sweep_results.json"))
        print(f"  Sweep across {sw['n_configs']} assumption sets: agent wins "
              f"{sw['wins']}/{sw['n_configs']}, worst +{sw['worst_pp']:.1f}pp.\n")
    except (FileNotFoundError, KeyError, ValueError):
        print("  Sweep not run yet -- `make sweep` for the robustness range.\n")


if __name__ == "__main__":
    main()
