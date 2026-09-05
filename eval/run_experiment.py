"""
Two-arm experiment: baseline vs agent, same held-out events.

The agent runs on PREDICTED causes from the classifier, never on ground
truth. That is what makes the money number a real end-to-end result rather
than an oracle result.

Usage:
    python -m eval.run_experiment
    python -m eval.run_experiment --sweep      # robustness across assumptions
"""
import argparse
import json
from collections import Counter, defaultdict

from agent.policy import baseline_decide, decide, MAX_TOUCHES
from sim.customer_sim import CustomerSim


def load_holdout(events_path, preds_path):
    preds = json.load(open(preds_path))
    events = [json.loads(l) for l in open(events_path)]
    held = [e for e in events if e["event_id"] in preds]
    return held, preds


def run_arm(events, preds, arm, seed=7, **sim_overrides):
    sim = CustomerSim(seed=seed, **sim_overrides)
    audit, refusals = [], Counter()
    tot = {"revenue": 0.0, "cost": 0.0, "recovered": 0, "optouts": 0,
           "touches": 0, "at_risk": 0.0, "escalations": 0}

    for e in events:
        tot["at_risk"] += e["amount_paise"] / 100
        opted = e["customer"]["opted_out"]
        done = False

        for touch in range(1, MAX_TOUCHES + 1):
            if done or opted:
                break
            if arm == "baseline":
                d = baseline_decide(e, touch)
            else:
                d = decide(e, preds[e["event_id"]], touch)

            if d["intervention"] == "none":
                if touch == 1:
                    refusals[d["rule"]] += 1
                break

            ev = dict(e)
            ev["customer"] = dict(e["customer"])
            ev["customer"]["opted_out"] = opted
            r = sim.respond(ev, d)

            tot["touches"] += 1
            tot["cost"] += r["cost_rupees"]
            tot["revenue"] += r["revenue_rupees"]
            if d.get("escalate"):
                tot["escalations"] += 1
            if r["recovered"]:
                tot["recovered"] += 1
                done = True
            if r["opted_out"]:
                opted = True
                tot["optouts"] += 1

            audit.append({
                "event_id": e["event_id"], "arm": arm, "touch": touch,
                "predicted_cause": preds[e["event_id"]] if arm == "agent" else None,
                "true_cause": e["true_cause"],
                "intervention": d["intervention"], "rule": d["rule"],
                "delay_min": d["delay_min"], "msg_lang": d["msg_lang"],
                "incentive_pct": d["incentive_pct"],
                "recovered": r["recovered"], "opted_out": r["opted_out"],
                "cost_rupees": r["cost_rupees"],
                "revenue_rupees": r["revenue_rupees"],
            })

    n = len(events)
    tot["n_events"] = n
    tot["recovery_rate"] = tot["recovered"] / n
    tot["value_recovery_rate"] = tot["revenue"] / tot["at_risk"]
    tot["net_rupees"] = tot["revenue"] - tot["cost"]
    tot["cost_per_recovery"] = (tot["cost"] / tot["recovered"]
                               if tot["recovered"] else 0.0)
    tot["optout_rate"] = tot["optouts"] / n
    tot["refusals"] = dict(refusals)
    return tot, audit


def summarise(base, agent):
    print(f"\n{'metric':<26}{'baseline':>14}{'agent':>14}{'delta':>14}")
    print("-" * 68)
    rows = [
        ("events", "n_events", "{:,.0f}"),
        ("value at risk (Rs)", "at_risk", "{:,.0f}"),
        ("recovered (count)", "recovered", "{:,.0f}"),
        ("recovery rate", "recovery_rate", "{:.2%}"),
        ("value recovered (Rs)", "revenue", "{:,.0f}"),
        ("value recovery rate", "value_recovery_rate", "{:.2%}"),
        ("cost (Rs)", "cost", "{:,.0f}"),
        ("net (Rs)", "net_rupees", "{:,.0f}"),
        ("cost / recovery (Rs)", "cost_per_recovery", "{:,.2f}"),
        ("messages sent", "touches", "{:,.0f}"),
        ("opt-out rate", "optout_rate", "{:.2%}"),
    ]
    for label, key, fmt in rows:
        b, a = base[key], agent[key]
        delta = a - b
        ds = fmt.format(delta)
        if delta > 0 and not ds.startswith("+"):
            ds = "+" + ds
        print(f"{label:<26}{fmt.format(b):>14}{fmt.format(a):>14}{ds:>14}")

    lift = (agent["value_recovery_rate"] - base["value_recovery_rate"]) * 100
    print(f"\nHEADLINE: +{lift:.1f} percentage points of value recovered "
          f"(Rs {agent['revenue']-base['revenue']:,.0f} more), "
          f"opt-out {agent['optout_rate']:.2%} vs {base['optout_rate']:.2%}")
    if agent["refusals"]:
        print("\nagent refused to act on:")
        for k, v in sorted(agent["refusals"].items(), key=lambda x: -x[1]):
            print(f"  {v:5d}  {k}")


def sweep(events, preds):
    """Does the agent still win if our assumptions are wrong? Show the range."""
    print("\n=== sensitivity sweep ===")
    print(f"{'lang_lift':>10}{'fatigue':>10}{'seed':>7}"
          f"{'base VRR':>11}{'agent VRR':>11}{'lift(pp)':>10}")
    lifts = []
    for lang_lift in (0.00, 0.06, 0.12, 0.20):
        for fatigue in (0.5, 0.65, 0.8):
            for seed in (7, 13):
                kw = {"lang_lift": lang_lift, "fatigue_decay": fatigue}
                b, _ = run_arm(events, preds, "baseline", seed=seed, **kw)
                a, _ = run_arm(events, preds, "agent", seed=seed, **kw)
                lift = (a["value_recovery_rate"] - b["value_recovery_rate"]) * 100
                lifts.append(lift)
                print(f"{lang_lift:>10.2f}{fatigue:>10.2f}{seed:>7}"
                      f"{b['value_recovery_rate']:>10.2%}"
                      f"{a['value_recovery_rate']:>10.2%}{lift:>10.1f}")
    print(f"\nlift range across {len(lifts)} configs: "
          f"{min(lifts):.1f} to {max(lifts):.1f} pp  "
          f"(agent wins in {sum(l>0 for l in lifts)}/{len(lifts)})")
    print("Quote the WORST case in your video, not the best one.")

    # Persist it so no downstream file has to hard-code a sweep number.
    with open("artifacts/sweep_results.json", "w") as f:
        json.dump({"n_configs": len(lifts),
                   "worst_pp": round(min(lifts), 1),
                   "best_pp": round(max(lifts), 1),
                   "wins": sum(l > 0 for l in lifts),
                   "lifts_pp": [round(l, 1) for l in lifts]}, f, indent=2)
    print("sweep -> artifacts/sweep_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="artifacts/events.jsonl")
    ap.add_argument("--preds", default="artifacts/test_predictions.json")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    events, preds = load_holdout(a.events, a.preds)
    print(f"held-out events: {len(events)}")

    base, base_audit = run_arm(events, preds, "baseline")
    agent, agent_audit = run_arm(events, preds, "agent")
    summarise(base, agent)

    with open("artifacts/audit_log.jsonl", "w") as f:
        for r in base_audit + agent_audit:
            f.write(json.dumps(r) + "\n")
    with open("artifacts/experiment_results.json", "w") as f:
        json.dump({"baseline": base, "agent": agent}, f, indent=2)
    print(f"\naudit log -> artifacts/audit_log.jsonl "
          f"({len(base_audit)+len(agent_audit)} decisions)")

    if a.sweep:
        sweep(events, preds)


if __name__ == "__main__":
    main()
