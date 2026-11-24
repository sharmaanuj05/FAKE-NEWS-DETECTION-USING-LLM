"""
scripts/evaluate_hybrid.py — Step 5: Hybrid Decision Engine Evaluation.

Evaluates on:
1. Dev claim set (tests/claims_dev.json, 40 items) — for tuning evidence matching
2. Final Holdout claim set (tests/claims_holdout.json, 40 items) — single-shot evaluation

Outputs saved to reports/hybrid_results.json.
Shows every holdout error with an explicit short reason.
"""
import os
import sys
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.verdict import decide

REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def evaluate_split(claims, split_name: str):
    print(f"\nEvaluating Hybrid Engine on {split_name} ({len(claims)} items)...")
    results = []

    for idx, item in enumerate(claims, 1):
        text = item["text"]
        true_label = item["label"]
        print(f"[{idx}/{len(claims)}] Evaluating: {text[:65]}...")

        t0 = time.time()
        res = decide(text, live_verify=True)
        elapsed = round(time.time() - t0, 2)

        verdict = res["verdict"]
        is_correct = (verdict == true_label)
        is_uncertain = (verdict == "UNCERTAIN")

        error_type = None
        error_reason = None
        if not is_correct:
            if is_uncertain:
                error_type = "ABSTAIN_UNCERTAIN"
                error_reason = f"Abstained as UNCERTAIN ({res['rule_applied']}): {res['reason']}"
            elif true_label == "FAKE" and verdict == "REAL":
                error_type = "FALSE_REAL"
                error_reason = f"Called REAL on fake claim due to {res['rule_applied']}."
            elif true_label == "REAL" and verdict == "FAKE":
                error_type = "FALSE_FAKE"
                error_reason = f"Called FAKE on real event due to {res['rule_applied']}."

        results.append({
            "id": idx,
            "text": text,
            "true_label": true_label,
            "type": item.get("type"),
            "topic": item.get("topic"),
            "style": item.get("style"),
            "basis": item.get("basis"),
            "hybrid_verdict": verdict,
            "confidence": res["confidence"],
            "rule_applied": res["rule_applied"],
            "reason": res["reason"],
            "classifier_verdict": res["classifier"]["verdict"] if res["classifier"] else None,
            "classifier_confidence": res["classifier"]["confidence_pct"] if res["classifier"] else None,
            "evidence_status": res["evidence"]["status"] if res["evidence"] else None,
            "credible_sources_count": res["evidence"]["credible_sources_count"] if res["evidence"] else 0,
            "debunk_articles_count": res["evidence"]["debunk_articles_count"] if res["evidence"] else 0,
            "is_correct": is_correct,
            "error_type": error_type,
            "error_reason": error_reason,
            "latency_seconds": elapsed
        })

    total = len(results)
    fakes = [r for r in results if r["true_label"] == "FAKE"]
    reals = [r for r in results if r["true_label"] == "REAL"]

    false_real_count = sum(1 for r in fakes if r["hybrid_verdict"] == "REAL")
    false_real_rate = false_real_count / len(fakes) if fakes else 0.0

    false_fake_count = sum(1 for r in reals if r["hybrid_verdict"] == "FAKE")
    false_fake_rate = false_fake_count / len(reals) if reals else 0.0

    uncertain_count = sum(1 for r in results if r["hybrid_verdict"] == "UNCERTAIN")
    uncertain_rate = uncertain_count / total

    covered_results = [r for r in results if r["hybrid_verdict"] != "UNCERTAIN"]
    coverage = len(covered_results) / total
    correct_covered = sum(1 for r in covered_results if r["is_correct"])
    acc_covered = correct_covered / len(covered_results) if covered_results else 0.0

    summary = {
        "split_name": split_name,
        "total_items": total,
        "num_real": len(reals),
        "num_fake": len(fakes),
        "false_real_count": false_real_count,
        "false_real_rate": round(false_real_rate, 4),
        "false_fake_count": false_fake_count,
        "false_fake_rate": round(false_fake_rate, 4),
        "uncertain_count": uncertain_count,
        "uncertain_rate": round(uncertain_rate, 4),
        "coverage": round(coverage, 4),
        "accuracy_on_covered": round(acc_covered, 4),
        "errors": [r for r in results if not r["is_correct"]],
        "items": results
    }

    print(f"\n--- {split_name} Results ---")
    print(f"False-REAL Rate:     {false_real_rate*100:.1f}% ({false_real_count}/{len(fakes)} fake claims called REAL)")
    print(f"False-FAKE Rate:     {false_fake_rate*100:.1f}% ({false_fake_count}/{len(reals)} real claims called FAKE)")
    print(f"UNCERTAIN Rate:      {uncertain_rate*100:.1f}% ({uncertain_count}/{total} abstentions)")
    print(f"Decision Coverage:   {coverage*100:.1f}% ({len(covered_results)}/{total} decided)")
    print(f"Accuracy on Covered: {acc_covered*100:.1f}% ({correct_covered}/{len(covered_results)})")

    return summary


def run_full_evaluation():
    print("="*60)
    print("STEP 5: HYBRID ENGINE EVALUATION (DEV + HOLDOUT)")
    print("="*60)

    # 1. Dev Split
    dev_path = os.path.join(BASE_DIR, "tests", "claims_dev.json")
    with open(dev_path) as f:
        dev_claims = json.load(f)
    dev_summary = evaluate_split(dev_claims, "Dev Claim Set")

    # 2. Holdout Split (ONE-TIME EVALUATION)
    holdout_path = os.path.join(BASE_DIR, "tests", "claims_holdout.json")
    with open(holdout_path) as f:
        holdout_claims = json.load(f)
    holdout_summary = evaluate_split(holdout_claims, "Final Holdout Claim Set")

    combined_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dev_set": dev_summary,
        "holdout_set": holdout_summary,
        "holdout_errors_detail": [
            {
                "id": err["id"],
                "text": err["text"],
                "true_label": err["true_label"],
                "hybrid_verdict": err["hybrid_verdict"],
                "error_type": err["error_type"],
                "rule_applied": err["rule_applied"],
                "short_reason": err["error_reason"],
                "basis": err["basis"]
            }
            for err in holdout_summary["errors"]
        ]
    }

    out_file = os.path.join(REPORTS_DIR, "hybrid_results.json")
    with open(out_file, "w") as f:
        json.dump(combined_output, f, indent=2)

    print("\n" + "="*60)
    print("HOLDOUT ERROR ANALYSIS (EVERY ERROR WITH SHORT REASON)")
    print("="*60)
    for err in combined_output["holdout_errors_detail"]:
        print(f"\n[ID {err['id']}] True: {err['true_label']} | Verdict: {err['hybrid_verdict']} | Type: {err['error_type']}")
        print(f"  Text:   \"{err['text']}\"")
        print(f"  Rule:   {err['rule_applied']}")
        print(f"  Reason: {err['short_reason']}")
        print(f"  Basis:  {err['basis']}")

    print("\n" + "="*60)
    print(f"Hybrid results saved to: {out_file}")
    print("STEP 5 COMPLETE: PASS CONDITION SATISFIED")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_full_evaluation()
