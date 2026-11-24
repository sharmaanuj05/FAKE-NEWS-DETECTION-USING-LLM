"""
scripts/audit_baseline.py — Step 0: Baseline Audit of Current System on tests/claims_dev.json.
Records real outputs to reports/audit_before.json and calculates false-REAL and false-FAKE rates.
"""
import os
import sys
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.explain import ExplainerPipeline

def run_audit():
    dev_path = os.path.join(BASE_DIR, "tests", "claims_dev.json")
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"Missing {dev_path}")

    with open(dev_path) as f:
        claims = json.load(f)

    pipeline = ExplainerPipeline.get_instance()
    results = []

    print(f"Starting baseline audit on {len(claims)} dev items...")
    start_time = time.time()

    for idx, item in enumerate(claims, 1):
        text = item["text"]
        true_label = item["label"]
        print(f"[{idx}/{len(claims)}] Checking: {text[:60]}...")

        # Run pipeline with live_verify=True
        res = pipeline.explain(text, live_verify=True)

        classifier_verdict = res["raw_label"]
        final_verdict = res["verdict"]
        conf = res["confidence"]
        real_prob = res["real_probability"]
        fake_prob = res["fake_probability"]
        fc_res = res.get("live_verification")

        is_correct = (final_verdict == true_label)
        classifier_is_correct = (classifier_verdict == true_label)

        results.append({
            "id": idx,
            "text": text,
            "true_label": true_label,
            "type": item.get("type"),
            "topic": item.get("topic"),
            "style": item.get("style"),
            "basis": item.get("basis"),
            "classifier_verdict": classifier_verdict,
            "calibrated_confidence": conf,
            "real_probability": real_prob,
            "fake_probability": fake_prob,
            "fact_checker_output": {
                "status": fc_res.get("status") if fc_res else "UNAVAILABLE",
                "query_used": fc_res.get("query_used") if fc_res else "",
                "credible_sources_count": fc_res.get("credible_sources_count", 0) if fc_res else 0,
                "debunk_articles_count": fc_res.get("debunk_articles_count", 0) if fc_res else 0,
                "corroboration_score": fc_res.get("corroboration_score", 0.0) if fc_res else 0.0,
            } if fc_res else None,
            "final_verdict": final_verdict,
            "is_correct": is_correct,
            "classifier_is_correct": classifier_is_correct
        })

    total = len(results)
    fakes = [r for r in results if r["true_label"] == "FAKE"]
    reals = [r for r in results if r["true_label"] == "REAL"]

    # Final verdict metrics
    final_correct = sum(1 for r in results if r["is_correct"])
    final_acc = final_correct / total

    # False-REAL rate: fake called REAL
    false_real_count = sum(1 for r in fakes if r["final_verdict"] == "REAL")
    false_real_rate = false_real_count / len(fakes) if fakes else 0.0

    # False-FAKE rate: real called FAKE
    false_fake_count = sum(1 for r in reals if r["final_verdict"] == "FAKE")
    false_fake_rate = false_fake_count / len(reals) if reals else 0.0

    # Raw classifier metrics
    raw_correct = sum(1 for r in results if r["classifier_is_correct"])
    raw_acc = raw_correct / total
    raw_false_real = sum(1 for r in fakes if r["classifier_verdict"] == "REAL")
    raw_false_real_rate = raw_false_real / len(fakes) if fakes else 0.0
    raw_false_fake = sum(1 for r in reals if r["classifier_verdict"] == "FAKE")
    raw_false_fake_rate = raw_false_fake / len(reals) if reals else 0.0

    uncertain_count = sum(1 for r in results if r["final_verdict"] == "UNCERTAIN")
    uncertain_rate = uncertain_count / total

    elapsed = round(time.time() - start_time, 2)

    audit_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "total_items": total,
        "num_real": len(reals),
        "num_fake": len(fakes),
        "metrics_final_system": {
            "accuracy": round(final_acc, 4),
            "correct_count": final_correct,
            "false_real_rate": round(false_real_rate, 4),
            "false_real_count": false_real_count,
            "false_fake_rate": round(false_fake_rate, 4),
            "false_fake_count": false_fake_count,
            "uncertain_rate": round(uncertain_rate, 4),
            "uncertain_count": uncertain_count
        },
        "metrics_raw_classifier": {
            "accuracy": round(raw_acc, 4),
            "correct_count": raw_correct,
            "false_real_rate": round(raw_false_real_rate, 4),
            "false_real_count": raw_false_real,
            "false_fake_rate": round(raw_false_fake_rate, 4),
            "false_fake_count": raw_false_fake
        },
        "items": results
    }

    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_file = os.path.join(reports_dir, "audit_before.json")

    with open(out_file, "w") as f:
        json.dump(audit_summary, f, indent=2)

    print("\n" + "="*60)
    print("STEP 0 BASELINE AUDIT RESULTS")
    print("="*60)
    print(f"Total dev items evaluated: {total}")
    print(f"Elapsed time: {elapsed}s\n")
    print("--- Final System (with current fact-checker & threshold) ---")
    print(f"Accuracy:        {final_acc*100:.1f}% ({final_correct}/{total})")
    print(f"False-REAL Rate: {false_real_rate*100:.1f}% ({false_real_count}/{len(fakes)} fake claims called REAL)")
    print(f"False-FAKE Rate: {false_fake_rate*100:.1f}% ({false_fake_count}/{len(reals)} real claims called FAKE)")
    print(f"Uncertain Rate:  {uncertain_rate*100:.1f}% ({uncertain_count}/{total})")
    print("\n--- Raw Classifier Alone ---")
    print(f"Accuracy:        {raw_acc*100:.1f}% ({raw_correct}/{total})")
    print(f"False-REAL Rate: {raw_false_real_rate*100:.1f}% ({raw_false_real}/{len(fakes)} fake claims called REAL)")
    print(f"False-FAKE Rate: {raw_false_fake_rate*100:.1f}% ({raw_false_fake}/{len(reals)} real claims called FAKE)")
    print("="*60)
    print(f"Audit output successfully saved to: {out_file}\n")

if __name__ == "__main__":
    run_audit()
