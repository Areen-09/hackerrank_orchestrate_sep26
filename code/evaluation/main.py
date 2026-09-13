"""
Evaluation and Validation Suite for HackerRank Orchestrate: Buy or Wait?
Tests the full hybrid pipeline against dataset/sample_requests.csv.
Computes field-by-field accuracy metrics and generates a validation scorecard.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from code.financial_engine import FinancialEngine
from code.ocr_extractor import get_event_amount_overrides
from code.message_processor import get_user_and_event_message_insights
from code.explanation_generator import generate_all_explanations
from code.gemini_client import GeminiRateLimitedClient


def run_evaluation(
    sample_csv_path: str = "dataset/sample_requests.csv",
    use_llm_explanations: bool = True
):
    print("=" * 70)
    print("BUY OR WAIT? - EVALUATION & VALIDATION HARNESS")
    print("=" * 70)

    if not os.path.exists(sample_csv_path):
        print(f"Sample file not found: {sample_csv_path}")
        return

    sample_df = pd.read_csv(sample_csv_path)
    total_samples = len(sample_df)
    print(f"Loaded {total_samples} ground-truth samples from {sample_csv_path}")

    # Load caches
    ocr_overrides = get_event_amount_overrides()
    print(f"Loaded {len(ocr_overrides)} OCR event overrides from cache.")
    msg_insights = get_user_and_event_message_insights()

    engine = FinancialEngine(
        ocr_overrides=ocr_overrides,
        message_insights=msg_insights.get("message_by_id", {})
    )

    evaluations = []
    for _, r in sample_df.iterrows():
        pred = engine.evaluate_request(r)
        evaluations.append(pred)

    # Explanations
    client = GeminiRateLimitedClient()
    if use_llm_explanations:
        explanations = generate_all_explanations(evaluations, client=client)
        for ev in evaluations:
            ev["decision_explanation"] = explanations.get(ev["request_id"], "")

    # Compare with ground truth
    status_matches = 0
    method_matches = 0
    plan_matches = 0
    earliest_matches = 0
    changes_matches = 0
    safe_amount_diffs = []

    print("\n--- Detailed Per-Request Results ---")
    for idx, r in sample_df.iterrows():
        rid = str(r["request_id"])
        pred = evaluations[idx]

        gt_safe = float(r["amount_safe_to_pay"])
        gt_status = str(r["affordability_status"])
        gt_method = str(r["recommended_payment_method"])
        gt_plan = str(r["payment_plan"])
        gt_earliest = str(r["earliest_date_for_full_payment"]) if pd.notna(r["earliest_date_for_full_payment"]) else ""
        gt_changes = str(r["spending_changes_needed"])

        p_safe = float(pred["amount_safe_to_pay"])
        p_status = str(pred["affordability_status"])
        p_method = str(pred["recommended_payment_method"])
        p_plan = str(pred["payment_plan"])
        p_earliest = str(pred["earliest_date_for_full_payment"])
        p_changes = str(pred["spending_changes_needed"])

        s_ok = (p_status == gt_status)
        m_ok = (p_method == gt_method)
        pl_ok = (p_plan == gt_plan)
        e_ok = (p_earliest == gt_earliest)
        c_ok = (p_changes == gt_changes)

        if s_ok: status_matches += 1
        if m_ok: method_matches += 1
        if pl_ok: plan_matches += 1
        if e_ok: earliest_matches += 1
        if c_ok: changes_matches += 1

        safe_diff = abs(p_safe - gt_safe)
        safe_amount_diffs.append(safe_diff)

        status_flag = "PASS" if (s_ok and m_ok and pl_ok and c_ok) else "DIFF"
        print(f"[{rid}] [{status_flag}]")
        print(f"  Status:   Pred='{p_status}' | GT='{gt_status}' {'[OK]' if s_ok else '[X]'}")
        print(f"  Method:   Pred='{p_method}' | GT='{gt_method}' {'[OK]' if m_ok else '[X]'}")
        print(f"  SafeAmt:  Pred={p_safe:,.2f} | GT={gt_safe:,.2f} (diff={safe_diff:,.2f})")
        print(f"  Plan:     Pred='{p_plan}' | GT='{gt_plan}' {'[OK]' if pl_ok else '[X]'}")
        print(f"  Earliest: Pred='{p_earliest}' | GT='{gt_earliest}' {'[OK]' if e_ok else '[X]'}")
        print(f"  Changes:  Pred='{p_changes}' | GT='{gt_changes}' {'[OK]' if c_ok else '[X]'}")
        print(f"  Explanation: {pred.get('decision_explanation', '')[:90]}...")
        print("-" * 65)

    print("\n" + "=" * 70)
    print("FINAL VALIDATION SCORECARD")
    print("=" * 70)
    print(f"Total Samples:                    {total_samples}")
    print(f"Affordability Status Accuracy:    {status_matches}/{total_samples} ({status_matches/total_samples*100:.1f}%)")
    print(f"Recommended Method Accuracy:      {method_matches}/{total_samples} ({method_matches/total_samples*100:.1f}%)")
    print(f"Payment Plan Accuracy:            {plan_matches}/{total_samples} ({plan_matches/total_samples*100:.1f}%)")
    print(f"Earliest Date Accuracy:           {earliest_matches}/{total_samples} ({earliest_matches/total_samples*100:.1f}%)")
    print(f"Spending Changes Needed Accuracy: {changes_matches}/{total_samples} ({changes_matches/total_samples*100:.1f}%)")
    print(f"Mean Safe Amount Abs Error:       {sum(safe_amount_diffs)/len(safe_amount_diffs):,.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
