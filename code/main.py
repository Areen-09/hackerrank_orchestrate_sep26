"""
Main Entry Point for HackerRank Orchestrate: Buy or Wait?
Generates dataset/output.csv, evaluation/usage_report.md, and code.zip.
Usage:
    .venv/Scripts/python.exe code/main.py
"""

import os
import sys
import zipfile
import pandas as pd

# Add both script directory and project root to sys.path
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [SCRIPT_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from code.gemini_client import GeminiRateLimitedClient
    from code.ocr_extractor import run_ocr_on_all_images, get_event_amount_overrides
    from code.message_processor import process_all_messages, get_user_and_event_message_insights
    from code.financial_engine import FinancialEngine, resolve_path
    from code.explanation_generator import generate_all_explanations
except (ImportError, ModuleNotFoundError):
    from gemini_client import GeminiRateLimitedClient
    from ocr_extractor import run_ocr_on_all_images, get_event_amount_overrides
    from message_processor import process_all_messages, get_user_and_event_message_insights
    from financial_engine import FinancialEngine, resolve_path
    from explanation_generator import generate_all_explanations

def find_file(filename: str, default_dir: str = "dataset") -> str:
    candidates = [
        os.path.join(default_dir, filename),
        os.path.join("..", default_dir, filename),
        os.path.join(PROJECT_ROOT, default_dir, filename),
        filename
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return candidates[0]

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation"
]


def create_submission_zip(output_zip_path: str = "code.zip"):
    """
    Zips ONLY the code/ directory (including uv files, cache, and evaluation) and nothing else.
    """
    code_dir = os.path.abspath(os.path.dirname(__file__))
    zip_target = os.path.abspath(output_zip_path)
    print(f"Creating submission package at {zip_target} from {code_dir}...")

    packaged_files = []
    with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(code_dir):
            # Exclude __pycache__
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith((".pyc", ".zip", ".log")):
                    continue
                # Do not package output.csv, lockfiles, or secrets in code.zip
                if f in [".env", "output.csv", "uv.lock"]:
                    continue

                full_path = os.path.join(root, f)
                # Store relative to parent of code_dir so the archive has 'code/...'
                rel_path = os.path.relpath(full_path, os.path.dirname(code_dir))
                zf.write(full_path, rel_path)
                packaged_files.append(rel_path)

    print(f"Successfully packaged {output_zip_path} ({os.path.getsize(zip_target):,} bytes, {len(packaged_files)} files).")



def main():
    print("=" * 75)
    print("HACKERRANK ORCHESTRATE: BUY OR WAIT? - FULL EVALUATION PIPELINE")
    print("=" * 75)

    client = GeminiRateLimitedClient()

    # Step 1: Multimodal OCR Extraction on Missing Amounts (Gemini Flash)
    print("\n[Step 1/6] Running Multimodal OCR on receipt/invoice images...")
    run_ocr_on_all_images(client=client)
    ocr_overrides = get_event_amount_overrides()
    print(f"Loaded {len(ocr_overrides)} extracted amounts for financial events.")

    # Step 2: Batched Message Processing (Gemini Flash-Lite)
    print("\n[Step 2/6] Processing unstructured employer and provider messages...")
    process_all_messages(client=client)
    msg_insights = get_user_and_event_message_insights()
    print(f"Loaded {len(msg_insights.get('message_by_id', {}))} processed message insights.")

    # Step 3: Initialize Financial Engine
    print("\n[Step 3/6] Initializing Deterministic 90-Day Financial Engine...")
    engine = FinancialEngine(
        ocr_overrides=ocr_overrides,
        message_insights=msg_insights.get("message_by_id", {})
    )

    # Step 4: Evaluate All 250 Requests
    requests_path = find_file("requests.csv")
    print(f"Reading requests from: {requests_path}")
    df_requests = pd.read_csv(requests_path)
    total_reqs = len(df_requests)
    print(f"\n[Step 4/6] Evaluating {total_reqs} requests with cash flow simulator...")

    evaluations = []
    for idx, r in df_requests.iterrows():
        pred = engine.evaluate_request(r)
        evaluations.append(pred)

    # Step 5: Batched Decision Explanation Generation (Gemini Flash-Lite)
    print("\n[Step 5/6] Generating grounded decision explanations...")
    explanations = generate_all_explanations(evaluations, client=client)

    # Build Output DataFrame
    output_rows = []
    for ev in evaluations:
        rid = ev["request_id"]
        expl = explanations.get(rid, "")
        output_rows.append({
            "request_id": rid,
            "amount_safe_to_pay": ev["amount_safe_to_pay"],
            "affordability_status": ev["affordability_status"],
            "recommended_payment_method": ev["recommended_payment_method"],
            "payment_plan": ev["payment_plan"],
            "earliest_date_for_full_payment": ev["earliest_date_for_full_payment"],
            "spending_changes_needed": ev["spending_changes_needed"],
            "decision_explanation": expl
        })

    df_output = pd.DataFrame(output_rows)[OUTPUT_COLUMNS]

    # Verification of output schema and writing to output locations
    output_paths = ["output.csv"]
    if os.path.abspath(PROJECT_ROOT) != os.path.abspath("."):
        output_paths.append(os.path.join(PROJECT_ROOT, "output.csv"))
    dataset_dir = os.path.dirname(requests_path)
    if os.path.exists(dataset_dir):
        output_paths.append(os.path.join(dataset_dir, "output.csv"))

    for out_p in set(output_paths):
        df_output.to_csv(out_p, index=False)
        print(f"Wrote predictions to {out_p} ({len(df_output)} rows)")


    # Step 6: Generate Token Usage Report & Packaging
    print("\n[Step 6/6] Generating usage report and submission package...")
    usage_path = os.path.join(PROJECT_ROOT, "evaluation", "usage_report.md")
    client.write_usage_report(usage_path, total_requests=total_reqs)

    # Also write to code/evaluation/usage_report.md
    code_usage_path = os.path.join(PROJECT_ROOT, "code", "evaluation", "usage_report.md")
    client.write_usage_report(code_usage_path, total_requests=total_reqs)

    # Create submission zip
    create_submission_zip(os.path.join(PROJECT_ROOT, "code.zip"))

    print("\n" + "=" * 75)
    print("ALL PIPELINE DELIVERABLES GENERATED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
