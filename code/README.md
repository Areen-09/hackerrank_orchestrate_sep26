# Buy or Wait? — AI Financial Decision Agent
**HackerRank Orchestrate (September 2026)**

An AI-powered, hybrid financial decision system that evaluates whether a user can safely afford a requested purchase. The agent accounts for current balance, recurring expenses, pending transactions, essential vs. flexible spending, confirmed salary, seller payment options, dated currency conversions, and evidence from messages and receipt images.

---

## 1. Approach & Architecture Overview

Our solution employs a **hybrid architecture** combining deterministic financial simulation with multimodal generative AI:

1. **Multimodal OCR Extractor (`ocr_extractor.py`)**:
   - Uses `gemini-3.8-flash` (with `gemini-3.7-flash` fallback) to extract financial figures from bills, payroll letters, and receipts.
   - Fully cached in `cache/ocr_cache.json` for offline reproducibility and instant execution.

2. **Unstructured Message Processor (`message_processor.py`)**:
   - Uses `gemini-3.5-flash-lite` (with `gemini-3.1-flash-lite` fallback) to parse employer/provider messages into structured financial modifications (salary dates, bonus settlements, fee waivers).
   - Fully cached in `cache/message_cache.json`.

3. **Prompt Injection & Security Guardrails (`guardrails.py`)**:
   - Validates all untrusted messages, image text, and user inputs against injection attacks, system override attempts, and malicious directives before inference.

4. **Deterministic 90-Day Cash Flow Engine (`financial_engine.py`)**:
   - Simulates day-by-day cash balance across 90 days from `request_date`.
   - Normalizes foreign currency transactions using fixed historical rates from `exchange_rates.csv`.
   - Reserves pending debits immediately; ignores unconfirmed pending credits.
   - Enforces `minimum_balance_to_keep` strictly on all days.
   - Optimizes payment recommendations (`full_payment`, `partial_payment`, `installments`, `wait`, `not_recommended`).
   - Identifies permitted spending reductions on non-protected flexible recurring subscriptions.

5. **Grounded Explanation Generator (`explanation_generator.py`)**:
   - Generates concise, grounded explanations citing exact financial facts, balances, dates, and decision rationales.
   - Cached in `cache/explanation_cache.json`.

6. **Token Usage & Evaluation (`evaluation/usage_report.md`)**:
   - Tracks all API calls, token counts, and cost metrics in accordance with competition requirements.

---

## 2. Setup & Installation

### Requirements
- Python 3.10 or higher
- Standard pip or uv

### Install Dependencies
```bash
pip install -r requirements.txt
```
*Dependencies:* `google-genai>=2.23.0`, `pandas>=3.0.5`, `pillow>=12.3.0`, `pydantic>=2.13.5`, `python-dotenv>=1.2.3`.

---

## 3. Configuration & API Keys

If running online to make new Gemini API calls, create a `.env` file or export the environment variable:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

> **Note**: For offline grading and testing, all model responses (OCR, message insights, and explanations) are pre-computed and stored in `cache/`. Running `main.py` runs 100% offline in < 5 seconds without needing an API key.

---

## 4. How to Run

### Standalone Run (from inside `code/`)
```bash
cd code
python main.py
```

### Or Run from Repository Root
```bash
python code/main.py
```

### Pipeline Execution Steps:
1. Loads OCR extractions from `cache/ocr_cache.json` (or calls Gemini Flash if cache missing).
2. Loads parsed message insights from `cache/message_cache.json`.
3. Reconstructs user balances and runs deterministic 90-day cash flow simulation.
4. Generates/loads grounded explanations for all 250 requests.
5. Emits predictions to `output.csv` with the exact required 8-column schema:
   `request_id,amount_safe_to_pay,affordability_status,recommended_payment_method,payment_plan,earliest_date_for_full_payment,spending_changes_needed,decision_explanation`
6. Updates `evaluation/usage_report.md` with complete token and cost tracking.

---

## 5. Directory Structure

```text
code/
├── README.md                      # Setup and approach overview
├── requirements.txt               # Dependencies for pip
├── pyproject.toml                 # Package metadata
├── .python-version                # Target Python version (3.12)
├── main.py                        # Main orchestration pipeline entry point
├── financial_engine.py            # Deterministic 90-day cash flow simulator
├── gemini_client.py               # Rate-limited Gemini client with fallbacks
├── guardrails.py                  # Prompt injection prevention guardrails
├── ocr_extractor.py               # Multimodal image OCR extraction
├── message_processor.py           # Message parsing and entity extraction
├── explanation_generator.py       # Grounded financial decision explanations
├── test_guardrails.py             # Security test suite
├── cache/                         # Offline deterministic cache
│   ├── ocr_cache.json             # 16 extracted invoice/receipt amounts
│   ├── message_cache.json         # 215 parsed message insights
│   └── explanation_cache.json     # 250 decision explanations
└── evaluation/                    # Token usage and cost analysis
    ├── usage_report.md            # Competition usage report
    ├── main.py                    # Evaluation report generator
    └── __init__.py
```
