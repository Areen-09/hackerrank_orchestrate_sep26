"""
Batched Decision Explanation Generator for HackerRank Orchestrate.
Uses Gemini 3.5 Flash-Lite (fallback 3.1) in batches of 5 requests to produce concise,
grounded decision_explanation strings matching sample_requests.csv style.
Results are cached and checkpointed in cache/explanation_cache.json.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional

try:
    from code.gemini_client import GeminiRateLimitedClient
    from code.guardrails import (
        UNTRUSTED_CONTENT_SYSTEM_INSTRUCTION,
        validate_explanation_grounding,
        generate_fallback_explanation
    )
except (ImportError, ModuleNotFoundError):
    from gemini_client import GeminiRateLimitedClient
    from guardrails import (
        UNTRUSTED_CONTENT_SYSTEM_INSTRUCTION,
        validate_explanation_grounding,
        generate_fallback_explanation
    )

def resolve_cache_path(filename: str = "explanation_cache.json") -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "cache", filename),
        os.path.join(os.path.dirname(__file__), "..", "cache", filename),
        os.path.join("cache", filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

CACHE_PATH = resolve_cache_path("explanation_cache.json")


def parse_explanation_batch_response(raw_text: str) -> List[Dict[str, str]]:
    """Parses JSON array of explanations."""
    match = re.search(r"\[\s*\{.*?\}\s*\]", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # Try matching individual JSON objects
    results = []
    for m in re.finditer(r"\{\s*\"request_id\"\s*:\s*\"([^\"]+)\"\s*,\s*\"decision_explanation\"\s*:\s*\"([^\"]+)\"\s*\}", raw_text):
        results.append({
            "request_id": m.group(1),
            "decision_explanation": m.group(2)
        })
    return results


def generate_all_explanations(
    evaluations: List[Dict[str, Any]],
    cache_path: str = CACHE_PATH,
    client: Optional[GeminiRateLimitedClient] = None,
    batch_size: int = 5
) -> Dict[str, str]:
    """
    Batches computed financial evaluation records to generate grounded explanations.
    Checkpoints progress to disk.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache: Dict[str, str] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load explanation cache: {e}. Starting fresh.")

    # Filter ungenerated
    needed = [ev for ev in evaluations if ev["request_id"] not in cache]
    if not needed:
        print(f"All {len(evaluations)} explanations are already cached in {cache_path}.")
        return cache

    if client is None:
        client = GeminiRateLimitedClient()

    print(f"Generating explanations for {len(needed)} requests in batches of {batch_size}...")

    system_prompt = (
        f"{UNTRUSTED_CONTENT_SYSTEM_INSTRUCTION}\n"
        "You are a professional financial advisor providing clear, concise, grounded recommendations.\n"
        "Given the mathematically calculated recommendation for each request, generate a 1-2 sentence decision_explanation.\n"
        "Style Guidelines:\n"
        "- For affordable_now: 'Pay <CURRENCY> <AMOUNT> today. This leaves at least <CURRENCY> <MIN_BALANCE> available over the next 90 days.'\n"
        "- For installments: 'Use <N> installments of <CURRENCY> <INST_AMT>, starting <FIRST_DATE>. This leaves at least <CURRENCY> <MIN_BALANCE> available.'\n"
        "- For wait: 'Pay <CURRENCY> <AMOUNT> in full on <DATE>. Paying earlier would take the balance below the <CURRENCY> <MIN_BALANCE> minimum.'\n"
        "- For not_affordable: 'Do not make this payment by <DATE>. None of the available options keeps the <CURRENCY> <MIN_BALANCE> minimum protected.'\n"
        "- For spending changes: 'Stop <item> [or Reduce <item> to <amount>], then pay <CURRENCY> <AMOUNT> today. This leaves at least <CURRENCY> <MIN_BALANCE> available.'\n"
        "Output STRICT JSON array of objects:\n"
        '[{"request_id": "<request_id>", "decision_explanation": "<concise explanation>"}]\n'
    )

    for i in range(0, len(needed), batch_size):
        batch = needed[i : i + batch_size]
        items_text = []
        for ev in batch:
            items_text.append(
                f"- RequestID: {ev['request_id']}\n"
                f"  Currency: {ev['home_currency']}\n"
                f"  Status: {ev['affordability_status']}\n"
                f"  Method: {ev['recommended_payment_method']}\n"
                f"  SafeAmount: {ev['amount_safe_to_pay']}\n"
                f"  RequestedAmount: {ev['requested_amount']}\n"
                f"  MinBalance: {ev['minimum_balance_to_keep']}\n"
                f"  Plan: {ev['payment_plan']}\n"
                f"  EarliestFullDate: {ev['earliest_date_for_full_payment']}\n"
                f"  SpendingChanges: {ev['spending_changes_needed']}\n"
                f"  CompletionDate: {ev['desired_completion_date']}"
            )

        prompt_content = (
            system_prompt + "\n" +
            "Requests to explain:\n\n" +
            "\n\n".join(items_text) +
            "\n\nOutput JSON array now:"
        )

        try:
            raw_response = client.generate_flash_lite(contents=prompt_content)
            parsed = parse_explanation_batch_response(raw_response)

            for item in parsed:
                rid = item.get("request_id")
                expl = item.get("decision_explanation")
                if rid and expl:
                    # Match with source ev
                    matching_ev = next((e for e in batch if e["request_id"] == rid), None)
                    if matching_ev:
                        grounded_expl = validate_explanation_grounding(
                            expl,
                            matching_ev["affordability_status"],
                            matching_ev["recommended_payment_method"],
                            matching_ev["amount_safe_to_pay"],
                            matching_ev["minimum_balance_to_keep"],
                            matching_ev["home_currency"]
                        )
                        cache[rid] = grounded_expl
                    else:
                        cache[rid] = expl

            # Fill any batch item that didn't get a response with fallback
            for ev in batch:
                rid = ev["request_id"]
                if rid not in cache:
                    cache[rid] = generate_fallback_explanation(
                        ev["affordability_status"],
                        ev["recommended_payment_method"],
                        ev["amount_safe_to_pay"],
                        ev["minimum_balance_to_keep"],
                        ev["home_currency"]
                    )

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)

            print(f"Generated explanations batch {i // batch_size + 1}/{(len(needed) + batch_size - 1) // batch_size}")

        except Exception as e:
            print(f"Error generating explanations for batch starting at {i}: {e}")
            # Use deterministic fallback for this batch
            for ev in batch:
                rid = ev["request_id"]
                if rid not in cache:
                    cache[rid] = generate_fallback_explanation(
                        ev["affordability_status"],
                        ev["recommended_payment_method"],
                        ev["amount_safe_to_pay"],
                        ev["minimum_balance_to_keep"],
                        ev["home_currency"]
                    )
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)

    return cache
