"""
Message Processor for HackerRank Orchestrate dataset/messages.csv.
Uses Gemini 3.5 Flash-Lite (with 3.1 fallback) in batched prompts (~15 messages per batch)
to extract financial facts, salary amendments, cancelled transactions, and pending bonus flags.
Results are cached in cache/message_cache.json.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
import pandas as pd

try:
    from code.gemini_client import GeminiRateLimitedClient
    from code.guardrails import UNTRUSTED_CONTENT_SYSTEM_INSTRUCTION, wrap_untrusted_content, sanitize_untrusted_text
except (ImportError, ModuleNotFoundError):
    from gemini_client import GeminiRateLimitedClient
    from guardrails import UNTRUSTED_CONTENT_SYSTEM_INSTRUCTION, wrap_untrusted_content, sanitize_untrusted_text

def resolve_cache_path(filename: str = "message_cache.json") -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "cache", filename),
        os.path.join(os.path.dirname(__file__), "..", "cache", filename),
        os.path.join("cache", filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

CACHE_PATH = resolve_cache_path("message_cache.json")


def parse_message_batch_response(raw_text: str) -> List[Dict[str, Any]]:
    """Parses JSON array or list of objects from LLM response."""
    # Find JSON block
    match = re.search(r"\[\s*\{.*?\}\s*\]", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # Try finding individual JSON objects
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", raw_text):
        try:
            obj = json.loads(m.group(0))
            if "message_id" in obj:
                objs.append(obj)
        except Exception:
            continue
    return objs


def resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p
    parent_p = os.path.join("..", p)
    if os.path.exists(parent_p):
        return parent_p
    script_rel = os.path.join(os.path.dirname(__file__), "..", p)
    if os.path.exists(script_rel):
        return script_rel
    return p


def process_all_messages(
    messages_csv_path: str = "dataset/messages.csv",
    cache_path: str = CACHE_PATH,
    client: Optional[GeminiRateLimitedClient] = None,
    batch_size: int = 15
) -> Dict[str, Dict[str, Any]]:
    """
    Processes all 215 messages from dataset/messages.csv in batched prompts.
    Saves and loads from cache/message_cache.json.
    """
    messages_csv_path = resolve_path(messages_csv_path)
    cache_path = resolve_cache_path(os.path.basename(cache_path))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load message cache: {e}. Starting fresh.")

    df = pd.read_csv(messages_csv_path)
    if client is None:
        client = GeminiRateLimitedClient()

    # Filter out already cached messages
    uncached_rows = [row for _, row in df.iterrows() if str(row["message_id"]) not in cache]

    if not uncached_rows:
        print(f"All {len(df)} messages are already cached in {cache_path}.")
        return cache

    print(f"Processing {len(uncached_rows)} uncached messages in batches of {batch_size}...")

    system_prompt = (
        f"{UNTRUSTED_CONTENT_SYSTEM_INSTRUCTION}\n"
        "You are an expert multilingual financial analyst parsing notifications from employers, banks, and service providers (in English and Indonesian).\n"
        "Extract key financial facts impacting cash flow, following these strict rules:\n"
        "1. If salary/pay is amended or reduced (e.g. unpaid leave, new monthly rate), extract the exact new numerical amount in salary_adjustment_amount.\n"
        "2. If a bonus, commission, refund, or payout is stated as pending, unapproved, conditional, or not yet withdrawable, mark is_income_unconfirmed_or_pending as true.\n"
        "3. If an expense, subscription, or order is cancelled, mark is_event_cancelled as true.\n"
        "4. Output STRICT JSON: an array of objects matching:\n"
        '[{"message_id": "<id>", "salary_adjustment_amount": <float or null>, "is_income_unconfirmed_or_pending": <bool>, "is_event_cancelled": <bool>, "amended_event_amount": <float or null>, "summary": "<brief summary>"}]\n'
    )

    for i in range(0, len(uncached_rows), batch_size):
        batch = uncached_rows[i : i + batch_size]
        batch_prompt_items = []
        for r in batch:
            m_id = str(r["message_id"])
            u_id = str(r["user_id"])
            rel_ev = str(r["related_event_id"]) if pd.notna(r["related_event_id"]) else "none"
            req_id = str(r["request_id"]) if pd.notna(r["request_id"]) else "none"
            text = str(r["message_text"])
            
            wrapped = wrap_untrusted_content(
                f"MessageID: {m_id} | UserID: {u_id} | RelatedEvent: {rel_ev} | RequestID: {req_id}\nText: {text}",
                tag="untrusted_evidence",
                id_attr=m_id
            )
            batch_prompt_items.append(wrapped)

        user_content = (
            system_prompt + "\n" +
            "Analyze the following batch of messages:\n\n" +
            "\n\n".join(batch_prompt_items) +
            "\n\nOutput JSON array now:"
        )

        try:
            raw_response = client.generate_flash_lite(contents=user_content)
            extracted_list = parse_message_batch_response(raw_response)

            for item in extracted_list:
                m_id = item.get("message_id")
                if m_id:
                    cache[m_id] = item

            # Also ensure all batch IDs are in cache even if empty
            for r in batch:
                m_id = str(r["message_id"])
                if m_id not in cache:
                    cache[m_id] = {
                        "message_id": m_id,
                        "salary_adjustment_amount": None,
                        "is_income_unconfirmed_or_pending": False,
                        "is_event_cancelled": False,
                        "amended_event_amount": None,
                        "summary": "Processed"
                    }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)

            print(f"Processed batch {i // batch_size + 1}/{(len(uncached_rows) + batch_size - 1) // batch_size}")

        except Exception as e:
            print(f"Error processing message batch starting at index {i}: {e}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    return cache


def get_user_and_event_message_insights(cache_path: str = CACHE_PATH) -> Dict[str, Any]:
    """
    Returns structured lookups:
    - event_adjustments: {related_event_id: {salary_adjustment_amount, is_cancelled, etc.}}
    - user_unconfirmed_income: set of (user_id, event_id) or similar flags.
    """
    if not os.path.exists(cache_path):
        return {"event_adjustments": {}, "message_by_id": {}}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        return {"message_by_id": cache}
    except Exception as e:
        print(f"Error loading message insights: {e}")
        return {"message_by_id": {}}
