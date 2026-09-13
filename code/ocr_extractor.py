"""
OCR Extractor for HackerRank Orchestrate dataset/media/images.
Extracts missing financial event amounts from the 16 receipt/invoice images using Gemini Flash.
Results are cached permanently in cache/ocr_cache.json.
"""

import os
import json
import re
from typing import Dict, Any, Optional
import pandas as pd
from PIL import Image

try:
    from code.gemini_client import GeminiRateLimitedClient
    from code.guardrails import UNTRUSTED_CONTENT_SYSTEM_INSTRUCTION
except (ImportError, ModuleNotFoundError):
    from gemini_client import GeminiRateLimitedClient
    from guardrails import UNTRUSTED_CONTENT_SYSTEM_INSTRUCTION

def resolve_cache_path(filename: str = "ocr_cache.json") -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "cache", filename),
        os.path.join(os.path.dirname(__file__), "..", "cache", filename),
        os.path.join("cache", filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

CACHE_PATH = resolve_cache_path("ocr_cache.json")


def extract_amount_from_text(raw_text: str) -> Optional[float]:
    """Helper to extract numerical amount from unstructured text or JSON."""
    # Try finding JSON first
    json_match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if "amount" in data and data["amount"] is not None:
                val = str(data["amount"]).replace(",", "").strip()
                return float(val)
        except Exception:
            pass

    # Fallback to regex pattern matching on currencies/numbers
    matches = re.findall(r"(?:total|amount|rs\.?|inr|\$|eur|€|idr|zar)?\s*([\d,]+(?:\.\d{1,2})?)", raw_text, re.IGNORECASE)
    for m in matches:
        clean = m.replace(",", "").strip()
        try:
            val = float(clean)
            if val > 0:
                return val
        except ValueError:
            continue

    return None


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


def run_ocr_on_all_images(
    images_csv_path: str = "dataset/images.csv",
    media_dir: str = "dataset/media/images",
    cache_path: str = CACHE_PATH,
    client: Optional[GeminiRateLimitedClient] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Processes all 16 images in images.csv.
    Uses cached values where available, and calls Gemini Flash for missing entries.
    """
    images_csv_path = resolve_path(images_csv_path)
    media_dir = resolve_path(media_dir)
    cache_path = resolve_cache_path(os.path.basename(cache_path))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load OCR cache: {e}. Starting fresh.")

    df_images = pd.read_csv(images_csv_path)
    if client is None:
        client = GeminiRateLimitedClient()

    prompt = (
        f"{UNTRUSTED_CONTENT_SYSTEM_INSTRUCTION}\n"
        "You are an OCR and financial document extraction specialist.\n"
        "Analyze this receipt / invoice / account statement image.\n"
        "Extract the exact transaction total amount, currency, and date.\n"
        "Output ONLY a JSON object with this exact schema:\n"
        '{"amount": <float>, "currency": "<currency code e.g. INR, USD, EUR, IDR, ZAR>", "date": "<YYYY-MM-DD or null>", "description": "<short text>"}\n'
        "Do not include markdown blocks, notes, or extra commentary."
    )

    updated = False
    for _, row in df_images.iterrows():
        image_id = str(row["image_id"])
        related_event_id = str(row["related_event_id"])
        
        if image_id in cache and cache[image_id].get("amount") is not None:
            # Already cached
            continue

        image_filename = f"{image_id}.png"
        image_path = os.path.join(media_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Image not found on disk: {image_path}")
            continue

        print(f"Extracting OCR for {image_id} ({related_event_id})...")
        try:
            img = Image.open(image_path)
            raw_response = client.generate_flash(contents=[prompt, img])
            amount = extract_amount_from_text(raw_response)

            cache[image_id] = {
                "image_id": image_id,
                "related_event_id": related_event_id,
                "user_id": str(row.get("user_id", "")),
                "request_id": str(row.get("request_id", "")),
                "amount": amount,
                "raw_response": raw_response
            }
            updated = True
            print(f"-> {image_id} ({related_event_id}): extracted amount = {amount}")

            # Save immediately to cache
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)

        except Exception as e:
            print(f"Error processing image {image_id}: {e}")

    if updated:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    return cache


def get_event_amount_overrides(cache_path: str = CACHE_PATH) -> Dict[str, float]:
    """
    Returns a dictionary mapping related_event_id -> extracted amount.
    Used by financial_engine to fill blank amounts in financial_events.csv.
    """
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        overrides = {}
        for img_id, data in cache.items():
            ev_id = data.get("related_event_id")
            amt = data.get("amount")
            if ev_id and amt is not None:
                overrides[ev_id] = float(amt)
        return overrides
    except Exception as e:
        print(f"Failed to load event overrides from OCR cache: {e}")
        return {}
