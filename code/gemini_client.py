"""
Gemini Client Wrapper with Model Fallbacks, Strict Rate Limiting, and Token Tracking.
Supports:
- OCR: gemini-3.8-flash (fallback: gemini-3.7-flash) with 5 req/min (12.5s sleep)
- Text/Reasoning: gemini-3.5-flash-lite (fallback: gemini-3.1-flash-lite) with 15 req/min (4.2s sleep)
- Automatic 429 retry with exponential backoff
- Usage and token logging for evaluation/usage_report.md
"""

import os
import time
import json
from typing import Optional, Dict, Any, List
from google import genai
from google.genai import types
import dotenv

dotenv.load_dotenv()

# Rate limit pacing (seconds)
FLASH_SLEEP_SECONDS = 12.5        # Max ~4.8 RPM (strictly <= 5 req/min)
FLASH_LITE_SLEEP_SECONDS = 4.2    # Max ~14.2 RPM (strictly <= 15 req/min)

FLASH_PRIMARY = os.environ.get("GEMINI_FLASH_PRIMARY", "gemini-3.8-flash")
FLASH_FALLBACKS = [os.environ.get("GEMINI_FLASH_FALLBACK", "gemini-3.7-flash"), "gemini-2.5-flash"]

FLASH_LITE_PRIMARY = os.environ.get("GEMINI_FLASH_LITE_PRIMARY", "gemini-3.5-flash-lite")
FLASH_LITE_FALLBACKS = [os.environ.get("GEMINI_FLASH_LITE_FALLBACK", "gemini-3.1-flash-lite"), "gemini-2.0-flash-lite"]

# Pricing estimates per million tokens ($)
PRICING = {
    "gemini-3.8-flash": {"input": 0.15, "output": 0.60},
    "gemini-3.7-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-3.5-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-3.1-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
}


class GeminiRateLimitedClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
        
        self.client = genai.Client(api_key=self.api_key)
        self._last_flash_call_time = 0.0
        self._last_flash_lite_call_time = 0.0

        # Usage records: list of dicts
        self.call_history: List[Dict[str, Any]] = []

    def _pace(self, is_flash: bool):
        now = time.time()
        if is_flash:
            elapsed = now - self._last_flash_call_time
            if elapsed < FLASH_SLEEP_SECONDS:
                time.sleep(FLASH_SLEEP_SECONDS - elapsed)
            self._last_flash_call_time = time.time()
        else:
            elapsed = now - self._last_flash_lite_call_time
            if elapsed < FLASH_LITE_SLEEP_SECONDS:
                time.sleep(FLASH_LITE_SLEEP_SECONDS - elapsed)
            self._last_flash_lite_call_time = time.time()

    def generate_content(
        self,
        contents: Any,
        is_flash: bool = False,
        config: Optional[types.GenerateContentConfig] = None,
        max_retries: int = 3
    ) -> str:
        models_to_try = [FLASH_PRIMARY] + FLASH_FALLBACKS if is_flash else [FLASH_LITE_PRIMARY] + FLASH_LITE_FALLBACKS

        last_err = None
        for model in models_to_try:
            for attempt in range(max_retries):
                self._pace(is_flash)
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config
                    )
                    
                    # Record usage
                    input_tokens = 0
                    output_tokens = 0
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

                    rates = PRICING.get(model, {"input": 0.10, "output": 0.40})
                    cost = (input_tokens / 1_000_000 * rates["input"]) + (output_tokens / 1_000_000 * rates["output"])

                    self.call_history.append({
                        "model": model,
                        "is_flash": is_flash,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "cost": cost,
                        "timestamp": time.time()
                    })

                    return response.text or ""
                except Exception as e:
                    err_str = str(e)
                    last_err = e
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait_time = (2 ** (attempt + 1)) * 5
                        print(f"Rate limit 429 on {model}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Error calling {model}: {e}. Trying fallback model...")
                        break  # Try next model in chain

        raise RuntimeError(f"All model attempts failed. Last error: {last_err}")

    def generate_flash(self, contents: Any, config: Optional[types.GenerateContentConfig] = None) -> str:
        """Call Flash model (for OCR)."""
        return self.generate_content(contents=contents, is_flash=True, config=config)

    def generate_flash_lite(self, contents: Any, config: Optional[types.GenerateContentConfig] = None) -> str:
        """Call Flash-Lite model (for reasoning, messages, explanations)."""
        return self.generate_content(contents=contents, is_flash=False, config=config)

    def get_usage_summary(self, total_requests: int = 250) -> Dict[str, Any]:
        """Summarize all recorded token usage."""
        summary_by_model: Dict[str, Dict[str, Any]] = {}
        total_input = 0
        total_output = 0
        total_cost = 0.0

        for record in self.call_history:
            m = record["model"]
            if m not in summary_by_model:
                summary_by_model[m] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0
                }
            summary_by_model[m]["calls"] += 1
            summary_by_model[m]["input_tokens"] += record["input_tokens"]
            summary_by_model[m]["output_tokens"] += record["output_tokens"]
            summary_by_model[m]["total_tokens"] += record["total_tokens"]
            summary_by_model[m]["cost"] += record["cost"]

            total_input += record["input_tokens"]
            total_output += record["output_tokens"]
            total_cost += record["cost"]

        total_tokens = total_input + total_output
        avg_tokens_per_request = total_tokens / max(1, total_requests)
        avg_cost_per_request = total_cost / max(1, total_requests)

        return {
            "by_model": summary_by_model,
            "total_calls": len(self.call_history),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "avg_tokens_per_request": avg_tokens_per_request,
            "total_cost": total_cost,
            "avg_cost_per_request": avg_cost_per_request
        }

    def write_usage_report(self, filepath: str, total_requests: int = 250):
        """Generates evaluation/usage_report.md matching challenge contract §6.5."""
        summary = self.get_usage_summary(total_requests)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        lines = [
            "# Final Token Usage & Cost Report",
            "",
            f"**Total Evaluation Requests**: {total_requests}",
            f"**Total Model Calls**: {summary['total_calls']}",
            f"**Total Tokens**: {summary['total_tokens']:,} (Input: {summary['total_input_tokens']:,}, Output: {summary['total_output_tokens']:,})",
            f"**Average Tokens Per Request**: {summary['avg_tokens_per_request']:,.2f}",
            f"**Estimated Total Cost**: ${summary['total_cost']:.4f}",
            f"**Estimated Cost Per Request**: ${summary['avg_cost_per_request']:.6f}",
            "",
            "## Model Breakdown",
            "",
            "| Provider | Model Name | Calls | Input Tokens | Output Tokens | Total Tokens | Estimated Cost ($) |",
            "|---|---|---|---|---|---|---|"
        ]

        for model, data in summary["by_model"].items():
            lines.append(
                f"| Google Gemini | `{model}` | {data['calls']} | {data['input_tokens']:,} | {data['output_tokens']:,} | {data['total_tokens']:,} | ${data['cost']:.4f} |"
            )

        lines.extend([
            "",
            "## Methodological Summary",
            "",
            "- **Multimodal OCR**: Executed using Gemini Flash for the 16 missing financial event amounts, paced at 5 req/min and cached persistently to disk.",
            "- **Message Processing**: Unstructured payroll/refund messages processed in batched prompts using Gemini Flash-Lite under the 15 req/min rate limit.",
            "- **Decision Explanations**: Generated in batched calls using Gemini Flash-Lite, strictly grounded on deterministic calculations.",
            "- **Financial Calculations**: Executed deterministically with zero token overhead.",
            ""
        ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Wrote usage report to {filepath}")
