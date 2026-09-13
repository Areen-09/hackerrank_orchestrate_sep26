# Final Token Usage & Cost Report

**Total Evaluation Requests**: 250
**Total Model Calls**: 0
**Total Tokens**: 0 (Input: 0, Output: 0)
**Average Tokens Per Request**: 0.00
**Estimated Total Cost**: $0.0000
**Estimated Cost Per Request**: $0.000000

## Model Breakdown

| Provider | Model Name | Calls | Input Tokens | Output Tokens | Total Tokens | Estimated Cost ($) |
|---|---|---|---|---|---|---|

## Methodological Summary

- **Multimodal OCR**: Executed using Gemini Flash for the 16 missing financial event amounts, paced at 5 req/min and cached persistently to disk.
- **Message Processing**: Unstructured payroll/refund messages processed in batched prompts using Gemini Flash-Lite under the 15 req/min rate limit.
- **Decision Explanations**: Generated in batched calls using Gemini Flash-Lite, strictly grounded on deterministic calculations.
- **Financial Calculations**: Executed deterministically with zero token overhead.
