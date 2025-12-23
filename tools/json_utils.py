# tools/json_utils.py
from __future__ import annotations

import json
import re
from typing import Any, Dict


def _extract_json_object(text: str) -> str:
    if not text:
        return ""
    text = text.strip()

    # strip ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # take first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _repair_jsonish(s: str) -> str:
    """
    Repairs common LLM "JSON-ish" mistakes:
    - Missing commas between fields/objects
    - NULL -> null
    - Arrays written like [ 0:{...} 1:{...} ] -> [ {...}, {...} ]
    - Missing commas between adjacent objects
    """
    if not s:
        return s

    s = s.strip()

    # NULL/True/False sometimes appear
    s = re.sub(r"\bNULL\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)

    # Convert [ 0:{...} 1:{...} ] style into [{...} {...}]
    # Replace "0:{" or "12:{" that occur after [ or , with "{"
    s = re.sub(r"(\[|\s|,)\s*\d+\s*:\s*\{", r"\1{", s)

    # Add missing commas between:
    #   ... } "nextKey": ...
    #   ... ] "nextKey": ...
    #   ... "value" "nextKey": ...
    # This regex inserts a comma when a value/brace/bracket/quote is followed by a quoted key
    s = re.sub(r'(?<=[0-9"\}\]])\s*(?="[^"]+"\s*:)', ",", s)

    # Add missing commas between adjacent objects/arrays in lists:
    #   } {  -> }, {
    #   ] {  -> ], {
    s = re.sub(r"\}\s*\{", "},{", s)
    s = re.sub(r"\]\s*\{", "],{", s)

    return s


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Best-effort parse:
    - extracts first JSON object
    - repairs common JSON-ish errors
    - json.loads
    Returns dict or {"error":..., "raw":...}
    """
    raw = _extract_json_object(text)
    if not raw:
        return {"error": "No JSON object found in model output.", "raw": (text or "")[:5000]}

    repaired = _repair_jsonish(raw)

    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj
        return {"error": "Parsed JSON but not an object.", "raw": repaired[:5000]}
    except Exception as e:
        return {"error": f"JSON parse failed after repair: {e}", "raw": repaired[:5000]}
