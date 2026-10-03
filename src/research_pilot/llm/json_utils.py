import ast
import json
import re
from typing import Any, Dict, List


def parse_json_object(text: str) -> Dict[str, Any] | None:
    """Extract the first JSON object from model output, tolerating fences and prose."""
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and _is_json_compatible(parsed):
                return parsed
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict) and _is_json_compatible(parsed):
                return parsed
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            continue
    return None


def extract_code_block(text: str, language: str = "python") -> str:
    """Return the first fenced code block for the language (or any fence) from text."""
    pattern = rf"```(?:{language}|py)\s*\n([\s\S]*?)```"
    match = re.search(pattern, text or "", flags=re.IGNORECASE)
    if match:
        return match.group(1).strip("\n")
    generic = re.search(r"```[a-zA-Z0-9_-]*\s*\n([\s\S]*?)```", text or "")
    if generic:
        return generic.group(1).strip("\n")
    return ""


def _json_candidates(text: str) -> List[str]:
    s = (text or "").strip()
    out: List[str] = []
    out.extend(re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, flags=re.IGNORECASE))
    out.append(s.strip("`"))
    balanced = _extract_balanced_object(s)
    if balanced:
        out.append(balanced)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        out.append(s[start : end + 1])
    seen = set()
    uniq = []
    for item in out:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            uniq.append(key)
    return uniq


def _extract_balanced_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            in_str, escaped = _string_step(ch, escaped)
            continue
        in_str = ch == '"'
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if ch == "}" and depth == 0:
            return text[start : idx + 1]
    return ""


def _string_step(ch: str, escaped: bool) -> tuple[bool, bool]:
    """Advance one character inside a string literal; returns (still in the string, escape pending)."""
    if escaped:
        return True, False
    if ch == "\\":
        return True, True
    return ch != '"', False


def _is_json_compatible(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_compatible(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_compatible(v) for k, v in value.items())
    return False
