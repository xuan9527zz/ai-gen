# -*- coding: utf-8 -*-
"""Resolve fuzzy character requests into verified Danbooru character tags.

The local language model is allowed to propose candidates, but only an exact
NAID Tag Suggest match in the ``character`` category may be applied
automatically.  Existing character tags are removed only when they are both
present in the supplied prompt and independently verified as character tags.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Iterable, List
from urllib.parse import urljoin

import requests

from .naid_verifier import NAID_SEARCH_URL, lookup_naid_exact


DEFAULT_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

MAX_CANDIDATES = 5
MAX_EXISTING_TAGS = 6


CHARACTER_SYSTEM_PROMPT = r"""
You resolve a user's fuzzy character request into real Danbooru character-tag
candidates for Illustrious image generation.

Return JSON only:
{
  "candidate_tags": ["complete character tag"],
  "existing_character_tags": ["exact fragment copied from existing prompt"],
  "note": "short Chinese explanation"
}

Rules:
- Understand Chinese, Japanese, English, abbreviations, common localized
  franchise names, omitted punctuation, and small naming mistakes.
- Map aliases semantically, not word-for-word.  宝可梦/神奇宝贝/口袋妖怪 mean
  Pokemon.  火焰纹章 and 火焰之纹章 mean Fire Emblem.
- A candidate must be the complete likely Danbooru character tag, normally in
  the disambiguated form "character name (franchise)" when that form exists.
- Example: 宝可梦 莉莉艾 -> ["lillie (pokemon)"], not separate "pokemon" and
  "lillie" tags.
- Return at most five candidates, most likely first.  If uncertain, return
  multiple candidates.  Do not invent a character when the request is too vague.
- existing_character_tags may contain only comma-separated fragments copied
  exactly from EXISTING PROMPT that identify another named character.
- Do not classify 1girl, hair, clothing, body, pose, artist, style, species, or
  generic franchise/copyright tags as existing character tags.
""".strip()


def _normalize_tag(value: Any) -> str:
    return " ".join(
        str(value or "")
        .strip()
        .lower()
        .replace("_", " ")
        .split()
    )


def _split_prompt(text: str) -> List[str]:
    return [
        " ".join(part.strip().split())
        for part in re.split(r"[\n,]+", str(text or ""))
        if part.strip()
    ]


def _dedupe(values: Iterable[Any], *, limit: int) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        tag = _normalize_tag(value)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        output.append(tag)
        if len(output) >= limit:
            break
    return output


def _json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise ValueError("角色解析模型没有返回 JSON 对象。")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("角色解析模型返回的 JSON 不是对象。")
    return value


def _is_verified_character(result: Dict[str, Any]) -> bool:
    return (
        bool(result.get("exact_match"))
        and str(result.get("status", "")) == "naid_verified"
        and str(result.get("category", "")).strip().lower() == "character"
    )


def _model_plan(
    query: str,
    base_prompt: str,
    *,
    model: str,
    ollama_url: str,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": CHARACTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CHARACTER REQUEST:\n{query}\n\n"
                    f"EXISTING PROMPT:\n{base_prompt}\n\n"
                    "Return the character-resolution JSON."
                ),
            },
        ],
        "options": {"temperature": 0.1},
    }
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    return _json_object(content)


def _naid_character_suggestions(
    candidates: Iterable[str],
) -> List[Dict[str, Any]]:
    """Use the verifier site's autocomplete only to refine failed candidates."""
    output: List[Dict[str, Any]] = []
    seen = set()
    endpoint = urljoin(NAID_SEARCH_URL, "/search")

    search_terms: List[str] = []
    for candidate in list(candidates)[:3]:
        term = re.sub(r"\s*\([^)]*\)\s*$", "", candidate).strip()
        if term:
            search_terms.append(term)
            # A translated name can be partially right but misspelled as a
            # whole ("edel jat").  The autocomplete index can still recover
            # the canonical full name from a distinctive token ("edel").
            search_terms.extend(
                token
                for token in re.findall(r"[a-z0-9]+", term.lower())
                if len(token) >= 4
            )

    for term in _dedupe(search_terms, limit=6):
        response = requests.get(
            endpoint,
            params={"term": term},
            timeout=20,
        )
        response.raise_for_status()
        rows = response.json().get("tags", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("d_category", "")).lower() != "character":
                continue
            tag = _normalize_tag(row.get("tag_name", ""))
            if not tag or tag in seen:
                continue
            seen.add(tag)
            output.append(
                {
                    "tag": tag,
                    "power": row.get("n_count"),
                    "danbooru_count": row.get("d_count"),
                }
            )
            if len(output) >= 20:
                return output
    return output


def _choose_naid_suggestion(
    *,
    query: str,
    failed_candidates: List[str],
    suggestions: List[Dict[str, Any]],
    model: str,
    ollama_url: str,
) -> str:
    allowed = [str(row.get("tag", "")) for row in suggestions]
    if not allowed:
        return ""
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Select the Danbooru character tag that exactly matches the "
                    "user's requested character. You may select only one exact "
                    "string from ALLOWED TAGS. Account for Chinese/localized "
                    "franchise aliases and full official names. If uncertain, "
                    "return an empty string. Return JSON only: "
                    '{"selected_tag":""}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"REQUEST:\n{query}\n\n"
                    f"FAILED CANDIDATES:\n{json.dumps(failed_candidates)}\n\n"
                    "ALLOWED TAGS:\n"
                    f"{json.dumps(allowed, ensure_ascii=False)}"
                ),
            },
        ],
        "options": {"temperature": 0.0},
    }
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    selected = _normalize_tag(_json_object(content).get("selected_tag", ""))
    allowed_by_key = {_normalize_tag(tag): tag for tag in allowed}
    return allowed_by_key.get(selected, "")


def resolve_character(
    *,
    query: str,
    base_prompt: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    lookup: Callable[[str], Dict[str, Any]] = lookup_naid_exact,
) -> Dict[str, Any]:
    """Resolve and verify one character override request."""
    query = str(query or "").strip()
    if not query:
        raise ValueError("请输入作品名和人物名，例如：宝可梦 莉莉艾。")

    plan = _model_plan(
        query,
        str(base_prompt or ""),
        model=str(model or DEFAULT_MODEL).strip(),
        ollama_url=ollama_url,
    )

    raw_candidates = plan.get("candidate_tags", [])
    if isinstance(raw_candidates, str):
        raw_candidates = [raw_candidates]
    candidates = _dedupe(
        raw_candidates if isinstance(raw_candidates, list) else [],
        limit=MAX_CANDIDATES,
    )

    prompt_fragments = _split_prompt(base_prompt)
    fragment_by_key = {
        _normalize_tag(fragment): fragment for fragment in prompt_fragments
    }
    raw_existing = plan.get("existing_character_tags", [])
    if isinstance(raw_existing, str):
        raw_existing = [raw_existing]
    proposed_existing = _dedupe(
        raw_existing if isinstance(raw_existing, list) else [],
        limit=MAX_EXISTING_TAGS,
    )

    attempts: List[Dict[str, Any]] = []
    resolved_tag = ""
    verification: Dict[str, Any] = {}
    for candidate in candidates:
        checked = dict(lookup(candidate))
        attempts.append(checked)
        if _is_verified_character(checked):
            resolved_tag = _normalize_tag(checked.get("tag", candidate))
            verification = checked
            break

    # A local model often knows the right character but emits a shortened tag
    # (for example ``edelgard (fire emblem)`` instead of the canonical full
    # name).  In that case, search the same NAID index for bounded suggestions,
    # let the model choose only from that allow-list, then exact-verify again.
    if candidates and not resolved_tag:
        try:
            suggestions = _naid_character_suggestions(candidates)
            selected = _choose_naid_suggestion(
                query=query,
                failed_candidates=candidates,
                suggestions=suggestions,
                model=str(model or DEFAULT_MODEL).strip(),
                ollama_url=ollama_url,
            )
        except Exception:
            # Autocomplete is only a refinement path.  Exact verification above
            # remains authoritative and lookup outages must fail closed.
            selected = ""
        if selected and selected not in candidates:
            candidates.append(selected)
            checked = dict(lookup(selected))
            attempts.append(checked)
            if _is_verified_character(checked):
                resolved_tag = _normalize_tag(checked.get("tag", selected))
                verification = checked

    remove_tags: List[str] = []
    removal_verifications: List[Dict[str, Any]] = []
    for proposed in proposed_existing:
        original_fragment = fragment_by_key.get(_normalize_tag(proposed))
        if not original_fragment:
            continue
        checked = dict(lookup(proposed))
        if not _is_verified_character(checked):
            continue
        if _normalize_tag(checked.get("tag")) == _normalize_tag(resolved_tag):
            continue
        remove_tags.append(original_fragment)
        removal_verifications.append(checked)

    suggestion = ""
    if not resolved_tag:
        for checked in attempts:
            if (
                checked.get("exact_match")
                and str(checked.get("category", "")).lower() == "character"
            ):
                suggestion = _normalize_tag(checked.get("tag", ""))
                break

    note = str(plan.get("note", "") or "").strip()
    error = ""
    if not candidates:
        error = "无法从这段输入确定人物，请同时输入作品名和人物名。"
    elif not resolved_tag and suggestion:
        error = (
            f"找到候选 {suggestion}，但它没有通过 NAID Tag Suggest 精确验证，"
            "因此未自动加入 Prompt。"
        )
    elif not resolved_tag:
        error = "候选人物没有通过 NAID character tag 精确验证，未修改 Prompt。"

    return {
        "query": query,
        "resolved_tag": resolved_tag,
        "remove_tags": remove_tags,
        "candidate_tags": candidates,
        "candidate_attempts": attempts,
        "verification": verification,
        "removal_verifications": removal_verifications,
        "suggested_tag": suggestion,
        "note": note,
        "error": error,
        "model": str(model or DEFAULT_MODEL).strip(),
    }


def validate_character_override(
    *,
    query: str,
    resolved_tag: str,
    remove_tags: Iterable[str],
    base_prompt: str,
    final_prompt: str,
    lookup: Callable[[str], Dict[str, Any]] = lookup_naid_exact,
) -> Dict[str, Any]:
    """Re-verify browser-submitted character fields before generation."""
    query = str(query or "").strip()
    resolved_tag = _normalize_tag(resolved_tag)
    remove_tags = _dedupe(remove_tags, limit=MAX_EXISTING_TAGS)

    if not query and not resolved_tag and not remove_tags:
        return {}
    if not query or not resolved_tag:
        raise ValueError("人物框尚未解析成功，请先点击“解析并替换人物”。")

    verification = dict(lookup(resolved_tag))
    if not _is_verified_character(verification):
        raise ValueError("人物 tag 未通过 NAID character 精确验证，不能自动生成。")

    final_keys = {_normalize_tag(x) for x in _split_prompt(final_prompt)}
    if resolved_tag not in final_keys:
        raise ValueError("最终 Prompt 中缺少已解析的人物 tag，请重新计算 Prompt。")

    base_by_key = {
        _normalize_tag(fragment): fragment for fragment in _split_prompt(base_prompt)
    }
    removal_verifications: List[Dict[str, Any]] = []
    safe_removals: List[str] = []
    for tag in remove_tags:
        if tag not in base_by_key:
            raise ValueError("人物替换的删除项已过期，请重新解析人物。")
        if tag in final_keys:
            raise ValueError("原人物 tag 仍在最终 Prompt 中，请重新计算 Prompt。")
        checked = dict(lookup(tag))
        if not _is_verified_character(checked):
            raise ValueError("原人物删除项未通过 character tag 验证。")
        if tag != resolved_tag:
            safe_removals.append(base_by_key[tag])
            removal_verifications.append(checked)

    return {
        "query": query,
        "resolved_tag": resolved_tag,
        "remove_tags": safe_removals,
        "verification": verification,
        "removal_verifications": removal_verifications,
    }
