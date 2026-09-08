# -*- coding: utf-8 -*-
r"""
NAID Tag Verifier v0.1

Purpose
-------
Resolve unknown Pixiv Japanese tags into English Danbooru/Illustrious-style
candidate tags with a local LLM, then verify candidates against:

    https://raideninfinity.pythonanywhere.com/naidv3_tag_search

The verifier is deliberately conservative:
- exact-match verification uses the site's "#" suffix semantics
- NAID Tag Suggest-backed exact matches can be auto-added
- Danbooru-fallback / ambiguous / unavailable results are NOT auto-added
  by default, but are preserved as suggestions for manual review
- all network results are cached locally to avoid unnecessary requests

Important
---------
The target site does not document a public API in its user guide.
This module therefore uses a best-effort, low-request discovery strategy:
1) inspect the site's search form
2) inspect same-origin JavaScript for likely search endpoints
3) cache a working endpoint strategy if found

If the site changes or automated lookup is unavailable, the module fails safe:
the translated candidate is preserved as "unverified" and is not silently
inserted into Final Prompt.

Dependencies:
- requests (already used by the local project)
- Python standard library only otherwise
"""

from __future__ import annotations

import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import requests


VERIFIER_VERSION = "0.2"

NAID_SEARCH_URL = (
    "https://raideninfinity.pythonanywhere.com/naidv3_tag_search"
)

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_TRANSLATION_MODEL = "qwen3:8b"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "naid_tag_cache.json"

DEFAULT_DISCOVERY_PATH = PROJECT_ROOT / "data" / "naid_endpoint_cache.json"

REQUEST_TIMEOUT = 20
MIN_REQUEST_INTERVAL = 0.8

# Conservative policy:
AUTO_ADD_DANBOORU_FALLBACK = False


# ============================================================
# GENERIC HELPERS
# ============================================================

def _norm_tag(tag: str) -> str:
    return " ".join(
        str(tag or "")
        .strip()
        .lower()
        .replace("_", " ")
        .split()
    )


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()

    for raw in items:
        item = _norm_tag(raw)

        if not item:
            continue

        if item in seen:
            continue

        seen.add(item)
        out.append(item)

    return out


def _read_json(
    path: Path,
    default: Any,
) -> Any:
    try:
        if not path.exists():
            return default

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return default


def _write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _parse_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.I,
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw,
    )

    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    match = re.search(
        r"\{.*\}",
        raw,
        flags=re.S,
    )

    if not match:
        raise ValueError(
            "LLM did not return a JSON object."
        )

    value = json.loads(
        match.group(0)
    )

    if not isinstance(value, dict):
        raise ValueError(
            "LLM JSON is not an object."
        )

    return value


# ============================================================
# PIXIV UNKNOWN -> ENGLISH TAG CANDIDATES
# ============================================================

TRANSLATION_SYSTEM_PROMPT = r"""
You convert unknown Pixiv Japanese tags into candidate English tags for
Danbooru/Illustrious-style anime image generation.

Return JSON only:
{
  "items": [
    {
      "source_tag": "original Pixiv tag",
      "candidate_tags": ["english candidate tag"],
      "kind": "general|character|copyright|artist|meta|unknown",
      "note": "short note"
    }
  ]
}

Rules:
- This is semantic tag mapping, not literal word-for-word translation.
- Prefer actual concise Danbooru-style tag wording.
- A source tag may map to more than one candidate tag only when clearly useful.
- Do not invent a tag if the Pixiv tag is a joke, community phrase, vague
  commentary, or cannot be mapped confidently; return candidate_tags=[].
- Keep proper character/copyright names as candidate tags only when the source
  tag itself clearly represents that entity.
- Do not create artist tags from an unknown phrase unless it is clearly an
  artist name.
- All sexual subjects in this workflow must be adults. Never introduce,
  infer, or sexualize minors or youth-like subjects.
""".strip()


def translate_pixiv_unknown_tags(
    unknown_tags: List[str],
    *,
    model: str = DEFAULT_TRANSLATION_MODEL,
    ollama_url: str = OLLAMA_URL,
) -> List[Dict[str, Any]]:
    clean = [
        str(x).strip()
        for x in unknown_tags
        if str(x).strip()
    ]

    if not clean:
        return []

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": TRANSLATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Map these Pixiv tags:\n"
                    + json.dumps(
                        clean,
                        ensure_ascii=False,
                    )
                ),
            },
        ],
        "options": {
            "temperature": 0.1,
        },
    }

    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    content = (
        response.json()
        .get(
            "message",
            {},
        )
        .get(
            "content",
            "",
        )
    )

    parsed = _parse_json_object(
        content
    )

    rows = parsed.get(
        "items",
        []
    )

    if not isinstance(rows, list):
        rows = []

    by_source = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        source_tag = str(
            row.get(
                "source_tag",
                "",
            )
            or ""
        ).strip()

        if not source_tag:
            continue

        raw_candidates = row.get(
            "candidate_tags",
            [],
        )

        if isinstance(raw_candidates, str):
            raw_candidates = [
                raw_candidates
            ]

        candidates = (
            _dedupe(
                raw_candidates
                if isinstance(
                    raw_candidates,
                    list,
                )
                else []
            )
        )

        by_source[
            source_tag
        ] = {
            "source_tag": source_tag,
            "candidate_tags": candidates,
            "kind": str(
                row.get(
                    "kind",
                    "unknown",
                )
                or "unknown"
            ),
            "note": str(
                row.get(
                    "note",
                    "",
                )
                or ""
            ),
        }

    # Preserve every requested source tag, even if the model omitted it.
    return [
        by_source.get(
            source,
            {
                "source_tag": source,
                "candidate_tags": [],
                "kind": "unknown",
                "note": (
                    "Translation model returned no candidate."
                ),
            },
        )
        for source in clean
    ]


# ============================================================
# HTML DISCOVERY / PARSING
# ============================================================

class _FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms: List[Dict[str, Any]] = []
        self.scripts: List[str] = []
        self._current_form: Optional[
            Dict[str, Any]
        ] = None

    def handle_starttag(
        self,
        tag: str,
        attrs: List[Tuple[str, Optional[str]]],
    ) -> None:
        attr = {
            k.lower(): (
                v
                if v is not None
                else ""
            )
            for k, v in attrs
        }

        if tag.lower() == "form":
            self._current_form = {
                "action": attr.get(
                    "action",
                    "",
                ),
                "method": attr.get(
                    "method",
                    "get",
                ).lower(),
                "inputs": [],
            }

            self.forms.append(
                self._current_form
            )

        elif (
            tag.lower()
            in {
                "input",
                "select",
                "textarea",
                "button",
            }
            and self._current_form
            is not None
        ):
            self._current_form[
                "inputs"
            ].append(
                attr
            )

        elif (
            tag.lower()
            == "script"
            and attr.get("src")
        ):
            self.scripts.append(
                attr[
                    "src"
                ]
            )

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        if tag.lower() == "form":
            self._current_form = None


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()

        self.rows: List[
            Dict[str, Any]
        ] = []

        self._in_row = False
        self._in_cell = False
        self._row_cells: List[
            Dict[str, Any]
        ] = []
        self._cell_text: List[str] = []
        self._cell_attrs: List[
            Dict[str, str]
        ] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: List[Tuple[str, Optional[str]]],
    ) -> None:
        attr = {
            k.lower(): (
                v
                if v is not None
                else ""
            )
            for k, v in attrs
        }

        if tag.lower() == "tr":
            self._in_row = True
            self._row_cells = []

        if (
            self._in_row
            and tag.lower()
            in {
                "td",
                "th",
            }
        ):
            self._in_cell = True
            self._cell_text = []
            self._cell_attrs = [
                attr
            ]

        elif self._in_cell:
            self._cell_attrs.append(
                attr
            )

    def handle_data(
        self,
        data: str,
    ) -> None:
        if self._in_cell:
            self._cell_text.append(
                data
            )

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        if (
            self._in_cell
            and tag.lower()
            in {
                "td",
                "th",
            }
        ):
            text = " ".join(
                " ".join(
                    self._cell_text
                ).split()
            )

            attrs = {}

            for item in self._cell_attrs:
                attrs.update(
                    item
                )

            self._row_cells.append({
                "text": text,
                "attrs": attrs,
            })

            self._in_cell = False
            self._cell_text = []
            self._cell_attrs = []

        if (
            self._in_row
            and tag.lower()
            == "tr"
        ):
            self.rows.append({
                "cells": (
                    self._row_cells
                )
            })

            self._in_row = False
            self._row_cells = []


def _extract_power(
    text: str,
) -> Optional[int]:
    raw = str(text or "")

    # Power is documented in the 0-10000 range. Prefer a plain
    # integer that fits that domain.
    values = re.findall(
        r"(?<![\d,.])"
        r"(\d{1,5})"
        r"(?![\d,.])",
        raw.replace(",", ""),
    )

    candidates = []

    for value in values:
        number = int(
            value
        )

        if (
            0
            <= number
            <= 10000
        ):
            candidates.append(
                number
            )

    return (
        candidates[-1]
        if candidates
        else None
    )


def _detect_power_source(
    row_text: str,
    attrs_text: str,
) -> str:
    haystack = (
        str(row_text)
        + " "
        + str(attrs_text)
    ).lower()

    if (
        "tag suggest"
        in haystack
        or "novelai"
        in haystack
        or "nai"
        in haystack
        or '"n_count"'
        in haystack
        or "'n_count'"
        in haystack
    ):
        return (
            "naid_tag_suggest"
        )

    if (
        "danbooru"
        in haystack
        or '"d_count"'
        in haystack
        or "'d_count'"
        in haystack
    ):
        return (
            "danbooru_fallback"
        )

    return "unknown"


def _parse_html_exact_match(
    html_text: str,
    candidate: str,
) -> Optional[Dict[str, Any]]:
    parser = _TableParser()
    parser.feed(
        html_text
    )

    target = _norm_tag(
        candidate
    )

    for row in parser.rows:
        cells = row.get(
            "cells",
            [],
        )

        if not cells:
            continue

        cell_texts = [
            str(
                cell.get(
                    "text",
                    "",
                )
            )
            for cell in cells
        ]

        normalized_cells = [
            _norm_tag(
                x
            )
            for x in cell_texts
        ]

        if target not in normalized_cells:
            continue

        tag_index = normalized_cells.index(
            target
        )

        category = (
            cell_texts[
                tag_index
                + 1
            ]
            if (
                tag_index
                + 1
                < len(
                    cell_texts
                )
            )
            else ""
        )

        joined = " | ".join(
            cell_texts
        )

        attrs_text = json.dumps(
            [
                cell.get(
                    "attrs",
                    {},
                )
                for cell in cells
            ],
            ensure_ascii=False,
        )

        power = _extract_power(
            joined
        )

        return {
            "exact_match": True,
            "tag": target,
            "category": category,
            "power": power,
            "power_source": (
                _detect_power_source(
                    joined,
                    attrs_text,
                )
            ),
            "raw_row_text": joined,
        }

    return None


def _walk_json_for_match(
    value: Any,
    candidate: str,
) -> Optional[Dict[str, Any]]:
    target = _norm_tag(
        candidate
    )

    if isinstance(value, dict):
        tag_value = None

        for key in (
            "tag",
            "tag_name",
            "name",
            "label",
        ):
            if key in value:
                tag_value = value[
                    key
                ]
                break

        if (
            tag_value is not None
            and _norm_tag(
                tag_value
            )
            == target
        ):
            power = None

            # The NAIDv3 Tag Search endpoint currently returns fields such as:
            #   tag_name, d_category, d_count, n_count
            # where n_count is the NovelAI Tag Suggest power/count and
            # d_count is the Danbooru fallback population.
            power = None
            power_source = "unknown"

            if "n_count" in value:
                try:
                    power = int(
                        value[
                            "n_count"
                        ]
                    )
                except Exception:
                    power = _extract_power(
                        str(
                            value[
                                "n_count"
                            ]
                        )
                    )

                if power is not None:
                    power_source = "naid_tag_suggest"

            if (
                power is None
                and "d_count" in value
            ):
                try:
                    power = int(
                        value[
                            "d_count"
                        ]
                    )
                except Exception:
                    power = _extract_power(
                        str(
                            value[
                                "d_count"
                            ]
                        )
                    )

                if power is not None:
                    power_source = "danbooru_fallback"

            if power is None:
                for key in (
                    "power",
                    "count",
                    "nai_count",
                    "novelai_count",
                    "tag_suggest_count",
                ):
                    if key in value:
                        try:
                            power = int(
                                value[
                                    key
                                ]
                            )
                        except Exception:
                            power = _extract_power(
                                str(
                                    value[
                                        key
                                    ]
                                )
                            )
                        break

            category = ""

            for key in (
                "d_category",
                "category",
                "type",
            ):
                if key in value:
                    category = str(
                        value[
                            key
                        ]
                    )
                    break

            packed = json.dumps(
                value,
                ensure_ascii=False,
            )

            if (
                power_source
                == "unknown"
            ):
                power_source = (
                    _detect_power_source(
                        packed,
                        packed,
                    )
                )

            return {
                "exact_match": True,
                "tag": target,
                "category": category,
                "power": power,
                "power_source": (
                    power_source
                ),
                "raw_row_text": packed,
            }

        for child in value.values():
            found = _walk_json_for_match(
                child,
                candidate,
            )

            if found:
                return found

    elif isinstance(value, list):
        for child in value:
            found = _walk_json_for_match(
                child,
                candidate,
            )

            if found:
                return found

    return None


# ============================================================
# ENDPOINT DISCOVERY
# ============================================================

def _request_with_delay(
    session: requests.Session,
    method: str,
    url: str,
    *,
    last_request_time: List[float],
    **kwargs: Any,
) -> requests.Response:
    elapsed = (
        time.time()
        - last_request_time[0]
    )

    if (
        last_request_time[0]
        and elapsed
        < MIN_REQUEST_INTERVAL
    ):
        time.sleep(
            MIN_REQUEST_INTERVAL
            - elapsed
        )

    response = session.request(
        method,
        url,
        timeout=REQUEST_TIMEOUT,
        **kwargs,
    )

    last_request_time[0] = (
        time.time()
    )

    return response


def _discover_form_strategy(
    html_text: str,
) -> Optional[Dict[str, Any]]:
    parser = _FormParser()
    parser.feed(
        html_text
    )

    for form in parser.forms:
        for inp in form.get(
            "inputs",
            [],
        ):
            placeholder = str(
                inp.get(
                    "placeholder",
                    "",
                )
            ).lower()

            name = str(
                inp.get(
                    "name",
                    "",
                )
            )

            iid = str(
                inp.get(
                    "id",
                    "",
                )
            ).lower()

            if (
                "search term"
                in placeholder
                or "search"
                in iid
                or "query"
                in iid
            ):
                if name:
                    return {
                        "type": "form",
                        "method": form.get(
                            "method",
                            "get",
                        ),
                        "action": form.get(
                            "action",
                            "",
                        ),
                        "search_field": name,
                    }

    return None


def _discover_js_candidates(
    html_text: str,
    *,
    base_url: str,
    session: requests.Session,
    last_request_time: List[float],
) -> List[Dict[str, Any]]:
    parser = _FormParser()
    parser.feed(
        html_text
    )

    candidates: List[
        Dict[str, Any]
    ] = []

    for src in parser.scripts:
        script_url = urljoin(
            base_url,
            src,
        )

        # Only inspect same-origin scripts.
        if (
            "raideninfinity.pythonanywhere.com"
            not in script_url
        ):
            continue

        try:
            response = _request_with_delay(
                session,
                "GET",
                script_url,
                last_request_time=(
                    last_request_time
                ),
            )

            if not response.ok:
                continue

            js = response.text

        except Exception:
            continue

        patterns = [
            (
                "fetch",
                r"""fetch\(\s*['"]([^'"]+)['"]""",
            ),
            (
                "axios_get",
                r"""axios\.get\(\s*['"]([^'"]+)['"]""",
            ),
            (
                "axios_post",
                r"""axios\.post\(\s*['"]([^'"]+)['"]""",
            ),
            (
                "url",
                r"""(?:url|endpoint)\s*[:=]\s*['"]([^'"]+)['"]""",
            ),
        ]

        for kind, pattern in patterns:
            for match in re.findall(
                pattern,
                js,
                flags=re.I,
            ):
                endpoint = urljoin(
                    base_url,
                    match,
                )

                low = endpoint.lower()

                if not any(
                    token in low
                    for token in (
                        "search",
                        "tag",
                        "api",
                    )
                ):
                    continue

                candidates.append({
                    "type": "endpoint",
                    "kind": kind,
                    "url": endpoint,
                })

    # stable dedupe
    seen = set()
    out = []

    for item in candidates:
        key = (
            item.get("kind"),
            item.get("url"),
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out[:8]


def _save_strategy(
    path: Path,
    strategy: Dict[str, Any],
) -> None:
    _write_json(
        path,
        {
            "verifier_version": (
                VERIFIER_VERSION
            ),
            "saved_at": (
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ),
            "strategy": strategy,
        },
    )


def _load_strategy(
    path: Path,
) -> Optional[Dict[str, Any]]:
    raw = _read_json(
        path,
        {},
    )

    if not isinstance(
        raw,
        dict,
    ):
        return None

    strategy = raw.get(
        "strategy"
    )

    return (
        strategy
        if isinstance(
            strategy,
            dict,
        )
        else None
    )


# ============================================================
# LOOKUP STRATEGIES
# ============================================================

def _parse_response(
    response: requests.Response,
    candidate: str,
) -> Optional[Dict[str, Any]]:
    content_type = str(
        response.headers.get(
            "Content-Type",
            "",
        )
    ).lower()

    if (
        "json"
        in content_type
    ):
        try:
            found = _walk_json_for_match(
                response.json(),
                candidate,
            )

            if found:
                return found
        except Exception:
            pass

    try:
        found = _parse_html_exact_match(
            response.text,
            candidate,
        )

        if found:
            return found
    except Exception:
        pass

    # Some endpoints return JSON but omit content-type.
    try:
        found = _walk_json_for_match(
            response.json(),
            candidate,
        )

        if found:
            return found
    except Exception:
        pass

    return None


def _try_strategy(
    strategy: Dict[str, Any],
    candidate: str,
    *,
    session: requests.Session,
    last_request_time: List[float],
) -> Optional[Dict[str, Any]]:
    query = (
        candidate
        + "#"
    )

    stype = strategy.get(
        "type"
    )

    if stype == "form":
        target = urljoin(
            NAID_SEARCH_URL,
            str(
                strategy.get(
                    "action",
                    "",
                )
            ),
        )

        field = str(
            strategy.get(
                "search_field",
                "",
            )
        )

        if not field:
            return None

        method = str(
            strategy.get(
                "method",
                "get",
            )
        ).lower()

        if method == "post":
            response = _request_with_delay(
                session,
                "POST",
                target,
                data={
                    field: query
                },
                last_request_time=(
                    last_request_time
                ),
            )
        else:
            response = _request_with_delay(
                session,
                "GET",
                target,
                params={
                    field: query
                },
                last_request_time=(
                    last_request_time
                ),
            )

        if not response.ok:
            return None

        return _parse_response(
            response,
            candidate,
        )

    if stype == "endpoint":
        url = str(
            strategy.get(
                "url",
                "",
            )
        )

        if not url:
            return None

        # Keep probes limited. These are the common field names used
        # by simple Flask/JS search backends.
        for field in (
            "q",
            "query",
            "search",
            "term",
        ):
            response = _request_with_delay(
                session,
                "GET",
                url,
                params={
                    field: query
                },
                last_request_time=(
                    last_request_time
                ),
            )

            if not response.ok:
                continue

            found = _parse_response(
                response,
                candidate,
            )

            if found:
                used = dict(
                    strategy
                )

                used[
                    "request_method"
                ] = "GET"

                used[
                    "search_field"
                ] = field

                found[
                    "_working_strategy"
                ] = used

                return found

        return None

    if stype == "endpoint_ready":
        url = str(
            strategy.get(
                "url",
                "",
            )
        )

        method = str(
            strategy.get(
                "request_method",
                "GET",
            )
        ).upper()

        field = str(
            strategy.get(
                "search_field",
                "q",
            )
        )

        kwargs: Dict[str, Any] = {}

        if method == "POST":
            kwargs[
                "data"
            ] = {
                field: query
            }
        else:
            kwargs[
                "params"
            ] = {
                field: query
            }

        response = _request_with_delay(
            session,
            method,
            url,
            last_request_time=(
                last_request_time
            ),
            **kwargs,
        )

        if not response.ok:
            return None

        return _parse_response(
            response,
            candidate,
        )

    return None


def lookup_naid_exact(
    candidate: str,
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    discovery_path: Path = DEFAULT_DISCOVERY_PATH,
) -> Dict[str, Any]:
    tag = _norm_tag(
        candidate
    )

    if not tag:
        return {
            "exact_match": False,
            "tag": "",
            "power": None,
            "power_source": "unknown",
            "status": "invalid_candidate",
            "lookup_error": "",
        }

    cache = _read_json(
        cache_path,
        {},
    )

    if not isinstance(
        cache,
        dict,
    ):
        cache = {}

    if tag in cache:
        cached = dict(
            cache[
                tag
            ]
        )

        cached[
            "cache_hit"
        ] = True

        return cached

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "IllustriousSourceVerifier/0.2 "
            "(local personal workflow; low request rate)"
        )
    })

    last_request_time = [
        0.0
    ]

    lookup_error = ""

    try:
        saved_strategy = _load_strategy(
            discovery_path
        )

        if saved_strategy:
            found = _try_strategy(
                saved_strategy,
                tag,
                session=session,
                last_request_time=(
                    last_request_time
                ),
            )

            if found:
                result = _finalize_lookup(
                    found
                )

                cache[
                    tag
                ] = result

                _write_json(
                    cache_path,
                    cache,
                )

                return result

        homepage = _request_with_delay(
            session,
            "GET",
            NAID_SEARCH_URL,
            last_request_time=(
                last_request_time
            ),
        )

        homepage.raise_for_status()

        html_text = homepage.text

        form_strategy = (
            _discover_form_strategy(
                html_text
            )
        )

        if form_strategy:
            found = _try_strategy(
                form_strategy,
                tag,
                session=session,
                last_request_time=(
                    last_request_time
                ),
            )

            if found:
                _save_strategy(
                    discovery_path,
                    form_strategy,
                )

                result = _finalize_lookup(
                    found
                )

                cache[
                    tag
                ] = result

                _write_json(
                    cache_path,
                    cache,
                )

                return result

        js_candidates = (
            _discover_js_candidates(
                html_text,
                base_url=(
                    NAID_SEARCH_URL
                ),
                session=session,
                last_request_time=(
                    last_request_time
                ),
            )
        )

        for strategy in js_candidates:
            found = _try_strategy(
                strategy,
                tag,
                session=session,
                last_request_time=(
                    last_request_time
                ),
            )

            if not found:
                continue

            working = found.pop(
                "_working_strategy",
                None,
            )

            if working:
                working[
                    "type"
                ] = (
                    "endpoint_ready"
                )

                _save_strategy(
                    discovery_path,
                    working,
                )

            result = _finalize_lookup(
                found
            )

            cache[
                tag
            ] = result

            _write_json(
                cache_path,
                cache,
            )

            return result

        lookup_error = (
            "No working automated NAID search strategy was discovered."
        )

    except Exception as exc:
        lookup_error = (
            f"{type(exc).__name__}: {exc}"
        )

    result = {
        "exact_match": False,
        "tag": tag,
        "category": "",
        "power": None,
        "power_source": "unknown",
        "status": "lookup_unavailable",
        "lookup_error": lookup_error,
        "cache_hit": False,
    }

    # Do not permanently cache lookup_unavailable: the site/network
    # may work later.
    return result


def _finalize_lookup(
    found: Dict[str, Any],
) -> Dict[str, Any]:
    exact = bool(
        found.get(
            "exact_match",
            False,
        )
    )

    power = found.get(
        "power"
    )

    source = str(
        found.get(
            "power_source",
            "unknown",
        )
        or "unknown"
    )

    if exact and (
        source
        == "naid_tag_suggest"
    ):
        status = (
            "naid_verified"
        )

    elif exact and (
        source
        == "danbooru_fallback"
    ):
        status = (
            "danbooru_fallback"
        )

    elif exact:
        status = (
            "exact_match_unknown_source"
        )

    else:
        status = (
            "not_found"
        )

    return {
        "exact_match": exact,
        "tag": _norm_tag(
            found.get(
                "tag",
                "",
            )
        ),
        "category": str(
            found.get(
                "category",
                "",
            )
            or ""
        ),
        "power": power,
        "power_source": source,
        "status": status,
        "lookup_error": "",
        "raw_row_text": str(
            found.get(
                "raw_row_text",
                "",
            )
            or ""
        ),
        "cache_hit": False,
    }


# ============================================================
# END-TO-END UNKNOWN PIXIV RESOLUTION
# ============================================================

def resolve_pixiv_unknown_tags(
    unknown_tags: List[str],
    *,
    translation_model: str = DEFAULT_TRANSLATION_MODEL,
    auto_add_danbooru_fallback: bool = (
        AUTO_ADD_DANBOORU_FALLBACK
    ),
) -> Dict[str, Any]:
    translations = (
        translate_pixiv_unknown_tags(
            unknown_tags,
            model=translation_model,
        )
    )

    records: List[
        Dict[str, Any]
    ] = []

    auto_added: List[str] = []
    suggested: List[str] = []
    unverified_source_tags: List[str] = []

    for translation in translations:
        source_tag = str(
            translation.get(
                "source_tag",
                "",
            )
            or ""
        )

        candidates = translation.get(
            "candidate_tags",
            [],
        )

        if not candidates:
            records.append({
                **translation,
                "verifications": [],
                "auto_added_tags": [],
                "suggested_tags": [],
                "status": (
                    "no_candidate"
                ),
            })

            unverified_source_tags.append(
                source_tag
            )
            continue

        verification_rows = []
        row_auto: List[str] = []
        row_suggested: List[str] = []

        for candidate in candidates:
            verification = lookup_naid_exact(
                candidate
            )

            verification_rows.append({
                "candidate": (
                    _norm_tag(
                        candidate
                    )
                ),
                **verification,
            })

            if (
                verification[
                    "status"
                ]
                == "naid_verified"
            ):
                auto_added.append(
                    candidate
                )
                row_auto.append(
                    candidate
                )

            elif (
                verification[
                    "status"
                ]
                == "danbooru_fallback"
                and auto_add_danbooru_fallback
            ):
                auto_added.append(
                    candidate
                )
                row_auto.append(
                    candidate
                )

            else:
                suggested.append(
                    candidate
                )
                row_suggested.append(
                    candidate
                )

        if row_auto:
            status = (
                "verified_and_added"
            )
        elif row_suggested:
            status = (
                "suggested_only"
            )
        else:
            status = (
                "unverified"
            )

        records.append({
            **translation,
            "verifications": (
                verification_rows
            ),
            "auto_added_tags": (
                _dedupe(
                    row_auto
                )
            ),
            "suggested_tags": (
                _dedupe(
                    row_suggested
                )
            ),
            "status": status,
        })

    return {
        "verifier_version": (
            VERIFIER_VERSION
        ),
        "translation_model": (
            translation_model
        ),
        "verification_source": (
            NAID_SEARCH_URL
        ),
        "auto_add_policy": {
            "naid_tag_suggest_exact": True,
            "danbooru_fallback_exact": (
                bool(
                    auto_add_danbooru_fallback
                )
            ),
            "unknown_source_exact": False,
            "lookup_unavailable": False,
        },
        "records": records,
        "auto_added_tags": (
            _dedupe(
                auto_added
            )
        ),
        "suggested_tags": (
            _dedupe(
                suggested
            )
        ),
        "unverified_source_tags": list(
            dict.fromkeys(
                unverified_source_tags
            )
        ),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "tags",
        nargs="*",
        help="Unknown Pixiv Japanese tags",
    )

    parser.add_argument(
        "--lookup",
        help=(
            "Test NAID exact lookup directly with an English tag, "
            "for example: --lookup \"1girl\""
        ),
    )

    args = parser.parse_args()

    if args.lookup:
        payload = lookup_naid_exact(
            args.lookup
        )

    else:
        if not args.tags:
            parser.error(
                "Provide Pixiv tags or use --lookup."
            )

        payload = resolve_pixiv_unknown_tags(
            args.tags
        )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
