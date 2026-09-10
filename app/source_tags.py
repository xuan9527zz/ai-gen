# -*- coding: utf-8 -*-
r"""
Source Tag Adapter v1.1 (project package)

Unifies source tags before they enter the Illustrious orchestrator.

Supported source_type:
- none
- danbooru
- pixiv
- raw

Danbooru:
- strips bullet markers
- strips the leading [?] wiki link
- extracts the actual tag link label
- strips trailing tag-count text such as 8.4M / 150k / 69k
- normalizes underscores to spaces
- de-duplicates while preserving order

Pixiv:
- accepts Japanese tags
- maps known tags through jp_to_danbooru_tags.json
- unknown Japanese tags are preserved separately for later translation,
  but are NOT added to the English final prompt automatically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ADAPTER_VERSION = "1.1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JP_MAPPING = PROJECT_ROOT / "config" / "jp_to_danbooru_tags.json"
DEFAULT_USER_JP_MAPPING = (
    PROJECT_ROOT / "data" / "pixiv_tag_overrides.json"
)

BULLET_RE = re.compile(r"^\s*[-*•·]\s*")
COUNT_SUFFIX_RE = re.compile(
    r"(?:\s|\u00a0)+"
    r"\d[\d,]*(?:\.\d+)?"
    r"\s*[kKmMbB]?"
    r"\s*$"
)
DANBOORU_EXACT_RE = re.compile(
    r"\[\?\]\([^)]+\)"
    r"\[([^\]]+)\]\([^)]+\)"
)
MARKDOWN_LINK_RE = re.compile(
    r"\[([^\]]+)\]\([^)]+\)"
)
URL_RE = re.compile(
    r"https?://\S+"
)


def normalize_tag(tag: str) -> str:
    tag = str(tag or "").strip()
    tag = tag.replace("_", " ")
    tag = re.sub(r"\s+", " ", tag)
    return tag.lower().strip()


def dedupe_keep_order(
    items: Iterable[str],
) -> List[str]:
    out: List[str] = []
    seen = set()

    for item in items:
        tag = normalize_tag(item)

        if not tag:
            continue

        if tag not in seen:
            seen.add(tag)
            out.append(tag)

    return out


def _strip_count_suffix(
    text: str,
) -> str:
    return COUNT_SUFFIX_RE.sub(
        "",
        str(text or ""),
    ).strip()


def _extract_danbooru_line(
    line: str,
) -> Optional[str]:
    line = str(line or "").strip()

    if not line:
        return None

    line = BULLET_RE.sub(
        "",
        line,
    ).strip()

    # Exact copied markdown form:
    # [?](wiki)[1girl](posts) 8.4M
    match = DANBOORU_EXACT_RE.search(
        line
    )

    if match:
        return normalize_tag(
            match.group(1)
        )

    # More general markdown:
    # choose the last non-"?" link label.
    labels = MARKDOWN_LINK_RE.findall(
        line
    )

    labels = [
        label.strip()
        for label in labels
        if label.strip()
        and label.strip() != "?"
    ]

    if labels:
        return normalize_tag(
            _strip_count_suffix(
                labels[-1]
            )
        )

    # Browser/plain-text fallback:
    # remove markdown URLs while preserving their labels.
    plain = MARKDOWN_LINK_RE.sub(
        lambda m: m.group(1),
        line,
    )

    plain = URL_RE.sub(
        "",
        plain,
    )

    plain = _strip_count_suffix(
        plain
    )

    plain = re.sub(
        r"^\s*\?\s*",
        "",
        plain,
    )

    plain = BULLET_RE.sub(
        "",
        plain,
    ).strip()

    if not plain:
        return None

    return normalize_tag(
        plain
    )


def parse_danbooru_tags(
    text: str,
) -> List[str]:
    """
    Parse copied Danbooru tag-list text.

    Also accepts simple comma-separated Danbooru tags.
    """
    text = str(text or "")

    tags: List[str] = []

    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    # Multi-line copied list.
    if len(lines) > 1:
        for line in lines:
            tag = _extract_danbooru_line(
                line
            )

            if tag:
                tags.append(
                    tag
                )

        return dedupe_keep_order(
            tags
        )

    # One line that still looks like markdown list.
    if lines and (
        "](" in lines[0]
        or lines[0].lstrip().startswith(
            ("-", "*", "•")
        )
    ):
        tag = _extract_danbooru_line(
            lines[0]
        )

        return (
            [tag]
            if tag
            else []
        )

    # Plain comma/newline separated input.
    parts = re.split(
        r"[,，\n]+",
        text,
    )

    for part in parts:
        part = _strip_count_suffix(
            part
        )

        part = re.sub(
            r"^\s*\?\s*",
            "",
            part,
        )

        if part.strip():
            tags.append(
                part
            )

    return dedupe_keep_order(
        tags
    )


def _read_mapping_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return (
            data
            if isinstance(
                data,
                dict,
            )
            else {}
        )

    except Exception:
        return {}


def load_jp_mapping(
    mapping_path: Optional[str | Path] = None,
    user_mapping_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Load the public mapping plus local user-confirmed overrides."""
    mapping = _read_mapping_file(
        Path(mapping_path or DEFAULT_JP_MAPPING)
    )
    user_mapping = _read_mapping_file(
        Path(user_mapping_path or DEFAULT_USER_JP_MAPPING)
    )
    mapping.update(user_mapping)
    return mapping


def save_pixiv_user_mapping(
    source_tag: str,
    mapped_prompt: str,
    *,
    user_mapping_path: Optional[str | Path] = None,
) -> List[str]:
    """Persist one user-confirmed Pixiv tag mapping in the local data folder."""
    source_tag = str(source_tag or "").strip().lstrip("#").strip()
    if not source_tag:
        raise ValueError("Pixiv 原始 tag 不能为空。")

    mapped_tags = dedupe_keep_order(
        re.split(r"[,，\n]+", str(mapped_prompt or ""))
    )
    if not mapped_tags:
        raise ValueError("转换后的 Prompt 至少需要一个 tag。")

    path = Path(user_mapping_path or DEFAULT_USER_JP_MAPPING)
    mapping = _read_mapping_file(path)
    mapping[source_tag] = mapped_tags
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return mapped_tags


def _split_pixiv_tags(
    text: str,
) -> List[str]:
    text = str(text or "").strip()

    if not text:
        return []

    # Hashtag form: #黒髪 #ツインテール
    hashtag_tags = re.findall(
        r"#([^#\s,，、]+)",
        text,
    )

    if hashtag_tags:
        return [
            x.strip()
            for x in hashtag_tags
            if x.strip()
        ]

    # Common copied-list forms.
    return [
        x.strip().lstrip("#").strip()
        for x in re.split(
            r"[\n,，、]+",
            text,
        )
        if x.strip()
    ]


def parse_pixiv_tags(
    text: str,
    mapping_path: Optional[str | Path] = None,
    user_mapping_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    public_mapping = _read_mapping_file(
        Path(mapping_path or DEFAULT_JP_MAPPING)
    )
    user_mapping = _read_mapping_file(
        Path(user_mapping_path or DEFAULT_USER_JP_MAPPING)
    )
    mapping = {**public_mapping, **user_mapping}

    raw_tags = _split_pixiv_tags(
        text
    )

    normalized: List[str] = []
    unknown: List[str] = []
    mapping_records: List[Dict[str, Any]] = []

    for raw in raw_tags:
        key = raw.strip()

        if not key:
            continue

        mapped = mapping.get(
            key
        )

        if mapped is None:
            unknown.append(
                key
            )
            continue

        if isinstance(
            mapped,
            str,
        ):
            mapped = [
                mapped
            ]

        if isinstance(
            mapped,
            list,
        ):
            mapped_tags: List[str] = []
            for tag in mapped:
                if isinstance(
                    tag,
                    str,
                ):
                    normalized_tag = normalize_tag(tag)
                    if normalized_tag:
                        normalized.append(normalized_tag)
                        mapped_tags.append(normalized_tag)

            mapping_records.append(
                {
                    "source_tag": key,
                    "mapped_tags": dedupe_keep_order(mapped_tags),
                    "mapping_source": (
                        "user_confirmed"
                        if key in user_mapping
                        else "public_mapping"
                    ),
                }
            )

    return {
        "source_tags_raw_list": (
            raw_tags
        ),
        "normalized_tags": (
            dedupe_keep_order(
                normalized
            )
        ),
        "unknown_tags": list(
            dict.fromkeys(
                unknown
            )
        ),
        "mapping_records": mapping_records,
    }


def adapt_source_tags(
    source_type: str,
    source_text: str = "",
    mapping_path: Optional[str | Path] = None,
    user_mapping_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    source_type = str(
        source_type
        or "none"
    ).strip().lower()

    source_text = str(
        source_text
        or ""
    )

    if source_type in {
        "",
        "none",
        "local",
    }:
        return {
            "adapter_version": (
                ADAPTER_VERSION
            ),
            "source_type": "none",
            "source_tags_raw": "",
            "source_tags_raw_list": [],
            "normalized_tags": [],
            "unknown_tags": [],
        }

    if source_type == "danbooru":
        tags = parse_danbooru_tags(
            source_text
        )

        return {
            "adapter_version": (
                ADAPTER_VERSION
            ),
            "source_type": (
                "danbooru"
            ),
            "source_tags_raw": (
                source_text
            ),
            "source_tags_raw_list": (
                tags
            ),
            "normalized_tags": (
                tags
            ),
            "unknown_tags": [],
        }

    if source_type == "pixiv":
        result = parse_pixiv_tags(
            source_text,
            mapping_path=(
                mapping_path
            ),
            user_mapping_path=(
                user_mapping_path
            ),
        )

        return {
            "adapter_version": (
                ADAPTER_VERSION
            ),
            "source_type": (
                "pixiv"
            ),
            "source_tags_raw": (
                source_text
            ),
            **result,
        }

    if source_type == "raw":
        tags = dedupe_keep_order(
            re.split(
                r"[,，\n]+",
                source_text,
            )
        )

        return {
            "adapter_version": (
                ADAPTER_VERSION
            ),
            "source_type": "raw",
            "source_tags_raw": (
                source_text
            ),
            "source_tags_raw_list": (
                tags
            ),
            "normalized_tags": (
                tags
            ),
            "unknown_tags": [],
        }

    raise ValueError(
        "source_type must be one of: "
        "none, danbooru, pixiv, raw"
    )


if __name__ == "__main__":
    sample = r"""
- [?](https://danbooru.donmai.us/wiki_pages/1girl?z=1)[1girl](https://danbooru.donmai.us/posts?tags=1girl\&z=1) 8.4M
- [?](https://danbooru.donmai.us/wiki_pages/beret?z=1)[beret](https://danbooru.donmai.us/posts?tags=beret\&z=1) 150k
- [?](https://danbooru.donmai.us/wiki_pages/black_hat?z=1)[black hat](https://danbooru.donmai.us/posts?tags=black_hat\&z=1) 252k
"""

    print(
        json.dumps(
            adapt_source_tags(
                "danbooru",
                sample,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
