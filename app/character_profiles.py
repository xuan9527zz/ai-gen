# -*- coding: utf-8 -*-
"""Local, user-confirmed character appearance profiles.

WD14 may suggest appearance tags from one or more clean reference images, but
the saved profile is always the user's editable result.  Generation-time merge
levels control only appearance conflicts; analysis provenance remains immutable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "data" / "character_profiles.json"
DEFAULT_DRAFT_DIR = PROJECT_ROOT / "data" / "character_profile_drafts"
PROFILE_VERSION = 1
PRIORITIES = {"a", "b", "c"}

COLORS = {
    "black", "blonde", "brown", "blue", "green", "red", "pink",
    "purple", "orange", "white", "grey", "gray", "silver", "aqua",
    "teal", "gold", "golden", "multicolored", "two-tone",
}

HAIR_STYLE_MARKERS = {
    "long hair", "short hair", "medium hair", "very long hair", "bob cut",
    "twintails", "ponytail", "side ponytail", "braid", "braided ponytail",
    "drill hair", "drill locks", "hime cut", "ahoge", "bangs",
    "blunt bangs", "parted bangs", "swept bangs", "hair over one eye",
    "hair between eyes", "low-tied long hair", "messy hair", "wavy hair",
    "curly hair", "straight hair", "twin braids", "single braid",
}

CLOTHING_MARKERS = {
    "dress", "shirt", "blouse", "skirt", "shorts", "pants", "trousers",
    "jacket", "coat", "sweater", "cardigan", "hoodie", "uniform",
    "kimono", "yukata", "robe", "armor", "apron", "vest", "suit",
    "leotard", "swimsuit", "bikini", "bodysuit", "cape", "cloak",
    "sleeves", "sleeveless", "off shoulder", "bare shoulders",
}

EXCLUDED_EXACT = {
    "1girl", "1boy", "solo", "multiple girls", "multiple boys",
    "looking at viewer", "looking away", "smile", "open mouth",
    "closed mouth", "blush", "standing", "sitting", "lying", "walking",
    "upper body", "full body", "cowboy shot", "portrait", "close-up",
    "simple background", "white background", "transparent background",
    "masterpiece", "best quality", "high quality", "absurdres",
    "official art", "character sheet", "reference sheet",
}

EXCLUDED_MARKERS = {
    "rating:", "score:", "source:", "artist:", "background", "scenery",
    "watermark", "signature", "text", "speech bubble", "censored",
    "sex", "nude", "nipples", "pussy", "penis", "anus", "cum",
    "masturbation", "oral", "vaginal", "anal", "spread", "bondage",
    "loli", "shota", "child", "underage", "young", "teen",
}


def normalize_tag(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().replace("_", " ").split())
    weighted = re.fullmatch(r"\(?\s*(.+?)\s*:\s*[0-9.]+\s*\)?", text)
    return weighted.group(1).strip() if weighted else text


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", text)


def split_tags(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = re.split(r"[,，、\n]+", str(value or ""))
    output: List[str] = []
    seen = set()
    for raw in parts:
        tag = normalize_tag(raw)
        if tag and tag not in seen:
            seen.add(tag)
            output.append(tag)
    return output


def tag_categories(tag: str) -> Set[str]:
    tag = normalize_tag(tag)
    if not tag:
        return set()
    words = set(tag.split())
    categories: Set[str] = set()

    if tag.endswith(" hair") and words & COLORS:
        categories.add("hair_color")
    if tag in HAIR_STYLE_MARKERS or (
        "hair" in words
        and not categories
        and not any(x in tag for x in ("ornament", "clip", "ribbon", "bow"))
    ):
        categories.add("hair_style")
    if tag.endswith(" eyes") and words & COLORS:
        categories.add("eye_color")
    elif "eyes" in words or "eye" in words:
        categories.add("eye_style")
    if any(marker in tag for marker in CLOTHING_MARKERS):
        categories.add("outfit")
    if any(marker in tag for marker in ("hair ornament", "hair clip", "hair bow", "hair ribbon", "hairband")):
        categories.add("hair_accessory")
    if any(marker in tag for marker in ("hat", "cap", "beret", "crown", "headwear", "headdress")):
        categories.add("headwear")
    if "glasses" in tag or "eyewear" in tag or "eyepatch" in tag:
        categories.add("eyewear")
    if any(marker in tag for marker in ("choker", "necktie", "bowtie", "scarf", "neck ribbon")):
        categories.add("neckwear")
    if any(marker in tag for marker in ("earring", "necklace", "bracelet", "ring", "jewelry")):
        categories.add("jewelry")
    if "gloves" in tag or "gauntlets" in tag:
        categories.add("handwear")
    if any(marker in tag for marker in ("thighhighs", "stockings", "socks", "pantyhose", "legwear")):
        categories.add("legwear")
    if any(marker in tag for marker in ("shoes", "boots", "sandals", "heels", "footwear")):
        categories.add("footwear")
    if "skin" in words or tag in {"tan", "pale", "freckles"}:
        categories.add("skin")
    for marker, category in (
        ("horn", "horns"), ("wing", "wings"), ("tail", "tail"),
        ("animal ears", "ears"), ("pointy ears", "ears"),
        ("halo", "halo"), ("fang", "fangs"), ("mole", "face_detail"),
    ):
        if marker in tag:
            categories.add(category)
    return categories


def is_allowed_profile_tag(tag: str) -> bool:
    tag = normalize_tag(tag)
    if not tag or tag in EXCLUDED_EXACT:
        return False
    if any(marker in tag for marker in EXCLUDED_MARKERS):
        return False
    return True


def is_profile_tag(tag: str) -> bool:
    return is_allowed_profile_tag(tag) and bool(tag_categories(tag))


def suggest_profile_tags(
    wd14_outputs: Sequence[str],
    *,
    character_tag: str = "",
) -> List[str]:
    character_key = normalize_tag(character_tag)
    per_image: List[List[str]] = []
    order: Dict[str, int] = {}
    for output in wd14_outputs:
        tags = [
            tag for tag in split_tags(output)
            if tag != character_key and is_profile_tag(tag)
        ]
        per_image.append(tags)
        for tag in tags:
            order.setdefault(tag, len(order))
    counts = Counter(tag for tags in per_image for tag in set(tags))
    return sorted(counts, key=lambda tag: (-counts[tag], order[tag]))


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else default
    except Exception:
        return default


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_profiles(path: Path = DEFAULT_PROFILE_PATH) -> Dict[str, Any]:
    value = _read_json(path, {})
    if value.get("version") != PROFILE_VERSION:
        return {"version": PROFILE_VERSION, "profiles": {}, "aliases": {}}
    value.setdefault("profiles", {})
    value.setdefault("aliases", {})
    return value


def get_profile(
    character_tag: str,
    *,
    path: Path = DEFAULT_PROFILE_PATH,
) -> Dict[str, Any]:
    key = normalize_tag(character_tag)
    profiles = load_profiles(path).get("profiles", {})
    row = profiles.get(key, {}) if isinstance(profiles, dict) else {}
    return dict(row) if isinstance(row, dict) else {}


def save_profile(
    *,
    character_tag: str,
    names: Iterable[str],
    prompt_tags: Any,
    wd14_outputs: Sequence[str] = (),
    source_images: Sequence[str] = (),
    path: Path = DEFAULT_PROFILE_PATH,
) -> Dict[str, Any]:
    character_tag = normalize_tag(character_tag)
    if not character_tag:
        raise ValueError("Danbooru 人物 tag 不能为空。")
    tags = split_tags(prompt_tags)
    if not tags:
        raise ValueError("人物原设加强 Prompt 不能为空。")
    unsafe = [tag for tag in tags if not is_allowed_profile_tag(tag)]
    if unsafe:
        raise ValueError(
            "人物原设库只保存外观、服装和标志性特征；请移除："
            + ", ".join(unsafe)
        )
    clean_names: List[str] = []
    seen_names = set()
    for raw in names:
        name = str(raw or "").strip()
        key = normalize_name(name)
        if name and key and key not in seen_names:
            seen_names.add(key)
            clean_names.append(name)
    if not clean_names:
        clean_names = [character_tag]

    data = load_profiles(path)
    profiles = data["profiles"]
    aliases = data["aliases"]
    previous = profiles.get(character_tag, {})
    created_at = (
        previous.get("created_at")
        if isinstance(previous, dict)
        else None
    ) or dt.datetime.now().astimezone().isoformat(timespec="seconds")
    row = {
        "character_tag": character_tag,
        "names": clean_names,
        "prompt_tags": tags,
        "wd14_outputs": [str(x) for x in wd14_outputs],
        "source_images": [str(x) for x in source_images],
        "created_at": created_at,
        "updated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "user_confirmed_wd14_profile",
    }
    profiles[character_tag] = row
    for name in clean_names:
        aliases[normalize_name(name)] = character_tag
    _write_json(path, data)
    return row


def create_draft(value: Dict[str, Any], *, directory: Path = DEFAULT_DRAFT_DIR) -> str:
    draft_id = uuid.uuid4().hex
    payload = dict(value)
    payload["draft_id"] = draft_id
    payload["created_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    _write_json(directory / f"{draft_id}.json", payload)
    return draft_id


def load_draft(draft_id: str, *, directory: Path = DEFAULT_DRAFT_DIR) -> Dict[str, Any]:
    draft_id = str(draft_id or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", draft_id):
        return {}
    return _read_json(directory / f"{draft_id}.json", {})


def profile_fingerprint(profile: Dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "character_tag": profile.get("character_tag", ""),
            "prompt_tags": profile.get("prompt_tags", []),
            "updated_at": profile.get("updated_at", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_profile(
    existing_prompt: str,
    profile: Dict[str, Any],
    priority: str,
) -> Dict[str, Any]:
    priority = str(priority or "b").strip().lower()
    if priority not in PRIORITIES:
        raise ValueError("人物原设冲突级别必须是 a、b 或 c。")
    existing = [
        " ".join(part.strip().split())
        for part in re.split(r"[,，\n]+", str(existing_prompt or ""))
        if part.strip()
    ]
    profile_tags = split_tags(profile.get("prompt_tags", []))
    existing_keys = {normalize_tag(tag) for tag in existing}
    conflicts_by_profile: Dict[str, List[str]] = {}
    all_conflicts: List[str] = []
    for profile_tag in profile_tags:
        categories = tag_categories(profile_tag)
        conflicts = [
            tag for tag in existing
            if (
                normalize_tag(tag) != profile_tag
                and categories & tag_categories(tag)
            )
        ]
        if conflicts:
            conflicts_by_profile[profile_tag] = conflicts
            for tag in conflicts:
                if tag not in all_conflicts:
                    all_conflicts.append(tag)

    if priority == "a":
        add_tags = [
            tag for tag in profile_tags
            if tag not in existing_keys and tag not in conflicts_by_profile
        ]
        remove_tags: List[str] = []
    elif priority == "b":
        add_tags = [tag for tag in profile_tags if tag not in existing_keys]
        remove_tags = []
    else:
        add_tags = [tag for tag in profile_tags if tag not in existing_keys]
        remove_tags = all_conflicts

    return {
        "priority": priority,
        "add_tags": add_tags,
        "remove_tags": remove_tags,
        "conflicts": conflicts_by_profile,
        "profile_tags": profile_tags,
        "profile_fingerprint": profile_fingerprint(profile),
    }
