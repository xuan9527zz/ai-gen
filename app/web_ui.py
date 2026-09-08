# -*- coding: utf-8 -*-
r"""
Illustrious Reconstruction Studio v2.1.1

Local/LAN browser workspace for:
1) image library + new-image analysis + re-analysis
2) AI natural-language prompt corrections
3) manual positive-prompt additions / direct final-prompt editing
4) ComfyUI generation with 3 LoRA slots
5) original-vs-generated comparison + 1-5 ratings
6) mobile-friendly UI

Expected sibling files:
    illustrious_web_ui.py
    illustrious_generate.py          # use Generator v1.1+
    illustrious_orchestrator.py      # Analyzer v2.3.1+
    illustrious_db.py                # DB v1.1+
    illustrious_generation_config.json
    anime.json
    source_tag_adapter.py
    jp_to_danbooru_tags.json

Run:
    py F:\OpenWebUI\illustrious_web_ui.py

Desktop:
    http://127.0.0.1:8765

Phone on same Wi-Fi/LAN:
    http://<YOUR_PC_LAN_IP>:8765

Security:
    This server has no authentication in v2.0.
    Use it only on a trusted LAN. Do NOT port-forward/expose port 8765 publicly.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import mimetypes
import os
import re
import socket
import subprocess
import ipaddress
import sqlite3
import sys
import threading
import traceback
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# ============================================================
# PATHS / MODULES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "generation.json"
)

UI_STATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "ui_state.json"
)

UPLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "uploads"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "data"
    / "runs"
)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BASE_DIR),
    )

from . import generator  # noqa: E402
from . import analyzer  # noqa: E402
from . import database as analysis_db  # noqa: E402


HOST = "0.0.0.0"
PORT = 8765

GENERATION_LOCK = threading.Lock()
ANALYSIS_LOCK = threading.Lock()
CORRECTION_LOCK = threading.Lock()

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

DEFAULT_CORRECTION_MODEL = os.getenv(
    "ILLUSTRIOUS_CORRECTION_MODEL",
    "qwen3:8b",
)
OLLAMA_URL = os.getenv(
    "ILLUSTRIOUS_OLLAMA_URL",
    "http://127.0.0.1:11434",
)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


# ============================================================
# GENERAL HELPERS
# ============================================================

def esc(value: Any) -> str:
    return html.escape(
        str(
            value
            if value is not None
            else ""
        )
    )


def now_iso() -> str:
    return (
        dt.datetime.now()
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )


def load_base_config() -> Dict[str, Any]:
    return generator.load_config(
        CONFIG_PATH
    )


def load_ui_state(
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    default_state = {
        "lora_slots": cfg.get(
            "lora_slots",
            [],
        ),
        "negative_prompt": cfg.get(
            "negative_prompt",
            "",
        ),
        "count": int(
            cfg.get(
                "images_per_run",
                5,
            )
        ),
        "last_run_id": None,
        "correction_model": (
            DEFAULT_CORRECTION_MODEL
        ),
    }

    if not UI_STATE_PATH.exists():
        return default_state

    try:
        raw = json.loads(
            UI_STATE_PATH.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            raw,
            dict,
        ):
            return default_state

        state = dict(
            default_state
        )
        state.update(
            raw
        )
        return state

    except Exception:
        return default_state


def save_ui_state(
    state: Dict[str, Any],
) -> None:
    UI_STATE_PATH.write_text(
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def first_value(
    form: Dict[str, List[str]],
    key: str,
    default: str = "",
) -> str:
    values = form.get(
        key
    )

    if not values:
        return default

    return values[0]


def normalize_tag_key(
    tag: str,
) -> str:
    return " ".join(
        str(tag)
        .strip()
        .lower()
        .replace(
            "_",
            " ",
        )
        .split()
    )


def split_prompt_fragments(
    text: str,
) -> List[str]:
    """
    Manual/AI prompt entry accepts commas or line breaks.

    Preserve prompt syntax such as:
        (tag:1.2)
    while only normalizing surrounding whitespace.
    """
    raw = str(
        text
        or ""
    ).replace(
        "\r",
        "\n",
    )

    pieces: List[str] = []

    for line in raw.split(
        "\n"
    ):
        for part in line.split(
            ","
        ):
            item = " ".join(
                part.strip().split()
            )

            if item:
                pieces.append(
                    item
                )

    return pieces


def dedupe_prompt_fragments(
    items: List[str],
) -> List[str]:
    out: List[str] = []
    seen = set()

    for item in items:
        key = normalize_tag_key(
            item
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(
            key
        )
        out.append(
            item.strip()
        )

    return out


def apply_prompt_edits(
    base_prompt: str,
    *,
    remove_tags_text: str = "",
    add_tags_text: str = "",
    manual_positive: str = "",
) -> str:
    base = split_prompt_fragments(
        base_prompt
    )

    remove_keys = {
        normalize_tag_key(
            x
        )
        for x in split_prompt_fragments(
            remove_tags_text
        )
        if normalize_tag_key(
            x
        )
    }

    kept = [
        item
        for item in base
        if normalize_tag_key(
            item
        )
        not in remove_keys
    ]

    added = split_prompt_fragments(
        add_tags_text
    )

    manual = split_prompt_fragments(
        manual_positive
    )

    final = dedupe_prompt_fragments(
        kept
        + added
        + manual
    )

    return ", ".join(
        final
    )


def safe_filename(
    filename: str,
) -> str:
    name = Path(
        str(
            filename
            or "upload.png"
        )
    ).name

    stem = re.sub(
        r"[^0-9A-Za-z._\-\u0080-\uffff]+",
        "_",
        Path(name).stem,
    ).strip(
        "._"
    )

    if not stem:
        stem = "upload"

    suffix = (
        Path(name)
        .suffix
        .lower()
    )

    if suffix not in (
        ALLOWED_IMAGE_EXTENSIONS
    ):
        suffix = ".png"

    return (
        f"{stem}{suffix}"
    )


def get_lan_ip() -> str:
    """
    Prefer a real private LAN address over VPN/TUN/WSL adapters.

    Windows first tries Get-NetIPConfiguration and scores:
      192.168.x.x > 10.x.x.x > 172.16-31.x.x

    Benchmark/test ranges such as 198.18.0.0/15 are excluded.
    """

    def score_ip(raw: str) -> int:
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            return -1000

        if not isinstance(
            ip,
            ipaddress.IPv4Address,
        ):
            return -1000

        if ip.is_loopback or ip.is_link_local:
            return -1000

        # RFC 2544 benchmarking range; often used by TUN/VPN software.
        if ip in ipaddress.ip_network(
            "198.18.0.0/15"
        ):
            return -1000

        text = str(ip)

        if text.startswith("192.168."):
            return 300

        if text.startswith("10."):
            return 200

        if ip in ipaddress.ip_network(
            "172.16.0.0/12"
        ):
            return 100

        if ip.is_private:
            return 50

        return 0

    candidates: List[str] = []

    if os.name == "nt":
        try:
            command = (
                "Get-NetIPConfiguration | "
                "Where-Object {"
                "$_.IPv4DefaultGateway -ne $null "
                "-and $_.NetAdapter.Status -eq 'Up'"
                "} | ForEach-Object {"
                "$_.IPv4Address.IPAddress"
                "}"
            )

            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )

            for line in result.stdout.splitlines():
                value = line.strip()

                if value:
                    candidates.append(
                        value
                    )

        except Exception:
            pass

    # Generic host lookup fallback.
    try:
        for item in socket.getaddrinfo(
            socket.gethostname(),
            None,
            family=socket.AF_INET,
        ):
            candidates.append(
                item[4][0]
            )
    except Exception:
        pass

    ranked = sorted(
        set(candidates),
        key=score_ip,
        reverse=True,
    )

    if ranked and score_ip(ranked[0]) >= 0:
        return ranked[0]

    # Last-resort route probe.
    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    try:
        sock.connect(
            (
                "8.8.8.8",
                80,
            )
        )

        candidate = str(
            sock.getsockname()[0]
        )

        if score_ip(candidate) >= 0:
            return candidate

    except Exception:
        pass

    finally:
        sock.close()

    return "127.0.0.1"


# ============================================================
# DATABASE
# ============================================================

STUDIO_SCHEMA = r"""
CREATE TABLE IF NOT EXISTS generation_prompt_edits (
    generation_run_id INTEGER PRIMARY KEY,

    parent_generation_id INTEGER,

    edit_kind TEXT NOT NULL DEFAULT 'base',
    base_prompt TEXT NOT NULL,
    user_instruction TEXT,
    correction_model TEXT,

    ai_add_tags_json TEXT NOT NULL DEFAULT '[]',
    ai_remove_tags_json TEXT NOT NULL DEFAULT '[]',

    manual_positive TEXT,
    final_generation_prompt TEXT NOT NULL,

    created_at TEXT NOT NULL,

    FOREIGN KEY(generation_run_id)
        REFERENCES generation_runs(id)
        ON DELETE CASCADE,

    FOREIGN KEY(parent_generation_id)
        REFERENCES generation_runs(id)
        ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_prompt_edits_parent
ON generation_prompt_edits(parent_generation_id);
"""


def db_path_from_config() -> Path:
    cfg = load_base_config()
    return Path(
        cfg[
            "database_path"
        ]
    )


def connect_db() -> sqlite3.Connection:
    db_path = db_path_from_config()

    # Fresh clone friendly: initialize the analysis DB automatically.
    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not db_path.exists():
        init_conn = analysis_db.connect(
            db_path
        )
        try:
            analysis_db.init_db(
                init_conn
            )
            init_conn.commit()
        finally:
            init_conn.close()

    conn = sqlite3.connect(
        str(db_path),
        timeout=60,
    )

    conn.row_factory = sqlite3.Row

    conn.execute(
        "PRAGMA foreign_keys=ON"
    )

    # Generator v1.0/1.1 creates these on first generation.
    # Create defensively so the web UI can open before generation.
    conn.executescript(
        generator.GENERATION_SCHEMA
    )

    conn.executescript(
        STUDIO_SCHEMA
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS generation_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_run_id INTEGER NOT NULL UNIQUE,
            overall_score INTEGER,
            comment TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(generation_run_id)
                REFERENCES generation_runs(id)
                ON DELETE CASCADE,
            CHECK(
                overall_score IS NULL
                OR overall_score BETWEEN 1 AND 5
            )
        )
        """
    )

    conn.commit()

    return conn


def image_library(
    conn: sqlite3.Connection,
    limit: int = 200,
) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            i.id AS image_id,
            i.filename,
            i.first_seen_path,
            COUNT(r.id) AS run_count,
            MAX(r.id) AS latest_run_id,
            MAX(r.analyzer_timestamp) AS latest_timestamp
        FROM images i
        LEFT JOIN analysis_runs r
          ON r.image_id=i.id
        GROUP BY
            i.id,
            i.filename,
            i.first_seen_path
        ORDER BY
            COALESCE(MAX(r.id), 0) DESC,
            i.id DESC
        LIMIT ?
        """,
        (
            limit,
        ),
    ).fetchall()


def recent_analysis_runs(
    conn: sqlite3.Connection,
    limit: int = 200,
) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            r.id,
            r.image_id,
            r.analyzer_timestamp AS timestamp,
            r.analyzer_version AS version,
            r.review_needed,
            r.elapsed_seconds,
            r.final_prompt,
            i.filename,
            i.first_seen_path
        FROM analysis_runs r
        JOIN images i
          ON i.id=r.image_id
        ORDER BY r.id DESC
        LIMIT ?
        """,
        (
            limit,
        ),
    ).fetchall()


def get_analysis_run(
    conn: sqlite3.Connection,
    run_id: int,
) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            r.*,
            i.filename,
            i.first_seen_path
        FROM analysis_runs r
        JOIN images i
          ON i.id=r.image_id
        WHERE r.id=?
        """,
        (
            run_id,
        ),
    ).fetchone()


def runs_for_image(
    conn: sqlite3.Connection,
    image_id: int,
) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            id,
            analyzer_timestamp,
            analyzer_version,
            final_prompt,
            review_needed
        FROM analysis_runs
        WHERE image_id=?
        ORDER BY id DESC
        """,
        (
            image_id,
        ),
    ).fetchall()


def get_generation_run(
    conn: sqlite3.Connection,
    generation_id: int,
) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM generation_runs
        WHERE id=?
        """,
        (
            generation_id,
        ),
    ).fetchone()


def get_generation_runs(
    conn: sqlite3.Connection,
    analysis_run_id: int,
    limit: int = 30,
) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            g.*,
            gr.overall_score,
            gr.comment AS rating_comment,
            pe.parent_generation_id,
            pe.edit_kind,
            pe.user_instruction,
            pe.correction_model,
            pe.ai_add_tags_json,
            pe.ai_remove_tags_json,
            pe.manual_positive
        FROM generation_runs g
        LEFT JOIN generation_ratings gr
          ON gr.generation_run_id=g.id
        LEFT JOIN generation_prompt_edits pe
          ON pe.generation_run_id=g.id
        WHERE g.analysis_run_id=?
          AND g.status='completed'
          AND g.generated_image_path IS NOT NULL
        ORDER BY g.id DESC
        LIMIT ?
        """,
        (
            analysis_run_id,
            limit,
        ),
    ).fetchall()


def save_rating(
    generation_run_id: int,
    score: int,
    comment: str,
) -> None:
    if score < 1 or score > 5:
        raise ValueError(
            "Score must be 1-5."
        )

    conn = connect_db()

    try:
        exists = conn.execute(
            """
            SELECT id
            FROM generation_runs
            WHERE id=?
            """,
            (
                generation_run_id,
            ),
        ).fetchone()

        if not exists:
            raise ValueError(
                f"Generation not found: "
                f"{generation_run_id}"
            )

        timestamp = now_iso()

        conn.execute(
            """
            INSERT INTO generation_ratings(
                generation_run_id,
                overall_score,
                comment,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(generation_run_id)
            DO UPDATE SET
                overall_score=excluded.overall_score,
                comment=excluded.comment,
                updated_at=excluded.updated_at
            """,
            (
                generation_run_id,
                score,
                comment.strip(),
                timestamp,
                timestamp,
            ),
        )

        conn.commit()

    finally:
        conn.close()


def save_prompt_edit_records(
    generation_results: List[Dict[str, Any]],
    *,
    parent_generation_id: Optional[int],
    edit_kind: str,
    base_prompt: str,
    user_instruction: str,
    correction_model: str,
    ai_add_tags: List[str],
    ai_remove_tags: List[str],
    manual_positive: str,
    final_prompt: str,
) -> None:
    if not generation_results:
        return

    conn = connect_db()

    try:
        for result in generation_results:
            generation_id = int(
                result[
                    "generation_id"
                ]
            )

            conn.execute(
                """
                INSERT INTO generation_prompt_edits(
                    generation_run_id,
                    parent_generation_id,
                    edit_kind,
                    base_prompt,
                    user_instruction,
                    correction_model,
                    ai_add_tags_json,
                    ai_remove_tags_json,
                    manual_positive,
                    final_generation_prompt,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(generation_run_id)
                DO UPDATE SET
                    parent_generation_id=excluded.parent_generation_id,
                    edit_kind=excluded.edit_kind,
                    base_prompt=excluded.base_prompt,
                    user_instruction=excluded.user_instruction,
                    correction_model=excluded.correction_model,
                    ai_add_tags_json=excluded.ai_add_tags_json,
                    ai_remove_tags_json=excluded.ai_remove_tags_json,
                    manual_positive=excluded.manual_positive,
                    final_generation_prompt=excluded.final_generation_prompt
                """,
                (
                    generation_id,
                    parent_generation_id,
                    edit_kind,
                    base_prompt,
                    user_instruction.strip(),
                    correction_model.strip(),
                    json.dumps(
                        ai_add_tags,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        ai_remove_tags,
                        ensure_ascii=False,
                    ),
                    manual_positive.strip(),
                    final_prompt,
                    now_iso(),
                ),
            )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# ANALYSIS PIPELINE
# ============================================================

def unique_run_json_path(
    image_path: Path,
) -> Path:
    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = (
        dt.datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    safe_stem = re.sub(
        r"[^0-9A-Za-z._\-\u0080-\uffff]+",
        "_",
        image_path.stem,
    )

    return (
        RUNS_DIR
        / f"{stamp}_{safe_stem}.json"
    )


def analyze_and_import(
    image_path: Path,
    *,
    source_type: str = "none",
    source_tags_text: str = "",
) -> Tuple[int, Dict[str, Any], Path]:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Not a file: {image_path}"
        )

    if (
        image_path.suffix.lower()
        not in ALLOWED_IMAGE_EXTENSIONS
    ):
        raise ValueError(
            "Unsupported image type: "
            f"{image_path.suffix}"
        )

    # The web layer owns the unique JSON filename so it can import
    # the exact analysis result deterministically.
    old_save_flag = getattr(
        analyzer,
        "SAVE_RUN_JSON",
        True,
    )

    analyzer.SAVE_RUN_JSON = False

    try:
        result = analyzer.analyze_image(
            str(
                image_path
            ),
            source_type=source_type,
            source_tags_text=(
                source_tags_text
            ),
        )

    finally:
        analyzer.SAVE_RUN_JSON = (
            old_save_flag
        )

    json_path = unique_run_json_path(
        image_path
    )

    json_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    db_path = db_path_from_config()

    conn = analysis_db.connect(
        db_path
    )

    try:
        analysis_db.init_db(
            conn
        )

        _status, run_id = (
            analysis_db.import_one(
                conn,
                json_path,
            )
        )

        conn.commit()

    finally:
        conn.close()

    return (
        int(
            run_id
        ),
        result,
        json_path,
    )


def save_uploaded_image(
    filename: str,
    content: bytes,
) -> Path:
    if not content:
        raise ValueError(
            "Empty upload."
        )

    if len(
        content
    ) > MAX_UPLOAD_BYTES:
        raise ValueError(
            "Image is too large. "
            "Maximum upload size is 50 MB."
        )

    safe = safe_filename(
        filename
    )

    suffix = (
        Path(
            safe
        )
        .suffix
        .lower()
    )

    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            "Unsupported image type."
        )

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = (
        dt.datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    target = (
        UPLOAD_DIR
        / f"{stamp}_{uuid.uuid4().hex[:8]}_{safe}"
    )

    target.write_bytes(
        content
    )

    return target


# ============================================================
# AI PROMPT CORRECTION
# ============================================================

CORRECTION_SYSTEM_PROMPT = r"""
You are a local Illustrious/Danbooru prompt-editing assistant.

TASK
Convert the user's natural-language correction instruction into a conservative
edit plan for an existing image-generation prompt.

All sexual subjects in this workflow must be adults.
Do not introduce, sexualize, or infer minors/youth-like subjects.
If the instruction explicitly asks for a minor or youth-like sexual subject,
return an error instead of editing the prompt.

OUTPUT
Return JSON only with exactly these fields:
{
  "add_tags": ["tag 1", "tag 2"],
  "remove_tags": ["conflicting existing tag"],
  "note": "short explanation",
  "error": ""
}

RULES
- Prefer concise English Danbooru/Illustrious-style tags.
- The user's correction is authoritative even if it changes the analyzed image.
- Only remove existing tags when they genuinely conflict with the requested edit.
- Do not remove unrelated details.
- Do not add quality filler unless the user explicitly asks for it.
- Do not invent artist names, LoRAs, checkpoints, seeds, samplers, or style names.
- Keep add_tags/remove_tags deduplicated.
- If there is no safe/useful edit, return empty arrays and explain in note.
""".strip()


def parse_json_object_from_text(
    text: str,
) -> Dict[str, Any]:
    raw = str(
        text
        or ""
    ).strip()

    if raw.startswith(
        "```"
    ):
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
        value = json.loads(
            raw
        )

        if isinstance(
            value,
            dict,
        ):
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
            "AI did not return a JSON object."
        )

    value = json.loads(
        match.group(0)
    )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "AI JSON is not an object."
        )

    return value


def normalize_ai_tag_list(
    value: Any,
) -> List[str]:
    if not isinstance(
        value,
        list,
    ):
        return []

    out: List[str] = []

    for raw in value:
        item = " ".join(
            str(raw)
            .strip()
            .replace(
                "_",
                " ",
            )
            .split()
        )

        if item:
            out.append(
                item
            )

    return dedupe_prompt_fragments(
        out
    )


def ai_prompt_correction(
    *,
    base_prompt: str,
    instruction: str,
    model: str,
) -> Dict[str, Any]:
    instruction = str(
        instruction
        or ""
    ).strip()

    if not instruction:
        raise ValueError(
            "请输入更正指令。"
        )

    model = str(
        model
        or DEFAULT_CORRECTION_MODEL
    ).strip()

    user_prompt = (
        "EXISTING PROMPT:\n"
        f"{base_prompt}\n\n"
        "USER CORRECTION:\n"
        f"{instruction}\n\n"
        "Return the JSON edit plan."
    )

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": (
                    CORRECTION_SYSTEM_PROMPT
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "options": {
            "temperature": 0.1,
        },
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    data = response.json()

    content = (
        data.get(
            "message",
            {}
        )
        .get(
            "content",
            ""
        )
    )

    parsed = parse_json_object_from_text(
        content
    )

    error = str(
        parsed.get(
            "error",
            "",
        )
        or ""
    ).strip()

    add_tags = normalize_ai_tag_list(
        parsed.get(
            "add_tags",
            [],
        )
    )

    remove_tags = normalize_ai_tag_list(
        parsed.get(
            "remove_tags",
            [],
        )
    )

    note = str(
        parsed.get(
            "note",
            "",
        )
        or ""
    ).strip()

    if error:
        return {
            "add_tags": [],
            "remove_tags": [],
            "note": note,
            "error": error,
            "model": model,
        }

    return {
        "add_tags": add_tags,
        "remove_tags": remove_tags,
        "note": note,
        "error": "",
        "model": model,
    }


# ============================================================
# COMFYUI / LORA
# ============================================================

def comfy_lora_names(
    cfg: Dict[str, Any],
) -> Tuple[List[str], Optional[str]]:
    base = str(
        cfg.get(
            "comfy_url",
            "http://127.0.0.1:8188",
        )
    ).rstrip(
        "/"
    )

    try:
        response = requests.get(
            f"{base}/object_info/LoraLoader",
            timeout=8,
        )

        response.raise_for_status()

        data = response.json()

        loader = data.get(
            "LoraLoader",
            data,
        )

        required = (
            loader.get(
                "input",
                {},
            )
            .get(
                "required",
                {},
            )
        )

        raw = required.get(
            "lora_name"
        )

        names: List[str] = []

        if (
            isinstance(
                raw,
                list,
            )
            and raw
            and isinstance(
                raw[0],
                list,
            )
        ):
            names = [
                str(x)
                for x in raw[0]
                if str(x).strip()
            ]

        return (
            sorted(
                set(
                    names
                ),
                key=str.lower,
            ),
            None,
        )

    except Exception as exc:
        return (
            [],
            f"{type(exc).__name__}: {exc}",
        )


def parse_lora_slots(
    form: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    slots = []

    for i in range(
        1,
        4,
    ):
        slots.append({
            "enabled": (
                first_value(
                    form,
                    f"lora{i}_enabled",
                    "",
                )
                == "on"
            ),
            "name": first_value(
                form,
                f"lora{i}_name",
                "",
            ).strip(),
            "strength_model": float(
                first_value(
                    form,
                    f"lora{i}_model",
                    "0.7",
                )
            ),
            "strength_clip": float(
                first_value(
                    form,
                    f"lora{i}_clip",
                    "0.7",
                )
            ),
        })

    return slots


def config_for_form(
    form: Dict[str, List[str]],
) -> Dict[str, Any]:
    cfg = load_base_config()

    cfg[
        "lora_slots"
    ] = parse_lora_slots(
        form
    )

    cfg[
        "negative_prompt"
    ] = first_value(
        form,
        "negative_prompt",
        str(
            cfg.get(
                "negative_prompt",
                "",
            )
        ),
    ).strip()

    cfg[
        "apply_lora_to_refiner"
    ] = True

    cfg[
        "use_lora_clip_for_negative"
    ] = True

    return cfg


def lora_slots_from_generation(
    raw_json: str,
) -> List[Dict[str, Any]]:
    try:
        rows = json.loads(
            raw_json
            or "[]"
        )
    except Exception:
        rows = []

    slots = [
        {
            "enabled": False,
            "name": "",
            "strength_model": 0.7,
            "strength_clip": 0.7,
        }
        for _ in range(
            3
        )
    ]

    if not isinstance(
        rows,
        list,
    ):
        return slots

    for item in rows:
        if not isinstance(
            item,
            dict,
        ):
            continue

        slot_num = int(
            item.get(
                "slot",
                0,
            )
            or 0
        )

        if slot_num not in {
            1,
            2,
            3,
        }:
            continue

        slots[
            slot_num
            - 1
        ] = {
            "enabled": True,
            "name": str(
                item.get(
                    "name",
                    "",
                )
                or ""
            ),
            "strength_model": float(
                item.get(
                    "strength_model",
                    0.7,
                )
            ),
            "strength_clip": float(
                item.get(
                    "strength_clip",
                    0.7,
                )
            ),
        }

    return slots


# ============================================================
# MULTIPART UPLOAD PARSER
# ============================================================

def parse_multipart(
    handler: BaseHTTPRequestHandler,
) -> Tuple[
    Dict[str, List[str]],
    Dict[str, List[Dict[str, Any]]],
]:
    content_type = handler.headers.get(
        "Content-Type",
        "",
    )

    match = re.search(
        r'boundary=(?:"([^"]+)"|([^;]+))',
        content_type,
        flags=re.I,
    )

    if not match:
        raise ValueError(
            "Missing multipart boundary."
        )

    boundary_text = (
        match.group(1)
        or match.group(2)
        or ""
    ).strip()

    boundary = (
        b"--"
        + boundary_text.encode(
            "utf-8"
        )
    )

    length = int(
        handler.headers.get(
            "Content-Length",
            "0",
        )
        or "0"
    )

    # Allow multiple files but keep a sane aggregate cap.
    if length > (
        MAX_UPLOAD_BYTES
        * 5
        + 2 * 1024 * 1024
    ):
        raise ValueError(
            "Upload request is too large."
        )

    body = handler.rfile.read(
        length
    )

    fields: Dict[
        str,
        List[str],
    ] = {}

    files: Dict[
        str,
        List[
            Dict[
                str,
                Any,
            ]
        ],
    ] = {}

    for chunk in body.split(
        boundary
    ):
        chunk = chunk.strip(
            b"\r\n"
        )

        if not chunk:
            continue

        if chunk == b"--":
            continue

        if chunk.endswith(
            b"--"
        ):
            chunk = chunk[:-2].rstrip(
                b"\r\n"
            )

        if b"\r\n\r\n" not in chunk:
            continue

        raw_headers, content = (
            chunk.split(
                b"\r\n\r\n",
                1,
            )
        )

        content = content.rstrip(
            b"\r\n"
        )

        headers_text = raw_headers.decode(
            "utf-8",
            errors="replace",
        )

        disposition_line = ""

        content_type_value = (
            "application/octet-stream"
        )

        for line in headers_text.split(
            "\r\n"
        ):
            lower = line.lower()

            if lower.startswith(
                "content-disposition:"
            ):
                disposition_line = (
                    line
                )

            elif lower.startswith(
                "content-type:"
            ):
                content_type_value = (
                    line.split(
                        ":",
                        1,
                    )[1]
                    .strip()
                )

        name_match = re.search(
            r'name="([^"]+)"',
            disposition_line,
        )

        if not name_match:
            continue

        name = name_match.group(
            1
        )

        filename_match = re.search(
            r'filename="([^"]*)"',
            disposition_line,
        )

        if filename_match:
            filename = (
                filename_match.group(
                    1
                )
            )

            if not filename:
                continue

            files.setdefault(
                name,
                [],
            ).append({
                "filename": filename,
                "content_type": (
                    content_type_value
                ),
                "content": content,
            })

        else:
            value = content.decode(
                "utf-8",
                errors="replace",
            )

            fields.setdefault(
                name,
                [],
            ).append(
                value
            )

    return (
        fields,
        files,
    )


# ============================================================
# HTML STYLES
# ============================================================

CSS = r"""
:root {
  color-scheme: dark;
  --bg: #0e1014;
  --panel: #171a21;
  --panel2: #1d212b;
  --border: #303746;
  --text: #f2f4f8;
  --muted: #aab2c0;
  --accent: #83a8ff;
  --accent2: #b8c9ff;
  --good: #8de2a5;
  --warn: #ffd166;
  --bad: #ff8383;
}

* { box-sizing: border-box; }

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  font-family:
    Inter, ui-sans-serif, system-ui,
    -apple-system, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
}

header {
  background: #11141a;
  border-bottom: 1px solid var(--border);
  padding: 15px 18px;
  position: sticky;
  top: 0;
  z-index: 20;
}

.header-inner {
  max-width: 1900px;
  margin: 0 auto;
  display: flex;
  justify-content: space-between;
  gap: 14px;
  align-items: center;
}

header h1 {
  font-size: 19px;
  margin: 0;
}

.header-sub {
  color: var(--muted);
  font-size: 12px;
  margin-top: 4px;
}

.lan-pill {
  padding: 7px 10px;
  border: 1px solid var(--border);
  background: var(--panel);
  border-radius: 999px;
  color: var(--muted);
  font-size: 12px;
}

main {
  max-width: 1900px;
  margin: 0 auto;
  padding: 16px;
  padding-bottom: 90px;
}

.panel {
  border: 1px solid var(--border);
  background: var(--panel);
  border-radius: 14px;
  padding: 14px;
  margin-bottom: 14px;
}

.panel h2, .panel h3 {
  margin-top: 0;
}

.message {
  background: #173321;
  border: 1px solid #2e7245;
  color: #c7f6d2;
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 14px;
}

.message.error {
  background: #3d1818;
  border-color: #824141;
  color: #ffd2d2;
}

.muted {
  color: var(--muted);
}

.small {
  font-size: 12px;
}

.toolbar {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items: end;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.field.grow {
  flex: 1;
  min-width: 240px;
}

.field label {
  color: var(--muted);
  font-size: 12px;
}

select,
input[type="text"],
input[type="number"],
textarea,
input[type="file"] {
  width: 100%;
  color: var(--text);
  background: #0f1218;
  border: 1px solid var(--border);
  border-radius: 9px;
  padding: 9px 10px;
  font: inherit;
}

textarea {
  min-height: 74px;
  resize: vertical;
}

button {
  border: 0;
  border-radius: 9px;
  padding: 10px 14px;
  background: var(--accent);
  color: #0d1320;
  font-weight: 750;
  cursor: pointer;
  font: inherit;
}

button.secondary {
  background: #323a4c;
  color: var(--text);
}

button.ghost {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text);
}

button.danger {
  background: #5f2c2c;
  color: #ffe4e4;
}

button:disabled {
  opacity: 0.55;
  cursor: wait;
}

details > summary {
  cursor: pointer;
  color: var(--accent2);
  user-select: none;
}

.image-library {
  display: flex;
  gap: 10px;
  overflow-x: auto;
  padding-bottom: 5px;
  scroll-snap-type: x proximity;
}

.image-card {
  flex: 0 0 135px;
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  background: #0d1015;
  scroll-snap-align: start;
}

.image-card.active {
  outline: 2px solid var(--accent);
}

.image-card a {
  display: block;
  text-decoration: none;
  color: inherit;
}

.image-thumb {
  width: 100%;
  height: 118px;
  object-fit: cover;
  display: block;
  background: #090b0e;
}

.image-card-text {
  padding: 7px;
  font-size: 11px;
}

.image-card-text strong {
  display: block;
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
}

.workspace {
  display: grid;
  grid-template-columns:
    minmax(350px, 0.82fr)
    minmax(560px, 1.35fr);
  gap: 16px;
  align-items: start;
}

.original-column {
  position: sticky;
  top: 86px;
}

.image-frame {
  border: 1px solid var(--border);
  border-radius: 12px;
  background: #080a0d;
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 250px;
}

.image-frame img {
  max-width: 100%;
  max-height: 73vh;
  display: block;
  object-fit: contain;
}

.prompt-box {
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.45;
  background: #0f1218;
  border: 1px solid var(--border);
  border-radius: 9px;
  padding: 9px;
  font-size: 12px;
  max-height: 190px;
  overflow: auto;
}

.lora-grid {
  display: grid;
  grid-template-columns:
    78px minmax(230px, 1fr)
    105px 105px;
  gap: 8px;
  align-items: center;
}

.lora-head {
  color: var(--muted);
  font-size: 11px;
}

.checkbox-inline {
  display: flex;
  align-items: center;
  gap: 6px;
}

.editor-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.ai-note {
  min-height: 22px;
  color: var(--good);
  font-size: 12px;
}

.final-prompt {
  min-height: 130px;
}

.action-row {
  display: flex;
  gap: 9px;
  flex-wrap: wrap;
  align-items: center;
}

.candidate {
  border: 1px solid var(--border);
  background: var(--panel);
  border-radius: 14px;
  overflow: hidden;
  margin-bottom: 16px;
}

.candidate-image {
  background: #07090c;
}

.candidate-image img {
  display: block;
  width: 100%;
  max-height: 80vh;
  object-fit: contain;
}

.candidate-meta {
  padding: 10px 12px;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 12px;
}

.rating-area,
.correction-area {
  padding: 11px 12px;
  border-top: 1px solid var(--border);
}

.rating-row {
  display: flex;
  gap: 7px;
  align-items: center;
  flex-wrap: wrap;
}

.score {
  position: relative;
}

.score input {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}

.score label {
  width: 38px;
  height: 34px;
  border: 1px solid var(--border);
  background: #10131a;
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.score input:checked + label {
  background: var(--accent);
  color: #0d1320;
  font-weight: 800;
}

.rating-comment {
  flex: 1;
  min-width: 180px;
}

.sticky-generate {
  position: sticky;
  bottom: 8px;
  z-index: 15;
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  pointer-events: none;
}

.sticky-generate button {
  pointer-events: auto;
  box-shadow: 0 8px 30px rgba(0,0,0,.45);
}

.status-good { color: var(--good); }
.status-bad { color: var(--bad); }
.status-warn { color: var(--warn); }

.tabline {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.chip {
  display: inline-flex;
  padding: 5px 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: #11151c;
  color: var(--muted);
  font-size: 11px;
}

@media (max-width: 950px) {
  header {
    position: static;
  }

  .header-inner {
    align-items: flex-start;
    flex-direction: column;
  }

  main {
    padding: 10px;
    padding-bottom: 100px;
  }

  .workspace {
    grid-template-columns: 1fr;
  }

  .original-column {
    position: static;
  }

  .image-frame img {
    max-height: 60vh;
  }

  .lora-grid {
    grid-template-columns: 1fr;
  }

  .lora-head {
    display: none;
  }

  .editor-grid {
    grid-template-columns: 1fr;
  }

  .field.grow {
    min-width: 100%;
  }

  button {
    min-height: 44px;
  }

  input,
  select,
  textarea {
    font-size: 16px !important;
  }

  .sticky-generate {
    position: fixed;
    left: 10px;
    right: 10px;
    bottom: 10px;
    display: grid;
    grid-template-columns: 1fr 1fr;
  }

  .sticky-generate button {
    width: auto;
    min-height: 50px;
    font-size: 16px;
  }

  .candidate-image img {
    max-height: 68vh;
  }
}
"""


# ============================================================
# PAGE DATA HELPERS
# ============================================================

def default_slot_values(
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    raw_slots = state.get(
        "lora_slots",
        [],
    )

    out = []

    for index in range(
        3
    ):
        raw = (
            raw_slots[
                index
            ]
            if (
                isinstance(
                    raw_slots,
                    list,
                )
                and index
                < len(
                    raw_slots
                )
                and isinstance(
                    raw_slots[
                        index
                    ],
                    dict,
                )
            )
            else {}
        )

        out.append({
            "enabled": bool(
                raw.get(
                    "enabled",
                    False,
                )
            ),
            "name": str(
                raw.get(
                    "name",
                    "",
                )
                or ""
            ),
            "strength_model": float(
                raw.get(
                    "strength_model",
                    0.7,
                )
            ),
            "strength_clip": float(
                raw.get(
                    "strength_clip",
                    0.7,
                )
            ),
        })

    return out


def lora_summary(
    raw_json: Any,
) -> str:
    try:
        rows = json.loads(
            raw_json
            or "[]"
        )
    except Exception:
        rows = []

    if not rows:
        return "LoRA: none"

    parts = []

    for row in rows:
        if not isinstance(
            row,
            dict,
        ):
            continue

        parts.append(
            "S"
            f"{row.get('slot', '?')} "
            f"{row.get('name', '?')} "
            "M"
            f"{row.get('strength_model', '?')} "
            "C"
            f"{row.get('strength_clip', '?')}"
        )

    return " | ".join(
        parts
    )


def diagnostic_summary(
    raw_json: Any,
) -> str:
    try:
        sampling = json.loads(
            raw_json
            or "{}"
        )
    except Exception:
        return ""

    if not isinstance(sampling, dict):
        return ""

    diagnostic = sampling.get(
        "diagnostic"
    )
    if not isinstance(diagnostic, dict):
        return ""

    variant = str(
        diagnostic.get(
            "variant",
            "",
        )
        or ""
    ).strip().upper()
    label = str(
        diagnostic.get(
            "label",
            "",
        )
        or ""
    ).strip()

    if not variant:
        return ""

    return (
        f"Diagnostic {variant}"
        + (
            f": {label}"
            if label
            else ""
        )
    )


def json_list_to_text(
    raw: Any,
) -> str:
    try:
        value = json.loads(
            raw
            or "[]"
        )
    except Exception:
        return ""

    if not isinstance(
        value,
        list,
    ):
        return ""

    return ", ".join(
        str(x)
        for x in value
    )


def pixiv_verification_html(
    raw_json: Any,
) -> str:
    try:
        payload = json.loads(
            raw_json
            or "{}"
        )
    except Exception:
        return ""

    if not isinstance(
        payload,
        dict,
    ):
        return ""

    source = payload.get(
        "source",
        {},
    )

    if not isinstance(
        source,
        dict,
    ):
        return ""

    if source.get(
        "source_type"
    ) != "pixiv":
        return ""

    verification = source.get(
        "pixiv_verification",
        {},
    )

    if not isinstance(
        verification,
        dict,
    ):
        return ""

    records = verification.get(
        "records",
        [],
    )

    status = str(
        verification.get(
            "status",
            "",
        )
        or ""
    )

    if (
        not records
        and status
        in {
            "",
            "not_applicable",
        }
    ):
        return ""

    rows = []

    for record in (
        records
        if isinstance(
            records,
            list,
        )
        else []
    ):
        if not isinstance(
            record,
            dict,
        ):
            continue

        source_tag = str(
            record.get(
                "source_tag",
                "",
            )
            or ""
        )

        candidates = record.get(
            "candidate_tags",
            [],
        )

        if not isinstance(
            candidates,
            list,
        ):
            candidates = []

        verifications = record.get(
            "verifications",
            [],
        )

        if not isinstance(
            verifications,
            list,
        ):
            verifications = []

        verification_bits = []

        for item in verifications:
            if not isinstance(
                item,
                dict,
            ):
                continue

            tag = str(
                item.get(
                    "candidate",
                    item.get(
                        "tag",
                        "",
                    ),
                )
                or ""
            )

            vstatus = str(
                item.get(
                    "status",
                    "",
                )
                or ""
            )

            power = item.get(
                "power"
            )

            power_source = str(
                item.get(
                    "power_source",
                    "",
                )
                or ""
            )

            cls = (
                "status-good"
                if vstatus
                == "naid_verified"
                else (
                    "status-warn"
                    if vstatus
                    in {
                        "danbooru_fallback",
                        "exact_match_unknown_source",
                    }
                    else "status-bad"
                )
            )

            power_text = (
                f" · Power {power}"
                if power is not None
                else ""
            )

            source_text = (
                f" · {power_source}"
                if power_source
                else ""
            )

            verification_bits.append(
                "<div class='small "
                + cls
                + "'>"
                + esc(tag)
                + " · "
                + esc(vstatus)
                + esc(power_text)
                + esc(source_text)
                + "</div>"
            )

        auto_added = record.get(
            "auto_added_tags",
            [],
        )

        if not isinstance(
            auto_added,
            list,
        ):
            auto_added = []

        suggested = record.get(
            "suggested_tags",
            [],
        )

        if not isinstance(
            suggested,
            list,
        ):
            suggested = []

        decision = ""

        if auto_added:
            decision += (
                "<div class='small status-good'>"
                "Auto-added: "
                + esc(
                    ", ".join(
                        str(x)
                        for x in auto_added
                    )
                )
                + "</div>"
            )

        if suggested:
            decision += (
                "<div class='small status-warn'>"
                "Suggested only: "
                + esc(
                    ", ".join(
                        str(x)
                        for x in suggested
                    )
                )
                + "</div>"
            )

        rows.append(
            """
            <div style="
              border-top:1px solid var(--border);
              padding:9px 0;
            ">
              <strong>{source_tag}</strong>
              <div class="small muted">
                AI candidate: {candidates}
              </div>
              {verification_bits}
              {decision}
            </div>
            """.format(
                source_tag=esc(
                    source_tag
                ),
                candidates=esc(
                    ", ".join(
                        str(x)
                        for x in candidates
                    )
                    or "none"
                ),
                verification_bits=(
                    "".join(
                        verification_bits
                    )
                    or (
                        "<div class='small status-bad'>"
                        "No verification result"
                        "</div>"
                    )
                ),
                decision=decision,
            )
        )

    error_text = str(
        verification.get(
            "error",
            "",
        )
        or ""
    )

    status_line = (
        "<div class='small muted'>"
        "Verifier status: "
        + esc(status)
        + "</div>"
    )

    if error_text:
        status_line += (
            "<div class='small status-bad'>"
            + esc(error_text)
            + "</div>"
        )

    if not rows:
        rows.append(
            "<div class='small muted'>"
            "No unknown Pixiv tags required verification."
            "</div>"
        )

    return (
        "<div class='panel'>"
        "<details open>"
        "<summary>Pixiv → NAID Tag Verification</summary>"
        + status_line
        + "".join(
            rows
        )
        + "</details>"
        "</div>"
    )


# ============================================================
# HTML PAGE
# ============================================================

def page_html(
    *,
    selected_run_id: Optional[int],
    message: str = "",
    error: str = "",
) -> str:
    cfg = load_base_config()
    state = load_ui_state(
        cfg
    )

    conn = connect_db()

    try:
        images = image_library(
            conn
        )

        all_runs = recent_analysis_runs(
            conn
        )

        valid_run_ids = {
            int(
                row[
                    "id"
                ]
            )
            for row in all_runs
        }

        if selected_run_id is None:
            saved = state.get(
                "last_run_id"
            )

            if (
                saved is not None
                and int(
                    saved
                )
                in valid_run_ids
            ):
                selected_run_id = int(
                    saved
                )

            elif all_runs:
                selected_run_id = int(
                    all_runs[0][
                        "id"
                    ]
                )

        selected = (
            get_analysis_run(
                conn,
                selected_run_id,
            )
            if selected_run_id
            is not None
            else None
        )

        selected_image_runs = (
            runs_for_image(
                conn,
                int(
                    selected[
                        "image_id"
                    ]
                ),
            )
            if selected
            else []
        )

        generations = (
            get_generation_runs(
                conn,
                int(
                    selected_run_id
                ),
                limit=30,
            )
            if selected_run_id
            is not None
            else []
        )

    finally:
        conn.close()

    lora_names, lora_error = (
        comfy_lora_names(
            cfg
        )
    )

    slots = default_slot_values(
        state
    )

    correction_model = str(
        state.get(
            "correction_model",
            DEFAULT_CORRECTION_MODEL,
        )
        or DEFAULT_CORRECTION_MODEL
    )

    msg_html = (
        f'<div class="message">'
        f'{esc(message)}'
        f'</div>'
        if message
        else ""
    )

    error_html = (
        f'<div class="message error">'
        f'{esc(error)}'
        f'</div>'
        if error
        else ""
    )

    # --------------------------------------------------------
    # Image library
    # --------------------------------------------------------

    image_cards = []

    selected_image_id = (
        int(
            selected[
                "image_id"
            ]
        )
        if selected
        else None
    )

    for row in images:
        image_id = int(
            row[
                "image_id"
            ]
        )

        latest_run_id = row[
            "latest_run_id"
        ]

        if latest_run_id is None:
            continue

        active = (
            " active"
            if (
                selected_image_id
                == image_id
            )
            else ""
        )

        image_cards.append(
            f"""
            <div class="image-card{active}">
              <a href="/?run_id={int(latest_run_id)}">
                <img
                  class="image-thumb"
                  src="/image?image_id={image_id}"
                  alt="{esc(row['filename'])}"
                  loading="lazy"
                >
                <div class="image-card-text">
                  <strong>{esc(row['filename'])}</strong>
                  <span class="muted">
                    {int(row['run_count'])} analysis run(s)
                  </span>
                </div>
              </a>
            </div>
            """
        )

    # --------------------------------------------------------
    # Analysis run select
    # --------------------------------------------------------

    run_options = []

    for row in selected_image_runs:
        rid = int(
            row[
                "id"
            ]
        )

        selected_attr = (
            " selected"
            if (
                selected_run_id
                == rid
            )
            else ""
        )

        label = (
            f"Run #{rid} · "
            f"v{row['analyzer_version']} · "
            f"{row['analyzer_timestamp']}"
        )

        run_options.append(
            f'<option value="{rid}"'
            f'{selected_attr}>'
            f'{esc(label)}'
            f'</option>'
        )

    # --------------------------------------------------------
    # Loras
    # --------------------------------------------------------

    lora_options = "\n".join(
        f'<option value="{esc(name)}">'
        f'</option>'
        for name in lora_names
    )

    lora_rows = []

    for i, slot in enumerate(
        slots,
        start=1,
    ):
        checked = (
            " checked"
            if slot[
                "enabled"
            ]
            else ""
        )

        lora_rows.append(
            f"""
            <div class="checkbox-inline">
              <input
                id="lora{i}_enabled"
                name="lora{i}_enabled"
                type="checkbox"
                {checked}
              >
              <label for="lora{i}_enabled">
                Slot {i}
              </label>
            </div>

            <input
              name="lora{i}_name"
              type="text"
              list="lora-options"
              value="{esc(slot['name'])}"
              placeholder="选择 LoRA"
            >

            <input
              name="lora{i}_model"
              type="number"
              value="{slot['strength_model']}"
              min="-4"
              max="4"
              step="0.05"
            >

            <input
              name="lora{i}_clip"
              type="number"
              value="{slot['strength_clip']}"
              min="-4"
              max="4"
              step="0.05"
            >
            """
        )

    comfy_status = (
        f'<span class="status-good">'
        f'ComfyUI OK · '
        f'{len(lora_names)} LoRA(s)'
        f'</span>'
        if not lora_error
        else (
            f'<span class="status-bad">'
            f'ComfyUI unavailable: '
            f'{esc(lora_error)}'
            f'</span>'
        )
    )

    # --------------------------------------------------------
    # Original / editor
    # --------------------------------------------------------

    if selected:
        base_prompt = str(
            selected[
                "final_prompt"
            ]
            or ""
        )

        pixiv_verify_panel = (
            pixiv_verification_html(
                selected[
                    "raw_json"
                ]
            )
        )

        original_column = f"""
        <div class="original-column">
          <div class="panel">
            <div class="toolbar">
              <div class="field grow">
                <label>Current image</label>
                <strong>{esc(selected['filename'])}</strong>
              </div>

              <div class="field grow">
                <label>Analysis history</label>
                <select
                  onchange="
                    window.location='/?run_id='+this.value
                  "
                >
                  {''.join(run_options)}
                </select>
              </div>
            </div>

            <div class="tabline" style="margin-top:10px">
              <span class="chip">
                Run #{int(selected['id'])}
              </span>

              <span class="chip">
                Analyzer {esc(selected['analyzer_version'])}
              </span>

              <span class="chip">
                Review:
                {'YES' if selected['review_needed'] else 'no'}
              </span>
            </div>
          </div>

          <div class="image-frame">
            <img
              src="/original?run_id={int(selected['id'])}"
              alt="Original"
            >
          </div>

          <div class="panel">
            <details>
              <summary>Analyzer Final Prompt</summary>
              <div
                id="base-prompt-display"
                class="prompt-box"
              >{esc(base_prompt)}</div>
            </details>
          </div>

          {pixiv_verify_panel}

          <div class="panel">
            <details>
              <summary>重新分析这张图片</summary>

              <form
                method="post"
                action="/reanalyze"
                onsubmit="
                  return markBusy(
                    this,
                    '重新分析中… VLM + WD14 可能需要几分钟'
                  )
                "
              >
                <input
                  type="hidden"
                  name="run_id"
                  value="{int(selected['id'])}"
                >

                <div class="field" style="margin-top:10px">
                  <label>Source type</label>
                  <select name="source_type">
                    <option value="none">none</option>
                    <option value="danbooru">danbooru</option>
                    <option value="pixiv">pixiv</option>
                    <option value="raw">raw</option>
                  </select>
                </div>

                <div class="field" style="margin-top:10px">
                  <label>Source Tags（可留空）</label>
                  <textarea
                    name="source_tags"
                    placeholder="Danbooru/Pixiv tags；不需要时留空"
                  ></textarea>
                </div>

                <button
                  type="submit"
                  style="margin-top:10px"
                >
                  重新分析并保留旧 Run
                </button>
              </form>
            </details>
          </div>
        </div>
        """
    else:
        base_prompt = ""

        original_column = """
        <div class="panel">
          还没有分析记录。请先上传图片。
        </div>
        """

    # --------------------------------------------------------
    # Candidate cards
    # --------------------------------------------------------

    candidate_cards = []

    for row in generations:
        gid = int(
            row[
                "id"
            ]
        )

        score = row[
            "overall_score"
        ]

        score_buttons = []

        for value in range(
            1,
            6,
        ):
            checked = (
                " checked"
                if score
                == value
                else ""
            )

            score_buttons.append(
                f"""
                <span class="score">
                  <input
                    id="g{gid}s{value}"
                    type="radio"
                    name="score"
                    value="{value}"
                    {checked}
                    required
                  >
                  <label for="g{gid}s{value}">
                    {value}
                  </label>
                </span>
                """
            )

        parent_text = (
            f" · revision of "
            f"#{row['parent_generation_id']}"
            if row[
                "parent_generation_id"
            ]
            else ""
        )

        correction_badge = ""

        if row[
            "user_instruction"
        ]:
            correction_badge = (
                "<div class='small status-good' "
                "style='margin-top:5px'>"
                "Correction: "
                f"{esc(row['user_instruction'])}"
                "</div>"
            )

        diagnostic_text = diagnostic_summary(
            row["sampling_json"]
        )
        diagnostic_badge = (
            "<div class='small status-warn' "
            "style='margin-top:5px'>"
            f"{esc(diagnostic_text)}"
            "</div>"
            if diagnostic_text
            else ""
        )

        candidate_cards.append(
            f"""
            <div class="candidate">
              <div class="candidate-image">
                <img
                  src="/generated?generation_id={gid}"
                  alt="Generation {gid}"
                  loading="lazy"
                >
              </div>

              <div class="candidate-meta">
                <strong>Generation #{gid}</strong>
                {esc(parent_text)}
                <br>
                Seed {esc(row['seed'])}
                · {esc(lora_summary(row['loras_json']))}
                · {esc(row['elapsed_seconds'])} sec
                {diagnostic_badge}
                {correction_badge}

                <details style="margin-top:7px">
                  <summary>实际生成 Prompt</summary>
                  <div class="prompt-box">
                    {esc(row['prompt'])}
                  </div>
                </details>
              </div>

              <div class="rating-area">
                <form
                  method="post"
                  action="/rate"
                >
                  <input
                    type="hidden"
                    name="run_id"
                    value="{int(selected_run_id)}"
                  >

                  <input
                    type="hidden"
                    name="generation_id"
                    value="{gid}"
                  >

                  <div class="rating-row">
                    <strong>相似度</strong>

                    {''.join(score_buttons)}

                    <input
                      class="rating-comment"
                      type="text"
                      name="comment"
                      value="{esc(row['rating_comment'] or '')}"
                      placeholder="评分备注（可选）"
                    >

                    <button type="submit">
                      保存评分
                    </button>
                  </div>
                </form>
              </div>

              <div class="correction-area">
                <form
                  method="post"
                  action="/correct-and-generate"
                  onsubmit="
                    return markBusy(
                      this,
                      'AI 正在修改 Prompt 并重新生成…'
                    )
                  "
                >
                  <input
                    type="hidden"
                    name="generation_id"
                    value="{gid}"
                  >

                  <input
                    type="hidden"
                    name="correction_model"
                    value="{esc(correction_model)}"
                  >

                  <div class="field">
                    <label>
                      对这张候选图的自然语言更正
                    </label>

                    <textarea
                      name="instruction"
                      placeholder="例如：她应该在舔腋下"
                      required
                    ></textarea>
                  </div>

                  <div class="action-row" style="margin-top:8px">
                    <button type="submit">
                      AI 更正 Prompt → 重新生成 1 张
                    </button>

                    <button
                      type="button"
                      class="secondary"
                      onclick='loadGenerationPrompt(
                        {json.dumps(str(row["prompt"]), ensure_ascii=False)}
                      )'
                    >
                      把这张 Prompt 载入上方编辑器
                    </button>
                  </div>
                </form>
              </div>
            </div>
            """
        )

    if not candidate_cards:
        candidate_cards.append(
            """
            <div class="panel muted">
              暂无候选图。配置 Prompt / LoRA 后点击 Generate。
            </div>
            """
        )

    # --------------------------------------------------------
    # Main editor
    # --------------------------------------------------------

    selected_run_hidden = (
        str(
            selected_run_id
        )
        if selected_run_id
        is not None
        else ""
    )

    count_default = int(
        state.get(
            "count",
            5,
        )
    )

    diagnostic_seed_default = str(
        state.get(
            "last_diagnostic_seed",
            "",
        )
        or ""
    )

    negative_prompt = str(
        state.get(
            "negative_prompt",
            cfg.get(
                "negative_prompt",
                "",
            ),
        )
        or ""
    )

    editor = (
        f"""
        <form
          id="generate-form"
          method="post"
          action="/generate"
          onsubmit="
            return markBusy(
              this,
              '生成中… 请等待 ComfyUI 完成'
            )
          "
        >
          <input
            type="hidden"
            name="run_id"
            value="{esc(selected_run_hidden)}"
          >

          <input
            id="base-prompt"
            type="hidden"
            value="{esc(base_prompt)}"
          >

          <div class="panel" id="prompt-editor">
            <h2>Prompt 编辑 / 更正</h2>

            <div class="field">
              <label>
                自然语言更正（中文即可）
              </label>

              <textarea
                id="correction-instruction"
                name="user_instruction"
                placeholder="例如：她应该在舔腋下；她不应该看镜头，应该闭眼"
              ></textarea>
            </div>

            <div class="toolbar" style="margin-top:8px">
              <div class="field grow">
                <label>Correction model</label>

                <input
                  id="correction-model"
                  name="correction_model"
                  type="text"
                  value="{esc(correction_model)}"
                >
              </div>

              <button
                type="button"
                onclick="runAICorrection()"
              >
                生成 AI 更正
              </button>
            </div>

            <div
              id="ai-note"
              class="ai-note"
              style="margin-top:8px"
            ></div>

            <div
              class="editor-grid"
              style="margin-top:10px"
            >
              <div class="field">
                <label>AI Add Tags</label>

                <textarea
                  id="ai-add-tags"
                  name="ai_add_tags"
                  placeholder="AI 建议加入的 tags"
                  oninput="refreshPromptPreview()"
                ></textarea>
              </div>

              <div class="field">
                <label>AI Remove Tags</label>

                <textarea
                  id="ai-remove-tags"
                  name="ai_remove_tags"
                  placeholder="AI 建议删除的冲突 tags"
                  oninput="refreshPromptPreview()"
                ></textarea>
              </div>
            </div>

            <div class="field" style="margin-top:10px">
              <label>
                Manual Positive Prompt
              </label>

              <textarea
                id="manual-positive"
                name="manual_positive"
                placeholder="你自己想直接追加的 positive tags / prompt"
                oninput="refreshPromptPreview()"
              ></textarea>
            </div>

            <div class="field" style="margin-top:10px">
              <label>
                最终实际生成 Positive Prompt
                （可直接手动编辑）
              </label>

              <textarea
                id="final-prompt"
                class="final-prompt"
                name="final_prompt_override"
                required
              >{esc(base_prompt)}</textarea>
            </div>

            <div class="action-row" style="margin-top:8px">
              <button
                type="button"
                class="secondary"
                onclick="refreshPromptPreview()"
              >
                根据更正重新计算 Prompt
              </button>

              <button
                type="button"
                class="ghost"
                onclick="resetPromptEditor()"
              >
                恢复 Analyzer Prompt
              </button>
            </div>
          </div>

          <div class="panel">
            <h2>Generation Settings</h2>

            <div class="toolbar">
              <div class="field">
                <label>生成数量</label>

                <select name="count">
                  <option
                    value="1"
                    {'selected' if count_default == 1 else ''}
                  >1</option>

                  <option
                    value="5"
                    {'selected' if count_default == 5 else ''}
                  >5</option>

                  <option
                    value="10"
                    {'selected' if count_default == 10 else ''}
                  >10</option>
                </select>
              </div>

              <div class="field grow">
                <label>ComfyUI</label>
                <div>{comfy_status}</div>
              </div>

              <div class="field grow">
                <label>A/B/C 固定 Seed（留空则随机）</label>
                <input
                  type="number"
                  name="diagnostic_seed"
                  min="0"
                  max="9223372036854775807"
                  value="{esc(diagnostic_seed_default)}"
                  placeholder="同一 seed 生成 A / B / C"
                >
              </div>
            </div>

            <div class="small" style="margin-top:10px">
              质量诊断会各生成 1 张：A 当前 LoRA + 双 sampler；
              B 关闭 LoRA + 双 sampler；C 关闭 LoRA + 单 sampler、CFG 6。
              三张严格共用当前 Prompt、seed、分辨率和 checkpoint。
            </div>

            <hr
              style="
                border:0;
                border-top:1px solid var(--border);
                margin:14px 0;
              "
            >

            <div class="lora-grid">
              <div class="lora-head">Enable</div>
              <div class="lora-head">LoRA</div>
              <div class="lora-head">Model</div>
              <div class="lora-head">CLIP</div>

              {''.join(lora_rows)}
            </div>

            <datalist id="lora-options">
              {lora_options}
            </datalist>

            <div class="field" style="margin-top:14px">
              <label>Negative Prompt</label>

              <textarea
                name="negative_prompt"
              >{esc(negative_prompt)}</textarea>
            </div>
          </div>

          <div class="sticky-generate">
            <button
              type="submit"
              class="secondary"
              formaction="/diagnose-generation"
              {'disabled' if selected is None else ''}
            >
              A/B/C 质量诊断
            </button>

            <button
              type="submit"
              {'disabled' if selected is None else ''}
            >
              Generate
            </button>
          </div>
        </form>
        """
        if selected
        else ""
    )

    # --------------------------------------------------------
    # Full page
    # --------------------------------------------------------

    lan_ip = get_lan_ip()

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta
    name="viewport"
    content="width=device-width, initial-scale=1"
  >
  <title>Illustrious Reconstruction Studio</title>
  <style>{CSS}</style>
</head>

<body>
<header>
  <div class="header-inner">
    <div>
      <h1>Illustrious Reconstruction Studio v2.1</h1>
      <div class="header-sub">
        Analyze · Correct · Generate · Compare · Learn
      </div>
    </div>

    <div class="lan-pill">
      手机：
      http://{esc(lan_ip)}:{PORT}
    </div>
  </div>
</header>

<main>
  {msg_html}
  {error_html}

  <div class="panel">
    <details>
      <summary>
        ＋ 添加新图片并分析
      </summary>

      <form
        method="post"
        action="/upload-analyze"
        enctype="multipart/form-data"
        onsubmit="
          return markBusy(
            this,
            '上传并分析中… 每张图片可能需要几分钟'
          )
        "
        style="margin-top:12px"
      >
        <div class="field">
          <label>
            从电脑或手机选择图片
          </label>

          <input
            type="file"
            name="images"
            accept="image/*"
            multiple
            required
          >
        </div>

        <div class="toolbar" style="margin-top:10px">
          <div class="field">
            <label>Source type</label>

            <select name="source_type">
              <option value="none">none</option>
              <option value="danbooru">danbooru</option>
              <option value="pixiv">pixiv</option>
              <option value="raw">raw</option>
            </select>
          </div>

          <div class="field grow">
            <label>
              Source Tags（可选）
            </label>

            <textarea
              name="source_tags"
              placeholder="多张图片一起上传时，建议 source_type=none；有来源 tags 时逐张分析更准确。"
            ></textarea>
          </div>
        </div>

        <button
          type="submit"
          style="margin-top:10px"
        >
          上传并开始分析
        </button>
      </form>
    </details>
  </div>

  <div class="panel">
    <div class="toolbar">
      <div class="field grow">
        <label>已分析图片库</label>
        <div class="small muted">
          点击缩略图切换；同一图片可保留多个 Analysis Run。
        </div>
      </div>
    </div>

    <div
      class="image-library"
      style="margin-top:10px"
    >
      {''.join(image_cards) if image_cards else '<span class="muted">暂无图片</span>'}
    </div>
  </div>

  <div class="workspace">
    <div>
      {original_column}
    </div>

    <div>
      {editor}

      <div class="panel">
        <h2 style="margin-bottom:4px">
          Generated Candidates
        </h2>
        <div class="small muted">
          每张图可评分，也可以直接输入自然语言更正后重新生成。
        </div>
      </div>

      {''.join(candidate_cards)}
    </div>
  </div>
</main>

<script>
function splitPrompt(text) {{
  return String(text || '')
    .replace(/\\r/g, '\\n')
    .split(/[\\n,]+/)
    .map(x => x.trim().replace(/\\s+/g, ' '))
    .filter(Boolean);
}}

function normTag(x) {{
  return String(x || '')
    .trim()
    .toLowerCase()
    .replace(/_/g, ' ')
    .replace(/\\s+/g, ' ');
}}

function dedupe(items) {{
  const out = [];
  const seen = new Set();

  for (const item of items) {{
    const key = normTag(item);

    if (!key || seen.has(key)) continue;

    seen.add(key);
    out.push(item.trim());
  }}

  return out;
}}

function buildEditedPrompt(
  base,
  removeText,
  addText,
  manualText
) {{
  const remove = new Set(
    splitPrompt(removeText).map(normTag)
  );

  const kept = splitPrompt(base).filter(
    item => !remove.has(normTag(item))
  );

  return dedupe(
    kept
      .concat(splitPrompt(addText))
      .concat(splitPrompt(manualText))
  ).join(', ');
}}

function refreshPromptPreview() {{
  const base = document.getElementById('base-prompt');
  const add = document.getElementById('ai-add-tags');
  const remove = document.getElementById('ai-remove-tags');
  const manual = document.getElementById('manual-positive');
  const finalBox = document.getElementById('final-prompt');

  if (!base || !finalBox) return;

  finalBox.value = buildEditedPrompt(
    base.value,
    remove ? remove.value : '',
    add ? add.value : '',
    manual ? manual.value : ''
  );
}}

function resetPromptEditor() {{
  const base = document.getElementById('base-prompt');
  const instruction = document.getElementById('correction-instruction');
  const add = document.getElementById('ai-add-tags');
  const remove = document.getElementById('ai-remove-tags');
  const manual = document.getElementById('manual-positive');
  const finalBox = document.getElementById('final-prompt');
  const note = document.getElementById('ai-note');

  if (instruction) instruction.value = '';
  if (add) add.value = '';
  if (remove) remove.value = '';
  if (manual) manual.value = '';
  if (note) note.textContent = '';
  if (base && finalBox) finalBox.value = base.value;
}}

async function runAICorrection() {{
  const instruction = document.getElementById(
    'correction-instruction'
  );

  const model = document.getElementById(
    'correction-model'
  );

  const finalBox = document.getElementById(
    'final-prompt'
  );

  const add = document.getElementById(
    'ai-add-tags'
  );

  const remove = document.getElementById(
    'ai-remove-tags'
  );

  const note = document.getElementById(
    'ai-note'
  );

  if (!instruction || !instruction.value.trim()) {{
    alert('请先输入更正指令。');
    return;
  }}

  note.textContent = 'AI 正在转换更正指令…';

  try {{
    const response = await fetch('/api/correct', {{
      method: 'POST',
      headers: {{
        'Content-Type': 'application/json'
      }},
      body: JSON.stringify({{
        base_prompt: finalBox.value,
        instruction: instruction.value,
        model: model.value
      }})
    }});

    const data = await response.json();

    if (!response.ok || data.error) {{
      throw new Error(
        data.error || 'Correction failed.'
      );
    }}

    add.value = (data.add_tags || []).join(', ');
    remove.value = (data.remove_tags || []).join(', ');

    note.textContent =
      data.note
      ? 'AI: ' + data.note
      : 'AI 更正已生成。';

    refreshPromptPreview();

  }} catch (err) {{
    note.textContent = '错误：' + err.message;
  }}
}}

function loadGenerationPrompt(prompt) {{
  const base = document.getElementById(
    'base-prompt'
  );

  const finalBox = document.getElementById(
    'final-prompt'
  );

  const instruction = document.getElementById(
    'correction-instruction'
  );

  const add = document.getElementById(
    'ai-add-tags'
  );

  const remove = document.getElementById(
    'ai-remove-tags'
  );

  const manual = document.getElementById(
    'manual-positive'
  );

  if (base) base.value = prompt;
  if (finalBox) finalBox.value = prompt;
  if (instruction) instruction.value = '';
  if (add) add.value = '';
  if (remove) remove.value = '';
  if (manual) manual.value = '';

  document.getElementById(
    'prompt-editor'
  ).scrollIntoView({{
    behavior: 'smooth',
    block: 'start'
  }});
}}

function markBusy(form, text) {{
  const buttons = form.querySelectorAll(
    'button[type="submit"]'
  );

  for (const button of buttons) {{
    button.disabled = true;
    button.dataset.oldText = button.textContent;
    button.textContent = text || '处理中…';
  }}

  return true;
}}

refreshPromptPreview();
</script>

</body>
</html>
"""


# ============================================================
# IMAGE SERVING
# ============================================================

def send_file(
    handler: BaseHTTPRequestHandler,
    path: Path,
) -> None:
    if (
        not path.exists()
        or not path.is_file()
    ):
        handler.send_error(
            404,
            "Image file not found.",
        )
        return

    mime = (
        mimetypes.guess_type(
            path.name
        )[0]
        or "application/octet-stream"
    )

    data = path.read_bytes()

    handler.send_response(
        200
    )
    handler.send_header(
        "Content-Type",
        mime,
    )
    handler.send_header(
        "Content-Length",
        str(
            len(
                data
            )
        ),
    )
    handler.send_header(
        "Cache-Control",
        "no-store",
    )
    handler.end_headers()
    handler.wfile.write(
        data
    )


def image_path_for_image_id(
    image_id: int,
) -> Optional[Path]:
    conn = connect_db()

    try:
        row = conn.execute(
            """
            SELECT first_seen_path
            FROM images
            WHERE id=?
            """,
            (
                image_id,
            ),
        ).fetchone()

        if not row:
            return None

        raw = str(
            row[
                "first_seen_path"
            ]
            or ""
        ).strip()

        return (
            Path(
                raw
            )
            if raw
            else None
        )

    finally:
        conn.close()


def original_path_for_run(
    run_id: int,
) -> Optional[Path]:
    conn = connect_db()

    try:
        row = get_analysis_run(
            conn,
            run_id,
        )

        if not row:
            return None

        raw = str(
            row[
                "first_seen_path"
            ]
            or ""
        ).strip()

        return (
            Path(
                raw
            )
            if raw
            else None
        )

    finally:
        conn.close()


def generated_path_for_id(
    generation_id: int,
) -> Optional[Path]:
    conn = connect_db()

    try:
        row = get_generation_run(
            conn,
            generation_id,
        )

        if not row:
            return None

        raw = str(
            row[
                "generated_image_path"
            ]
            or ""
        ).strip()

        return (
            Path(
                raw
            )
            if raw
            else None
        )

    finally:
        conn.close()


# ============================================================
# WEB SERVER
# ============================================================

class Handler(
    BaseHTTPRequestHandler
):

    def log_message(
        self,
        format: str,
        *args: Any,
    ) -> None:
        # Keep terminal readable.
        pass

    def send_html(
        self,
        body: str,
        status: int = 200,
    ) -> None:
        raw = body.encode(
            "utf-8"
        )

        self.send_response(
            status
        )
        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(
                len(
                    raw
                )
            ),
        )
        self.send_header(
            "Cache-Control",
            "no-store",
        )
        self.end_headers()
        self.wfile.write(
            raw
        )

    def send_json(
        self,
        payload: Dict[str, Any],
        status: int = 200,
    ) -> None:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode(
            "utf-8"
        )

        self.send_response(
            status
        )
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(
                len(
                    raw
                )
            ),
        )
        self.send_header(
            "Cache-Control",
            "no-store",
        )
        self.end_headers()
        self.wfile.write(
            raw
        )

    def redirect(
        self,
        path: str,
        params: Dict[str, Any],
    ) -> None:
        query = urllib.parse.urlencode(
            {
                key: value
                for key, value
                in params.items()
                if value is not None
            }
        )

        target = (
            path
            + (
                "?"
                + query
                if query
                else ""
            )
        )

        self.send_response(
            303
        )
        self.send_header(
            "Location",
            target,
        )
        self.end_headers()

    def parse_urlencoded_form(
        self,
    ) -> Dict[str, List[str]]:
        length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
            or "0"
        )

        raw = self.rfile.read(
            length
        ).decode(
            "utf-8",
            errors="replace",
        )

        return urllib.parse.parse_qs(
            raw,
            keep_blank_values=True,
        )

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def do_GET(
        self,
    ) -> None:
        parsed = urllib.parse.urlparse(
            self.path
        )

        params = urllib.parse.parse_qs(
            parsed.query
        )

        if parsed.path == "/":
            raw_run = first_value(
                params,
                "run_id",
                "",
            )

            run_id = (
                int(
                    raw_run
                )
                if raw_run.strip()
                else None
            )

            self.send_html(
                page_html(
                    selected_run_id=(
                        run_id
                    ),
                    message=first_value(
                        params,
                        "message",
                        "",
                    ),
                    error=first_value(
                        params,
                        "error",
                        "",
                    ),
                )
            )
            return

        if parsed.path == "/image":
            raw = first_value(
                params,
                "image_id",
                "",
            )

            if not raw:
                self.send_error(
                    400
                )
                return

            path = image_path_for_image_id(
                int(
                    raw
                )
            )

            if path is None:
                self.send_error(
                    404
                )
                return

            send_file(
                self,
                path,
            )
            return

        if parsed.path == "/original":
            raw = first_value(
                params,
                "run_id",
                "",
            )

            if not raw:
                self.send_error(
                    400
                )
                return

            path = original_path_for_run(
                int(
                    raw
                )
            )

            if path is None:
                self.send_error(
                    404
                )
                return

            send_file(
                self,
                path,
            )
            return

        if parsed.path == "/generated":
            raw = first_value(
                params,
                "generation_id",
                "",
            )

            if not raw:
                self.send_error(
                    400
                )
                return

            path = generated_path_for_id(
                int(
                    raw
                )
            )

            if path is None:
                self.send_error(
                    404
                )
                return

            send_file(
                self,
                path,
            )
            return

        self.send_error(
            404
        )

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    def do_POST(
        self,
    ) -> None:
        parsed = urllib.parse.urlparse(
            self.path
        )

        # ----------------------------------------------------
        # JSON: AI correction preview
        # ----------------------------------------------------

        if parsed.path == "/api/correct":
            try:
                length = int(
                    self.headers.get(
                        "Content-Length",
                        "0",
                    )
                    or "0"
                )

                payload = json.loads(
                    self.rfile.read(
                        length
                    ).decode(
                        "utf-8"
                    )
                    or "{}"
                )

                if not CORRECTION_LOCK.acquire(
                    blocking=False
                ):
                    self.send_json(
                        {
                            "error": (
                                "另一个 AI 更正任务正在运行。"
                            )
                        },
                        status=409,
                    )
                    return

                try:
                    result = (
                        ai_prompt_correction(
                            base_prompt=str(
                                payload.get(
                                    "base_prompt",
                                    "",
                                )
                            ),
                            instruction=str(
                                payload.get(
                                    "instruction",
                                    "",
                                )
                            ),
                            model=str(
                                payload.get(
                                    "model",
                                    DEFAULT_CORRECTION_MODEL,
                                )
                            ),
                        )
                    )

                finally:
                    CORRECTION_LOCK.release()

                status = (
                    400
                    if result.get(
                        "error"
                    )
                    else 200
                )

                self.send_json(
                    result,
                    status=status,
                )

            except Exception as exc:
                traceback.print_exc()

                self.send_json(
                    {
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )
                    },
                    status=500,
                )

            return

        # ----------------------------------------------------
        # Multipart: upload + analyze
        # ----------------------------------------------------

        if parsed.path == "/upload-analyze":
            try:
                fields, files = (
                    parse_multipart(
                        self
                    )
                )

                uploads = files.get(
                    "images",
                    [],
                )

                if not uploads:
                    raise ValueError(
                        "没有收到图片。"
                    )

                source_type = first_value(
                    fields,
                    "source_type",
                    "none",
                ).strip().lower()

                source_tags = first_value(
                    fields,
                    "source_tags",
                    "",
                )

                if (
                    len(
                        uploads
                    )
                    > 1
                    and source_type
                    != "none"
                    and source_tags.strip()
                ):
                    raise ValueError(
                        "批量图片不能共用同一组 Source Tags。"
                        "请 source_type=none 批量分析，"
                        "或逐张上传来源 tags。"
                    )

                if not ANALYSIS_LOCK.acquire(
                    blocking=False
                ):
                    raise RuntimeError(
                        "已有分析任务正在运行。"
                    )

                newest_run_id = None

                try:
                    for upload in uploads:
                        image_path = (
                            save_uploaded_image(
                                upload[
                                    "filename"
                                ],
                                upload[
                                    "content"
                                ],
                            )
                        )

                        (
                            newest_run_id,
                            _result,
                            _json_path,
                        ) = analyze_and_import(
                            image_path,
                            source_type=(
                                source_type
                            ),
                            source_tags_text=(
                                source_tags
                            ),
                        )

                finally:
                    ANALYSIS_LOCK.release()

                self.redirect(
                    "/",
                    {
                        "run_id": (
                            newest_run_id
                        ),
                        "message": (
                            f"已分析 "
                            f"{len(uploads)} 张图片。"
                        ),
                    },
                )

            except Exception as exc:
                traceback.print_exc()

                self.redirect(
                    "/",
                    {
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )
                    },
                )

            return

        # Remaining POST routes are urlencoded.
        form = self.parse_urlencoded_form()

        # ----------------------------------------------------
        # Re-analyze current original
        # ----------------------------------------------------

        if parsed.path == "/reanalyze":
            old_run_id = int(
                first_value(
                    form,
                    "run_id",
                )
            )

            try:
                if not ANALYSIS_LOCK.acquire(
                    blocking=False
                ):
                    raise RuntimeError(
                        "已有分析任务正在运行。"
                    )

                try:
                    image_path = (
                        original_path_for_run(
                            old_run_id
                        )
                    )

                    if image_path is None:
                        raise FileNotFoundError(
                            "找不到原图。"
                        )

                    (
                        new_run_id,
                        _result,
                        _json_path,
                    ) = analyze_and_import(
                        image_path,
                        source_type=first_value(
                            form,
                            "source_type",
                            "none",
                        ),
                        source_tags_text=(
                            first_value(
                                form,
                                "source_tags",
                                "",
                            )
                        ),
                    )

                finally:
                    ANALYSIS_LOCK.release()

                self.redirect(
                    "/",
                    {
                        "run_id": (
                            new_run_id
                        ),
                        "message": (
                            f"重新分析完成："
                            f"Run #{new_run_id}"
                        ),
                    },
                )

            except Exception as exc:
                traceback.print_exc()

                self.redirect(
                    "/",
                    {
                        "run_id": (
                            old_run_id
                        ),
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    },
                )

            return

        # ----------------------------------------------------
        # Controlled A/B/C generation-quality diagnostic
        # ----------------------------------------------------

        if parsed.path == "/diagnose-generation":
            run_id = int(
                first_value(
                    form,
                    "run_id",
                )
            )

            try:
                if not GENERATION_LOCK.acquire(
                    blocking=False
                ):
                    raise RuntimeError(
                        "已有生成任务正在运行。"
                    )

                try:
                    cfg = config_for_form(
                        form
                    )

                    final_prompt = first_value(
                        form,
                        "final_prompt_override",
                        "",
                    ).strip()
                    if not final_prompt:
                        raise ValueError(
                            "最终 Positive Prompt 为空。"
                        )

                    raw_seed = first_value(
                        form,
                        "diagnostic_seed",
                        "",
                    ).strip()
                    fixed_seed = (
                        int(raw_seed)
                        if raw_seed
                        else generator.make_seeds(1)[0]
                    )
                    if fixed_seed < 0 or fixed_seed >= 2**63:
                        raise ValueError(
                            "Seed 必须在 0 到 2^63 - 1 之间。"
                        )

                    correction_model = first_value(
                        form,
                        "correction_model",
                        DEFAULT_CORRECTION_MODEL,
                    ).strip()
                    user_instruction = first_value(
                        form,
                        "user_instruction",
                        "",
                    )
                    ai_add_tags = split_prompt_fragments(
                        first_value(
                            form,
                            "ai_add_tags",
                            "",
                        )
                    )
                    ai_remove_tags = split_prompt_fragments(
                        first_value(
                            form,
                            "ai_remove_tags",
                            "",
                        )
                    )
                    manual_positive = first_value(
                        form,
                        "manual_positive",
                        "",
                    )

                    conn = connect_db()
                    try:
                        analysis = get_analysis_run(
                            conn,
                            run_id,
                        )
                        if not analysis:
                            raise ValueError(
                                f"Run #{run_id} not found."
                            )
                        analyzer_prompt = str(
                            analysis["final_prompt"]
                            or ""
                        )
                    finally:
                        conn.close()

                    results = generator.generate_diagnostic_abc(
                        run_id=run_id,
                        cfg=cfg,
                        prompt_override=final_prompt,
                        seed=fixed_seed,
                    )

                    for result in results:
                        variant = str(
                            result.get(
                                "diagnostic_variant",
                                "unknown",
                            )
                        ).lower()
                        save_prompt_edit_records(
                            [result],
                            parent_generation_id=None,
                            edit_kind=f"diagnostic_{variant}",
                            base_prompt=analyzer_prompt,
                            user_instruction=user_instruction,
                            correction_model=correction_model,
                            ai_add_tags=ai_add_tags,
                            ai_remove_tags=ai_remove_tags,
                            manual_positive=manual_positive,
                            final_prompt=final_prompt,
                        )

                    state = load_ui_state(
                        load_base_config()
                    )
                    state.update({
                        "lora_slots": cfg["lora_slots"],
                        "negative_prompt": cfg["negative_prompt"],
                        "last_run_id": run_id,
                        "last_diagnostic_seed": fixed_seed,
                        "correction_model": correction_model,
                    })
                    save_ui_state(state)

                finally:
                    GENERATION_LOCK.release()

                self.redirect(
                    "/",
                    {
                        "run_id": run_id,
                        "message": (
                            "A/B/C diagnostic completed: "
                            f"{len(results)}/3 · seed {fixed_seed}"
                        ),
                    },
                )

            except Exception as exc:
                traceback.print_exc()
                self.redirect(
                    "/",
                    {
                        "run_id": run_id,
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    },
                )

            return

        # ----------------------------------------------------
        # Main generation
        # ----------------------------------------------------

        if parsed.path == "/generate":
            run_id = int(
                first_value(
                    form,
                    "run_id",
                )
            )

            count = int(
                first_value(
                    form,
                    "count",
                    "5",
                )
            )

            if count not in {
                1,
                5,
                10,
            }:
                self.redirect(
                    "/",
                    {
                        "run_id": run_id,
                        "error": (
                            "count must be 1, 5 or 10."
                        ),
                    },
                )
                return

            try:
                if not GENERATION_LOCK.acquire(
                    blocking=False
                ):
                    raise RuntimeError(
                        "已有生成任务正在运行。"
                    )

                try:
                    cfg = config_for_form(
                        form
                    )

                    final_prompt = (
                        first_value(
                            form,
                            "final_prompt_override",
                            "",
                        ).strip()
                    )

                    if not final_prompt:
                        raise ValueError(
                            "最终 Positive Prompt 为空。"
                        )

                    correction_model = (
                        first_value(
                            form,
                            "correction_model",
                            DEFAULT_CORRECTION_MODEL,
                        ).strip()
                    )

                    user_instruction = (
                        first_value(
                            form,
                            "user_instruction",
                            "",
                        )
                    )

                    ai_add_tags = (
                        split_prompt_fragments(
                            first_value(
                                form,
                                "ai_add_tags",
                                "",
                            )
                        )
                    )

                    ai_remove_tags = (
                        split_prompt_fragments(
                            first_value(
                                form,
                                "ai_remove_tags",
                                "",
                            )
                        )
                    )

                    manual_positive = (
                        first_value(
                            form,
                            "manual_positive",
                            "",
                        )
                    )

                    conn = connect_db()

                    try:
                        analysis = get_analysis_run(
                            conn,
                            run_id,
                        )

                        if not analysis:
                            raise ValueError(
                                f"Run #{run_id} not found."
                            )

                        analyzer_prompt = str(
                            analysis[
                                "final_prompt"
                            ]
                            or ""
                        )

                    finally:
                        conn.close()

                    results = (
                        generator.generate_candidates(
                            run_id=run_id,
                            count=count,
                            cfg=cfg,
                            prompt_override=(
                                final_prompt
                            ),
                        )
                    )

                    edit_kind = (
                        "edited"
                        if (
                            final_prompt.strip()
                            != analyzer_prompt.strip()
                        )
                        else "base"
                    )

                    save_prompt_edit_records(
                        results,
                        parent_generation_id=None,
                        edit_kind=edit_kind,
                        base_prompt=(
                            analyzer_prompt
                        ),
                        user_instruction=(
                            user_instruction
                        ),
                        correction_model=(
                            correction_model
                        ),
                        ai_add_tags=(
                            ai_add_tags
                        ),
                        ai_remove_tags=(
                            ai_remove_tags
                        ),
                        manual_positive=(
                            manual_positive
                        ),
                        final_prompt=(
                            final_prompt
                        ),
                    )

                    state = load_ui_state(
                        load_base_config()
                    )

                    state.update({
                        "lora_slots": cfg[
                            "lora_slots"
                        ],
                        "negative_prompt": cfg[
                            "negative_prompt"
                        ],
                        "count": count,
                        "last_run_id": run_id,
                        "correction_model": (
                            correction_model
                        ),
                    })

                    save_ui_state(
                        state
                    )

                finally:
                    GENERATION_LOCK.release()

                self.redirect(
                    "/",
                    {
                        "run_id": run_id,
                        "message": (
                            "Generation completed: "
                            f"{len(results)}/{count}"
                        ),
                    },
                )

            except Exception as exc:
                traceback.print_exc()

                self.redirect(
                    "/",
                    {
                        "run_id": run_id,
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    },
                )

            return

        # ----------------------------------------------------
        # Correct a specific candidate and regenerate 1
        # ----------------------------------------------------

        if parsed.path == "/correct-and-generate":
            generation_id = int(
                first_value(
                    form,
                    "generation_id",
                )
            )

            instruction = first_value(
                form,
                "instruction",
                "",
            ).strip()

            model = first_value(
                form,
                "correction_model",
                DEFAULT_CORRECTION_MODEL,
            ).strip()

            run_id: Optional[int] = None

            try:
                if not GENERATION_LOCK.acquire(
                    blocking=False
                ):
                    raise RuntimeError(
                        "已有生成任务正在运行。"
                    )

                try:
                    conn = connect_db()

                    try:
                        parent = (
                            get_generation_run(
                                conn,
                                generation_id,
                            )
                        )

                        if not parent:
                            raise ValueError(
                                "找不到这条 generation。"
                            )

                        run_id = int(
                            parent[
                                "analysis_run_id"
                            ]
                        )

                        parent_prompt = str(
                            parent[
                                "prompt"
                            ]
                            or ""
                        )

                        parent_negative = str(
                            parent[
                                "negative_prompt"
                            ]
                            or ""
                        )

                        parent_loras = str(
                            parent[
                                "loras_json"
                            ]
                            or "[]"
                        )

                    finally:
                        conn.close()

                    correction = (
                        ai_prompt_correction(
                            base_prompt=(
                                parent_prompt
                            ),
                            instruction=(
                                instruction
                            ),
                            model=model,
                        )
                    )

                    if correction[
                        "error"
                    ]:
                        raise ValueError(
                            correction[
                                "error"
                            ]
                        )

                    add_text = ", ".join(
                        correction[
                            "add_tags"
                        ]
                    )

                    remove_text = ", ".join(
                        correction[
                            "remove_tags"
                        ]
                    )

                    corrected_prompt = (
                        apply_prompt_edits(
                            parent_prompt,
                            remove_tags_text=(
                                remove_text
                            ),
                            add_tags_text=(
                                add_text
                            ),
                        )
                    )

                    cfg = load_base_config()

                    cfg[
                        "lora_slots"
                    ] = (
                        lora_slots_from_generation(
                            parent_loras
                        )
                    )

                    cfg[
                        "negative_prompt"
                    ] = (
                        parent_negative
                    )

                    cfg[
                        "apply_lora_to_refiner"
                    ] = True

                    cfg[
                        "use_lora_clip_for_negative"
                    ] = True

                    results = (
                        generator.generate_candidates(
                            run_id=run_id,
                            count=1,
                            cfg=cfg,
                            prompt_override=(
                                corrected_prompt
                            ),
                        )
                    )

                    save_prompt_edit_records(
                        results,
                        parent_generation_id=(
                            generation_id
                        ),
                        edit_kind=(
                            "ai_revision"
                        ),
                        base_prompt=(
                            parent_prompt
                        ),
                        user_instruction=(
                            instruction
                        ),
                        correction_model=model,
                        ai_add_tags=(
                            correction[
                                "add_tags"
                            ]
                        ),
                        ai_remove_tags=(
                            correction[
                                "remove_tags"
                            ]
                        ),
                        manual_positive="",
                        final_prompt=(
                            corrected_prompt
                        ),
                    )

                finally:
                    GENERATION_LOCK.release()

                self.redirect(
                    "/",
                    {
                        "run_id": run_id,
                        "message": (
                            f"Generation #{generation_id} "
                            "已按自然语言更正并生成新候选图。"
                        ),
                    },
                )

            except Exception as exc:
                traceback.print_exc()

                self.redirect(
                    "/",
                    {
                        "run_id": run_id,
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    },
                )

            return

        # ----------------------------------------------------
        # Rating
        # ----------------------------------------------------

        if parsed.path == "/rate":
            run_id = int(
                first_value(
                    form,
                    "run_id",
                )
            )

            generation_id = int(
                first_value(
                    form,
                    "generation_id",
                )
            )

            try:
                score = int(
                    first_value(
                        form,
                        "score",
                    )
                )

                comment = first_value(
                    form,
                    "comment",
                    "",
                )

                save_rating(
                    generation_id,
                    score,
                    comment,
                )

                state = load_ui_state(
                    load_base_config()
                )

                state[
                    "last_run_id"
                ] = run_id

                save_ui_state(
                    state
                )

                self.redirect(
                    "/",
                    {
                        "run_id": run_id,
                        "message": (
                            f"Generation #{generation_id} "
                            f"评分已保存：{score}/5"
                        ),
                    },
                )

            except Exception as exc:
                self.redirect(
                    "/",
                    {
                        "run_id": run_id,
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    },
                )

            return

        self.send_error(
            404
        )


# ============================================================
# START
# ============================================================

def main() -> None:
    cfg = load_base_config()
    lan_ip = get_lan_ip()

    print()
    print("=" * 76)
    print("ILLUSTRIOUS RECONSTRUCTION STUDIO V2")
    print("=" * 76)
    print(
        f"Database:  {cfg['database_path']}"
    )
    print(
        f"Workflow:  {cfg['workflow_path']}"
    )
    print(
        f"ComfyUI:   {cfg['comfy_url']}"
    )
    print(
        f"Ollama:    {OLLAMA_URL}"
    )
    print()
    print(
        f"Desktop:   http://127.0.0.1:{PORT}"
    )
    print(
        f"Phone/LAN: http://{lan_ip}:{PORT}"
    )
    print()
    print(
        "WARNING: LAN UI has no authentication. "
        "Do not expose port 8765 to the public Internet."
    )
    print(
        "Press Ctrl+C to stop."
    )
    print()

    server = ThreadingHTTPServer(
        (
            HOST,
            PORT,
        ),
        Handler,
    )

    try:
        webbrowser.open(
            f"http://127.0.0.1:{PORT}"
        )
    except Exception:
        pass

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print()
        print(
            "Stopping Studio..."
        )

    finally:
        server.server_close()


if __name__ == "__main__":
    main()
