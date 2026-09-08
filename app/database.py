# -*- coding: utf-8 -*-
r"""
Illustrious SQLite Manager v1.0
Standard-library only.

Default DB:
    F:\OpenWebUI\data\illustrious.sqlite3

Default Analyzer JSON folder:
    F:\OpenWebUI\runs

Examples:
    py F:\OpenWebUI\illustrious_db.py import
    py F:\OpenWebUI\illustrious_db.py stats
    py F:\OpenWebUI\illustrious_db.py recent --limit 10
    py F:\OpenWebUI\illustrious_db.py inspect 12
    py F:\OpenWebUI\illustrious_db.py rate 12 --rating 5 --accept
    py F:\OpenWebUI\illustrious_db.py export
    py F:\OpenWebUI\illustrious_db.py backup
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import ntpath
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


APP_VERSION = "1.1"
SCHEMA_VERSION = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DB = PROJECT_ROOT / "data" / "illustrious.sqlite3"
DEFAULT_RUNS = PROJECT_ROOT / "data" / "runs"
DEFAULT_EXPORTS = PROJECT_ROOT / "data" / "exports"
DEFAULT_BACKUPS = PROJECT_ROOT / "data" / "backups"


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def jd(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def jl(value: Optional[str], fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_path(text: str) -> str:
    text = str(text or "").strip().strip('"')
    if not text:
        return ""
    return ntpath.normcase(ntpath.normpath(text))


def normalize_tag(tag: str) -> str:
    return " ".join(str(tag).strip().lower().replace("_", " ").split())


def parse_prompt(prompt: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in str(prompt or "").split(","):
        tag = normalize_tag(raw)
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def prompt_diff(ai_prompt: str, corrected_prompt: str) -> Tuple[List[str], List[str]]:
    ai = parse_prompt(ai_prompt)
    corrected = parse_prompt(corrected_prompt)
    ai_set = set(ai)
    corrected_set = set(corrected)
    added = [x for x in corrected if x not in ai_set]
    removed = [x for x in ai if x not in corrected_set]
    return added, removed


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError:
        pass
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


SCHEMA = r"""
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_key TEXT NOT NULL UNIQUE,
    sha256 TEXT UNIQUE,
    filename TEXT,
    extension TEXT,
    first_seen_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS image_locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    path TEXT NOT NULL UNIQUE,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uid TEXT NOT NULL UNIQUE,
    image_id INTEGER NOT NULL,

    analyzer_timestamp TEXT,
    analyzer_version TEXT,
    prompt_kind TEXT NOT NULL DEFAULT 'pseudo_prompt',

    vlm_model TEXT,
    merger_model TEXT,
    target_adult INTEGER,
    target_uncensored INTEGER,

    vlm_caption TEXT,
    wd14_tags_text TEXT,

    visual_tags_json TEXT NOT NULL DEFAULT '[]',
    final_tags_json TEXT NOT NULL DEFAULT '[]',
    final_prompt TEXT,
    removed_tags_json TEXT NOT NULL DEFAULT '[]',

    participant_visibility_json TEXT NOT NULL DEFAULT '{}',
    review_needed INTEGER NOT NULL DEFAULT 0,
    review_reasons_json TEXT NOT NULL DEFAULT '[]',

    major_disagreements_json TEXT NOT NULL DEFAULT '[]',
    conflicts_json TEXT NOT NULL DEFAULT '[]',
    suspected_false_positives_json TEXT NOT NULL DEFAULT '[]',
    uncertain_tags_json TEXT NOT NULL DEFAULT '[]',
    rejected_visual_features_json TEXT NOT NULL DEFAULT '[]',
    notes_json TEXT NOT NULL DEFAULT '[]',

    elapsed_seconds REAL,

    source_json_path TEXT,
    source_json_sha256 TEXT,
    raw_json TEXT NOT NULL,

    imported_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_image ON analysis_runs(image_id);
CREATE INDEX IF NOT EXISTS idx_runs_review ON analysis_runs(review_needed);
CREATE INDEX IF NOT EXISTS idx_runs_time ON analysis_runs(analyzer_timestamp);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL UNIQUE,
    rating INTEGER,
    accepted INTEGER,
    corrected_prompt TEXT,
    added_tags_json TEXT NOT NULL DEFAULT '[]',
    removed_tags_json TEXT NOT NULL DEFAULT '[]',
    comment TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES analysis_runs(id) ON DELETE CASCADE,
    CHECK(rating IS NULL OR (rating BETWEEN 1 AND 5)),
    CHECK(accepted IS NULL OR accepted IN (0, 1))
);

CREATE TABLE IF NOT EXISTS image_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    source_platform TEXT,
    source_url TEXT,
    source_page_id TEXT,
    source_tags_json TEXT NOT NULL DEFAULT '[]',
    original_prompt_text TEXT,
    original_prompt_status TEXT NOT NULL DEFAULT 'unknown',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
    UNIQUE(image_id, source_url)
);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    entity_type TEXT,
    entity_id INTEGER,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    t = now_iso()
    for key, value in (("schema_version", SCHEMA_VERSION), ("app_version", APP_VERSION)):
        conn.execute(
            """
            INSERT INTO meta(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=excluded.updated_at
            """,
            (key, value, t),
        )
    conn.commit()


def audit(
    conn: sqlite3.Connection,
    event_type: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO audit_events(event_type, entity_type, entity_id, details_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (event_type, entity_type, entity_id, jd(details or {}), now_iso()),
    )


def get_or_create_image(conn: sqlite3.Connection, image_path_text: str) -> Tuple[int, Optional[str]]:
    cpath = canonical_path(image_path_text)
    actual_path = Path(image_path_text) if image_path_text else None
    image_hash: Optional[str] = None

    if actual_path is not None:
        try:
            if actual_path.is_file():
                image_hash = sha256_file(actual_path)
        except OSError:
            pass

    if cpath:
        row = conn.execute(
            "SELECT image_id FROM image_locations WHERE path=?",
            (cpath,),
        ).fetchone()
        if row:
            image_id = int(row["image_id"])
            conn.execute(
                "UPDATE image_locations SET last_seen_at=? WHERE path=?",
                (now_iso(), cpath),
            )
            if image_hash:
                old = conn.execute(
                    "SELECT sha256 FROM images WHERE id=?",
                    (image_id,),
                ).fetchone()
                if old and not old["sha256"]:
                    collision = conn.execute(
                        "SELECT id FROM images WHERE sha256=?",
                        (image_hash,),
                    ).fetchone()
                    if not collision:
                        conn.execute(
                            """
                            UPDATE images
                            SET sha256=?, identity_key=?, updated_at=?
                            WHERE id=?
                            """,
                            (image_hash, f"sha256:{image_hash}", now_iso(), image_id),
                        )
            return image_id, image_hash

    if image_hash:
        row = conn.execute(
            "SELECT id FROM images WHERE sha256=?",
            (image_hash,),
        ).fetchone()
        if row:
            image_id = int(row["id"])
            if cpath:
                t = now_iso()
                conn.execute(
                    """
                    INSERT INTO image_locations(image_id, path, first_seen_at, last_seen_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET last_seen_at=excluded.last_seen_at
                    """,
                    (image_id, cpath, t, t),
                )
            return image_id, image_hash

    identity_key = (
        f"sha256:{image_hash}"
        if image_hash
        else "path:" + sha256_text(cpath or str(image_path_text))
    )
    filename = ntpath.basename(image_path_text) if image_path_text else None
    ext = ntpath.splitext(filename or "")[1].lower() or None
    t = now_iso()

    cur = conn.execute(
        """
        INSERT INTO images(
            identity_key, sha256, filename, extension,
            first_seen_path, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (identity_key, image_hash, filename, ext, cpath or None, t, t),
    )
    image_id = int(cur.lastrowid)

    if cpath:
        conn.execute(
            """
            INSERT INTO image_locations(image_id, path, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?)
            """,
            (image_id, cpath, t, t),
        )

    return image_id, image_hash


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def extract_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    merger = data.get("merger_data", {})
    if not isinstance(merger, dict):
        merger = {}

    models = data.get("models", {})
    if not isinstance(models, dict):
        models = {}

    target = data.get("target", {})
    if not isinstance(target, dict):
        target = {}

    visibility = data.get(
        "participant_visibility",
        merger.get("participant_visibility", {}),
    )
    if not isinstance(visibility, dict):
        visibility = {}

    visual_tags = data.get(
        "visual_tags_added",
        merger.get("visual_tags_accepted", []),
    )

    return {
        "image": str(data.get("image", "") or ""),
        "timestamp": str(data.get("timestamp", "") or ""),
        "version": str(data.get("version", "") or ""),
        "vlm_model": str(models.get("vlm", "") or ""),
        "merger_model": str(models.get("merger", "") or ""),
        "adult": 1 if bool(target.get("adult", False)) else 0,
        "uncensored": 1 if bool(target.get("uncensored", False)) else 0,
        "vlm_caption": str(data.get("vlm_caption", "") or ""),
        "wd14": str(data.get("wd14_tags", "") or ""),
        "visual_tags": safe_list(visual_tags),
        "final_tags": safe_list(data.get("final_tags", [])),
        "final_prompt": str(data.get("final_prompt", "") or ""),
        "removed_tags": safe_list(data.get("removed_tags", [])),
        "visibility": visibility,
        "review_needed": 1 if bool(
            data.get("review_needed", merger.get("review_needed", False))
        ) else 0,
        "review_reasons": safe_list(
            data.get("review_reasons", merger.get("review_reasons", []))
        ),
        "major_disagreements": safe_list(
            data.get("major_disagreements", merger.get("major_disagreements", []))
        ),
        "conflicts": safe_list(
            data.get("conflicts", merger.get("conflicts", []))
        ),
        "suspected_false_positives": safe_list(
            data.get(
                "suspected_false_positives",
                merger.get("suspected_false_positives", []),
            )
        ),
        "uncertain_tags": safe_list(
            data.get("uncertain_tags", merger.get("uncertain_tags", []))
        ),
        "rejected_visual_features": safe_list(
            data.get(
                "rejected_visual_features",
                merger.get("visual_features_rejected", []),
            )
        ),
        "notes": safe_list(data.get("notes", merger.get("notes", []))),
        "elapsed_seconds": data.get("elapsed_seconds"),
    }



def extract_source_record(
    data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Read V2.3+ source provenance.

    Expected analyzer structure:
    {
      "source": {
        "adapter_version": "...",
        "source_type": "danbooru|pixiv|raw|none",
        "source_tags_raw": "...",
        "source_tags_raw_list": [...],
        "normalized_tags": [...],
        "unknown_tags": [...]
      }
    }

    Future optional keys such as source_url / source_page_id /
    original_prompt_text are also accepted.
    """
    source = data.get(
        "source",
        {},
    )

    if not isinstance(
        source,
        dict,
    ):
        return None

    source_type = str(
        source.get(
            "source_type",
            "",
        )
        or ""
    ).strip().lower()

    if source_type in {
        "",
        "none",
        "local",
    }:
        return None

    normalized_tags = source.get(
        "normalized_tags",
        [],
    )

    if not isinstance(
        normalized_tags,
        list,
    ):
        normalized_tags = []

    raw_list = source.get(
        "source_tags_raw_list",
        [],
    )

    if not isinstance(
        raw_list,
        list,
    ):
        raw_list = []

    unknown_tags = source.get(
        "unknown_tags",
        [],
    )

    if not isinstance(
        unknown_tags,
        list,
    ):
        unknown_tags = []

    original_prompt_status = str(
        source.get(
            "original_prompt_status",
            "unknown",
        )
        or "unknown"
    ).strip().lower()

    if original_prompt_status not in {
        "unknown",
        "unavailable",
        "embedded",
        "source_page",
        "verified_external",
    }:
        original_prompt_status = "unknown"

    metadata = {
        "adapter_version": source.get(
            "adapter_version"
        ),
        "source_tags_raw": source.get(
            "source_tags_raw",
            "",
        ),
        "source_tags_raw_list": (
            raw_list
        ),
        "unknown_tags": (
            unknown_tags
        ),
    }

    # Preserve any extra source fields without changing the core
    # provenance columns.
    known_keys = {
        "adapter_version",
        "source_type",
        "source_tags_raw",
        "source_tags_raw_list",
        "normalized_tags",
        "unknown_tags",
        "source_url",
        "source_page_id",
        "original_prompt_text",
        "original_prompt_status",
    }

    extra = {
        key: value
        for key, value in source.items()
        if key not in known_keys
    }

    if extra:
        metadata[
            "extra"
        ] = extra

    return {
        "source_platform": source_type,
        "source_url": (
            str(
                source.get(
                    "source_url",
                    "",
                )
                or ""
            ).strip()
            or None
        ),
        "source_page_id": (
            str(
                source.get(
                    "source_page_id",
                    "",
                )
                or ""
            ).strip()
            or None
        ),
        "source_tags": [
            str(tag)
            for tag in normalized_tags
            if str(tag).strip()
        ],
        "original_prompt_text": (
            str(
                source.get(
                    "original_prompt_text",
                    "",
                )
                or ""
            ).strip()
            or None
        ),
        "original_prompt_status": (
            original_prompt_status
        ),
        "metadata": metadata,
    }


def upsert_source_from_analysis_json(
    conn: sqlite3.Connection,
    image_id: int,
    data: Dict[str, Any],
) -> Tuple[Optional[int], bool]:
    """
    Backfill / update image_sources from an Analyzer V2.3+ JSON.

    Important:
    This runs even if the analysis_run itself is SKIPPED because it
    was already imported earlier. That lets a newer DB importer
    backfill provenance into an existing database.
    """
    source = extract_source_record(
        data
    )

    if source is None:
        return None, False

    platform = source[
        "source_platform"
    ]

    source_url = source[
        "source_url"
    ]

    if source_url:
        existing = conn.execute(
            """
            SELECT *
            FROM image_sources
            WHERE image_id=?
              AND source_url=?
            ORDER BY id
            LIMIT 1
            """,
            (
                image_id,
                source_url,
            ),
        ).fetchone()
    else:
        existing = conn.execute(
            """
            SELECT *
            FROM image_sources
            WHERE image_id=?
              AND source_platform=?
              AND source_url IS NULL
            ORDER BY id
            LIMIT 1
            """,
            (
                image_id,
                platform,
            ),
        ).fetchone()

    source_tags_json = jd(
        source[
            "source_tags"
        ]
    )

    metadata_json = jd(
        source[
            "metadata"
        ]
    )

    values_for_compare = {
        "source_platform": platform,
        "source_url": source_url,
        "source_page_id": source[
            "source_page_id"
        ],
        "source_tags_json": (
            source_tags_json
        ),
        "original_prompt_text": source[
            "original_prompt_text"
        ],
        "original_prompt_status": source[
            "original_prompt_status"
        ],
        "metadata_json": metadata_json,
    }

    if existing:
        unchanged = all(
            existing[
                key
            ]
            == value
            for key, value
            in values_for_compare.items()
        )

        source_id = int(
            existing[
                "id"
            ]
        )

        if unchanged:
            return source_id, False

        conn.execute(
            """
            UPDATE image_sources
            SET
                source_platform=?,
                source_url=?,
                source_page_id=?,
                source_tags_json=?,
                original_prompt_text=?,
                original_prompt_status=?,
                metadata_json=?,
                updated_at=?
            WHERE id=?
            """,
            (
                platform,
                source_url,
                source[
                    "source_page_id"
                ],
                source_tags_json,
                source[
                    "original_prompt_text"
                ],
                source[
                    "original_prompt_status"
                ],
                metadata_json,
                now_iso(),
                source_id,
            ),
        )

        audit(
            conn,
            "image_source_updated",
            "image_source",
            source_id,
            {
                "image_id": image_id,
                "source_platform": (
                    platform
                ),
            },
        )

        return source_id, True

    t = now_iso()

    cur = conn.execute(
        """
        INSERT INTO image_sources(
            image_id,
            source_platform,
            source_url,
            source_page_id,
            source_tags_json,
            original_prompt_text,
            original_prompt_status,
            metadata_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_id,
            platform,
            source_url,
            source[
                "source_page_id"
            ],
            source_tags_json,
            source[
                "original_prompt_text"
            ],
            source[
                "original_prompt_status"
            ],
            metadata_json,
            t,
            t,
        ),
    )

    source_id = int(
        cur.lastrowid
    )

    audit(
        conn,
        "image_source_imported",
        "image_source",
        source_id,
        {
            "image_id": image_id,
            "source_platform": platform,
        },
    )

    return source_id, True


def import_one(conn: sqlite3.Connection, json_path: Path) -> Tuple[str, int]:
    raw_bytes = json_path.read_bytes()
    source_sha = hashlib.sha256(raw_bytes).hexdigest()
    data = json.loads(raw_bytes.decode("utf-8-sig"))

    if not isinstance(data, dict):
        raise ValueError("JSON top level is not an object")

    f = extract_fields(data)
    image_id, image_hash = get_or_create_image(conn, f["image"])

    # V1.1: import / backfill source provenance even when the
    # analysis run itself already exists and will be skipped.
    upsert_source_from_analysis_json(
        conn,
        image_id,
        data,
    )

    identity = conn.execute(
        "SELECT identity_key FROM images WHERE id=?",
        (image_id,),
    ).fetchone()["identity_key"]

    run_uid = sha256_text(
        "\n".join(
            [
                str(identity),
                f["timestamp"],
                f["version"],
                f["final_prompt"],
            ]
        )
    )

    raw_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    t = now_iso()

    existing = conn.execute(
        "SELECT id, source_json_sha256 FROM analysis_runs WHERE run_uid=?",
        (run_uid,),
    ).fetchone()

    values = (
        image_id,
        f["timestamp"],
        f["version"],
        f["vlm_model"],
        f["merger_model"],
        f["adult"],
        f["uncensored"],
        f["vlm_caption"],
        f["wd14"],
        jd(f["visual_tags"]),
        jd(f["final_tags"]),
        f["final_prompt"],
        jd(f["removed_tags"]),
        jd(f["visibility"]),
        f["review_needed"],
        jd(f["review_reasons"]),
        jd(f["major_disagreements"]),
        jd(f["conflicts"]),
        jd(f["suspected_false_positives"]),
        jd(f["uncertain_tags"]),
        jd(f["rejected_visual_features"]),
        jd(f["notes"]),
        f["elapsed_seconds"],
        canonical_path(str(json_path)),
        source_sha,
        raw_json,
        t,
    )

    if existing:
        run_id = int(existing["id"])
        if existing["source_json_sha256"] == source_sha:
            return "skipped", run_id

        conn.execute(
            """
            UPDATE analysis_runs SET
                image_id=?,
                analyzer_timestamp=?,
                analyzer_version=?,
                vlm_model=?,
                merger_model=?,
                target_adult=?,
                target_uncensored=?,
                vlm_caption=?,
                wd14_tags_text=?,
                visual_tags_json=?,
                final_tags_json=?,
                final_prompt=?,
                removed_tags_json=?,
                participant_visibility_json=?,
                review_needed=?,
                review_reasons_json=?,
                major_disagreements_json=?,
                conflicts_json=?,
                suspected_false_positives_json=?,
                uncertain_tags_json=?,
                rejected_visual_features_json=?,
                notes_json=?,
                elapsed_seconds=?,
                source_json_path=?,
                source_json_sha256=?,
                raw_json=?,
                updated_at=?
            WHERE id=?
            """,
            values + (run_id,),
        )
        audit(
            conn,
            "analysis_run_updated",
            "analysis_run",
            run_id,
            {"json": str(json_path), "image_sha256": image_hash},
        )
        return "updated", run_id

    cur = conn.execute(
        """
        INSERT INTO analysis_runs(
            run_uid,
            image_id,
            analyzer_timestamp,
            analyzer_version,
            vlm_model,
            merger_model,
            target_adult,
            target_uncensored,
            vlm_caption,
            wd14_tags_text,
            visual_tags_json,
            final_tags_json,
            final_prompt,
            removed_tags_json,
            participant_visibility_json,
            review_needed,
            review_reasons_json,
            major_disagreements_json,
            conflicts_json,
            suspected_false_positives_json,
            uncertain_tags_json,
            rejected_visual_features_json,
            notes_json,
            elapsed_seconds,
            source_json_path,
            source_json_sha256,
            raw_json,
            imported_at,
            updated_at
        )
        VALUES(
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
        """,
        (run_uid,) + values[:-1] + (t, t),
    )
    run_id = int(cur.lastrowid)

    audit(
        conn,
        "analysis_run_imported",
        "analysis_run",
        run_id,
        {"json": str(json_path), "image_sha256": image_hash},
    )
    return "inserted", run_id


def import_folder(conn: sqlite3.Connection, runs_dir: Path) -> Dict[str, Any]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs folder not found: {runs_dir}")

    files = sorted(runs_dir.glob("*.json"), key=lambda p: p.name.lower())
    summary: Dict[str, Any] = {
        "found": len(files),
        "inserted": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }

    for i, path in enumerate(files, 1):
        try:
            status, run_id = import_one(conn, path)
            summary[status] += 1
            print(f"[{i}/{len(files)}] {status.upper():8s} run_id={run_id} {path.name}")
        except Exception as exc:
            summary["failed"] += 1
            msg = f"{path.name}: {type(exc).__name__}: {exc}"
            summary["errors"].append(msg)
            print(f"[{i}/{len(files)}] FAILED   {msg}")

    conn.commit()
    return summary


def get_run(conn: sqlite3.Connection, run_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            r.*,
            i.filename,
            i.sha256 AS image_sha256,
            i.first_seen_path
        FROM analysis_runs r
        JOIN images i ON i.id=r.image_id
        WHERE r.id=?
        """,
        (run_id,),
    ).fetchone()


def save_feedback(
    conn: sqlite3.Connection,
    run_id: int,
    rating: Optional[int],
    accepted: Optional[bool],
    corrected_prompt: Optional[str],
    comment: Optional[str],
) -> None:
    run = get_run(conn, run_id)
    if not run:
        raise ValueError(f"Run not found: {run_id}")

    if rating is not None and rating not in {1, 2, 3, 4, 5}:
        raise ValueError("rating must be 1-5")

    old = conn.execute(
        "SELECT * FROM feedback WHERE run_id=?",
        (run_id,),
    ).fetchone()

    final_rating = rating if rating is not None else (old["rating"] if old else None)
    final_accepted = (
        (1 if accepted else 0)
        if accepted is not None
        else (old["accepted"] if old else None)
    )
    final_corrected = (
        corrected_prompt
        if corrected_prompt is not None
        else (old["corrected_prompt"] if old else None)
    )
    final_comment = (
        comment
        if comment is not None
        else (old["comment"] if old else None)
    )

    if final_corrected:
        added, removed = prompt_diff(run["final_prompt"] or "", final_corrected)
    else:
        added, removed = [], []

    t = now_iso()

    conn.execute(
        """
        INSERT INTO feedback(
            run_id, rating, accepted, corrected_prompt,
            added_tags_json, removed_tags_json,
            comment, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            rating=excluded.rating,
            accepted=excluded.accepted,
            corrected_prompt=excluded.corrected_prompt,
            added_tags_json=excluded.added_tags_json,
            removed_tags_json=excluded.removed_tags_json,
            comment=excluded.comment,
            updated_at=excluded.updated_at
        """,
        (
            run_id,
            final_rating,
            final_accepted,
            final_corrected,
            jd(added),
            jd(removed),
            final_comment,
            t,
            t,
        ),
    )
    audit(
        conn,
        "feedback_saved",
        "analysis_run",
        run_id,
        {
            "rating": final_rating,
            "accepted": final_accepted,
            "added_tags": added,
            "removed_tags": removed,
        },
    )
    conn.commit()


def stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    def scalar(sql: str) -> Any:
        return conn.execute(sql).fetchone()[0]

    return {
        "images": scalar("SELECT COUNT(*) FROM images"),
        "analysis_runs": scalar("SELECT COUNT(*) FROM analysis_runs"),
        "review_needed": scalar(
            "SELECT COUNT(*) FROM analysis_runs WHERE review_needed=1"
        ),
        "feedback_rows": scalar("SELECT COUNT(*) FROM feedback"),
        "rated_runs": scalar(
            "SELECT COUNT(*) FROM feedback WHERE rating IS NOT NULL"
        ),
        "corrected_prompts": scalar(
            """
            SELECT COUNT(*) FROM feedback
            WHERE corrected_prompt IS NOT NULL
              AND TRIM(corrected_prompt)<>''
            """
        ),
        "source_records": scalar("SELECT COUNT(*) FROM image_sources"),
        "avg_rating": scalar(
            "SELECT ROUND(AVG(rating), 2) FROM feedback WHERE rating IS NOT NULL"
        ),
    }


def print_stats(conn: sqlite3.Connection, db_path: Path) -> None:
    s = stats(conn)
    print("\n" + "=" * 60)
    print("ILLUSTRIOUS DATABASE")
    print("=" * 60)
    print(f"Database:          {db_path}")
    print(f"Images:            {s['images']}")
    print(f"Analysis runs:     {s['analysis_runs']}")
    print(f"Review needed:     {s['review_needed']}")
    print(f"Feedback rows:     {s['feedback_rows']}")
    print(f"Rated runs:        {s['rated_runs']}")
    print(f"Corrected prompts: {s['corrected_prompts']}")
    print(f"Source records:    {s['source_records']}")
    print(f"Average rating:    {s['avg_rating']}")


def show_recent(conn: sqlite3.Connection, limit: int) -> None:
    rows = conn.execute(
        """
        SELECT
            r.id,
            r.analyzer_timestamp,
            r.analyzer_version,
            r.review_needed,
            r.elapsed_seconds,
            i.filename,
            f.rating
        FROM analysis_runs r
        JOIN images i ON i.id=r.image_id
        LEFT JOIN feedback f ON f.run_id=r.id
        ORDER BY COALESCE(r.analyzer_timestamp, r.imported_at) DESC, r.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    print("\n" + "=" * 95)
    print("RECENT RUNS")
    print("=" * 95)
    print(f"{'ID':>5}  {'Timestamp':19}  {'Ver':7}  {'Review':6}  {'Rate':4}  {'Sec':7}  Filename")
    print("-" * 95)

    for r in rows:
        elapsed = "-" if r["elapsed_seconds"] is None else f"{float(r['elapsed_seconds']):.1f}"
        rating = "-" if r["rating"] is None else str(r["rating"])
        print(
            f"{int(r['id']):>5}  "
            f"{str(r['analyzer_timestamp'] or '')[:19]:19}  "
            f"{str(r['analyzer_version'] or '')[:7]:7}  "
            f"{'YES' if r['review_needed'] else 'no':6}  "
            f"{rating:4}  {elapsed:>7}  {r['filename'] or ''}"
        )


def inspect(conn: sqlite3.Connection, run_id: int) -> None:
    r = get_run(conn, run_id)
    if not r:
        raise ValueError(f"Run not found: {run_id}")

    fb = conn.execute(
        "SELECT * FROM feedback WHERE run_id=?",
        (run_id,),
    ).fetchone()

    print("\n" + "=" * 70)
    print(f"RUN {run_id}")
    print("=" * 70)
    print(f"Image:     {r['filename']}")
    print(f"Path:      {r['first_seen_path']}")
    print(f"SHA-256:   {r['image_sha256'] or '(not available at import time)'}")
    print(f"Timestamp: {r['analyzer_timestamp']}")
    print(f"Version:   {r['analyzer_version']}")
    print(f"Review:    {bool(r['review_needed'])}")
    print(f"Elapsed:   {r['elapsed_seconds']}")

    print("\nParticipant visibility:")
    print(json.dumps(jl(r["participant_visibility_json"], {}), ensure_ascii=False, indent=2))

    print("\nWD14:")
    print(r["wd14_tags_text"] or "")

    print("\nVisual tags:")
    print(", ".join(jl(r["visual_tags_json"], [])))

    print("\nFinal pseudo-prompt:")
    print(r["final_prompt"] or "")

    if r["review_needed"]:
        print("\nReview reasons:")
        print(json.dumps(jl(r["review_reasons_json"], []), ensure_ascii=False, indent=2))

    print("\nFeedback:")
    if not fb:
        print("None")
    else:
        print(f"Rating:   {fb['rating']}")
        print(f"Accepted: {fb['accepted']}")
        print(f"Comment:  {fb['comment'] or ''}")
        print("Added:    " + ", ".join(jl(fb["added_tags_json"], [])))
        print("Removed:  " + ", ".join(jl(fb["removed_tags_json"], [])))
        if fb["corrected_prompt"]:
            print("\nCorrected prompt:")
            print(fb["corrected_prompt"])


def export_feedback(
    conn: sqlite3.Connection,
    output: Path,
    min_rating: Optional[int],
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)

    sql = """
        SELECT
            r.id AS run_id,
            r.analyzer_version,
            r.vlm_model,
            r.merger_model,
            r.vlm_caption,
            r.wd14_tags_text,
            r.visual_tags_json,
            r.final_prompt,
            r.review_needed,
            r.participant_visibility_json,
            i.sha256 AS image_sha256,
            i.filename,
            f.rating,
            f.accepted,
            f.corrected_prompt,
            f.added_tags_json,
            f.removed_tags_json,
            f.comment
        FROM feedback f
        JOIN analysis_runs r ON r.id=f.run_id
        JOIN images i ON i.id=r.image_id
        WHERE (
            f.rating IS NOT NULL
            OR f.accepted IS NOT NULL
            OR f.corrected_prompt IS NOT NULL
        )
    """
    params: List[Any] = []
    if min_rating is not None:
        sql += " AND f.rating>=?"
        params.append(min_rating)
    sql += " ORDER BY r.id"

    rows = conn.execute(sql, params).fetchall()
    count = 0

    with output.open("w", encoding="utf-8") as f:
        for r in rows:
            obj = {
                "run_id": r["run_id"],
                "image_sha256": r["image_sha256"],
                "filename": r["filename"],
                "analyzer_version": r["analyzer_version"],
                "models": {
                    "vlm": r["vlm_model"],
                    "merger": r["merger_model"],
                },
                "vlm_caption": r["vlm_caption"],
                "wd14_tags": r["wd14_tags_text"],
                "visual_tags": jl(r["visual_tags_json"], []),
                "ai_prompt": r["final_prompt"],
                "prompt_kind": "pseudo_prompt",
                "review_needed": bool(r["review_needed"]),
                "participant_visibility": jl(
                    r["participant_visibility_json"], {}
                ),
                "feedback": {
                    "rating": r["rating"],
                    "accepted": None if r["accepted"] is None else bool(r["accepted"]),
                    "corrected_prompt": r["corrected_prompt"],
                    "added_tags": jl(r["added_tags_json"], []),
                    "removed_tags": jl(r["removed_tags_json"], []),
                    "comment": r["comment"],
                },
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    conn.commit()
    return count


def backup(conn: sqlite3.Connection, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = backup_dir / f"illustrious_{stamp}.sqlite3"
    dst = sqlite3.connect(str(out))
    try:
        conn.backup(dst)
    finally:
        dst.close()
    return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Illustrious SQLite Manager")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)

    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("init")
    sp.add_parser("import")
    sp.add_parser("stats")

    r = sp.add_parser("recent")
    r.add_argument("--limit", type=int, default=10)

    i = sp.add_parser("inspect")
    i.add_argument("run_id", type=int)

    rate = sp.add_parser("rate")
    rate.add_argument("run_id", type=int)
    rate.add_argument("--rating", type=int, choices=[1, 2, 3, 4, 5])
    g = rate.add_mutually_exclusive_group()
    g.add_argument("--accept", action="store_true")
    g.add_argument("--reject", action="store_true")
    rate.add_argument("--corrected-prompt")
    rate.add_argument("--corrected-prompt-file", type=Path)
    rate.add_argument("--comment")

    e = sp.add_parser("export")
    e.add_argument("--output", type=Path)
    e.add_argument("--min-rating", type=int, choices=[1, 2, 3, 4, 5])

    b = sp.add_parser("backup")
    b.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUPS)

    return p


def main() -> int:
    args = parser().parse_args()
    conn = connect(args.db)

    try:
        init_db(conn)

        if args.cmd == "init":
            print(f"Initialized: {args.db}")

        elif args.cmd == "import":
            summary = import_folder(conn, args.runs_dir)
            print("\nIMPORT SUMMARY")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print_stats(conn, args.db)

        elif args.cmd == "stats":
            print_stats(conn, args.db)

        elif args.cmd == "recent":
            show_recent(conn, args.limit)

        elif args.cmd == "inspect":
            inspect(conn, args.run_id)

        elif args.cmd == "rate":
            corrected = args.corrected_prompt
            if args.corrected_prompt_file:
                corrected = args.corrected_prompt_file.read_text(
                    encoding="utf-8"
                ).strip()

            accepted: Optional[bool]
            if args.accept:
                accepted = True
            elif args.reject:
                accepted = False
            else:
                accepted = None

            save_feedback(
                conn,
                run_id=args.run_id,
                rating=args.rating,
                accepted=accepted,
                corrected_prompt=corrected,
                comment=args.comment,
            )
            print(f"Feedback saved for run {args.run_id}.")
            inspect(conn, args.run_id)

        elif args.cmd == "export":
            out = args.output
            if out is None:
                stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                out = DEFAULT_EXPORTS / f"feedback_training_{stamp}.jsonl"

            count = export_feedback(conn, out, args.min_rating)
            print(f"Exported {count} feedback rows:")
            print(out)

        elif args.cmd == "backup":
            out = backup(conn, args.backup_dir)
            print("Backup created:")
            print(out)

        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
