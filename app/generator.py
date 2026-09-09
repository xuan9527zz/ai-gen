# -*- coding: utf-8 -*-
r"""
Illustrious ComfyUI Generator v1.1

Purpose
-------
Generate reconstruction candidates from one Analyzer run_id.

Current behavior:
- Reads final_prompt from SQLite analysis_runs.
- Loads the user's ComfyUI API workflow template.
- Replaces positive prompt in BOTH base/refiner text nodes.
- Replaces negative prompt in BOTH base/refiner text nodes.
- Supports 3 logical LoRA slots.
- Applies the same enabled LoRA stack to BOTH base and refiner stages.
- Generates 5 images by default, each with a different random seed.
- Downloads generated images into:
      F:\OpenWebUI\generated\run_<RUN_ID>\
- Stores each generation recipe/result in SQLite generation_runs.
- Creates generation_ratings table now for the later comparison UI.

Designed for the uploaded workflow structure:
    base checkpoint     node 4
    latent              node 5
    base positive       node 6
    base negative       node 7
    base sampler        node 10
    refiner sampler     node 11
    refiner checkpoint  node 12
    refiner positive    node 15
    refiner negative    node 16
    VAE decode          node 17
    SaveImage           node 19

The script dynamically removes template LoraLoader nodes and rebuilds
the active LoRA stack from config, so stale disabled LoRA nodes cannot
accidentally affect generation.

Run:
    py F:\OpenWebUI\illustrious_generate.py

Or:
    py F:\OpenWebUI\illustrious_generate.py --run-id 30

Optional:
    py F:\OpenWebUI\illustrious_generate.py --run-id 30 --count 5

Config:
    F:\OpenWebUI\illustrious_generation_config.json
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import mimetypes
import secrets
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "generation.json"

GENERATOR_VERSION = "1.3"

DIAGNOSTIC_STYLE_PROMPT = (
    "(semi-realistic illustration:1.2), "
    "(highly detailed digital painting:1.2), "
    "soft realistic rendering, painterly shading, "
    "realistic skin shading, intricate fabric texture, "
    "volumetric lighting, soft bloom, "
    "cinematic depth of field"
)

GENERATION_PRESETS = {
    "current": {
        "label": "当前设置",
        "description": "保留当前 LoRA、workflow 分辨率和双 sampler 设置。",
    },
    "balanced": {
        "label": "平衡重建",
        "description": "关闭 LoRA，单 sampler，35 steps，CFG 6，使用原图纵横比。",
    },
    "semi_realistic": {
        "label": "半写实增强",
        "description": "平衡重建设置，并追加加权的半写实绘画式风格提示词。",
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        data = json.load(
            f
        )

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            f"JSON top level must be object: {path}"
        )

    return data


def save_json(
    path: Path,
    data: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# CONFIG
# ============================================================

def load_config(
    path: Path,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}"
        )

    cfg = load_json(
        path
    )

    # Project package: relative paths in config are resolved against repo root.
    for key in (
        "workflow_path",
        "database_path",
        "output_root",
    ):
        raw = cfg.get(key)
        if raw:
            candidate = Path(str(raw))
            if not candidate.is_absolute():
                cfg[key] = str(
                    (PROJECT_ROOT / candidate).resolve()
                )

    required = (
        "workflow_path",
        "database_path",
        "output_root",
        "comfy_url",
        "negative_prompt",
        "lora_slots",
        "workflow_nodes",
    )

    for key in required:
        if key not in cfg:
            raise KeyError(
                f"Config missing key: {key}"
            )

    slots = cfg.get(
        "lora_slots"
    )

    if not isinstance(
        slots,
        list,
    ) or len(
        slots
    ) != 3:
        raise ValueError(
            "lora_slots must contain exactly 3 slots."
        )

    return cfg


def apply_generation_preset(
    cfg: Dict[str, Any],
    preset_id: str,
) -> Dict[str, Any]:
    """Apply one UI generation preset without mutating caller config."""

    preset_id = str(
        preset_id
        or "current"
    ).strip().lower()
    if preset_id not in GENERATION_PRESETS:
        raise ValueError(
            f"Unknown generation preset: {preset_id}"
        )

    value = copy.deepcopy(cfg)
    definition = GENERATION_PRESETS[preset_id]
    value["generation_preset"] = {
        "id": preset_id,
        "label": definition["label"],
        "description": definition["description"],
    }

    if preset_id == "current":
        return value

    for slot in value.get("lora_slots", []):
        if isinstance(slot, dict):
            slot["enabled"] = False

    value["sampling_mode"] = "single"
    value["sampling_overrides"] = {
        "base": {
            "steps": 35,
            "cfg": 6.0,
            "start_at_step": 0,
            "end_at_step": 35,
            "add_noise": "enable",
            "return_with_leftover_noise": "disable",
        }
    }
    value["resolution"] = {
        "mode": "source",
        "multiple": 64,
    }
    value.pop("positive_prompt_append", None)

    if preset_id == "semi_realistic":
        value["positive_prompt_append"] = (
            DIAGNOSTIC_STYLE_PROMPT
        )

    return value


def enabled_loras(
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out = []

    for index, raw in enumerate(
        cfg.get(
            "lora_slots",
            [],
        ),
        start=1,
    ):
        if not isinstance(
            raw,
            dict,
        ):
            continue

        enabled = bool(
            raw.get(
                "enabled",
                False,
            )
        )

        name = str(
            raw.get(
                "name",
                "",
            )
            or ""
        ).strip()

        if not enabled:
            continue

        if not name:
            raise ValueError(
                f"LoRA slot {index} is enabled but name is empty."
            )

        out.append({
            "slot": index,
            "name": name,
            "strength_model": float(
                raw.get(
                    "strength_model",
                    1.0,
                )
            ),
            "strength_clip": float(
                raw.get(
                    "strength_clip",
                    1.0,
                )
            ),
        })

    return out


# ============================================================
# DATABASE
# ============================================================

GENERATION_SCHEMA = r"""
CREATE TABLE IF NOT EXISTS generation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    analysis_run_id INTEGER NOT NULL,

    generator_version TEXT NOT NULL,

    status TEXT NOT NULL,

    prompt TEXT NOT NULL,
    negative_prompt TEXT NOT NULL,

    seed INTEGER NOT NULL,

    width INTEGER,
    height INTEGER,

    base_checkpoint TEXT,
    refiner_checkpoint TEXT,

    loras_json TEXT NOT NULL DEFAULT '[]',
    sampling_json TEXT NOT NULL DEFAULT '{}',

    workflow_path TEXT,
    workflow_sha256 TEXT,
    workflow_snapshot_json TEXT NOT NULL,

    comfy_prompt_id TEXT,
    comfy_output_json TEXT NOT NULL DEFAULT '{}',

    generated_image_path TEXT,
    generated_image_sha256 TEXT,

    elapsed_seconds REAL,

    created_at TEXT NOT NULL,
    finished_at TEXT,

    FOREIGN KEY(analysis_run_id)
        REFERENCES analysis_runs(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_generation_runs_analysis
ON generation_runs(analysis_run_id);

CREATE INDEX IF NOT EXISTS idx_generation_runs_status
ON generation_runs(status);

CREATE INDEX IF NOT EXISTS idx_generation_runs_seed
ON generation_runs(seed);

CREATE TABLE IF NOT EXISTS generation_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    generation_run_id INTEGER NOT NULL UNIQUE,

    overall_score INTEGER,
    semantic_score INTEGER,
    style_score INTEGER,
    composition_score INTEGER,
    comment TEXT,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY(generation_run_id)
        REFERENCES generation_runs(id)
        ON DELETE CASCADE,

    CHECK(
        overall_score IS NULL
        OR overall_score BETWEEN 1 AND 5
    ),
    CHECK(
        semantic_score IS NULL
        OR semantic_score BETWEEN 1 AND 5
    ),
    CHECK(
        style_score IS NULL
        OR style_score BETWEEN 1 AND 5
    ),
    CHECK(
        composition_score IS NULL
        OR composition_score BETWEEN 1 AND 5
    )
);
"""


GENERATION_RATING_MIGRATIONS = {
    "semantic_score": "INTEGER",
    "style_score": "INTEGER",
    "composition_score": "INTEGER",
}


def migrate_generation_schema(
    conn: sqlite3.Connection,
) -> None:
    """Add backward-compatible columns missing from older local DBs."""

    existing = {
        str(row[1])
        for row in conn.execute(
            "PRAGMA table_info(generation_ratings)"
        ).fetchall()
    }

    for column, sql_type in GENERATION_RATING_MIGRATIONS.items():
        if column in existing:
            continue
        conn.execute(
            f"ALTER TABLE generation_ratings "
            f"ADD COLUMN {column} {sql_type}"
        )


def connect_db(
    db_path: Path,
) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}"
        )

    conn = sqlite3.connect(
        str(
            db_path
        ),
        timeout=60,
    )

    conn.row_factory = sqlite3.Row

    conn.execute(
        "PRAGMA foreign_keys=ON"
    )

    conn.executescript(
        GENERATION_SCHEMA
    )

    migrate_generation_schema(
        conn
    )

    conn.commit()

    return conn


def get_analysis_run(
    conn: sqlite3.Connection,
    run_id: int,
) -> sqlite3.Row:
    row = conn.execute(
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

    if not row:
        raise ValueError(
            f"Analysis run not found: {run_id}"
        )

    return row


def create_generation_row(
    conn: sqlite3.Connection,
    *,
    analysis_run_id: int,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: Optional[int],
    height: Optional[int],
    base_checkpoint: Optional[str],
    refiner_checkpoint: Optional[str],
    loras: List[Dict[str, Any]],
    sampling: Dict[str, Any],
    workflow_path: Path,
    workflow_sha256: str,
    workflow_snapshot: Dict[str, Any],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO generation_runs(
            analysis_run_id,
            generator_version,
            status,
            prompt,
            negative_prompt,
            seed,
            width,
            height,
            base_checkpoint,
            refiner_checkpoint,
            loras_json,
            sampling_json,
            workflow_path,
            workflow_sha256,
            workflow_snapshot_json,
            created_at
        )
        VALUES(
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
        """,
        (
            analysis_run_id,
            GENERATOR_VERSION,
            "queued",
            prompt,
            negative_prompt,
            seed,
            width,
            height,
            base_checkpoint,
            refiner_checkpoint,
            json_dumps(
                loras
            ),
            json_dumps(
                sampling
            ),
            str(
                workflow_path
            ),
            workflow_sha256,
            json_dumps(
                workflow_snapshot
            ),
            now_iso(),
        ),
    )

    conn.commit()

    return int(
        cur.lastrowid
    )


def finish_generation_row(
    conn: sqlite3.Connection,
    generation_id: int,
    *,
    status: str,
    prompt_id: Optional[str] = None,
    comfy_output: Optional[Dict[str, Any]] = None,
    generated_path: Optional[Path] = None,
    generated_sha256: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
) -> None:
    conn.execute(
        """
        UPDATE generation_runs
        SET
            status=?,
            comfy_prompt_id=?,
            comfy_output_json=?,
            generated_image_path=?,
            generated_image_sha256=?,
            elapsed_seconds=?,
            finished_at=?
        WHERE id=?
        """,
        (
            status,
            prompt_id,
            json_dumps(
                comfy_output
                or {}
            ),
            (
                str(
                    generated_path
                )
                if generated_path
                else None
            ),
            generated_sha256,
            elapsed_seconds,
            now_iso(),
            generation_id,
        ),
    )

    conn.commit()


# ============================================================
# COMFY WORKFLOW
# ============================================================

def validate_node(
    workflow: Dict[str, Any],
    node_id: str,
    expected_class: Optional[str] = None,
) -> Dict[str, Any]:
    if node_id not in workflow:
        raise KeyError(
            f"Workflow missing node {node_id}"
        )

    node = workflow[
        node_id
    ]

    if not isinstance(
        node,
        dict,
    ):
        raise ValueError(
            f"Workflow node {node_id} is invalid."
        )

    if expected_class:
        actual = node.get(
            "class_type"
        )

        if actual != expected_class:
            raise ValueError(
                f"Node {node_id}: expected {expected_class}, got {actual}"
            )

    if "inputs" not in node:
        node[
            "inputs"
        ] = {}

    return node


def validate_workflow(
    workflow: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, str]:
    ids = {
        key: str(
            value
        )
        for key, value
        in cfg[
            "workflow_nodes"
        ].items()
    }

    expected = {
        "base_checkpoint": (
            "CheckpointLoaderSimple"
        ),
        "latent": (
            "EmptyLatentImage"
        ),
        "base_positive": (
            "CLIPTextEncode"
        ),
        "base_negative": (
            "CLIPTextEncode"
        ),
        "base_sampler": (
            "KSamplerAdvanced"
        ),
        "refiner_sampler": (
            "KSamplerAdvanced"
        ),
        "refiner_checkpoint": (
            "CheckpointLoaderSimple"
        ),
        "refiner_positive": (
            "CLIPTextEncode"
        ),
        "refiner_negative": (
            "CLIPTextEncode"
        ),
        "vae_decode": (
            "VAEDecode"
        ),
        "save_image": (
            "SaveImage"
        ),
    }

    for key, class_type in expected.items():
        if key not in ids:
            raise KeyError(
                f"workflow_nodes missing: {key}"
            )

        validate_node(
            workflow,
            ids[key],
            class_type,
        )

    return ids


def remove_existing_lora_nodes(
    workflow: Dict[str, Any],
) -> List[str]:
    removed = []

    for node_id in list(
        workflow.keys()
    ):
        node = workflow[
            node_id
        ]

        if (
            isinstance(
                node,
                dict,
            )
            and node.get(
                "class_type"
            )
            == "LoraLoader"
        ):
            removed.append(
                node_id
            )

            del workflow[
                node_id
            ]

    return removed


def add_lora_chain(
    workflow: Dict[str, Any],
    *,
    start_model: List[Any],
    start_clip: List[Any],
    loras: List[Dict[str, Any]],
    node_id_start: int,
    title_prefix: str,
) -> Tuple[List[Any], List[Any], List[str]]:
    model_ref = copy.deepcopy(
        start_model
    )

    clip_ref = copy.deepcopy(
        start_clip
    )

    added_ids: List[str] = []

    for offset, lora in enumerate(
        loras
    ):
        node_id = str(
            node_id_start
            + offset
        )

        workflow[
            node_id
        ] = {
            "inputs": {
                "lora_name": lora[
                    "name"
                ],
                "strength_model": lora[
                    "strength_model"
                ],
                "strength_clip": lora[
                    "strength_clip"
                ],
                "model": model_ref,
                "clip": clip_ref,
            },
            "class_type": (
                "LoraLoader"
            ),
            "_meta": {
                "title": (
                    f"{title_prefix} "
                    f"LoRA Slot "
                    f"{lora['slot']}"
                )
            },
        }

        model_ref = [
            node_id,
            0,
        ]

        clip_ref = [
            node_id,
            1,
        ]

        added_ids.append(
            node_id
        )

    return (
        model_ref,
        clip_ref,
        added_ids,
    )


def apply_sampler_overrides(
    sampler: Dict[str, Any],
    overrides: Dict[str, Any],
) -> None:
    """Apply explicit, provenance-visible KSampler input overrides."""

    allowed = {
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "start_at_step",
        "end_at_step",
        "add_noise",
        "return_with_leftover_noise",
    }

    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(
            "Unsupported sampler override(s): "
            + ", ".join(sorted(unknown))
        )

    sampler["inputs"].update(
        copy.deepcopy(overrides)
    )


def append_generation_prompt(
    prompt: str,
    cfg: Dict[str, Any],
) -> str:
    suffix = str(
        cfg.get(
            "positive_prompt_append",
            "",
        )
        or ""
    ).strip(" ,\n\t")

    if not suffix:
        return prompt.strip()

    return (
        prompt.strip(" ,\n\t")
        + ", "
        + suffix
    )


def fit_source_resolution(
    source_width: int,
    source_height: int,
    *,
    target_pixels: int,
    multiple: int = 64,
    min_dimension: int = 512,
    max_dimension: int = 1536,
) -> Tuple[int, int]:
    """Fit source aspect ratio near a fixed pixel budget."""

    if source_width < 1 or source_height < 1:
        raise ValueError(
            "Source dimensions must be positive."
        )
    if target_pixels < 1 or multiple < 1:
        raise ValueError(
            "target_pixels and multiple must be positive."
        )

    scale = math.sqrt(
        target_pixels
        / (source_width * source_height)
    )

    width = int(
        round(
            source_width * scale / multiple
        )
        * multiple
    )
    height = int(
        round(
            source_height * scale / multiple
        )
        * multiple
    )

    width = min(
        max(width, min_dimension),
        max_dimension,
    )
    height = min(
        max(height, min_dimension),
        max_dimension,
    )

    return width, height


def resolve_resolution_config(
    cfg: Dict[str, Any],
    template: Dict[str, Any],
    *,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve workflow/fixed/source resolution into an effective config."""

    resolved = copy.deepcopy(cfg)
    raw = resolved.get(
        "resolution",
        {"mode": "workflow"},
    )
    if not isinstance(raw, dict):
        raise ValueError(
            "resolution must be an object."
        )

    mode = str(
        raw.get(
            "mode",
            "workflow",
        )
    ).strip().lower()

    if mode == "workflow":
        resolved["resolution"] = {
            "mode": "workflow"
        }
        return resolved

    if mode == "fixed":
        width = int(raw.get("width", 0))
        height = int(raw.get("height", 0))
        if width < 64 or height < 64:
            raise ValueError(
                "Fixed resolution requires positive width and height."
            )
        resolved["resolution"] = {
            "mode": "fixed",
            "width": width,
            "height": height,
        }
        return resolved

    if mode != "source":
        raise ValueError(
            "resolution.mode must be workflow, fixed, or source."
        )

    if source_path is None or not source_path.exists():
        raise FileNotFoundError(
            f"Source image not found for resolution fitting: {source_path}"
        )

    latent_id = str(
        resolved["workflow_nodes"]["latent"]
    )
    latent = validate_node(
        template,
        latent_id,
        "EmptyLatentImage",
    )
    workflow_width = int(
        latent["inputs"].get("width", 1024)
    )
    workflow_height = int(
        latent["inputs"].get("height", 1024)
    )

    with Image.open(source_path) as image:
        source_width, source_height = image.size

    multiple = int(raw.get("multiple", 64))
    width, height = fit_source_resolution(
        source_width,
        source_height,
        target_pixels=(
            workflow_width
            * workflow_height
        ),
        multiple=multiple,
    )

    resolved["resolution"] = {
        "mode": "source",
        "width": width,
        "height": height,
        "source_width": source_width,
        "source_height": source_height,
        "target_pixels": (
            workflow_width
            * workflow_height
        ),
        "multiple": multiple,
    }
    return resolved


def build_workflow_for_generation(
    template: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    prompt: str,
    seed: int,
    filename_prefix: str,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:
    workflow = copy.deepcopy(
        template
    )

    ids = validate_workflow(
        workflow,
        cfg,
    )

    loras = enabled_loras(
        cfg
    )

    remove_existing_lora_nodes(
        workflow
    )

    base_ckpt = validate_node(
        workflow,
        ids[
            "base_checkpoint"
        ],
    )

    ref_ckpt = validate_node(
        workflow,
        ids[
            "refiner_checkpoint"
        ],
    )

    base_start_model = [
        ids[
            "base_checkpoint"
        ],
        0,
    ]

    base_start_clip = [
        ids[
            "base_checkpoint"
        ],
        1,
    ]

    (
        base_model_ref,
        base_clip_ref,
        base_lora_nodes,
    ) = add_lora_chain(
        workflow,
        start_model=(
            base_start_model
        ),
        start_clip=(
            base_start_clip
        ),
        loras=loras,
        node_id_start=9001,
        title_prefix="BASE",
    )

    if bool(
        cfg.get(
            "apply_lora_to_refiner",
            True,
        )
    ):
        (
            refiner_model_ref,
            refiner_clip_ref,
            refiner_lora_nodes,
        ) = add_lora_chain(
            workflow,
            start_model=[
                ids[
                    "refiner_checkpoint"
                ],
                0,
            ],
            start_clip=[
                ids[
                    "refiner_checkpoint"
                ],
                1,
            ],
            loras=loras,
            node_id_start=9011,
            title_prefix=(
                "REFINER"
            ),
        )
    else:
        refiner_model_ref = [
            ids[
                "refiner_checkpoint"
            ],
            0,
        ]

        refiner_clip_ref = [
            ids[
                "refiner_checkpoint"
            ],
            1,
        ]

        refiner_lora_nodes = []

    negative_prompt = str(
        cfg[
            "negative_prompt"
        ]
    ).strip()

    base_positive = validate_node(
        workflow,
        ids[
            "base_positive"
        ],
    )

    base_negative = validate_node(
        workflow,
        ids[
            "base_negative"
        ],
    )

    refiner_positive = validate_node(
        workflow,
        ids[
            "refiner_positive"
        ],
    )

    refiner_negative = validate_node(
        workflow,
        ids[
            "refiner_negative"
        ],
    )

    base_positive[
        "inputs"
    ][
        "text"
    ] = prompt

    refiner_positive[
        "inputs"
    ][
        "text"
    ] = prompt

    base_negative[
        "inputs"
    ][
        "text"
    ] = negative_prompt

    refiner_negative[
        "inputs"
    ][
        "text"
    ] = negative_prompt

    base_positive[
        "inputs"
    ][
        "clip"
    ] = base_clip_ref

    refiner_positive[
        "inputs"
    ][
        "clip"
    ] = refiner_clip_ref

    use_lora_clip_for_negative = bool(
        cfg.get(
            "use_lora_clip_for_negative",
            True,
        )
    )

    if use_lora_clip_for_negative:
        base_negative[
            "inputs"
        ][
            "clip"
        ] = base_clip_ref

        refiner_negative[
            "inputs"
        ][
            "clip"
        ] = refiner_clip_ref
    else:
        base_negative[
            "inputs"
        ][
            "clip"
        ] = [
            ids[
                "base_checkpoint"
            ],
            1,
        ]

        refiner_negative[
            "inputs"
        ][
            "clip"
        ] = [
            ids[
                "refiner_checkpoint"
            ],
            1,
        ]

    base_sampler = validate_node(
        workflow,
        ids[
            "base_sampler"
        ],
    )

    refiner_sampler = validate_node(
        workflow,
        ids[
            "refiner_sampler"
        ],
    )

    sampling_mode = str(
        cfg.get(
            "sampling_mode",
            "dual",
        )
    ).strip().lower()

    if sampling_mode not in {
        "dual",
        "single",
    }:
        raise ValueError(
            "sampling_mode must be 'dual' or 'single'."
        )

    sampling_overrides = cfg.get(
        "sampling_overrides",
        {},
    )

    if not isinstance(
        sampling_overrides,
        dict,
    ):
        raise ValueError(
            "sampling_overrides must be an object."
        )

    for stage, sampler in (
        ("base", base_sampler),
        ("refiner", refiner_sampler),
    ):
        overrides = sampling_overrides.get(
            stage,
            {},
        )
        if not isinstance(overrides, dict):
            raise ValueError(
                f"sampling_overrides.{stage} must be an object."
            )
        apply_sampler_overrides(
            sampler,
            overrides,
        )

    base_sampler[
        "inputs"
    ][
        "model"
    ] = base_model_ref

    base_sampler[
        "inputs"
    ][
        "noise_seed"
    ] = int(
        seed
    )

    refiner_sampler[
        "inputs"
    ][
        "model"
    ] = refiner_model_ref

    # This refiner has add_noise=disable in the template,
    # so its seed is operationally irrelevant. Keep it explicit.
    refiner_sampler[
        "inputs"
    ][
        "noise_seed"
    ] = int(
        seed
    )

    vae_decode = validate_node(
        workflow,
        ids[
            "vae_decode"
        ],
    )

    if sampling_mode == "single":
        # Leave the template refiner nodes intact but disconnect them from the
        # SaveImage execution graph.  This is a true base-only sampling path.
        vae_decode["inputs"]["samples"] = [
            ids["base_sampler"],
            0,
        ]
        vae_decode["inputs"]["vae"] = [
            ids["base_checkpoint"],
            2,
        ]

    save_node = validate_node(
        workflow,
        ids[
            "save_image"
        ],
    )

    save_node[
        "inputs"
    ][
        "filename_prefix"
    ] = filename_prefix

    latent = validate_node(
        workflow,
        ids[
            "latent"
        ],
    )

    resolution = cfg.get(
        "resolution",
        {"mode": "workflow"},
    )
    resolution_mode = str(
        resolution.get(
            "mode",
            "workflow",
        )
    ).strip().lower()

    if resolution_mode in {
        "fixed",
        "source",
    }:
        latent["inputs"]["width"] = int(
            resolution["width"]
        )
        latent["inputs"]["height"] = int(
            resolution["height"]
        )
    elif resolution_mode != "workflow":
        raise ValueError(
            "resolution.mode must be workflow, fixed, or source."
        )

    width = latent[
        "inputs"
    ].get(
        "width"
    )

    height = latent[
        "inputs"
    ].get(
        "height"
    )

    sampling = {
        "mode": sampling_mode,
        "resolution": {
            "mode": resolution_mode,
            "width": (
                int(width)
                if width is not None
                else None
            ),
            "height": (
                int(height)
                if height is not None
                else None
            ),
            **(
                {
                    key: resolution[key]
                    for key in (
                        "source_width",
                        "source_height",
                        "target_pixels",
                        "multiple",
                    )
                    if key in resolution
                }
            ),
        },
        "base": {
            key: base_sampler[
                "inputs"
            ].get(
                key
            )
            for key in (
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "start_at_step",
                "end_at_step",
                "add_noise",
                "return_with_leftover_noise",
            )
        },
        "refiner": {
            key: refiner_sampler[
                "inputs"
            ].get(
                key
            )
            for key in (
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "start_at_step",
                "end_at_step",
                "add_noise",
                "return_with_leftover_noise",
            )
        },
    }

    diagnostic = cfg.get(
        "diagnostic"
    )
    if isinstance(diagnostic, dict):
        sampling["diagnostic"] = copy.deepcopy(
            diagnostic
        )

    generation_preset = cfg.get(
        "generation_preset"
    )
    if isinstance(generation_preset, dict):
        sampling["preset"] = copy.deepcopy(
            generation_preset
        )

    metadata = {
        "width": (
            int(
                width
            )
            if width is not None
            else None
        ),
        "height": (
            int(
                height
            )
            if height is not None
            else None
        ),
        "base_checkpoint": (
            base_ckpt[
                "inputs"
            ].get(
                "ckpt_name"
            )
        ),
        "refiner_checkpoint": (
            ref_ckpt[
                "inputs"
            ].get(
                "ckpt_name"
            )
        ),
        "loras": loras,
        "sampling": sampling,
        "resolution": copy.deepcopy(
            sampling["resolution"]
        ),
        "diagnostic": (
            copy.deepcopy(diagnostic)
            if isinstance(diagnostic, dict)
            else None
        ),
        "generation_preset": (
            copy.deepcopy(generation_preset)
            if isinstance(generation_preset, dict)
            else None
        ),
        "removed_template_loras": (
            True
        ),
        "base_lora_nodes": (
            base_lora_nodes
        ),
        "refiner_lora_nodes": (
            refiner_lora_nodes
        ),
    }

    return (
        workflow,
        metadata,
    )


# ============================================================
# COMFYUI HTTP
# ============================================================

def check_comfy(
    comfy_url: str,
) -> None:
    response = requests.get(
        f"{comfy_url}/system_stats",
        timeout=15,
    )

    response.raise_for_status()


def submit_workflow(
    comfy_url: str,
    workflow: Dict[str, Any],
) -> str:
    payload = {
        "prompt": workflow,
        "client_id": str(
            uuid.uuid4()
        ),
    }

    response = requests.post(
        f"{comfy_url}/prompt",
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    prompt_id = data.get(
        "prompt_id"
    )

    if not prompt_id:
        raise RuntimeError(
            f"ComfyUI returned no prompt_id: {data}"
        )

    return str(
        prompt_id
    )


def wait_for_history(
    comfy_url: str,
    prompt_id: str,
    *,
    timeout_seconds: int = 1800,
) -> Dict[str, Any]:
    start = time.time()

    while (
        time.time()
        - start
        < timeout_seconds
    ):
        response = requests.get(
            f"{comfy_url}/history/{prompt_id}",
            timeout=30,
        )

        response.raise_for_status()

        history = response.json()

        if prompt_id in history:
            record = history[
                prompt_id
            ]

            status = record.get(
                "status",
                {}
            )

            status_str = str(
                status.get(
                    "status_str",
                    ""
                )
            ).lower()

            completed = bool(
                status.get(
                    "completed",
                    False,
                )
            )

            if completed:
                return record

            if status_str in {
                "error",
                "failed",
            }:
                raise RuntimeError(
                    f"ComfyUI generation failed: {record}"
                )

        time.sleep(
            1
        )

    raise TimeoutError(
        f"ComfyUI generation timed out: {prompt_id}"
    )


def extract_saved_images(
    history_record: Dict[str, Any],
    save_node_id: str,
) -> List[Dict[str, Any]]:
    outputs = history_record.get(
        "outputs",
        {}
    )

    if not isinstance(
        outputs,
        dict,
    ):
        return []

    node = outputs.get(
        save_node_id,
        {}
    )

    if not isinstance(
        node,
        dict,
    ):
        return []

    images = node.get(
        "images",
        []
    )

    if not isinstance(
        images,
        list,
    ):
        return []

    return [
        x
        for x in images
        if isinstance(
            x,
            dict,
        )
    ]


def download_comfy_image(
    comfy_url: str,
    image_info: Dict[str, Any],
    output_path: Path,
) -> Path:
    filename = str(
        image_info.get(
            "filename",
            "",
        )
        or ""
    )

    if not filename:
        raise ValueError(
            "Comfy image info has no filename."
        )

    params = {
        "filename": filename,
        "subfolder": str(
            image_info.get(
                "subfolder",
                "",
            )
            or ""
        ),
        "type": str(
            image_info.get(
                "type",
                "output",
            )
            or "output"
        ),
    }

    url = (
        f"{comfy_url}/view?"
        + urlencode(
            params
        )
    )

    response = requests.get(
        url,
        timeout=120,
    )

    response.raise_for_status()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        response.content
    )

    return output_path


# ============================================================
# SEEDS
# ============================================================

def make_seeds(
    count: int,
    mode: str = "random",
) -> List[int]:
    if count < 1:
        raise ValueError(
            "count must be >= 1"
        )

    if mode != "random":
        raise ValueError(
            "Only seed_mode=random is supported in v1.0."
        )

    seeds: List[int] = []
    seen = set()

    while len(
        seeds
    ) < count:
        seed = secrets.randbelow(
            2**63
            - 1
        )

        if seed not in seen:
            seen.add(
                seed
            )

            seeds.append(
                seed
            )

    return seeds


# ============================================================
# GENERATION
# ============================================================

def generate_candidates(
    *,
    run_id: int,
    count: int,
    cfg: Dict[str, Any],
    prompt_override: Optional[str] = None,
    seeds_override: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    db_path = Path(
        cfg[
            "database_path"
        ]
    )

    workflow_path = Path(
        cfg[
            "workflow_path"
        ]
    )

    output_root = Path(
        cfg[
            "output_root"
        ]
    )

    comfy_url = str(
        cfg[
            "comfy_url"
        ]
    ).rstrip(
        "/"
    )

    conn = connect_db(
        db_path
    )

    try:
        analysis = get_analysis_run(
            conn,
            run_id,
        )

        base_analysis_prompt = str(
            analysis[
                "final_prompt"
            ]
            or ""
        ).strip()

        if prompt_override is not None:
            prompt = str(
                prompt_override
            ).strip()
        else:
            prompt = base_analysis_prompt

        if not prompt:
            raise ValueError(
                f"Analysis run {run_id} has empty generation prompt."
            )

        template = load_json(
            workflow_path
        )

        cfg = resolve_resolution_config(
            cfg,
            template,
            source_path=Path(
                str(
                    analysis[
                        "first_seen_path"
                    ]
                )
            ),
        )

        prompt = append_generation_prompt(
            prompt,
            cfg,
        )

        workflow_hash = sha256_file(
            workflow_path
        )

        ids = validate_workflow(
            template,
            cfg,
        )

        check_comfy(
            comfy_url
        )

        if seeds_override is None:
            seeds = make_seeds(
                count,
                str(
                    cfg.get(
                        "seed_mode",
                        "random",
                    )
                ),
            )
        else:
            seeds = [
                int(seed)
                for seed in seeds_override
            ]

            if len(seeds) != count:
                raise ValueError(
                    "seeds_override length must equal count."
                )

            if any(
                seed < 0 or seed >= 2**63
                for seed in seeds
            ):
                raise ValueError(
                    "Each seed must be between 0 and 2^63 - 1."
                )

        loras = enabled_loras(
            cfg
        )

        print()
        print("=" * 70)
        print("ILLUSTRIOUS COMFYUI GENERATOR")
        print("=" * 70)
        print(
            f"Analysis run:  {run_id}"
        )
        print(
            f"Image:         {analysis['filename']}"
        )
        print(
            f"Workflow:      {workflow_path}"
        )
        print(
            f"Candidates:    {count}"
        )
        print(
            "Prompt source:  "
            + (
                "UI override"
                if prompt_override is not None
                else "Analyzer Final Prompt"
            )
        )
        print(
            f"LoRA slots on: {len(loras)} / 3"
        )

        for lora in loras:
            print(
                f"  Slot {lora['slot']}: "
                f"{lora['name']} "
                f"(model={lora['strength_model']}, "
                f"clip={lora['strength_clip']})"
            )

        print()
        print("Negative prompt:")
        print(
            cfg[
                "negative_prompt"
            ]
        )
        print()

        results = []

        run_dir = (
            output_root
            / f"run_{run_id}"
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, seed in enumerate(
            seeds,
            start=1,
        ):
            print(
                f"[{index}/{count}] seed={seed}"
            )

            stamp = dt.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            diagnostic_info = cfg.get(
                "diagnostic"
            )
            diagnostic_variant = (
                str(
                    diagnostic_info.get(
                        "variant",
                        "",
                    )
                ).strip().upper()
                if isinstance(diagnostic_info, dict)
                else ""
            )
            variant_suffix = (
                f"{diagnostic_variant.lower()}_"
                if diagnostic_variant
                else ""
            )

            prefix = (
                f"recon_run{run_id}_"
                f"{variant_suffix}"
                f"{index:02d}_seed{seed}"
            )

            workflow, meta = (
                build_workflow_for_generation(
                    template,
                    cfg,
                    prompt=prompt,
                    seed=seed,
                    filename_prefix=(
                        prefix
                    ),
                )
            )

            generation_id = (
                create_generation_row(
                    conn,
                    analysis_run_id=run_id,
                    prompt=prompt,
                    negative_prompt=str(
                        cfg[
                            "negative_prompt"
                        ]
                    ),
                    seed=seed,
                    width=meta[
                        "width"
                    ],
                    height=meta[
                        "height"
                    ],
                    base_checkpoint=meta[
                        "base_checkpoint"
                    ],
                    refiner_checkpoint=meta[
                        "refiner_checkpoint"
                    ],
                    loras=meta[
                        "loras"
                    ],
                    sampling=meta[
                        "sampling"
                    ],
                    workflow_path=workflow_path,
                    workflow_sha256=workflow_hash,
                    workflow_snapshot=workflow,
                )
            )

            start = time.time()
            prompt_id: Optional[str] = None

            try:
                prompt_id = submit_workflow(
                    comfy_url,
                    workflow,
                )

                history = wait_for_history(
                    comfy_url,
                    prompt_id,
                )

                images = extract_saved_images(
                    history,
                    ids[
                        "save_image"
                    ],
                )

                if not images:
                    raise RuntimeError(
                        "ComfyUI completed but SaveImage returned no image."
                    )

                # Current workflow has batch_size=1, so use first result.
                image_info = images[
                    0
                ]

                suffix = Path(
                    str(
                        image_info.get(
                            "filename",
                            "",
                        )
                    )
                ).suffix

                if not suffix:
                    suffix = ".png"

                local_path = (
                    run_dir
                    / (
                        f"generation_"
                        f"{generation_id:06d}_"
                        f"seed_{seed}"
                        f"{suffix}"
                    )
                )

                download_comfy_image(
                    comfy_url,
                    image_info,
                    local_path,
                )

                image_hash = sha256_file(
                    local_path
                )

                elapsed = (
                    time.time()
                    - start
                )

                finish_generation_row(
                    conn,
                    generation_id,
                    status="completed",
                    prompt_id=prompt_id,
                    comfy_output=history,
                    generated_path=(
                        local_path
                    ),
                    generated_sha256=(
                        image_hash
                    ),
                    elapsed_seconds=(
                        round(
                            elapsed,
                            2,
                        )
                    ),
                )

                results.append({
                    "generation_id": (
                        generation_id
                    ),
                    "seed": seed,
                    "prompt": prompt,
                    "image_path": str(
                        local_path
                    ),
                    "elapsed_seconds": round(
                        elapsed,
                        2,
                    ),
                    "diagnostic_variant": (
                        diagnostic_variant
                        or None
                    ),
                })

                print(
                    f"  completed -> "
                    f"{local_path}"
                )

            except Exception as exc:
                elapsed = (
                    time.time()
                    - start
                )

                finish_generation_row(
                    conn,
                    generation_id,
                    status="failed",
                    prompt_id=prompt_id,
                    comfy_output={
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )
                    },
                    elapsed_seconds=(
                        round(
                            elapsed,
                            2,
                        )
                    ),
                )

                print(
                    f"  FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        print()
        print("=" * 70)
        print("GENERATION SUMMARY")
        print("=" * 70)
        print(
            f"Completed: {len(results)} / {count}"
        )

        if results:
            print(
                f"Output folder: {run_dir}"
            )

        return results

    finally:
        conn.close()


def diagnostic_variant_configs(
    cfg: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Build the controlled A-E configs without mutating caller state."""

    def base_variant(
        variant: str,
        label: str,
    ) -> Dict[str, Any]:
        value = copy.deepcopy(cfg)
        value["sampling_mode"] = "dual"
        value.pop("sampling_overrides", None)
        value.pop("positive_prompt_append", None)
        value["diagnostic"] = {
            "suite": "generation_quality_abcde_v1",
            "variant": variant,
            "label": label,
        }
        return value

    variant_a = base_variant(
        "A",
        "current LoRA + dual sampler",
    )

    variant_b = base_variant(
        "B",
        "LoRA off + dual sampler",
    )
    for slot in variant_b["lora_slots"]:
        slot["enabled"] = False

    variant_c = copy.deepcopy(
        variant_b
    )
    variant_c["sampling_mode"] = "single"
    variant_c["sampling_overrides"] = {
        "base": {
            "steps": 35,
            "cfg": 6.0,
            "start_at_step": 0,
            "end_at_step": 35,
            "add_noise": "enable",
            "return_with_leftover_noise": "disable",
        }
    }
    variant_c["diagnostic"] = {
        "suite": "generation_quality_abcde_v1",
        "variant": "C",
        "label": "LoRA off + single sampler + CFG 6",
    }

    variant_d = copy.deepcopy(
        variant_c
    )
    variant_d["positive_prompt_append"] = (
        DIAGNOSTIC_STYLE_PROMPT
    )
    variant_d["diagnostic"] = {
        "suite": "generation_quality_abcde_v1",
        "variant": "D",
        "label": "C + semi-realistic painterly style prompt",
    }

    variant_e = copy.deepcopy(
        variant_d
    )
    variant_e["resolution"] = {
        "mode": "source",
        "multiple": 64,
    }
    variant_e["diagnostic"] = {
        "suite": "generation_quality_abcde_v1",
        "variant": "E",
        "label": "D + source aspect-ratio resolution",
    }

    return [
        ("A", variant_a),
        ("B", variant_b),
        ("C", variant_c),
        ("D", variant_d),
        ("E", variant_e),
    ]


def generate_diagnostic_abcde(
    *,
    run_id: int,
    cfg: Dict[str, Any],
    prompt_override: Optional[str] = None,
    seed: Optional[int] = None,
    variants: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Generate a controlled A-E comparison with one fixed seed."""

    fixed_seed = (
        make_seeds(1)[0]
        if seed is None
        else int(seed)
    )

    results: List[Dict[str, Any]] = []

    selected = (
        {value.upper() for value in variants}
        if variants is not None
        else None
    )

    for variant, variant_cfg in diagnostic_variant_configs(cfg):
        if selected is not None and variant not in selected:
            continue
        print()
        print(
            f"[Diagnostic {variant}] seed={fixed_seed}"
        )
        generated = generate_candidates(
            run_id=run_id,
            count=1,
            cfg=variant_cfg,
            prompt_override=prompt_override,
            seeds_override=[fixed_seed],
        )
        results.extend(generated)

    return results


def generate_diagnostic_abc(
    *,
    run_id: int,
    cfg: Dict[str, Any],
    prompt_override: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Backward-compatible A/B/C-only diagnostic entry point."""

    return generate_diagnostic_abcde(
        run_id=run_id,
        cfg=cfg,
        prompt_override=prompt_override,
        seed=seed,
        variants=["A", "B", "C"],
    )


# ============================================================
# CLI
# ============================================================

def build_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 5 ComfyUI reconstruction candidates "
            "from one Analyzer run_id."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--run-id",
        type=int,
    )

    parser.add_argument(
        "--count",
        type=int,
    )

    return parser


def main(
) -> int:
    args = build_parser().parse_args()

    cfg = load_config(
        args.config
    )

    run_id = args.run_id

    if run_id is None:
        raw = input(
            "请输入 analysis run_id: "
        ).strip()

        run_id = int(
            raw
        )

    count = (
        args.count
        if args.count is not None
        else int(
            cfg.get(
                "images_per_run",
                5,
            )
        )
    )

    generate_candidates(
        run_id=run_id,
        count=count,
        cfg=cfg,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
