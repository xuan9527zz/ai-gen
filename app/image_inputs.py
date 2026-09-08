# -*- coding: utf-8 -*-
"""Image input normalization helpers.

The uploaded original is never modified.  Animated inputs are flattened to a
temporary first-frame PNG only for consumers that would otherwise process the
whole frame batch (currently WD14 through ComfyUI).
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from PIL import Image


@contextmanager
def prepare_wd14_image(
    image_path: Path | str,
) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    """Yield a safe WD14 input path and provenance metadata.

    Static images are passed through unchanged.  GIF, APNG, animated WebP, or
    any other Pillow-readable multi-frame image is represented by frame zero
    in a temporary PNG.  The temporary file is removed after the caller has
    finished uploading it to ComfyUI.
    """

    source_path = Path(image_path)
    temporary_path: Path | None = None

    with Image.open(source_path) as image:
        source_format = str(image.format or "unknown").upper()
        frame_count = int(getattr(image, "n_frames", 1) or 1)
        is_animated = bool(
            getattr(image, "is_animated", False)
            and frame_count > 1
        )

        metadata: Dict[str, Any] = {
            "strategy": "original",
            "source_format": source_format,
            "source_frame_count": frame_count,
            "source_is_animated": is_animated,
            "selected_frame_index": None,
            "wd14_input_format": source_format,
        }

        if not is_animated:
            yield source_path, metadata
            return

        image.seek(0)

        has_alpha = (
            "A" in image.getbands()
            or "transparency" in image.info
        )

        first_frame = image.convert(
            "RGBA" if has_alpha else "RGB"
        )

        handle = tempfile.NamedTemporaryFile(
            prefix="illustrious_wd14_first_frame_",
            suffix=".png",
            delete=False,
        )
        handle.close()
        temporary_path = Path(handle.name)

        first_frame.save(
            temporary_path,
            format="PNG",
        )

        metadata.update({
            "strategy": "first_frame_png",
            "selected_frame_index": 0,
            "wd14_input_format": "PNG",
        })

    try:
        yield temporary_path, metadata
    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True,
            )
