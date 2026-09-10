import base64
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from app import database, web_ui


class AutoImproveTests(unittest.TestCase):
    def test_prompt_plan_can_replace_with_weighted_tag(self):
        result = web_ui.apply_optimizer_prompt_plan(
            "1girl, spread anus, car interior, soft shading",
            {
                "replace_tags": [
                    {"from": "spread anus", "to": "(spread anus:1.25)"}
                ],
                "remove_tags": ["soft shading"],
                "add_tags": ["low-key lighting"],
            },
        )
        self.assertEqual(
            result,
            "1girl, (spread anus:1.25), car interior, low-key lighting",
        )

    def test_prompt_plan_normalizes_model_weight_syntax_and_deduplicates(self):
        result = web_ui.apply_optimizer_prompt_plan(
            "1girl, spread anus, (semi-realistic illustration:1.2)",
            {
                "replace_tags": [],
                "remove_tags": ["semi-realistic illustration"],
                "add_tags": ["spread anus:1.35", "from below:1.2"],
            },
        )
        self.assertEqual(
            result,
            "1girl, (spread anus:1.35), (from below:1.2)",
        )

    def test_optimizer_settings_are_bounded_and_can_disable_loras(self):
        cfg = {
            "img2img": {"enabled": True, "denoise": 0.6},
            "sampling_overrides": {"base": {"cfg": 6.0}},
            "lora_slots": [{"enabled": True}, {"enabled": False}],
        }
        result = web_ui.apply_optimizer_settings(
            cfg,
            {
                "recommended_settings": {
                    "img2img_denoise": 99,
                    "cfg": 99,
                    "disable_loras": True,
                }
            },
        )
        self.assertEqual(
            result["img2img"]["denoise"],
            web_ui.generator.MAX_UI_IMG2IMG_DENOISE,
        )
        self.assertEqual(result["sampling_overrides"]["base"]["cfg"], 8.0)
        self.assertFalse(any(slot["enabled"] for slot in result["lora_slots"]))
        self.assertEqual(result["generation_mode"]["id"], "ai_auto_improve")

    def test_preferred_generation_keeps_current_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "preference.sqlite3"
            conn = database.connect(db_path)
            try:
                database.init_db(conn)
                image_id = conn.execute(
                    """
                    INSERT INTO images(identity_key, created_at, updated_at)
                    VALUES ('preference-image', 'now', 'now')
                    """
                ).lastrowid
                analysis_id = conn.execute(
                    """
                    INSERT INTO analysis_runs(
                        run_uid, image_id, raw_json, imported_at, updated_at
                    ) VALUES ('preference-run', ?, '{}', 'now', 'now')
                    """,
                    (image_id,),
                ).lastrowid
                conn.commit()
            finally:
                conn.close()

            with mock.patch.object(
                web_ui,
                "db_path_from_config",
                return_value=db_path,
            ):
                conn = web_ui.connect_db()
                try:
                    first_id = conn.execute(
                        """
                        INSERT INTO generation_runs(
                            analysis_run_id, generator_version, status, prompt,
                            negative_prompt, seed, workflow_snapshot_json, created_at
                        ) VALUES (?, 'test', 'completed', 'first', '', 1, '{}', 'now')
                        """,
                        (analysis_id,),
                    ).lastrowid
                    second_id = conn.execute(
                        """
                        INSERT INTO generation_runs(
                            analysis_run_id, generator_version, status, prompt,
                            negative_prompt, seed, workflow_snapshot_json, created_at
                        ) VALUES (?, 'test', 'completed', 'second', '', 2, '{}', 'now')
                        """,
                        (analysis_id,),
                    ).lastrowid
                    conn.commit()
                finally:
                    conn.close()

                web_ui.set_preferred_generation(first_id, "first choice")
                web_ui.set_preferred_generation(second_id, "changed mind")

                conn = web_ui.connect_db()
                try:
                    current = conn.execute(
                        """
                        SELECT preferred_generation_id
                        FROM generation_preferences
                        WHERE analysis_run_id=?
                        """,
                        (analysis_id,),
                    ).fetchone()[0]
                    events = conn.execute(
                        """
                        SELECT preferred_generation_id, previous_generation_id
                        FROM generation_preference_events
                        WHERE analysis_run_id=? ORDER BY id
                        """,
                        (analysis_id,),
                    ).fetchall()
                finally:
                    conn.close()

        self.assertEqual(current, second_id)
        self.assertEqual(
            [tuple(row) for row in events],
            [(first_id, None), (second_id, first_id)],
        )

    def test_comparison_board_contains_both_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            generated = root / "generated.png"
            Image.new("RGB", (100, 200), (255, 0, 0)).save(source)
            Image.new("RGB", (100, 200), (0, 0, 255)).save(generated)

            with mock.patch.object(web_ui, "RUNS_DIR", root / "runs"):
                output, encoded = web_ui.build_generation_comparison_image(
                    source,
                    generated,
                    12,
                )

            self.assertTrue(output.is_file())
            self.assertGreater(len(base64.b64decode(encoded)), 100)
            with Image.open(output) as board:
                self.assertGreater(board.width, board.height)


if __name__ == "__main__":
    unittest.main()
