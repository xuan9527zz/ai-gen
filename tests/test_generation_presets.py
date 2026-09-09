import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import database, generator, web_ui


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class GenerationPresetTests(unittest.TestCase):
    def setUp(self):
        self.cfg = generator.load_config(
            PROJECT_ROOT / "config" / "generation.example.json"
        )
        self.cfg["workflow_path"] = str(
            PROJECT_ROOT / "workflows" / "anime.example.json"
        )
        self.cfg["lora_slots"][0].update({
            "enabled": True,
            "name": "shape-test.safetensors",
            "strength_model": 0.7,
            "strength_clip": 0.7,
        })

    def test_presets_do_not_mutate_manual_settings(self):
        before = copy.deepcopy(self.cfg)
        current = generator.apply_generation_preset(
            self.cfg,
            "current",
        )
        balanced = generator.apply_generation_preset(
            self.cfg,
            "balanced",
        )
        styled = generator.apply_generation_preset(
            self.cfg,
            "semi_realistic",
        )

        self.assertEqual(self.cfg, before)
        self.assertTrue(current["lora_slots"][0]["enabled"])
        self.assertTrue(
            all(
                not slot["enabled"]
                for slot in balanced["lora_slots"]
            )
        )
        self.assertEqual(balanced["sampling_mode"], "single")
        self.assertEqual(
            balanced["sampling_overrides"]["base"]["cfg"],
            6.0,
        )
        self.assertEqual(balanced["resolution"]["mode"], "source")
        self.assertNotIn("positive_prompt_append", balanced)
        self.assertEqual(
            styled["positive_prompt_append"],
            generator.DIAGNOSTIC_STYLE_PROMPT,
        )

    def test_preset_is_recorded_in_sampling_metadata(self):
        cfg = generator.apply_generation_preset(
            self.cfg,
            "current",
        )
        template = generator.load_json(
            Path(cfg["workflow_path"])
        )
        _workflow, metadata = generator.build_workflow_for_generation(
            template,
            cfg,
            prompt="1girl",
            seed=123,
            filename_prefix="preset_test",
        )

        self.assertEqual(metadata["sampling"]["preset"]["id"], "current")
        self.assertEqual(metadata["generation_preset"]["label"], "当前设置")

    def test_img2img_mode_applies_rated_g_recipe_without_mutating_config(self):
        before = copy.deepcopy(self.cfg)
        configured = generator.apply_generation_mode(
            self.cfg,
            "img2img",
            denoise=0.60,
        )

        self.assertEqual(self.cfg, before)
        self.assertEqual(configured["generation_mode"]["id"], "img2img")
        self.assertTrue(configured["img2img"]["enabled"])
        self.assertEqual(configured["img2img"]["denoise"], 0.60)
        self.assertEqual(configured["sampling_mode"], "single")
        self.assertEqual(
            configured["sampling_overrides"]["base"]["cfg"],
            6.0,
        )
        self.assertEqual(configured["resolution"]["mode"], "source")
        self.assertEqual(
            configured["positive_prompt_append"],
            generator.DIAGNOSTIC_STYLE_PROMPT,
        )
        self.assertTrue(
            all(not slot["enabled"] for slot in configured["lora_slots"])
        )

    def test_normal_img2img_mode_rejects_denoise_outside_ui_range(self):
        for denoise in (0.49, 0.71, float("nan"), float("inf")):
            with self.subTest(denoise=denoise):
                with self.assertRaises(ValueError):
                    generator.apply_generation_mode(
                        self.cfg,
                        "img2img",
                        denoise=denoise,
                    )

    def test_form_img2img_mode_uses_g_recipe(self):
        form = {
            "generation_preset": ["current"],
            "generation_mode": ["img2img"],
            "img2img_denoise": ["0.62"],
            "negative_prompt": ["low quality"],
        }
        with mock.patch.object(
            web_ui,
            "load_base_config",
            return_value=copy.deepcopy(self.cfg),
        ):
            configured = web_ui.config_for_form(form)

        self.assertEqual(configured["generation_mode"]["id"], "img2img")
        self.assertEqual(configured["img2img"]["denoise"], 0.62)
        self.assertEqual(configured["sampling_mode"], "single")
        self.assertEqual(configured["generation_preset"]["id"], "semi_realistic")

    def test_old_rating_table_migrates_without_losing_columns(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                """
                CREATE TABLE generation_ratings (
                    id INTEGER PRIMARY KEY,
                    generation_run_id INTEGER,
                    overall_score INTEGER,
                    comment TEXT
                )
                """
            )
            generator.migrate_generation_schema(conn)
            generator.migrate_generation_schema(conn)
            columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(generation_ratings)"
                )
            }
        finally:
            conn.close()

        self.assertTrue({
            "overall_score",
            "semantic_score",
            "style_score",
            "composition_score",
            "comment",
        }.issubset(columns))

    def test_candidate_revision_restores_effective_preset_settings(self):
        generation = {
            "loras_json": "[]",
            "negative_prompt": "low quality",
            "sampling_json": json.dumps({
                "mode": "single",
                "resolution": {
                    "mode": "source",
                    "width": 832,
                    "height": 1280,
                    "multiple": 64,
                },
                "base": {
                    "steps": 35,
                    "cfg": 6.0,
                    "sampler_name": "dpmpp_2m_sde",
                    "scheduler": "karras",
                    "start_at_step": 0,
                    "end_at_step": 35,
                    "add_noise": "enable",
                    "return_with_leftover_noise": "disable",
                },
                "preset": {
                    "id": "semi_realistic",
                    "label": "半写实增强",
                },
                "generation_mode": {
                    "id": "img2img",
                    "label": "Img2img（推荐）",
                },
                "img2img": {
                    "enabled": True,
                    "denoise": 0.60,
                    "source_sha256": "not-needed-on-revision",
                },
            }),
        }

        restored = web_ui.config_from_generation(generation)

        self.assertEqual(restored["sampling_mode"], "single")
        self.assertEqual(
            restored["sampling_overrides"]["base"]["cfg"],
            6.0,
        )
        self.assertEqual(restored["resolution"]["mode"], "source")
        self.assertEqual(
            restored["generation_preset"]["id"],
            "semi_realistic",
        )
        self.assertEqual(restored["generation_mode"]["id"], "img2img")
        self.assertTrue(restored["img2img"]["enabled"])
        self.assertEqual(restored["img2img"]["denoise"], 0.60)
        self.assertNotIn("source_sha256", restored["img2img"])

    def test_multidimensional_rating_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "ratings.sqlite3"
            conn = database.connect(db_path)
            try:
                database.init_db(conn)
                image_id = conn.execute(
                    """
                    INSERT INTO images(identity_key, created_at, updated_at)
                    VALUES ('test-image', 'now', 'now')
                    """
                ).lastrowid
                analysis_id = conn.execute(
                    """
                    INSERT INTO analysis_runs(
                        run_uid,
                        image_id,
                        raw_json,
                        imported_at,
                        updated_at
                    )
                    VALUES ('test-run', ?, '{}', 'now', 'now')
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
                    generation_id = conn.execute(
                        """
                        INSERT INTO generation_runs(
                            analysis_run_id,
                            generator_version,
                            status,
                            prompt,
                            negative_prompt,
                            seed,
                            workflow_snapshot_json,
                            created_at
                        )
                        VALUES (?, 'test', 'completed', '1girl', '', 123, '{}', 'now')
                        """,
                        (analysis_id,),
                    ).lastrowid
                    conn.commit()
                finally:
                    conn.close()

                web_ui.save_rating(
                    generation_id,
                    4,
                    "test rating",
                    semantic_score=5,
                    style_score=3,
                    composition_score=2,
                )

                conn = web_ui.connect_db()
                try:
                    rating = conn.execute(
                        """
                        SELECT
                            overall_score,
                            semantic_score,
                            style_score,
                            composition_score,
                            comment
                        FROM generation_ratings
                        WHERE generation_run_id=?
                        """,
                        (generation_id,),
                    ).fetchone()
                finally:
                    conn.close()

        self.assertEqual(tuple(rating), (4, 5, 3, 2, "test rating"))


if __name__ == "__main__":
    unittest.main()
