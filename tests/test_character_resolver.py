import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import character_resolver
from app import web_ui


def verified(tag):
    return {
        "exact_match": True,
        "tag": tag,
        "category": "character",
        "power": 10000,
        "power_source": "naid_tag_suggest",
        "status": "naid_verified",
    }


class CharacterResolverTests(unittest.TestCase):
    def test_canonical_mode_removes_only_appearance_conflicts(self):
        plan = {
            "candidate_tags": ["lillie (pokemon)"],
            "existing_character_tags": [],
            "conflicting_appearance_tags": [
                "brown hair",
                "red eyes",
                "pink dress",
                "holding basket",
                "forest background",
                "closed eyes",
                "hair pulling",
                "painterly hair rendering",
            ],
        }

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(character_resolver, "_model_plan", return_value=plan),
        ):
            result = character_resolver.resolve_character(
                query="宝可梦 莉莉艾",
                base_prompt=(
                    "1girl, brown hair, red eyes, pink dress, "
                    "holding basket, forest background, closed eyes, "
                    "hair pulling, painterly hair rendering"
                ),
                mode="canonical",
                lookup=verified,
                cache_path=Path(temp_dir) / "characters.json",
            )

        self.assertEqual(
            result["appearance_remove_tags"],
            ["brown hair", "red eyes", "pink dress"],
        )
        self.assertNotIn("holding basket", result["remove_tags"])
        self.assertNotIn("forest background", result["remove_tags"])
        self.assertNotIn("closed eyes", result["remove_tags"])
        self.assertNotIn("hair pulling", result["remove_tags"])
        self.assertNotIn("painterly hair rendering", result["remove_tags"])

    def test_identical_request_and_prompt_use_plan_cache(self):
        plan = {
            "candidate_tags": ["lillie (pokemon)"],
            "existing_character_tags": [],
            "conflicting_appearance_tags": ["brown hair"],
            "note": "cached plan",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "characters.json"
            with patch.object(
                character_resolver,
                "_model_plan",
                return_value=plan,
            ) as model_plan:
                first = character_resolver.resolve_character(
                    query="宝可梦 莉莉艾",
                    base_prompt="1girl, brown hair",
                    mode="canonical",
                    lookup=verified,
                    cache_path=cache_path,
                )
                second = character_resolver.resolve_character(
                    query="神奇宝贝莉莉艾",
                    base_prompt="1girl, brown hair",
                    mode="canonical",
                    lookup=verified,
                    cache_path=cache_path,
                )

        self.assertEqual(first["resolved_tag"], "lillie (pokemon)")
        self.assertEqual(second["cache_hit"], "plan")
        self.assertEqual(model_plan.call_count, 1)

    def test_franchise_prefixed_and_short_name_share_canonical_plan(self):
        plan = {
            "candidate_tags": ["lusamine (pokemon)"],
            "existing_character_tags": [],
            "note": "same person",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "characters.json"
            with patch.object(
                character_resolver,
                "_model_plan",
                return_value=plan,
            ) as model_plan:
                first = character_resolver.resolve_character(
                    query="宝可梦露莎米奈",
                    base_prompt="1girl, blonde hair",
                    lookup=verified,
                    cache_path=cache_path,
                )
                second = character_resolver.resolve_character(
                    query="露莎米奈",
                    base_prompt="1girl, blonde hair",
                    lookup=verified,
                    cache_path=cache_path,
                )

        self.assertEqual(first["resolved_tag"], "lusamine (pokemon)")
        self.assertEqual(second["resolved_tag"], "lusamine (pokemon)")
        self.assertEqual(second["cache_hit"], "alias_plan")
        self.assertEqual(model_plan.call_count, 1)

    def test_manual_tag_and_aliases_are_verified_and_saved(self):
        plan = {
            "candidate_tags": ["wrong character"],
            "existing_character_tags": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "characters.json"
            with patch.object(character_resolver, "_model_plan", return_value=plan):
                result = character_resolver.resolve_character(
                    query="宝可梦露莎米奈",
                    base_prompt="1girl",
                    preferred_tag="lusamine (pokemon)",
                    user_aliases="露莎米奈，宝可梦妈妈",
                    lookup=verified,
                    cache_path=cache_path,
                )
            cache = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(result["resolved_tag"], "lusamine (pokemon)")
        self.assertTrue(result["manual_tag_used"])
        self.assertEqual(
            set(result["saved_aliases"]),
            {"宝可梦露莎米奈", "露莎米奈", "宝可梦妈妈"},
        )
        self.assertEqual(
            cache["aliases"]["露莎米奈"]["tag"],
            "lusamine (pokemon)",
        )

    def test_ambiguous_short_alias_does_not_guess(self):
        aliases = {
            "作品甲小雪": {"tag": "character a"},
            "作品乙小雪": {"tag": "character b"},
        }
        self.assertEqual(
            character_resolver._cached_alias_for_query("小雪", aliases),
            "",
        )

    def test_verified_character_replaces_only_verified_existing_character(self):
        plan = {
            "candidate_tags": ["lillie (pokemon)"],
            "existing_character_tags": [
                "aerith gainsborough",
                "brown hair",
            ],
            "note": "mapped",
        }

        def lookup(tag):
            if tag == "lillie (pokemon)":
                return verified(tag)
            if tag == "aerith gainsborough":
                return verified(tag)
            return {
                "exact_match": True,
                "tag": tag,
                "category": "general",
                "status": "naid_verified",
            }

        with patch.object(character_resolver, "_model_plan", return_value=plan):
            with tempfile.TemporaryDirectory() as temp_dir:
                result = character_resolver.resolve_character(
                    query="宝可梦 莉莉艾",
                    base_prompt=(
                        "1girl, aerith gainsborough, brown hair, green eyes"
                    ),
                    lookup=lookup,
                    cache_path=Path(temp_dir) / "characters.json",
                )

        self.assertEqual(result["resolved_tag"], "lillie (pokemon)")
        self.assertEqual(result["remove_tags"], ["aerith gainsborough"])
        self.assertEqual(result["error"], "")

    def test_fuzzy_refinement_is_still_exact_verified(self):
        plan = {
            "candidate_tags": ["edel jat (fire emblem)"],
            "existing_character_tags": [],
        }

        def lookup(tag):
            if tag == "edelgard von hresvelg":
                return verified(tag)
            return {
                "exact_match": False,
                "tag": tag,
                "category": "",
                "status": "not_found",
            }

        with (
            patch.object(character_resolver, "_model_plan", return_value=plan),
            patch.object(
                character_resolver,
                "_naid_character_suggestions",
                return_value=[{"tag": "edelgard von hresvelg"}],
            ),
            patch.object(
                character_resolver,
                "_choose_naid_suggestion",
                return_value="edelgard von hresvelg",
            ),
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                result = character_resolver.resolve_character(
                    query="火焰纹章 艾黛尔贾特",
                    base_prompt="1girl",
                    lookup=lookup,
                    cache_path=Path(temp_dir) / "characters.json",
                )

        self.assertEqual(result["resolved_tag"], "edelgard von hresvelg")
        self.assertEqual(result["error"], "")

    def test_danbooru_fallback_is_suggestion_only(self):
        plan = {
            "candidate_tags": ["some character"],
            "existing_character_tags": [],
        }

        def lookup(tag):
            return {
                "exact_match": True,
                "tag": tag,
                "category": "character",
                "status": "danbooru_fallback",
            }

        with (
            patch.object(character_resolver, "_model_plan", return_value=plan),
            patch.object(
                character_resolver,
                "_naid_character_suggestions",
                return_value=[],
            ),
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                result = character_resolver.resolve_character(
                    query="测试角色",
                    base_prompt="1girl",
                    lookup=lookup,
                    cache_path=Path(temp_dir) / "characters.json",
                )

        self.assertEqual(result["resolved_tag"], "")
        self.assertEqual(result["suggested_tag"], "some character")
        self.assertTrue(result["error"])

    def test_generation_validation_rejects_stale_final_prompt(self):
        with self.assertRaisesRegex(ValueError, "缺少已解析"):
            character_resolver.validate_character_override(
                query="宝可梦 莉莉艾",
                resolved_tag="lillie (pokemon)",
                remove_tags=[],
                base_prompt="1girl",
                final_prompt="1girl, blonde hair",
                lookup=verified,
            )

    def test_generation_validation_accepts_canonical_appearance_removal(self):
        result = character_resolver.validate_character_override(
            query="宝可梦 莉莉艾",
            resolved_tag="lillie (pokemon)",
            remove_tags=[],
            appearance_remove_tags=["brown hair", "red eyes"],
            mode="canonical",
            base_prompt="1girl, brown hair, red eyes, forest background",
            final_prompt="1girl, forest background, lillie (pokemon)",
            lookup=verified,
        )
        self.assertEqual(
            result["appearance_remove_tags"],
            ["brown hair", "red eyes"],
        )

    def test_prompt_edit_keeps_scene_and_action_when_applying_character(self):
        final_prompt = web_ui.apply_prompt_edits(
            (
                "1girl, aerith gainsborough, brown hair, pink dress, "
                "holding basket, forest background, soft lighting"
            ),
            remove_tags_text=(
                "aerith gainsborough, brown hair, pink dress"
            ),
            add_tags_text="lillie (pokemon)",
        )
        self.assertEqual(
            final_prompt,
            (
                "1girl, holding basket, forest background, soft lighting, "
                "lillie (pokemon)"
            ),
        )


class StudioSchemaMigrationTests(unittest.TestCase):
    def test_old_prompt_edit_table_gains_character_provenance(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE generation_prompt_edits (
                generation_run_id INTEGER PRIMARY KEY,
                base_prompt TEXT NOT NULL,
                final_generation_prompt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        web_ui.migrate_studio_schema(conn)
        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(generation_prompt_edits)"
            ).fetchall()
        }
        conn.close()

        self.assertIn("character_query", columns)
        self.assertIn("character_tag", columns)
        self.assertIn("character_mode", columns)
        self.assertIn("character_identity_remove_tags_json", columns)
        self.assertIn("character_appearance_remove_tags_json", columns)
        self.assertIn("character_remove_tags_json", columns)
        self.assertIn("character_verification_json", columns)

    def test_prompt_edit_save_persists_character_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "studio.sqlite3"
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE generation_runs (id INTEGER PRIMARY KEY)"
            )
            conn.executescript(web_ui.STUDIO_SCHEMA)
            conn.execute("INSERT INTO generation_runs(id) VALUES (1)")
            conn.commit()
            conn.close()

            def connect_temp():
                opened = sqlite3.connect(db_path)
                opened.row_factory = sqlite3.Row
                return opened

            with patch.object(web_ui, "connect_db", side_effect=connect_temp):
                web_ui.save_prompt_edit_records(
                    [{"generation_id": 1}],
                    parent_generation_id=None,
                    edit_kind="edited",
                    base_prompt="1girl, aerith gainsborough",
                    user_instruction="",
                    correction_model="qwen3:8b",
                    ai_add_tags=[],
                    ai_remove_tags=[],
                    manual_positive="",
                    final_prompt="1girl, lillie (pokemon)",
                    character_query="宝可梦 莉莉艾",
                    character_tag="lillie (pokemon)",
                    character_mode="canonical",
                    character_identity_remove_tags=["aerith gainsborough"],
                    character_appearance_remove_tags=["brown hair"],
                    character_remove_tags=["aerith gainsborough"],
                    character_verification={
                        "verification": verified("lillie (pokemon)")
                    },
                )

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM generation_prompt_edits WHERE generation_run_id=1"
            ).fetchone()
            conn.close()

            self.assertEqual(row["character_tag"], "lillie (pokemon)")
            self.assertEqual(row["character_mode"], "canonical")
            self.assertIn("brown hair", row["character_appearance_remove_tags_json"])
            self.assertIn("aerith gainsborough", row["character_remove_tags_json"])


if __name__ == "__main__":
    unittest.main()
