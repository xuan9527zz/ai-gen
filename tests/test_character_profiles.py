import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import character_profiles
from app import web_ui


class CharacterProfileTests(unittest.TestCase):
    def test_wd14_suggestion_keeps_design_tags_only(self):
        result = character_profiles.suggest_profile_tags(
            [
                "1girl, solo, long hair, black hair, red eyes, white dress, "
                "hair ornament, halo, looking at viewer, smile, nude, breasts, "
                "simple background, masterpiece"
            ],
            character_tag="example character",
        )
        self.assertEqual(
            result,
            [
                "long hair",
                "black hair",
                "red eyes",
                "white dress",
                "hair ornament",
                "halo",
            ],
        )

    def test_conflict_priorities_are_distinct(self):
        profile = {
            "character_tag": "example character",
            "prompt_tags": [
                "black hair", "long hair", "red eyes", "white dress",
                "halo", "wings",
            ],
            "updated_at": "now",
        }
        base = "1girl, brown hair, short hair, blue eyes, pink dress, halo"
        level_a = character_profiles.apply_profile(base, profile, "a")
        level_b = character_profiles.apply_profile(base, profile, "b")
        level_c = character_profiles.apply_profile(base, profile, "c")

        self.assertEqual(level_a["add_tags"], ["wings"])
        self.assertEqual(level_a["remove_tags"], [])
        self.assertEqual(
            level_b["add_tags"],
            ["black hair", "long hair", "red eyes", "white dress", "wings"],
        )
        self.assertEqual(level_b["remove_tags"], [])
        self.assertEqual(
            level_c["remove_tags"],
            ["brown hair", "short hair", "blue eyes", "pink dress"],
        )

    def test_user_confirmed_profile_round_trips_to_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profiles.json"
            saved = character_profiles.save_profile(
                character_tag="example_character",
                names=["示例人物", "作品示例人物"],
                prompt_tags="black_hair, red eyes, white dress, forehead jewel",
                wd14_outputs=["black hair, red eyes, white dress"],
                source_images=["reference.png"],
                path=path,
            )
            raw = json.loads(path.read_text(encoding="utf-8"))
            loaded = character_profiles.get_profile(
                "example character",
                path=path,
            )

        self.assertEqual(
            saved["prompt_tags"],
            ["black hair", "red eyes", "white dress", "forehead jewel"],
        )
        self.assertEqual(loaded["names"], ["示例人物", "作品示例人物"])
        self.assertEqual(raw["aliases"]["示例人物"], "example character")

    def test_parenthesized_character_tag_is_preserved(self):
        self.assertEqual(
            character_profiles.normalize_tag("lusamine_(pokemon)"),
            "lusamine (pokemon)",
        )

    def test_profile_conflict_removal_preserves_weighted_source_fragment(self):
        result = character_profiles.apply_profile(
            "1girl, (brown hair:1.2), blue eyes",
            {
                "character_tag": "example character",
                "prompt_tags": ["black hair", "red eyes"],
                "updated_at": "now",
            },
            "c",
        )
        self.assertEqual(
            result["remove_tags"],
            ["(brown hair:1.2)", "blue eyes"],
        )

    def test_profile_rejects_non_design_or_explicit_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "只保存外观"):
                character_profiles.save_profile(
                    character_tag="example character",
                    names=["示例人物"],
                    prompt_tags="black hair, looking at viewer, nude",
                    path=Path(temp_dir) / "profiles.json",
                )

    def test_generation_revalidates_profile_fingerprint_and_merge(self):
        profile = {
            "character_tag": "example character",
            "names": ["示例人物"],
            "prompt_tags": ["black hair", "red eyes"],
            "updated_at": "now",
        }
        application = character_profiles.apply_profile(
            "1girl, brown hair, blue eyes",
            profile,
            "c",
        )
        form = {
            "character_query": ["示例人物"],
            "resolved_character_tag": ["example character"],
            "character_mode": ["canonical"],
            "character_profile_priority": ["c"],
            "character_profile_add_tags": ["black hair, red eyes"],
            "character_profile_remove_tags": ["brown hair, blue eyes"],
            "character_profile_fingerprint": [
                application["profile_fingerprint"]
            ],
        }
        validated = {
            "query": "示例人物",
            "resolved_tag": "example character",
            "mode": "canonical",
            "identity_remove_tags": [],
            "appearance_remove_tags": ["brown hair", "blue eyes"],
            "remove_tags": ["brown hair", "blue eyes"],
            "verification": {},
            "removal_verifications": [],
        }
        with (
            patch.object(
                web_ui.character_resolver,
                "validate_character_override",
                return_value=validated,
            ),
            patch.object(
                web_ui.character_profiles,
                "get_profile",
                return_value=profile,
            ),
        ):
            result = web_ui.character_override_from_form(
                form,
                base_prompt="1girl, brown hair, blue eyes",
                final_prompt=(
                    "1girl, example character, black hair, red eyes"
                ),
            )

        self.assertEqual(result["profile"]["priority"], "c")
        self.assertEqual(
            result["profile"]["remove_tags"],
            ["brown hair", "blue eyes"],
        )


if __name__ == "__main__":
    unittest.main()
