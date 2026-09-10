import json
import tempfile
import unittest
from pathlib import Path

from app import source_tags


class PixivUserMappingTests(unittest.TestCase):
    def test_user_mapping_overrides_public_mapping_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            public_path = root / "public.json"
            user_path = root / "user.json"
            public_path.write_text(
                json.dumps({"黒髪": "black hair"}, ensure_ascii=False),
                encoding="utf-8",
            )

            source_tags.save_pixiv_user_mapping(
                "黒髪",
                "dark brown hair, long hair",
                user_mapping_path=user_path,
            )
            result = source_tags.parse_pixiv_tags(
                "黒髪, 未知",
                mapping_path=public_path,
                user_mapping_path=user_path,
            )

        self.assertEqual(
            result["normalized_tags"],
            ["dark brown hair", "long hair"],
        )
        self.assertEqual(result["unknown_tags"], ["未知"])
        self.assertEqual(
            result["mapping_records"],
            [
                {
                    "source_tag": "黒髪",
                    "mapped_tags": ["dark brown hair", "long hair"],
                    "mapping_source": "user_confirmed",
                }
            ],
        )

    def test_saved_mapping_is_valid_json_and_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            public_path = root / "missing-public.json"
            user_path = root / "pixiv-overrides.json"
            saved = source_tags.save_pixiv_user_mapping(
                "寝転ぶ",
                "lying, on bed, lying",
                user_mapping_path=user_path,
            )
            loaded = json.loads(user_path.read_text(encoding="utf-8"))
            result = source_tags.parse_pixiv_tags(
                "寝転ぶ",
                mapping_path=public_path,
                user_mapping_path=user_path,
            )

        self.assertEqual(saved, ["lying", "on bed"])
        self.assertEqual(loaded["寝転ぶ"], ["lying", "on bed"])
        self.assertEqual(result["normalized_tags"], ["lying", "on bed"])


if __name__ == "__main__":
    unittest.main()
