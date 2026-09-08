import copy
import unittest

from app import analyzer


SOURCE_STYLE_CAPTION = """
The lighting is soft and diffused, with a warm, golden glow that creates
subtle shadows. The background is slightly blurred, creating an ethereal
atmosphere. The overall composition is rich in detail. The illustration
style is highly detailed and realistic, with a painterly quality that
captures the textures of the fabric. Small details include delicate lace
patterns and the texture of the woven basket.
"""


class AnalyzerVisualFeatureTests(unittest.TestCase):
    def test_recovers_explicit_style_material_and_lighting(self):
        features = analyzer.extract_caption_visual_features(
            SOURCE_STYLE_CAPTION
        )
        tags = {item["tag"] for item in features}

        self.assertIn("realistic illustration", tags)
        self.assertIn("highly detailed illustration", tags)
        self.assertIn("painterly rendering", tags)
        self.assertIn("detailed fabric texture", tags)
        self.assertIn("intricate lace texture", tags)
        self.assertIn("woven texture", tags)
        self.assertIn("soft diffused lighting", tags)
        self.assertIn("warm golden lighting", tags)
        self.assertIn("subtle shadows", tags)
        self.assertIn("slightly blurred background", tags)

        for item in features:
            self.assertEqual(item["origin"], "vlm_caption_rule")
            self.assertTrue(item["evidence"])

    def test_combiner_keeps_model_feature_when_rule_matches(self):
        model_features = [
            {
                "category": "lighting",
                "tag": "soft diffused lighting",
            }
        ]
        deterministic = analyzer.extract_caption_visual_features(
            "The lighting is soft and diffused."
        )
        combined = analyzer.combine_visual_features(
            model_features,
            deterministic,
        )

        self.assertEqual(combined, model_features)

    def test_combiner_deduplicates_rendering_style_suffix(self):
        model_features = [
            {
                "category": "rendering",
                "tag": "realistic illustration style",
            }
        ]
        deterministic = [
            {
                "category": "rendering",
                "tag": "realistic illustration",
                "origin": "vlm_caption_rule",
            }
        ]

        self.assertEqual(
            analyzer.combine_visual_features(
                model_features,
                deterministic,
            ),
            model_features,
        )

    def test_negated_bokeh_is_not_recovered(self):
        features = analyzer.extract_caption_visual_features(
            "There are no visible reflections or bokeh effects."
        )

        self.assertNotIn(
            "bokeh",
            {item["tag"] for item in features},
        )

    def test_negation_does_not_cross_contrast_boundary(self):
        features = analyzer.extract_caption_visual_features(
            "There are no harsh shadows, but the rendering is painterly."
        )

        self.assertIn(
            "painterly rendering",
            {item["tag"] for item in features},
        )

    def test_sanitizer_preserves_recovery_provenance(self):
        recovered = analyzer.extract_caption_visual_features(
            "The image has a painterly quality."
        )
        before = copy.deepcopy(recovered)
        tags, accepted, rejected = analyzer.sanitize_visual_features(
            recovered
        )

        self.assertEqual(recovered, before)
        self.assertEqual(tags, ["painterly rendering"])
        self.assertEqual(rejected, [])
        self.assertEqual(
            accepted[0]["origin"],
            "vlm_caption_rule",
        )
        self.assertEqual(accepted[0]["evidence"], "painterly")

    def test_does_not_invent_style_when_caption_has_none(self):
        features = analyzer.extract_caption_visual_features(
            "A woman holds a basket beside a tree."
        )

        self.assertEqual(features, [])


if __name__ == "__main__":
    unittest.main()
