from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from PIL import Image, features

from app.image_inputs import prepare_wd14_image
from app import analyzer


class PrepareWd14ImageTests(unittest.TestCase):
    def test_static_image_is_not_reencoded(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "static.png"
            Image.new("RGB", (4, 3), "red").save(source)

            with prepare_wd14_image(source) as (prepared, metadata):
                self.assertEqual(prepared, source)
                self.assertEqual(metadata["strategy"], "original")
                self.assertFalse(metadata["source_is_animated"])
                self.assertEqual(metadata["source_frame_count"], 1)

    def test_animated_gif_is_flattened_to_first_frame(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "animated.gif"
            frames = [
                Image.new("RGB", (4, 3), "red"),
                Image.new("RGB", (4, 3), "blue"),
            ]
            frames[0].save(
                source,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )

            with prepare_wd14_image(source) as (prepared, metadata):
                temporary = prepared
                self.assertNotEqual(prepared, source)
                self.assertEqual(prepared.suffix, ".png")
                self.assertTrue(prepared.exists())
                self.assertEqual(metadata["strategy"], "first_frame_png")
                self.assertEqual(metadata["source_format"], "GIF")
                self.assertEqual(metadata["source_frame_count"], 2)
                self.assertEqual(metadata["selected_frame_index"], 0)

                with Image.open(prepared) as image:
                    self.assertEqual(image.n_frames, 1)
                    self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 0, 0))

            self.assertFalse(temporary.exists())

    @unittest.skipUnless(features.check("webp"), "WebP unavailable")
    def test_animated_webp_is_flattened(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "animated.webp"
            frames = [
                Image.new("RGB", (4, 3), "green"),
                Image.new("RGB", (4, 3), "yellow"),
            ]
            frames[0].save(
                source,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
                lossless=True,
            )

            with prepare_wd14_image(source) as (prepared, metadata):
                self.assertEqual(metadata["source_format"], "WEBP")
                self.assertEqual(metadata["strategy"], "first_frame_png")
                self.assertEqual(metadata["source_frame_count"], 2)
                self.assertEqual(prepared.suffix, ".png")

    def test_analyzer_uploads_only_one_frame_to_wd14(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "animated.gif"
            frames = [
                Image.new("RGB", (4, 3), "red"),
                Image.new("RGB", (4, 3), "blue"),
            ]
            frames[0].save(
                source,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )

            def inspect_upload(path):
                with Image.open(path) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertEqual(image.n_frames, 1)
                return "first-frame.png"

            workflow = {
                analyzer.WD14_LOAD_IMAGE_NODE: {
                    "inputs": {
                        "image": "old.png",
                    }
                }
            }
            history = {
                "prompt-id": {
                    "outputs": {
                        analyzer.WD14_OUTPUT_NODE: {
                            "text": ["1girl, red hair"]
                        }
                    }
                }
            }

            with (
                mock.patch.object(
                    analyzer,
                    "upload_to_comfy",
                    side_effect=inspect_upload,
                ),
                mock.patch.object(
                    analyzer,
                    "load_wd14_workflow",
                    return_value=workflow,
                ),
                mock.patch.object(
                    analyzer,
                    "queue_comfy_workflow",
                    return_value="prompt-id",
                ),
                mock.patch.object(
                    analyzer,
                    "get_comfy_history",
                    return_value=history,
                ),
            ):
                tags, metadata = analyzer.run_wd14_with_metadata(source)

            self.assertEqual(tags, "1girl, red hair")
            self.assertEqual(metadata["strategy"], "first_frame_png")
            self.assertEqual(metadata["source_frame_count"], 2)

    def test_apng_is_flattened_when_supported(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "animated.png"
            frames = [
                Image.new("RGBA", (4, 3), "red"),
                Image.new("RGBA", (4, 3), "blue"),
            ]
            frames[0].save(
                source,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )

            with Image.open(source) as image:
                if not getattr(image, "is_animated", False):
                    self.skipTest("APNG unavailable in this Pillow build")

            with prepare_wd14_image(source) as (prepared, metadata):
                self.assertEqual(metadata["source_format"], "PNG")
                self.assertEqual(metadata["strategy"], "first_frame_png")
                self.assertEqual(metadata["source_frame_count"], 2)
                self.assertEqual(prepared.suffix, ".png")


if __name__ == "__main__":
    unittest.main()
