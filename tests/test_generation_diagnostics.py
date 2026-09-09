import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from app import generator


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class GenerationDiagnosticTests(unittest.TestCase):
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
        self.template = generator.load_json(
            Path(self.cfg["workflow_path"])
        )

    def test_variant_builder_does_not_mutate_config(self):
        before = copy.deepcopy(self.cfg)
        variants = dict(
            generator.diagnostic_variant_configs(self.cfg)
        )

        self.assertEqual(self.cfg, before)
        self.assertTrue(variants["A"]["lora_slots"][0]["enabled"])
        self.assertFalse(variants["B"]["lora_slots"][0]["enabled"])
        self.assertFalse(variants["C"]["lora_slots"][0]["enabled"])
        self.assertFalse(variants["D"]["lora_slots"][0]["enabled"])
        self.assertFalse(variants["E"]["lora_slots"][0]["enabled"])
        self.assertEqual(
            variants["D"]["positive_prompt_append"],
            generator.DIAGNOSTIC_STYLE_PROMPT,
        )
        self.assertEqual(variants["E"]["resolution"]["mode"], "source")

    def test_single_sampler_variant_bypasses_refiner(self):
        variants = dict(
            generator.diagnostic_variant_configs(self.cfg)
        )
        workflow, metadata = generator.build_workflow_for_generation(
            self.template,
            variants["C"],
            prompt="test prompt",
            seed=123,
            filename_prefix="test",
        )
        ids = variants["C"]["workflow_nodes"]

        self.assertEqual(metadata["loras"], [])
        self.assertEqual(metadata["sampling"]["mode"], "single")
        self.assertEqual(metadata["sampling"]["base"]["cfg"], 6.0)
        self.assertEqual(metadata["sampling"]["base"]["steps"], 35)
        self.assertEqual(
            workflow[ids["vae_decode"]]["inputs"]["samples"],
            [ids["base_sampler"], 0],
        )
        self.assertEqual(
            workflow[ids["vae_decode"]]["inputs"]["vae"],
            [ids["base_checkpoint"], 2],
        )

    def test_dual_sampler_variant_keeps_refiner_path(self):
        variants = dict(
            generator.diagnostic_variant_configs(self.cfg)
        )
        workflow, metadata = generator.build_workflow_for_generation(
            self.template,
            variants["B"],
            prompt="test prompt",
            seed=123,
            filename_prefix="test",
        )
        ids = variants["B"]["workflow_nodes"]

        self.assertEqual(metadata["sampling"]["mode"], "dual")
        self.assertEqual(
            workflow[ids["vae_decode"]]["inputs"]["samples"],
            [ids["refiner_sampler"], 0],
        )

    def test_source_resolution_preserves_portrait_aspect_ratio(self):
        self.assertEqual(
            generator.fit_source_resolution(
                853,
                1280,
                target_pixels=1024 * 1024,
            ),
            (832, 1280),
        )

    def test_variant_e_resolves_source_dimensions_and_provenance(self):
        variants = dict(
            generator.diagnostic_variant_configs(self.cfg)
        )

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "portrait.png"
            Image.new("RGB", (853, 1280)).save(source_path)
            resolved = generator.resolve_resolution_config(
                variants["E"],
                self.template,
                source_path=source_path,
            )

        prompt = generator.append_generation_prompt(
            "1girl",
            resolved,
        )
        workflow, metadata = generator.build_workflow_for_generation(
            self.template,
            resolved,
            prompt=prompt,
            seed=123,
            filename_prefix="test",
        )
        latent_id = resolved["workflow_nodes"]["latent"]

        self.assertEqual(prompt, f"1girl, {generator.DIAGNOSTIC_STYLE_PROMPT}")
        self.assertEqual(workflow[latent_id]["inputs"]["width"], 832)
        self.assertEqual(workflow[latent_id]["inputs"]["height"], 1280)
        self.assertEqual(metadata["resolution"]["mode"], "source")
        self.assertEqual(metadata["resolution"]["source_width"], 853)
        self.assertEqual(metadata["resolution"]["source_height"], 1280)

    def test_img2img_variants_share_baseline_and_only_change_conditioning(self):
        before = copy.deepcopy(self.cfg)
        variants = dict(
            generator.img2img_diagnostic_variant_configs(self.cfg)
        )

        self.assertEqual(self.cfg, before)
        self.assertEqual(list(variants), ["F", "G", "H"])
        for variant in variants.values():
            self.assertEqual(variant["sampling_mode"], "single")
            self.assertEqual(
                variant["sampling_overrides"]["base"]["cfg"],
                6.0,
            )
            self.assertEqual(variant["resolution"]["mode"], "source")
            self.assertFalse(
                any(slot["enabled"] for slot in variant["lora_slots"])
            )
            self.assertEqual(
                variant["positive_prompt_append"],
                generator.DIAGNOSTIC_STYLE_PROMPT,
            )

        self.assertFalse(variants["F"]["img2img"]["enabled"])
        self.assertEqual(variants["G"]["img2img"]["denoise"], 0.60)
        self.assertEqual(variants["H"]["img2img"]["denoise"], 0.75)

    def test_img2img_workflow_uses_source_latent_and_records_denoise(self):
        variants = dict(
            generator.img2img_diagnostic_variant_configs(self.cfg)
        )

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "portrait.png"
            Image.new("RGB", (853, 1280)).save(source_path)
            resolved = generator.resolve_resolution_config(
                variants["G"],
                self.template,
                source_path=source_path,
            )

        resolved["img2img"].update({
            "comfy_image": "diagnostic-source.png",
            "source_path": str(source_path),
            "source_sha256": "abc123",
            "frame": 0,
            "resize_method": "lanczos_exact",
        })
        workflow, metadata = generator.build_workflow_for_generation(
            self.template,
            resolved,
            prompt="test prompt",
            seed=123,
            filename_prefix="test",
        )
        ids = resolved["workflow_nodes"]

        self.assertEqual(workflow["9101"]["class_type"], "LoadImage")
        self.assertEqual(
            workflow["9101"]["inputs"]["image"],
            "diagnostic-source.png",
        )
        self.assertEqual(workflow["9102"]["class_type"], "VAEEncode")
        self.assertEqual(
            workflow["9102"]["inputs"]["vae"],
            [ids["base_checkpoint"], 2],
        )
        self.assertEqual(
            workflow[ids["base_sampler"]]["inputs"]["latent_image"],
            ["9102", 0],
        )
        self.assertEqual(
            workflow[ids["base_sampler"]]["inputs"]["start_at_step"],
            14,
        )
        self.assertTrue(metadata["sampling"]["img2img"]["enabled"])
        self.assertEqual(metadata["sampling"]["img2img"]["denoise"], 0.60)

    def test_img2img_upload_uses_first_frame_and_exact_target_size(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "animated.gif"
            first = Image.new("RGB", (40, 60), "red")
            second = Image.new("RGB", (40, 60), "blue")
            first.save(
                source_path,
                save_all=True,
                append_images=[second],
                duration=100,
                loop=0,
            )

            response = mock.Mock()
            response.json.return_value = {
                "name": "uploaded.png",
                "subfolder": "diagnostics",
                "type": "input",
            }
            with mock.patch(
                "app.generator.requests.post",
                return_value=response,
            ) as post:
                metadata = generator.upload_img2img_source(
                    "http://127.0.0.1:8188",
                    source_path,
                    width=64,
                    height=128,
                )

            response.raise_for_status.assert_called_once_with()
            uploaded = post.call_args.kwargs["files"]["image"][1]
            uploaded.seek(0)
            with Image.open(uploaded) as prepared:
                self.assertEqual(prepared.size, (64, 128))
                self.assertEqual(prepared.getpixel((0, 0)), (255, 0, 0))

        self.assertEqual(
            metadata["comfy_image"],
            "diagnostics/uploaded.png",
        )
        self.assertEqual(metadata["frame"], 0)


if __name__ == "__main__":
    unittest.main()
