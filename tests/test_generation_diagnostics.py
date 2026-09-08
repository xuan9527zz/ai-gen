import copy
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
