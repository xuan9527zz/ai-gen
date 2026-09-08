import copy
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
