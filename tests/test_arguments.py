import unittest
from argparse import ArgumentParser
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock


def load_model_params_without_cuda_dependencies():
    distribution_config = types.ModuleType("gaussian_renderer.distribution_config")
    distribution_config.init_image_distribution_config = lambda *args, **kwargs: None
    stubs = {
        "gaussian_renderer": types.ModuleType("gaussian_renderer"),
        "gaussian_renderer.distribution_config": distribution_config,
        "utils": types.ModuleType("utils"),
        "utils.general_utils": types.ModuleType("utils.general_utils"),
        "diff_gaussian_rasterization": types.ModuleType(
            "diff_gaussian_rasterization"
        ),
    }
    module_path = Path(__file__).parents[1] / "arguments" / "__init__.py"
    spec = importlib.util.spec_from_file_location("_arguments_parser_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


arguments_module = load_model_params_without_cuda_dependencies()
ModelParams = arguments_module.ModelParams
AuxiliaryParams = arguments_module.AuxiliaryParams


class SentinelModelParamsTests(unittest.TestCase):
    def test_auxiliary_and_model_options_do_not_conflict(self):
        parser = ArgumentParser()
        AuxiliaryParams(parser)
        ModelParams(parser, sentinel=True)

        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("--detect_anomaly", option_strings)
        self.assertNotIn("--etect_anomaly", option_strings)
        self.assertIn("-s", option_strings)
        self.assertIn("-m", option_strings)

    def test_list_default_is_unchanged_without_sentinel(self):
        parser = ArgumentParser()
        ModelParams(parser, sentinel=False)

        args = parser.parse_args([])

        self.assertEqual(args.resolution_scales, [1.0])

    def test_list_default_is_none_when_sentinel_is_enabled(self):
        parser = ArgumentParser()
        ModelParams(parser, sentinel=True)

        args = parser.parse_args([])

        self.assertIsNone(args.resolution_scales)

    def test_explicit_list_uses_original_element_type(self):
        parser = ArgumentParser()
        ModelParams(parser, sentinel=True)

        args = parser.parse_args(["--resolution_scales", "0.5", "1.0"])

        self.assertEqual(args.resolution_scales, [0.5, 1.0])

    def test_explicit_nullable_integer_uses_declared_type(self):
        parser = ArgumentParser()
        ModelParams(parser, sentinel=True)

        args = parser.parse_args(["--load_iteration", "-1"])

        self.assertEqual(args.load_iteration, -1)
        self.assertIsInstance(args.load_iteration, int)

    def test_old_config_receives_defaults_for_new_model_parameters(self):
        parser = ArgumentParser()
        ModelParams(parser, sentinel=True)

        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "cfg_args").write_text(
                "Namespace(model_path={!r}, resolution=4)".format(model_path),
                encoding="utf-8",
            )
            argv = [
                "train_update.py",
                "--model_path",
                model_path,
                "--load_iteration",
                "-1",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = arguments_module.get_combined_args(parser)

        self.assertEqual(args.load_iteration, -1)
        self.assertEqual(args.block_manifest, "")
        self.assertEqual(args.block_id, "")
        self.assertFalse(args.block_core_only)


if __name__ == "__main__":
    unittest.main()
