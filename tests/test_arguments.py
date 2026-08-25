import unittest
from argparse import ArgumentParser
import importlib.util
from pathlib import Path
import sys
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
    return module.ModelParams


ModelParams = load_model_params_without_cuda_dependencies()


class SentinelModelParamsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
