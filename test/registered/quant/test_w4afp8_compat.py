import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.fp8 import Fp8KVCacheMethod
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    load_kv_cache_scales,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestW4AFp8Qwen35Compatibility(CustomTestCase):
    def _make_config(self):
        return W4AFp8Config.from_config(
            {
                "quant_method": "w4afp8",
                "ignore": [
                    "lm_head",
                    "linear_attn.in_proj_a",
                    "linear_attn.in_proj_b",
                    "linear_attn.in_proj_ba",
                ],
            }
        )

    def _make_attention(self, config):
        return RadixAttention(
            num_heads=2,
            head_dim=8,
            scaling=1.0,
            num_kv_heads=2,
            layer_id=0,
            quant_config=config,
            prefix="model.layers.0.attn",
        )

    def test_parses_checkpoint_ignore_alias(self):
        config = self._make_config()

        self.assertIn("linear_attn.in_proj_ba", config.ignored_layers)
        self.assertIn("model.linear_attn.in_proj_ba", config.ignored_layers)

    def test_registers_fp8_kv_cache_scales(self):
        attention = self._make_attention(self._make_config())

        self.assertIsInstance(attention.quant_method, Fp8KVCacheMethod)
        self.assertEqual(attention.k_scale.item(), -1.0)
        self.assertEqual(attention.v_scale.item(), -1.0)

    def test_reports_embedded_scales_loaded_from_checkpoint(self):
        attention = self._make_attention(self._make_config())
        attention.k_scale.data.fill_(0.25)
        attention.v_scale.data.fill_(0.125)
        attention.quant_method.process_weights_after_loading(attention)
        model = torch.nn.Module()
        model.attention = attention
        server_args = SimpleNamespace(
            kv_cache_dtype="fp8_e4m3", quantization_param_path=None
        )

        with self.assertLogs(
            "sglang.srt.model_executor.model_runner_components.load_model_utils",
            level="INFO",
        ) as logs:
            load_kv_cache_scales(model=model, server_args=server_args)

        self.assertIn("Loaded embedded FP8 KV cache scales", "\n".join(logs.output))

    def test_reports_missing_checkpoint_scales(self):
        attention = self._make_attention(self._make_config())
        attention.quant_method.process_weights_after_loading(attention)
        model = torch.nn.Module()
        model.attention = attention
        server_args = SimpleNamespace(
            kv_cache_dtype="fp8_e4m3", quantization_param_path=None
        )

        with self.assertLogs(
            "sglang.srt.model_executor.model_runner_components.load_model_utils",
            level="WARNING",
        ) as logs:
            load_kv_cache_scales(model=model, server_args=server_args)

        self.assertIn(
            "missing or incomplete (0/1 valid attention layers)",
            "\n".join(logs.output),
        )
        self.assertEqual(attention.k_scale.item(), 1.0)
        self.assertEqual(attention.v_scale.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
