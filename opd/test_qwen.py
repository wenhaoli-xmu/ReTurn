import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, Qwen3_5TextConfig

from opd.data import Unit
from opd.forward import Runtime, build_rope
from opd.replay import State, replay
from opd.test_parallel import reference_attention


class QwenTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        config = Qwen3_5TextConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            linear_conv_kernel_dim=3, linear_key_head_dim=4, linear_value_head_dim=4,
            linear_num_key_heads=1, linear_num_value_heads=2,
            layer_types=['full_attention', 'linear_attention', 'full_attention'],
            max_position_embeddings=256, partial_rotary_factor=0.5)
        self.model = AutoModelForCausalLM.from_config(config).train()
        self.run = Runtime.of(self.model, torch.device('cpu'), 256)
        self.units = [Unit([1, 2, 3], 0, []), Unit([4, 5], 3, [0]),
                      Unit([6, 7, 8], 5, [0]), Unit([9, 10], 8, [0, 2])]

    @patch('opd.forward.flash_prefill_func', side_effect=reference_attention)
    def test_dynamic_replay_matches_sequential_parameter_gradients(self, attention):
        states, hidden = [], []
        for unit in self.units:
            ctx = None if not unit.ctx else [
                (torch.cat([states[j].kv[a][0] for j in unit.ctx]),
                 torch.cat([states[j].kv[a][1] for j in unit.ctx])) for a in range(2)]
            h, kv, gdn = self.run.forward(unit.ids, ctx, states[-1].gdn if states else None)
            hidden.append(h)
            states.append(State(kv, gdn))
        def loss(h, i):
            return h.sin().square().mean() if i in (1, 3) else None
        expected = loss(hidden[1], 1) + loss(hidden[3], 3)
        expected.backward()
        gradients = {name: p.grad.clone() for name, p in self.model.named_parameters()
                     if p.grad is not None}
        self.model.zero_grad(set_to_none=True)
        out = self.run.forward_parallel(self.units)
        for left, right in zip(hidden, out):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=3e-5)
        actual = replay(self.run, self.units, loss)
        torch.testing.assert_close(expected.detach(), actual, atol=2e-6, rtol=3e-5)
        for name, parameter in self.model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(gradients[name], parameter.grad, atol=4e-6, rtol=1e-4)

    @patch('opd.forward.flash_prefill_func', side_effect=reference_attention)
    def test_full_context_matches_native_transformers(self, attention):
        self.run.cos, self.run.sin = build_rope(self.model, 256, 'cpu', dtype=torch.float32)
        units = [Unit(unit.ids, unit.begin, list(range(i))) for i, unit in enumerate(self.units)]
        with torch.no_grad():
            tokens = torch.tensor([t for unit in units for t in unit.ids])[None]
            native = self.model.model(input_ids=tokens, use_cache=False).last_hidden_state[0]
            got = torch.cat(self.run.forward_parallel(units))
        torch.testing.assert_close(native, got, atol=3e-6, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
