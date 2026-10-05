import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from opd.data import Unit
from opd.forward import ReplayLayout, Runtime
from opd.replay import State, replay
from rollout.constant import PAGE_SIZE


def reference_attention(q, kp, vp, table, masks, cu_pages, cu_q, qpos, page_pos,
                        cos, sin, nkv, num_seg, scale):
    def rotate(x, positions):
        half = cos.shape[-1]
        c = cos[positions, None]
        s = sin[positions, None]
        left, right = x[..., :half], x[..., half:2 * half]
        return torch.cat([left * c - right * s, right * c + left * s,
                          x[..., 2 * half:]], dim=-1)

    out = []
    for seg in range(num_seg):
        keys, values, positions = [], [], []
        for index in range(int(cu_pages[seg]), int(cu_pages[seg + 1])):
            page, count = int(table[index]), int(masks[index])
            keys.append(kp[page, :count])
            values.append(vp[page, :count])
            positions.extend(range(int(page_pos[index]), int(page_pos[index]) + count))
        k, v = torch.cat(keys), torch.cat(values)
        begin, end = int(cu_q[seg]), int(cu_q[seg + 1])
        qpositions = torch.arange(int(qpos[seg]), int(qpos[seg]) + end - begin)
        kpositions = torch.tensor(positions)
        qr = rotate(q[begin:end], qpositions)
        kr = rotate(k, kpositions).repeat_interleave(q.size(1) // nkv, dim=1)
        v = v.repeat_interleave(q.size(1) // nkv, dim=1)
        score = torch.einsum('qhd,khd->hqk', qr, kr) * scale
        score = score.masked_fill(kpositions[None, :] > qpositions[:, None], -torch.inf)
        out.append(torch.einsum('hqk,khd->qhd', score.softmax(-1), v))
    return torch.cat(out)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 4
        self.scaling = 0.5
        self.q_proj = nn.Linear(8, 16)
        self.k_proj = nn.Linear(8, 4)
        self.v_proj = nn.Linear(8, 4)
        self.o_proj = nn.Linear(8, 8)
        self.q_norm = nn.LayerNorm(4)
        self.k_norm = nn.LayerNorm(4)


class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_kernel_size = 3
        self.key_dim = self.value_dim = 4
        self.head_k_dim = self.head_v_dim = 4
        self.num_k_heads = self.num_v_heads = 1
        self.in_proj_qkv = nn.Linear(8, 12)
        self.conv1d = nn.Conv1d(12, 12, 3, groups=12)
        self.in_proj_z = nn.Linear(8, 4)
        self.in_proj_b = nn.Linear(8, 1)
        self.in_proj_a = nn.Linear(8, 1)
        self.out_proj = nn.Linear(4, 8)
        self.A_log = nn.Parameter(torch.zeros(1))
        self.dt_bias = nn.Parameter(torch.zeros(1))

    def norm(self, x, z):
        return x * z.sigmoid()

    def chunk_gated_delta_rule(self, q, k, v, g, beta, initial_state,
                              output_final_state, use_qk_l2norm_in_kernel):
        state = q.new_zeros(1, 1, 4, 4) if initial_state is None else initial_state
        out = []
        for i in range(q.shape[1]):
            state = state * g[:, i, :, None, None].exp()
            state = state + beta[:, i, :, None, None] * torch.einsum('bhd,bhe->bhde', k[:, i], v[:, i])
            out.append(torch.einsum('bhd,bhde->bhe', q[:, i], state))
        return torch.stack(out, dim=1), state


class Layer(nn.Module):
    def __init__(self, linear=False):
        super().__init__()
        self.layer_type = 'linear_attention' if linear else 'full_attention'
        self.input_layernorm = nn.LayerNorm(8)
        self.post_attention_layernorm = nn.LayerNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8, 12), nn.SiLU(), nn.Linear(12, 8))
        if linear:
            self.linear_attn = LinearAttention()
        else:
            self.self_attn = Attention()


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([Layer(), Layer(True), Layer()])
        self.norm = nn.LayerNorm(8)


class ParallelTests(unittest.TestCase):
    def test_layout_shared_pages_and_positions(self):
        units = [Unit([1] * 65, 0, []), Unit([2] * 3, 65, [0]),
                 Unit([3] * 2, 68, [0]), Unit([4] * 4, 70, [0, 2])]
        layout = ReplayLayout.of(units, 'cpu')
        self.assertEqual(layout.pool_pages, 5)
        self.assertEqual(layout.qpos.tolist(), [0, 65, 65, 67])
        self.assertEqual(layout.page_table.tolist(), [0, 1, 0, 1, 2, 0, 1, 3, 0, 1, 3, 4])
        self.assertEqual(layout.page_pos.tolist(), [0, 64, 0, 64, 65, 0, 64, 65, 0, 64, 65, 67])

    def test_invalid_history(self):
        for history in ([0, 0], [1], [-1]):
            with self.assertRaises(ValueError):
                ReplayLayout.of([Unit([1], 0, []), Unit([2], 1, history)], 'cpu')

    def test_parallel_forward_and_gradients_match_sequential(self):
        torch.manual_seed(9)
        body = Body()
        freq = torch.outer(torch.arange(256).float(), torch.tensor([0.4, 0.03]))
        run = Runtime(body, nn.Linear(8, 32), freq.cos(), freq.sin(), 1, torch.device('cpu'))
        units = [Unit([1, 2, 3], 0, []), Unit([4, 5], 3, [0]),
                 Unit([6, 7, 8, 9], 5, [0]), Unit([10, 11], 9, [0, 2])]
        states, hidden = [], []
        with patch('opd.forward.flash_prefill_func', side_effect=reference_attention):
            for i, unit in enumerate(units):
                ctx = None if not unit.ctx else [
                    (torch.cat([states[j].kv[a][0] for j in unit.ctx]),
                     torch.cat([states[j].kv[a][1] for j in unit.ctx])) for a in range(2)]
                value, kv, gdn = run.forward(unit.ids, ctx, states[-1].gdn if states else None)
                hidden.append(value)
                states.append(State(kv, gdn))
            def loss_fn(value, i):
                return value.sin().square().mean() if i in (1, 3) else None
            loss = loss_fn(hidden[1], 1) + loss_fn(hidden[3], 3)
            loss.backward()
            expected = [p.grad.clone() for p in body.parameters()]
            body.zero_grad(set_to_none=True)
            with patch('opd.forward.flash_prefill_func', side_effect=reference_attention) as attention:
                got_hidden = run.forward_parallel(units)
                self.assertEqual(attention.call_count, 2)
                self.assertTrue(all(call.args[-2] == 4 for call in attention.call_args_list))
            for left, right in zip(hidden, got_hidden):
                torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
            actual = replay(run, units, loss_fn)
        torch.testing.assert_close(actual, loss.detach(), atol=2e-6, rtol=2e-5)
        for left, parameter in zip(expected, body.parameters()):
            torch.testing.assert_close(left, parameter.grad, atol=3e-6, rtol=3e-5)


if __name__ == '__main__':
    unittest.main()
