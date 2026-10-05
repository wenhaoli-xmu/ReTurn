import unittest

import torch

from opd.teacher import annotate_row, targets


class TeacherTests(unittest.TestCase):
    def test_sampled_log_probability_outside_topk(self):
        head = torch.nn.Identity()
        logits = torch.tensor([[8., 4., 1., -2.], [9., 3., 2., -1.]])
        sampled = torch.tensor([3, 2])
        ids, values, at_sampled = targets(head, logits, sampled, 2, 1)
        expected = logits.log_softmax(-1)
        self.assertEqual(ids, [[0, 1], [0, 1]])
        torch.testing.assert_close(torch.tensor(at_sampled), expected.gather(1, sampled[:, None]).squeeze(1))

    def test_full_context_excludes_probes_and_only_labels_assistant(self):
        class Runtime:
            device = torch.device('cpu')
            head = torch.nn.Identity()

            def __init__(self):
                self.calls = []

            def forward(self, ids, ctx, gdn):
                length = 0 if ctx is None else len(ctx[0][0])
                self.calls.append((list(ids), length, gdn))
                hidden = torch.tensor(ids, dtype=torch.float32)[:, None] * torch.arange(6)[None]
                kv = [(torch.zeros(len(ids), 1, 1), torch.zeros(len(ids), 1, 1))]
                return hidden, kv, length + len(ids)

        row = {
            'tokens': [0, 1, 2, 3, 5, 4, 3, 2, 1, 0, 4, 2],
            'loss_mask': [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
            'rollout_logprobs': [0.] * 12,
            'paras': [
                {'start': 0, 'kind': 'prompt', 'pid': 0},
                {'start': 2, 'kind': 'assistant', 'pid': 1, 'folded': True},
                {'start': 4, 'kind': 'obs', 'pid': 2},
                {'start': 6, 'kind': 'probe', 'pid': None, 'swap': [1]},
                {'start': 8, 'kind': 'assistant', 'pid': 3},
            ],
        }
        run = Runtime()
        annotate_row(row, run, top_k=2, chunk_size=1)
        self.assertEqual(run.calls, [([0, 1], 0, None), ([2, 3], 2, 2),
                                    ([5, 4], 4, 4), ([1, 0, 4, 2], 6, 6)])
        self.assertEqual([i for i, ids in enumerate(row['teacher_topk_ids']) if ids], [3, 10])


if __name__ == '__main__':
    unittest.main()
