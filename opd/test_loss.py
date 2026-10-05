import unittest

import torch

from opd.loss import OPD, chunked_opd_loss, opd_loss


class LossTests(unittest.TestCase):
    def inputs(self, vocab=13, top_k=10):
        torch.manual_seed(4)
        logits = torch.randn(4, vocab, requires_grad=True)
        teacher = torch.randn(4, vocab).log_softmax(-1)
        values, ids = teacher.topk(top_k, dim=-1)
        sampled = torch.tensor([0, 1, 2, 3])
        rollout = logits.detach().log_softmax(-1).gather(1, sampled[:, None]).squeeze(1)
        sampled_teacher = teacher.gather(1, sampled[:, None]).squeeze(1)
        return logits, sampled, rollout, ids, values, sampled_teacher

    def test_residual_bucket_value_and_gradient(self):
        args = self.inputs()
        logits, sampled, rollout, ids, values, teacher_logp = args
        loss, metrics = opd_loss(*args, forward_coef=1, reverse_coef=0, entropy_coef=0)
        student = logits.log_softmax(-1).gather(1, ids).exp()
        teacher = values.exp()
        p = torch.cat([teacher, 1 - teacher.sum(-1, keepdim=True)], dim=-1)
        q = torch.cat([student, 1 - student.sum(-1, keepdim=True)], dim=-1)
        reference = (p * (p.log() - q.log())).sum(-1).mean()
        expected_grad = torch.autograd.grad(reference, logits, retain_graph=True)[0]
        loss.backward()
        torch.testing.assert_close(loss, reference)
        torch.testing.assert_close(logits.grad, expected_grad)
        self.assertGreaterEqual(float(metrics['forward_kl'].detach()), 0)

    def test_full_vocabulary_has_finite_gradients(self):
        args = self.inputs(vocab=10, top_k=10)
        loss, _ = opd_loss(*args)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(args[0].grad).all())

    def test_entropy_bonus_sign(self):
        args = self.inputs()
        loss, metrics = opd_loss(*args, forward_coef=0, reverse_coef=0)
        torch.testing.assert_close(loss, -0.01 * metrics['entropy'])
        loss.backward()
        self.assertTrue(torch.isfinite(args[0].grad).all())

    def test_reverse_policy_gradient(self):
        args = self.inputs()
        logits, sampled, rollout, ids, values, teacher = args
        loss, _ = opd_loss(*args, forward_coef=0, reverse_coef=1, entropy_coef=0)
        probs = logits.detach().softmax(-1)
        selected = logits.detach().log_softmax(-1).gather(1, sampled[:, None]).squeeze(1)
        advantage = selected - teacher
        expected = -probs
        expected[torch.arange(len(sampled)), sampled] += 1
        expected *= advantage[:, None] / len(sampled)
        loss.backward()
        torch.testing.assert_close(logits.grad, expected)

    def test_chunked_matches_direct(self):
        torch.manual_seed(8)
        head = torch.nn.Linear(7, 13)
        hidden = torch.randn(4, 7, requires_grad=True)
        args = self.inputs()
        expected, _ = opd_loss(head(hidden), *args[1:])
        grad = torch.autograd.grad(expected, (hidden, head.weight, head.bias), retain_graph=True)
        actual, _ = chunked_opd_loss(hidden, head, *args[1:], chunk_size=2)
        got = torch.autograd.grad(actual, (hidden, head.weight, head.bias))
        torch.testing.assert_close(actual, expected)
        for left, right in zip(grad, got):
            torch.testing.assert_close(left, right)

    def test_paper_defaults(self):
        self.assertEqual((OPD().forward_coef, OPD().reverse_coef, OPD().entropy_coef), (3, 1, 0.01))


if __name__ == '__main__':
    unittest.main()
