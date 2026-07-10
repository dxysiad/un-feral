import torch

from feral.loops import _global_grad_norm


def _param(grad_vals):
    p = torch.nn.Parameter(torch.zeros(len(grad_vals)))
    p.grad = torch.tensor(grad_vals)
    return p


class TestGlobalGradNorm:
    def test_matches_clip_grad_norm_value(self):
        # Read-only norm must equal clip_grad_norm_'s returned pre-clip norm.
        ours = _global_grad_norm([_param([3.0, 4.0])])
        ref = torch.nn.utils.clip_grad_norm_([_param([3.0, 4.0])], max_norm=float('inf'))
        assert ours.item() == ref.item() == 5.0

    def test_does_not_mutate_finite_grads(self):
        p = _param([3.0, 4.0])
        _global_grad_norm([p])
        assert p.grad.tolist() == [3.0, 4.0]

    def test_does_not_corrupt_on_overflow(self):
        # The whole point: an inf gradient must be left untouched (NOT turned to
        # NaN the way clip_grad_norm_(max_norm=inf) would).
        p = _param([3.0, float('inf')])
        norm = _global_grad_norm([p])
        assert p.grad[0].item() == 3.0
        assert torch.isinf(p.grad[1])
        assert torch.isinf(norm)

    def test_none_when_no_grads(self):
        p = torch.nn.Parameter(torch.zeros(2))  # grad is None
        assert _global_grad_norm([p]) is None
