"""CPU-only tests for the chunk-level head: attention pooling, the per-frame MLP,
and the triplet loss over per-chunk vectors.

These drive the head modules directly (no backbone weights, no fixtures), so they
run anywhere — the full backbone -> chunk shape check lives in test_backbones.py.
"""
import pytest
import torch
from torch import nn

from feral.contrastive import triplet_feature_loss
from feral.model import AttentionPoolingBlockCustom, FeralModel


B, T, D, EMBED_DIM = 3, 8, 32, 16


def _head():
    """The mlp + chunk_pooler stack FeralModel builds, standalone."""
    mlp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, EMBED_DIM))
    chunk_pooler = AttentionPoolingBlockCustom(embed_dim=EMBED_DIM, num_heads=4, out_tokens=1)
    return mlp, chunk_pooler


class TestChunkHeadShapes:
    def test_frame_pooling_gives_one_vector_per_frame(self):
        pooler = AttentionPoolingBlockCustom(embed_dim=D, num_heads=4, out_tokens=T)
        out = pooler(torch.randn(B, 20, D))       # (B, num_tokens, D)
        assert out.shape == (B * T, D)
        assert out.reshape(B, T, D).shape == (B, T, D)

    def test_chunk_pooling_collapses_frames(self):
        mlp, chunk_pooler = _head()
        out = chunk_pooler(mlp(torch.randn(B, T, D)))
        assert out.shape == (B, EMBED_DIM)

    def test_chunk_vector_depends_on_every_frame(self):
        # Attention pooling must actually attend over the frame axis: perturbing a
        # single frame has to move the chunk vector.
        torch.manual_seed(0)
        mlp, chunk_pooler = _head()
        frames = torch.randn(B, T, D)
        base = chunk_pooler(mlp(frames))
        perturbed = frames.clone()
        # Replace the frame outright — a uniform shift would be erased by the
        # MLP's leading LayerNorm and prove nothing.
        perturbed[:, T - 1] = torch.randn(B, D) * 3.0
        assert not torch.allclose(base, chunk_pooler(mlp(perturbed)), atol=1e-4)

    def test_chunk_pooler_is_trainable_through_the_loss(self):
        torch.manual_seed(0)
        mlp, chunk_pooler = _head()
        frames = [torch.randn(B, T, D) for _ in range(3)]
        q = [chunk_pooler(mlp(f)) for f in frames]
        triplet_feature_loss(*q, margin=1.0).backward()
        assert chunk_pooler.x_q.grad is not None
        assert chunk_pooler.x_q.grad.abs().sum() > 0


class TestTripletOverChunks:
    def test_scalar_and_nonnegative_under_hinge(self):
        torch.manual_seed(0)
        q = [torch.randn(B, EMBED_DIM) for _ in range(3)]
        loss = triplet_feature_loss(*q, margin=1.0)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_identical_positive_beats_random_negative(self):
        # anchor == positive: d_pos is 0, so the hinge sits at max(margin - d_neg, 0),
        # strictly below the margin for a negative that is not also identical.
        torch.manual_seed(0)
        anchor = torch.randn(B, EMBED_DIM)
        loss = triplet_feature_loss(anchor, anchor.clone(), torch.randn(B, EMBED_DIM), margin=1.0)
        assert loss.item() < 1.0

    def test_accepts_legacy_per_frame_tensors(self):
        # Distances are taken over the last axis, so (B, T, D) still works.
        q = [torch.randn(B, T, EMBED_DIM) for _ in range(3)]
        assert triplet_feature_loss(*q, margin=1.0).ndim == 0


class TestConfigValidation:
    def test_embed_dim_must_divide_chunk_pool_heads(self):
        # Raised before the backbone is constructed, so this stays cheap.
        with pytest.raises(ValueError, match="divisible"):
            FeralModel(backbone='vjepa2_vitl', predict_per_item=T,
                       embed_dim=30, chunk_pool_heads=8, pretrained=False)
