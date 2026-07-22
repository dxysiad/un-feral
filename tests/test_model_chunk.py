"""CPU-only tests for the chunk-level head: single-query attention pooling, the
MLP projection, and the triplet loss over per-chunk vectors.

These drive the head modules directly (no backbone weights, no fixtures), so they
run anywhere — the full backbone -> chunk shape check lives in test_backbones.py.
"""
import torch
from torch import nn

from feral.contrastive import triplet_feature_loss
from feral.model import AttentionPoolingBlockCustom

B, N, D, EMBED_DIM = 3, 12, 32, 16


def _head():
    """The clip_projector + mlp stack FeralModel builds, standalone."""
    pooler = AttentionPoolingBlockCustom(embed_dim=D, num_heads=4, out_tokens=1)
    mlp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, EMBED_DIM))
    return pooler, mlp


class TestChunkHead:
    def test_pooling_collapses_tokens_to_one_vector_per_chunk(self):
        pooler, _ = _head()
        # out_tokens=1 means the block's internal flatten already yields (B, D).
        assert pooler(torch.randn(B, N, D)).shape == (B, D)

    def test_head_projects_into_embed_dim(self):
        pooler, mlp = _head()
        assert mlp(pooler(torch.randn(B, N, D))).shape == (B, EMBED_DIM)

    def test_chunk_vector_depends_on_every_token(self):
        # Attention pooling must actually attend over the token axis: perturbing a
        # single token has to move the chunk vector.
        torch.manual_seed(0)
        pooler, mlp = _head()
        tokens = torch.randn(B, N, D)
        base = mlp(pooler(tokens))
        perturbed = tokens.clone()
        # Replace the token outright — a uniform shift would be erased by the
        # pooler's leading LayerNorm and prove nothing.
        perturbed[:, N - 1] = torch.randn(B, D) * 3.0
        assert not torch.allclose(base, mlp(pooler(perturbed)), atol=1e-4)

    def test_query_token_is_trainable_through_the_loss(self):
        torch.manual_seed(0)
        pooler, mlp = _head()
        q = [mlp(pooler(torch.randn(B, N, D))) for _ in range(3)]
        triplet_feature_loss(*q, margin=1.0).backward()
        assert pooler.x_q.grad is not None
        assert pooler.x_q.grad.abs().sum() > 0


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

    def test_accepts_stacked_per_token_tensors(self):
        # Distances are taken over the last axis, so (B, T, D) still works.
        q = [torch.randn(B, N, EMBED_DIM) for _ in range(3)]
        assert triplet_feature_loss(*q, margin=1.0).ndim == 0
