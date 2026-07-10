import torch
from torch import nn

from feral.backbones import BackboneAdapter


class AttentionPoolingBlockCustom(nn.Module):
    def __init__(self, embed_dim, num_heads, out_tokens, **kwargs):
        """Build the attention pooling block. If out_tokens > 0, allocate that many learnable
        query tokens; if out_tokens == 0, mean-pooling is used as the query at forward time."""
        super().__init__()
        self.out_tokens = out_tokens
        if out_tokens > 0:
            self.x_q = nn.Parameter(torch.empty(out_tokens, embed_dim))
            nn.init.xavier_uniform_(self.x_q.data)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_x = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        """Cross-attend learnable (or mean-pooled) queries over the token sequence x and flatten
        the result. x is (B, N, embed_dim); returns (B * num_queries, embed_dim), where num_queries
        is out_tokens (or 1 when out_tokens == 0)."""
        if self.out_tokens == 0:
            x_q = x.mean(1, keepdim=True)
        else:
            x_q = self.x_q.unsqueeze(0).expand(x.size(0), -1, -1)
        x_q = self.ln_q(x_q)
        x_kv = self.ln_x(x)
        attn_output, _ = self.attn(x_q, x_kv, x_kv, need_weights=False)
        attn_output = attn_output.reshape(-1, x.shape[2])
        return attn_output


class FeralModel(nn.Module):
    """Unsupervised video encoder: a backbone plus an attention-pooling projector.

    The model has no classification head — it maps a chunk to per-frame feature
    vectors (the representation the contrastive objective is trained on and the
    representation ``feral infer`` extracts as embeddings). ``forward`` and
    ``forward_features`` are the same computation.
    """

    def __init__(self,
            backbone,
            predict_per_item,
            freeze_encoder_layers=0,
            pretrained=True,
            gradient_checkpointing=False,
            **kwargs):
        """Assemble the encoder: a backbone plus an attention-pooling projector
        (out_tokens = predict_per_item). Freezes the first freeze_encoder_layers
        backbone layers. Extra kwargs (leftover config keys) are ignored."""
        super().__init__()
        self.backbone = BackboneAdapter(backbone, pretrained=pretrained, gradient_checkpointing=gradient_checkpointing)
        d = self.backbone.hidden_dim

        self.clip_projector = AttentionPoolingBlockCustom(
            embed_dim=d, num_heads=16, out_tokens=predict_per_item
        )
        self.backbone.freeze_encoder(freeze_encoder_layers)

    def forward_features(self, x):
        """Return per-frame feature vectors, shape (B, predict_per_item, D)."""
        tokens = self.backbone(x)             # (B, N, D) -> (batch size, num of tokens per chunk, feature embedding dimension)
        pooled = self.clip_projector(tokens)  # (B * predict_per_item, D)
        return pooled.reshape(x.shape[0], -1, tokens.shape[-1])  # (B, predict_per_item, D)

    def forward(self, x):
        """Alias for ``forward_features`` — the encoder has no classification head."""
        return self.forward_features(x)
