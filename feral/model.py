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
    """Unsupervised video encoder: a backbone plus an attention-pooling head.

    The model has no classification head — it maps a whole chunk to a single
    feature vector (the representation the contrastive objective is trained on
    and the representation ``feral infer`` extracts as embeddings):

        backbone tokens   (B, N, d)
        -> clip_projector (B, d)          attention pooling, one learned query
        -> mlp            (B, embed_dim)  projection head

    The pooling attends over the backbone's spatiotemporal tokens directly, so
    there is no intermediate per-frame stage; ``forward_tokens`` exposes the raw
    tokens. ``forward`` and ``forward_features`` are the same computation.
    """

    def __init__(self,
            backbone,
            predict_per_item=1,
            embed_dim=256,
            mlp_hidden_dim=None,
            freeze_encoder_layers=0,
            pretrained=True,
            gradient_checkpointing=False,
            **kwargs):

        """Assemble the encoder: a backbone, an attention-pooling projector with a
        single learned query (one vector per chunk), and an MLP into embed_dim.
        mlp_hidden_dim defaults to the backbone's hidden dim. Freezes the first
        freeze_encoder_layers backbone layers.

        Extra kwargs (leftover config keys) are ignored — notably
        ``predict_per_item``, which sized the old per-frame pooling and no longer
        affects the encoder now that it emits one vector per chunk."""

        super().__init__()

        self.backbone = BackboneAdapter(backbone, pretrained=pretrained, gradient_checkpointing=gradient_checkpointing)
        d = self.backbone.hidden_dim
        mlp_hidden = d if mlp_hidden_dim is None else mlp_hidden_dim

        self.clip_projector = AttentionPoolingBlockCustom(
            embed_dim=d, num_heads=16, out_tokens=predict_per_item
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

        self.backbone.freeze_encoder(freeze_encoder_layers)

    def forward_features(self, x):
        """Return one feature vector per chunk, shape (B, embed_dim)."""
        tokens = self.backbone(x)       # (B, N, d) -> (batch size, num of tokens per chunk, backbone hidden dim)
        pooled = self.clip_projector(tokens)  # (B * 1, d) == (B, d)
        return self.mlp(pooled)               # (B, embed_dim)

    def forward(self, x):
        """Alias for ``forward_features`` — the encoder has no classification head."""
        return self.forward_features(x)
