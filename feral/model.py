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
    """Unsupervised video encoder: a backbone plus a two-stage pooling head.

    The model has no classification head — it maps a chunk to a single feature
    vector (the representation the contrastive objective is trained on and the
    representation ``feral infer`` extracts as embeddings):

        backbone tokens   (B, N, d)
        -> clip_projector (B, T, d)          frame attention pooling, T = predict_per_item
        -> mlp            (B, T, embed_dim)  per-frame projection head
        -> chunk_pooler   (B, embed_dim)     chunk attention pooling

    ``forward`` and ``forward_features`` are the same computation. The per-frame
    tap is still available via ``forward_frames``.
    """

    def __init__(self,
            backbone,
            predict_per_item,
            embed_dim=256,
            mlp_hidden_dim=None,
            chunk_pool_heads=8,
            freeze_encoder_layers=0,
            pretrained=True,
            gradient_checkpointing=False,
            **kwargs):
        
        """Assemble the encoder: a backbone, a frame attention-pooling projector
        (out_tokens = predict_per_item), a per-frame MLP into embed_dim, and a
        chunk attention-pooling block (out_tokens = 1). mlp_hidden_dim defaults
        to the backbone's hidden dim. Freezes the first freeze_encoder_layers
        backbone layers. Extra kwargs (leftover config keys) are ignored."""

        super().__init__()

        if embed_dim % chunk_pool_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by chunk_pool_heads "
                f"({chunk_pool_heads}) for the chunk pooler's multi-head attention."
            )
        
        self.backbone = BackboneAdapter(backbone, pretrained=pretrained, gradient_checkpointing=gradient_checkpointing)
        d = self.backbone.hidden_dim
        mlp_hidden = d if mlp_hidden_dim is None else mlp_hidden_dim

        self.clip_projector = AttentionPoolingBlockCustom(
            embed_dim=d, num_heads=16, out_tokens=predict_per_item
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

        self.chunk_pooler = AttentionPoolingBlockCustom(
            embed_dim=embed_dim, num_heads=chunk_pool_heads, out_tokens=1
        )
        self.backbone.freeze_encoder(freeze_encoder_layers)

    def forward_frames(self, x):
        """Return per-frame feature vectors, shape (B, predict_per_item, D)."""
        tokens = self.backbone(x)             # (B, N, D) -> (batch size, num of tokens per chunk, feature embedding dimension)
        pooled = self.clip_projector(tokens)  # (B * predict_per_item, D)
        return pooled.reshape(x.shape[0], -1, tokens.shape[-1])  # (B, predict_per_item, D)

    def forward_features(self, x):
        """Return one feature vector per chunk, shape (B, embed_dim)."""
        frames = self.forward_frames(x)       # (B, T, D)
        return self.chunk_pooler(self.mlp(frames))  # (B * 1, embed_dim) == (B, embed_dim)

    def forward(self, x):
        """Alias for ``forward_features`` — the encoder has no classification head."""
        return self.forward_features(x)
