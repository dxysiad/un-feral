"""Contrastive (self-supervised) objectives over FeralModel feature vectors.

Standalone helpers used to train the encoder on triplets from
``ContrastiveVideoDataset`` ({vid1, vid2, vid3}). Parameter-free and independent
of the training loop, so they can be driven from ``loops.train_contrastive_epoch``
or directly from a notebook.
"""
import torch
import torch.nn.functional as F


def triplet_feature_loss(q_anchor, q_pos, q_neg, margin=1.0, normalize=True, hinge=True):
    """Per-chunk L2 triplet loss over (B, D) feature tensors.

    Pulls anchor<->pos together and pushes anchor<->neg apart, per chunk:

        mean_b  relu( ||q1 - q2||^2 - ||q1 - q3||^2 + margin )

    q_*: (B, D) from ``FeralModel.forward_features`` — q1=anchor (vid1),
         q2=positive (vid2, same clip / different augmentation), q3=negative
         (vid3, a different clip). The distances are taken over the last axis,
         so per-frame (B, T, D) tensors still work (mean over frames as well).
    normalize: L2-normalize each chunk vector first (bounds distances in
               [0, 4], avoids trivial scale collapse). Recommended.
    hinge:     relu+margin (stable, standard triplet). ``hinge=False, margin=0``
               gives the literal form ``mean(d_pos - d_neg)`` — unbounded below
               and will collapse; for inspection only.
    """
    if normalize:
        q_anchor = F.normalize(q_anchor, dim=-1)
        q_pos    = F.normalize(q_pos,    dim=-1)
        q_neg    = F.normalize(q_neg,    dim=-1)
    d_pos = (q_anchor - q_pos).pow(2).sum(-1)   # (B,)
    d_neg = (q_anchor - q_neg).pow(2).sum(-1)   # (B,)
    diff = d_pos - d_neg
    return F.relu(diff + margin).mean() if hinge else diff.mean()


def contrastive_step(model, batch, margin=1.0, normalize=True, hinge=True):
    """Run a triplet batch through the model's feature tap and return the loss.

    batch: dict of {'vid1', 'vid2', 'vid3'} tensors, each (B, T, C, H, W)
           (as default-collated from ``ContrastiveVideoDataset``).

    ``forward_features`` returns one (B, D) vector per chunk, so the triplet is
    scored at the chunk level — the chunk attention pooling is trained by it.

    Returns ``(loss, diag)`` where diag holds the mean per-chunk squared
    distances ``d_pos`` / ``d_neg`` (post-normalization if enabled) for logging.
    """
    q1 = model.forward_features(batch['vid1'])
    q2 = model.forward_features(batch['vid2'])
    q3 = model.forward_features(batch['vid3'])
    loss = triplet_feature_loss(q1, q2, q3, margin=margin, normalize=normalize, hinge=hinge)
    with torch.no_grad():
        if normalize:
            a, p, n = (F.normalize(t, dim=-1) for t in (q1, q2, q3))
        else:
            a, p, n = q1, q2, q3
        diag = {
            'd_pos': (a - p).pow(2).sum(-1).mean().item(),
            'd_neg': (a - n).pow(2).sum(-1).mean().item(),
        }
    return loss, diag
