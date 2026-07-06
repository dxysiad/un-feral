#!/usr/bin/env python
"""Throwaway prototype: de-risk contrastive feature learning on FERAL.

This does NOT touch the training pipeline. It only proves, end to end, that:
  1. FERAL's existing model already emits 64 per-frame feature vectors per clip
     (tap `clip_projector(backbone(x))`, skipping fc_norm/head), and
  2. an InfoNCE / NT-Xent contrastive loss over those features has the right
     shapes, produces a finite loss, and backpropagates a gradient.

It mirrors the design in the plan: clip-level InfoNCE with in-batch negatives,
plus the explicit "third video" C folded in as extra hard negatives. A 64-frame
anchor, an augmented + 32-frame-shifted positive, and a different-section/video
negative.

Run ladders (fastest first):

  # (1) Pure loss/wiring test — no network, no real model, instant:
  python scratch_contrastive_prototype.py --fake-backbone

  # (2) Real FeralModel tap on random video tensors (fetches backbone config;
  #     --no-pretrained skips the weight download):
  python scratch_contrastive_prototype.py --synthetic --no-pretrained

  # (3) Full realistic smoke test on real clips:
  python scratch_contrastive_prototype.py \
      --videos /path/a.mp4 /path/b.mp4 --pretrained

Success = printed shapes are (B, 64, D) and (B, D), the loss is finite, the
backward step runs, and positive cosine-sim > negative cosine-sim on the pooled
features (only expected to hold cleanly with a pretrained backbone).
"""
import argparse

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
# Contrastive loss (clip-level NT-Xent / InfoNCE)
# --------------------------------------------------------------------------- #
def info_nce(z_a, z_p, temperature, extra_neg=None):
    """NT-Xent over L2-normalized clip embeddings.

    z_a, z_p : (B, D) — anchor and positive embeddings (a matched pair per row).
    extra_neg: (M, D) — optional embeddings used ONLY as negatives (the explicit
               third-video C). They never carry a positive target.

    In-batch negatives: for each of the 2B views, its positive is its partner and
    every other view (plus every extra_neg) is a negative.
    """
    B = z_a.shape[0]
    z = torch.cat([z_a, z_p], dim=0)                     # (2B, D)
    sim = (z @ z.t()) / temperature                      # (2B, 2B)
    sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))
    targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)   # partner index
    if extra_neg is not None and extra_neg.numel():
        sim = torch.cat([sim, (z @ extra_neg.t()) / temperature], dim=1)  # (2B, 2B+M)
    return F.cross_entropy(sim, targets)


# --------------------------------------------------------------------------- #
# Feature tap: FERAL's 64 per-frame features, before fc_norm/head
# --------------------------------------------------------------------------- #
def feral_features(model, x):
    """Return (B, predict_per_item, D) — the clip_projector output for a batch.

    Mirrors FeralModel.forward up to (but not including) fc_norm/head:
        backbone(x) -> (B, N, D)   then   clip_projector -> (B * out_tokens, D).
    """
    tokens = model.backbone(x)                # (B, N, D)
    pooled = model.clip_projector(tokens)     # (B * predict_per_item, D)
    B = x.shape[0]
    return pooled.reshape(B, -1, tokens.shape[-1])   # (B, predict_per_item, D)


class FakeBackbone(nn.Module):
    """Offline stand-in exposing the same tap surface as FeralModel.

    Lets us test the loss + training step with zero downloads. Produces
    (B, frames, D) features from raw pixels via a trivial linear encoder, so
    identical clips map to identical features (the loss can actually learn).
    """
    def __init__(self, frames, dim, in_feats):
        super().__init__()
        self.frames = frames
        self.enc = nn.Linear(in_feats, dim)

    def forward(self, x):                     # x: (B, T, C, H, W)
        B, T = x.shape[:2]
        return self.enc(x.reshape(B, T, -1))  # (B, T, D)  (T == frames)

    # match the tap helper's expectations
    def features(self, x):
        return self(x)


def build_encoder(args, device):
    """Return (feature_fn, projection_head, trainable_params_desc).

    feature_fn(x) -> (B, frames, D). Either the real FERAL tap or the offline
    FakeBackbone.
    """
    proj_dim = args.proj_dim
    if args.fake_backbone:
        D = 256
        in_feats = 3 * args.resize * args.resize
        enc = FakeBackbone(args.frames, D, in_feats).to(device)
        feature_fn = enc.features
        params = list(enc.parameters())
        desc = "FakeBackbone (offline)"
    else:
        from feral.model import FeralModel
        model = FeralModel(
            backbone=args.backbone,
            num_classes=2,                 # irrelevant to the tap; head unused
            predict_per_item=args.frames,  # 64 -> one feature token per frame
            fc_drop_rate=0.0,
            freeze_encoder_layers=args.freeze,
            pretrained=args.pretrained,
        ).to(device)
        D = model.backbone.hidden_dim
        feature_fn = lambda x: feral_features(model, x)  # noqa: E731
        params = [p for p in model.parameters() if p.requires_grad]
        desc = f"FeralModel[{args.backbone}] pretrained={args.pretrained}"

    proj = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, proj_dim)).to(device)
    params = params + list(proj.parameters())
    return feature_fn, proj, params, D, desc


# --------------------------------------------------------------------------- #
# Data: three views (anchor, shifted+augmented positive, negative)
# --------------------------------------------------------------------------- #
def load_real_clips(args, device):
    """Load (anchor, positive, negative) as (1, T, C, H, W) each from real video.

    anchor   = video[0] frames [s, s+T)
    positive = video[0] frames [s+shift, s+shift+T)  (independent augmentation)
    negative = video[1] (or a far section of video[0]) frames [0, T)
    """
    from torchvision.transforms.v2 import TrivialAugmentWide, Normalize
    from feral.dataset import read_range_video_decord, compute_decode_size, get_video_dims

    norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    aug = TrivialAugmentWide()
    scale = 1.0 / 255.0
    T, shift = args.frames, args.shift

    vids = args.videos
    neg_path = vids[1] if len(vids) > 1 else vids[0]
    neg_start = 0 if len(vids) > 1 else max(0, args.anchor_start + 4 * T)

    def load(path, start, augment):
        w, h = get_video_dims(path)
        dw, dh = compute_decode_size(w, h, args.resize, "square")
        frames = list(range(start, start + T))
        clip = read_range_video_decord(path, frames, width=dw, height=dh)  # (T,C,H,W) uint8
        if augment:
            clip = aug(clip)
        clip = norm(clip.float() * scale)
        return clip.unsqueeze(0).to(device)

    s = args.anchor_start
    anchor = load(vids[0], s, augment=True)             # first aug draw
    positive = load(vids[0], s + shift, augment=True)   # independent aug draw + 32-frame shift
    negative = load(neg_path, neg_start, augment=True)
    return anchor, positive, negative


def synthetic_clips(args, device):
    """Random tensors shaped like FERAL input: (1, T, C, H, W).

    The positive is a lightly perturbed copy of the anchor so the loss has a
    real (if weak) signal even without a trained encoder.
    """
    T, R = args.frames, args.resize
    anchor = torch.randn(1, T, 3, R, R, device=device)
    positive = anchor + 0.05 * torch.randn_like(anchor)   # near-duplicate view
    negative = torch.randn(1, T, 3, R, R, device=device)
    return anchor, positive, negative


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="vjepa2_vitl_diving48")
    ap.add_argument("--videos", nargs="*", default=[],
                    help="0/1/2+ video paths. If empty, uses synthetic tensors.")
    ap.add_argument("--anchor-start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=64, help="clip_length == predict_per_item")
    ap.add_argument("--shift", type=int, default=32, help="temporal shift for the positive")
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--freeze", type=int, default=0, help="freeze_encoder_layers")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--fake-backbone", action="store_true",
                    help="offline stand-in; no model/weights download")
    ap.add_argument("--synthetic", action="store_true",
                    help="use random video tensors instead of real files")
    ap.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    print(f"== building encoder on {device} ==")
    feature_fn, proj, params, D, desc = build_encoder(args, device)
    print(f"encoder: {desc}   feature_dim D={D}   proj_dim={args.proj_dim}")

    print("== loading three views (anchor / shifted+aug positive / negative) ==")
    if args.videos and not args.synthetic and not args.fake_backbone:
        anchor, positive, negative = load_real_clips(args, device)
    else:
        anchor, positive, negative = synthetic_clips(args, device)
    print(f"input clip shape (B,T,C,H,W): {tuple(anchor.shape)}")

    # --- feature tap: the "64 features per video" ---
    f_a = feature_fn(anchor)     # (B, frames, D)
    f_p = feature_fn(positive)
    f_n = feature_fn(negative)
    print(f"per-frame feature shape (B, frames, D): {tuple(f_a.shape)}  "
          f"<- this is the '64 features per video'")

    # clip-level embeddings: mean-pool over frames -> project -> L2-normalize
    def embed(f):
        pooled = f.mean(dim=1)                      # (B, D)
        z = F.normalize(proj(pooled), dim=-1)       # (B, proj_dim)
        return pooled, z

    pooled_a, z_a = embed(f_a)
    pooled_p, z_p = embed(f_p)
    pooled_n, z_n = embed(f_n)
    print(f"clip embedding shape (B, proj_dim): {tuple(z_a.shape)}")

    # sanity: on POOLED backbone features, is the positive closer than the negative?
    cos = lambda a, b: F.cosine_similarity(a, b, dim=-1).mean().item()  # noqa: E731
    print(f"[sanity] cos(anchor, positive)={cos(pooled_a, pooled_p):+.4f}   "
          f"cos(anchor, negative)={cos(pooled_a, pooled_n):+.4f}   "
          f"(expect pos > neg with a pretrained backbone)")

    # --- contrastive loss + one backward step ---
    loss = info_nce(z_a, z_p, args.temperature, extra_neg=z_n)
    print(f"InfoNCE loss: {loss.item():.4f}")

    opt = torch.optim.AdamW(params, lr=1e-4)
    opt.zero_grad()
    loss.backward()
    gnorm = torch.norm(torch.stack([
        p.grad.norm() for p in params if p.grad is not None
    ])).item()
    opt.step()
    print(f"backward OK. global grad-norm over {len(params)} trainable tensors: {gnorm:.4f}")
    print("== prototype wiring verified ==")


if __name__ == "__main__":
    main()
