import logging

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from feral.backbones import warn_if_resize_mismatch
from feral.model import FeralModel

logger = logging.getLogger(__name__)


def _load_starting_checkpoint(model, checkpoint_path, device):
    """Load weights from a prior checkpoint into ``model`` for resuming / transfer.

    Loads only tensors whose name AND shape match the model, so an encoder saved
    by an older or differently-configured run still transfers as far as it
    matches; anything skipped or left at init is logged. Call on the un-compiled
    model (clean, unprefixed keys) before compile.
    """
    raw = torch.load(checkpoint_path, map_location=device)
    state_dict = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
    model_sd = model.state_dict()
    matched = {k: v for k, v in state_dict.items()
               if k in model_sd and v.shape == model_sd[k].shape}
    skipped = [k for k in state_dict if k not in matched]
    missing = [k for k in model_sd if k not in matched]
    model.load_state_dict(matched, strict=False)
    logger.info("Loaded %d/%d tensors from starting checkpoint %s",
                len(matched), len(model_sd), checkpoint_path)
    if skipped:
        logger.warning("Starting checkpoint: skipped %d mismatched/unexpected tensors: %s",
                       len(skipped), skipped)
    if missing:
        logger.warning("Starting checkpoint: %d model tensors left at init (e.g. head): %s",
                       len(missing), missing)


def build_model(cfg, device):
    """Construct the headless FeralModel encoder, move to device, optionally load a
    starting checkpoint, then compile.

    If ``cfg['starting_checkpoint']`` is set, its matching weights are loaded into
    the model before compile.

    Returns the model.
    """
    warn_if_resize_mismatch(cfg)
    model = FeralModel(
        backbone=cfg['backbone'],
        predict_per_item=cfg['predict_per_item'],
        **cfg['model'],
    )
    model.to(device)

    if cfg.get('starting_checkpoint') is not None:
        _load_starting_checkpoint(model, cfg['starting_checkpoint'], device)

    if cfg['training']['compile']:
        model = torch.compile(model, mode="reduce-overhead")

    n_params = sum(el.numel() for el in model.state_dict().values())
    logger.info("parameters: %s", f"{n_params:_d}")

    return model


def load_model_from_checkpoint(cfg, device, checkpoint_path):
    """Build a fresh encoder and load weights from checkpoint_path.

    Supports both new-style checkpoints (dict with 'state_dict' + embedded 'cfg')
    and legacy checkpoints (bare state_dict).

    Returns (model, metadata) where metadata is ``{'cfg': ...}`` for new-style
    checkpoints, or None for legacy checkpoints.
    """
    raw = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(raw, dict) and 'state_dict' in raw:
        state_dict = raw['state_dict']
        metadata = {'cfg': raw.get('cfg')}
    else:
        logging.warning(
            "Checkpoint '%s' is a legacy format (bare state_dict).",
            checkpoint_path,
        )
        state_dict = raw
        metadata = None

    warn_if_resize_mismatch(cfg)
    model = FeralModel(
        backbone=cfg['backbone'],
        predict_per_item=cfg['predict_per_item'],
        **cfg['model'],
    )
    # Load every encoder tensor that matches by name+shape; ignore extras (e.g. a
    # classification head from a pre-headless checkpoint). Assert the encoder is
    # fully covered so a genuinely mismatched backbone still errors loudly.
    model_sd = model.state_dict()
    matched = {k: v for k, v in state_dict.items()
               if k in model_sd and v.shape == model_sd[k].shape}
    missing = [k for k in model_sd if k not in matched]
    ignored = [k for k in state_dict if k not in matched]
    # Checkpoints predating the chunk head (mlp + chunk_pooler) still cover the
    # backbone and frame pooler; let those load with the new head at init.
    chunk_head_missing = [k for k in missing if k.startswith(('mlp.', 'chunk_pooler.'))]
    encoder_missing = [k for k in missing if k not in chunk_head_missing]
    if encoder_missing:
        logging.error(
            "Checkpoint '%s' is missing weights for %d encoder tensors (e.g. %s). "
            "This usually means it was saved from a different backbone/model.",
            checkpoint_path, len(encoder_missing), encoder_missing[:6],
        )
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' does not cover the encoder: "
            f"{len(encoder_missing)} tensors missing."
        )
    if chunk_head_missing:
        logging.warning(
            "Checkpoint '%s' predates the chunk head: %d tensors (mlp/chunk_pooler) are "
            "left RANDOMLY INITIALIZED. Per-chunk embeddings from this model are "
            "meaningless until it is retrained; use embedding_pool='mean' for the "
            "legacy per-frame representation.",
            checkpoint_path, len(chunk_head_missing),
        )
    model.load_state_dict(matched, strict=False)
    if ignored:
        logging.info(
            "Loaded encoder from '%s'; ignored %d non-encoder tensors (e.g. head): %s",
            checkpoint_path, len(ignored), ignored[:6],
        )
    model.to(device)
    if cfg['training']['compile']:
        model = torch.compile(model, mode="max-autotune", dynamic=True)
    model.eval()
    return model, metadata


def build_contrastive_objects(cfg, model, train_loader):
    """Build the optimizer + cosine schedule for the self-supervised training run.

    No criterion: the triplet loss (``feral.contrastive``) is parameter-free. Only
    parameters with ``requires_grad`` (unfrozen backbone layers + clip_projector)
    are optimized.
    """
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['training'].get('contrastive_lr', cfg['training']['lr']),
        weight_decay=cfg['training']['weight_decay'],
    )
    total_steps = len(train_loader) * cfg['training']['epochs']
    warmup_steps = round(total_steps * cfg['training']['part_warmup'])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    return optimizer, lr_scheduler
