import logging

import torch
from torch.optim import AdamW
from timm.utils import ModelEma
from torchvision.transforms.v2 import MixUp
from transformers import get_cosine_schedule_with_warmup

from feral.backbones import warn_if_resize_mismatch
from feral.model import FeralModel
from feral.utils import get_weights

logger = logging.getLogger(__name__)


def _load_starting_checkpoint(model, checkpoint_path, device):
    """Load weights from a prior checkpoint into ``model`` for resuming / transfer.

    Loads only tensors whose name AND shape match the model, so an encoder
    pretrained with a differently-sized or untrained classification head (e.g. a
    contrastive ``*_pretrained.pt``) still transfers — the mismatched head stays
    at init and is (re)trained. Anything skipped or left at init is logged.
    Call on the un-compiled model (clean, unprefixed keys) before compile/EMA.
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


def build_model(cfg, num_classes, device, *, with_ema=True):
    """Construct FeralModel, move to device, optionally load a starting checkpoint,
    then compile and wrap in EMA.

    If ``cfg['starting_checkpoint']`` is set, its matching weights are loaded into
    the model before compile/EMA (encoder transfers even if the head size differs).

    Returns (model, model_ema). model_ema is None if cfg['ema_decay'] is None
    or with_ema is False.
    """
    warn_if_resize_mismatch(cfg)
    model = FeralModel(
        backbone=cfg['backbone'],
        num_classes=num_classes,
        predict_per_item=cfg['predict_per_item'],
        **cfg['model'],
    )
    model.to(device)

    if cfg.get('starting_checkpoint') is not None:
        _load_starting_checkpoint(model, cfg['starting_checkpoint'], device)

    if cfg['training']['compile']:
        model = torch.compile(model, mode="reduce-overhead")

    model_ema = None
    if with_ema and cfg['ema_decay'] is not None:
        model_ema = ModelEma(model, decay=cfg['ema_decay'], device=device)

    n_params = sum(el.numel() for el in model.state_dict().values())
    logger.info("parameters: %s", f"{n_params:_d}")

    return model, model_ema


def load_model_from_checkpoint(cfg, device, checkpoint_path, num_classes=None):
    """Build a fresh model and load weights from checkpoint_path.

    Supports both new-style checkpoints (dict with 'state_dict', 'class_names',
    'is_multilabel') and legacy checkpoints (bare state_dict).

    For new-style checkpoints, num_classes is derived from the metadata and the
    argument is ignored. For legacy checkpoints, num_classes must be provided.

    Returns (model, metadata) where metadata is a dict with 'class_names' and
    'is_multilabel', or None for legacy checkpoints.
    """
    raw = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(raw, dict) and 'state_dict' in raw:
        state_dict = raw['state_dict']
        metadata = {
            'class_names': raw['class_names'],
            'is_multilabel': raw['is_multilabel'],
            'cfg': raw.get('cfg'),
        }
        num_classes = len(metadata['class_names'])
    else:
        logging.warning(
            "Checkpoint '%s' is a legacy format (bare state_dict) with no "
            "embedded class_names/is_multilabel. Falling back to labels_json.",
            checkpoint_path,
        )
        if num_classes is None:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' is legacy format and num_classes was not provided."
            )
        state_dict = raw
        metadata = None

    warn_if_resize_mismatch(cfg)
    model = FeralModel(
        backbone=cfg['backbone'],
        num_classes=num_classes,
        predict_per_item=cfg['predict_per_item'],
        **cfg['model'],
    )
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logging.error(
            "Checkpoint '%s' does not match the current model. "
            "This usually means the checkpoint was saved from a different model "
            "or a different number of classes.",
            checkpoint_path,
        )
        raise
    model.to(device)
    if cfg['training']['compile']:
        model = torch.compile(model, mode="max-autotune", dynamic=True)
    model.eval()
    return model, metadata


def build_contrastive_objects(cfg, model, contrastive_loader):
    """Build the optimizer + cosine schedule for the self-supervised pretraining phase.

    No criterion: the triplet loss (``feral.contrastive``) is parameter-free. Only
    parameters with ``requires_grad`` (unfrozen backbone layers + clip_projector)
    are optimized; fc_norm/head are absent from the forward_features path and so
    receive no gradient in phase 1.
    """
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['training'].get('contrastive_lr', cfg['training']['lr']),
        weight_decay=cfg['training']['weight_decay'],
    )
    total_steps = len(contrastive_loader) * cfg['training']['contrastive_epochs']
    warmup_steps = round(total_steps * cfg['training']['part_warmup'])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    return optimizer, lr_scheduler


def build_training_objects(cfg, model, train_dataset, train_loader, labels_json, device):
    """Build criterion, optimizer, lr_scheduler, mixup. Returns dict-of-objects."""
    class_weights = get_weights(
        train_dataset.json_data, cfg['model']['class_weights'], device,
        max_weight=cfg['model'].get('max_class_weight'),
    )
    if labels_json['is_multilabel']:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=cfg['training']['label_smoothing'],
            weight=class_weights,
        )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )

    total_steps = len(train_loader) * cfg['training']['epochs']
    warmup_steps = round(total_steps * cfg['training']['part_warmup'])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    mixup = (None if cfg['mixup_alpha'] is None
             else MixUp(alpha=cfg['mixup_alpha'], num_classes=cfg['training']['train_bs']))

    return criterion, optimizer, lr_scheduler, mixup
