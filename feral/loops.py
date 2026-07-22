import logging
import time

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def _global_grad_norm(parameters, norm_type=2.0):
    """Read-only total gradient norm (matches clip_grad_norm_'s return value)
    WITHOUT mutating any gradient.

    Note: clip_grad_norm_(max_norm=inf) is NOT a safe read-only measurement — on
    an overflowing batch total_norm is inf, the clip coefficient is inf/inf=NaN,
    and it multiplies every gradient by NaN in place, turning a recoverable
    overflow into corrupted weights. This only reads the gradients.

    Returns None if no parameter has a gradient.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return None
    return torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type) for g in grads]),
        norm_type,
    )


def train_contrastive_epoch(model, loader, optimizer, scheduler, *, device,
                            margin=1.0, normalize=True, hinge=True,
                            log_fn=None, max_batches=None,
                            grad_clip_norm=None, log_grad_norm=True):
    """Run one epoch of self-supervised contrastive training. Returns avg_loss.

    The loader yields {vid1, vid2, vid3} triplet batches; the loss is the
    per-chunk triplet loss from ``feral.contrastive`` (no criterion/metrics), and
    only the encoder (unfrozen backbone + clip_projector + mlp) receives gradients.
    """
    from feral.contrastive import contrastive_step
    model.train()
    losses = []
    t_end = time.perf_counter()
    for i, batch in enumerate(tqdm(loader, total=len(loader))):
        t_data = time.perf_counter() - t_end
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            loss, diag = contrastive_step(model, batch, margin=margin,
                                          normalize=normalize, hinge=hinge)
        loss.backward()

        grad_norm = None # scales down vectors that grow past a set threshold to prevent exploding gradients
        if grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        elif log_grad_norm:
            grad_norm = _global_grad_norm(model.parameters())

        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        if log_fn is not None:
            batch_logs = {
                'contrastive/batch_loss': loss_val,
                'contrastive/d_pos': diag['d_pos'],
                'contrastive/d_neg': diag['d_neg'],
                'contrastive/lr': scheduler.get_last_lr()[0],
                'perf/step_time': time.perf_counter() - t_end,
                'perf/data_time': t_data,
            }
            if grad_norm is not None:
                batch_logs['contrastive/grad_norm'] = grad_norm.item()
            log_fn(batch_logs)

        losses.append(loss_val)
        if max_batches is not None and i + 1 >= max_batches:
            break
        t_end = time.perf_counter()

    return sum(losses) / len(losses) if losses else 0.0


@torch.no_grad()
def evaluate_contrastive(model, loader, *, device, margin=1.0, normalize=True,
                         hinge=True, max_batches=None):
    """Average held-out triplet loss over a contrastive loader (val/test).

    The unsupervised analogue of the old classification ``evaluate``: no labels,
    no metrics — just the mean per-chunk triplet loss with grads disabled.
    Returns avg_loss (0.0 for an empty loader).
    """
    from feral.contrastive import contrastive_step
    model.eval()
    losses = []
    for i, batch in enumerate(tqdm(loader, total=len(loader))):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            loss, _ = contrastive_step(model, batch, margin=margin,
                                       normalize=normalize, hinge=hinge)
        losses.append(loss.item())
        if max_batches is not None and i + 1 >= max_batches:
            break
    return sum(losses) / len(losses) if losses else 0.0
