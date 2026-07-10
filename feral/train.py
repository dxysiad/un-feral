import datetime
import logging
import os
import random

import numpy as np
import torch
import wandb

from feral.data import build_datasets_and_loaders, resolve_splits
from feral.embeddings import extract_embeddings
from feral.loops import train_contrastive_epoch, evaluate_contrastive
from feral.modeling import build_model, build_contrastive_objects, load_model_from_checkpoint
from feral.utils import save_model, check_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# force pytorch to use flash-attention kernel for scaled dot product attention
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


def _str_now():
    """Return the current local time as a filename-safe 'YYYY-MM-DD_HH-MM-SS' string."""
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def main(cfg):
    """Run the fully unsupervised contrastive training pipeline for one config.

    Builds the label-free splits, the headless encoder, and contrastive triplet
    loaders, then trains the encoder with the per-frame triplet loss. Val/test
    report held-out contrastive loss (no labels/metrics). The best encoder (lowest
    val loss) is checkpointed; if an inference split is present its per-chunk
    embeddings are extracted to an ``.npz``. Returns None.
    """
    check_environment(compile_enabled=cfg['training']['compile'])

    os.makedirs("answers", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = True

    wandb.init(
        project=cfg.get('wandb', {}).get('project'),
        config=cfg,
        mode='disabled' if cfg.get('wandb') is None else 'online',
    )

    model_save_metadata = {'cfg': cfg}

    splits = resolve_splits(cfg)
    datasets, loaders = build_datasets_and_loaders(cfg, splits)
    train_dataset = datasets.get('train')
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    test_loader = loaders.get('test')
    inference_loader = loaders.get('inference')

    device = torch.device(cfg.get('device', 'cuda'))

    # Contrastive-loss knobs shared by train and eval.
    margin = cfg['training'].get('contrastive_margin', 1.0)
    loss_kwargs = dict(margin=margin)

    # The model is built whenever there's something to train or evaluate.
    if train_loader is not None or test_loader is not None or inference_loader is not None:
        model = build_model(cfg, device)
        if wandb.run is not None:
            wandb.run.summary['n_params'] = sum(p.numel() for p in model.parameters())

    best_checkpoint_path = os.path.join("checkpoints", f"{cfg['run_name']}_best_checkpoint.pt")

    if train_loader is not None:
        logger.info("Contrastive training for %d epochs (%d samples/epoch)",
                    cfg['training']['epochs'], len(train_dataset))
        optimizer, lr_scheduler = build_contrastive_objects(cfg, model, train_loader)

        best_loss = float('inf')
        epochs_without_updates = 0

        for epoch in range(cfg['training']['epochs']):
            train_dataset.resample()
            avg_loss = train_contrastive_epoch(
                model, train_loader, optimizer, lr_scheduler, device=device,
                log_fn=wandb.log, max_batches=cfg.get('max_batches'),
                grad_clip_norm=cfg['training'].get('grad_clip_norm'),
                log_grad_norm=cfg['training'].get('log_grad_norm', True),
                **loss_kwargs,
            )
            logs = {'contrastive/epoch_loss': avg_loss, 'contrastive/epoch': epoch}
            if torch.cuda.is_available():
                logs['perf/gpu_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
            logger.info("Epoch %d: train contrastive loss %.4f", epoch, avg_loss)

            if val_loader is None:
                save_model(model, best_checkpoint_path, model_save_metadata)
                wandb.log(logs)
                logger.info("Epoch %d: Saved model", epoch)
                continue

            val_loss = evaluate_contrastive(
                model, val_loader, device=device,
                max_batches=cfg.get('max_batches'), **loss_kwargs,
            )
            logs['val/contrastive_loss'] = val_loss
            logger.info("Epoch %d: val contrastive loss %.4f", epoch, val_loss)
            wandb.log(logs)

            # Lower val contrastive loss is better.
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_updates = 0
                save_model(model, best_checkpoint_path, model_save_metadata)
                logger.info("Epoch %d: Saved best checkpoint with val/contrastive_loss=%.4f", epoch, val_loss)
            else:
                epochs_without_updates += 1
                logger.info("Epoch %d: Didnt improve for %d epochs", epoch, epochs_without_updates)
                patience = cfg['training'].get('patience')
                if patience is not None and epochs_without_updates >= patience:
                    logger.info("Epoch %d: Early stopping: no improvement for %d epochs", epoch, patience)
                    break

        logger.info("Finished training. Best checkpoint: %s (best val loss: %.4f)",
                    best_checkpoint_path, best_loss)
        del model, optimizer, lr_scheduler
        torch.cuda.empty_cache()

    if test_loader is None and inference_loader is None:
        return

    logger.info("Loading model for test/inference")
    test_checkpoint_path = best_checkpoint_path if train_loader is not None else cfg['starting_checkpoint']
    best_model, _meta = load_model_from_checkpoint(cfg, device, test_checkpoint_path)

    if test_loader is not None:
        logger.info("Running test (held-out contrastive loss)...")
        test_loss = evaluate_contrastive(
            best_model, test_loader, device=device,
            max_batches=cfg.get('max_batches'), **loss_kwargs,
        )
        logger.info("test/contrastive_loss: %.4f", test_loss)
        wandb.log({'test/contrastive_loss': test_loss})

    if inference_loader is not None:
        logger.info("Extracting embeddings for the inference split...")
        emb, ids = extract_embeddings(
            best_model, inference_loader, device=device,
            pool=cfg['data'].get('embedding_pool', 'mean'),
            max_batches=cfg.get('max_batches'),
        )
        out_pth = os.path.join("answers", f"_embeddings_{cfg['run_name']}_{_str_now()}.npz")
        files = np.array([f for f, _ in ids])
        starts = np.array([s for _, s in ids])
        np.savez(out_pth, emb=emb.numpy(), files=files, starts=starts)
        logger.info("Saved %d embeddings to %s", len(files), out_pth)


if __name__ == '__main__':
    from feral.cli import main as cli_main
    cli_main()
