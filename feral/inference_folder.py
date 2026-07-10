"""Extract per-chunk embeddings for all videos in a folder using a saved checkpoint."""
import importlib.resources
import logging
import os

import torch
import yaml

_DEFAULT_CONFIG = importlib.resources.files("feral").joinpath("default_config.yaml")

from feral.modeling import load_model_from_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def find_videos(folder):
    """Return sorted video filenames in folder; warn on skipped non-videos, raise if none found."""
    all_files = [fn for fn in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, fn))]
    videos = [fn for fn in all_files if os.path.splitext(fn)[1].lower() in VIDEO_EXTENSIONS]
    logger.info("Found %d video files out of %d files in %s", len(videos), len(all_files), folder)
    if not videos:
        raise FileNotFoundError(f"No video files found in {folder}")
    if len(videos) < len(all_files):
        skipped = [fn for fn in all_files if fn not in videos]
        logger.warning("WARNING: %d FILES IN THE FOLDER ARE NOT VIDEOS AND WILL BE SKIPPED: %s",
                        len(skipped), skipped)
    return videos


def build_inference_labels_json(video_filenames):
    """Build a minimal splits-only dict for the inference chunk enumerator (no labels)."""
    return {'splits': {'inference': list(video_filenames)}}


def _load_default_cfg():
    """Load and return the packaged default_config.yaml as a dict."""
    with importlib.resources.as_file(_DEFAULT_CONFIG) as cfg_path:
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)


def run_inference_folder(checkpoint_path, video_folder, output=None,
                         batch_size=8, num_workers=4, compile=False,
                         mode=None, resolution=None):
    """Extract per-chunk embeddings for every video in video_folder and save an .npz.

    Reads the training cfg embedded in the checkpoint (falling back to
    default_config.yaml for legacy checkpoints), applies optional inference-time
    overrides (compile, mode -> chunk_shift, resolution -> resize_to), loads the
    encoder, taps ``forward_features`` over every chunk, and writes an ``.npz``
    with ``emb`` (N, D), ``files`` (N,), ``starts`` (N,) to ``output`` (defaults
    to embeddings_<folder>.npz).
    """
    # Peek at the checkpoint to grab the training cfg (saved since v0.2.1).
    raw = torch.load(checkpoint_path, map_location="cpu")
    stored_cfg = raw.get('cfg') if isinstance(raw, dict) else None
    if stored_cfg is not None:
        cfg = stored_cfg
        logger.info("Using training cfg embedded in checkpoint")
    else:
        logger.warning(
            "Checkpoint has no embedded training cfg (legacy format). Falling "
            "back to default_config.yaml — model/data params may not match how "
            "this checkpoint was trained."
        )
        cfg = _load_default_cfg()

    cfg['training']['compile'] = compile

    # Embedding extraction uses the eval-time chunk overlap (eval_chunk_shift) the
    # checkpoint was configured with, falling back to the training chunk_shift.
    eval_chunk_shift = cfg['data'].get('eval_chunk_shift')
    if eval_chunk_shift is not None:
        logger.info("eval_chunk_shift: chunk_shift %s -> %s (%.0f%% overlap)",
                    cfg['data']['chunk_shift'], eval_chunk_shift,
                    100 * (1 - eval_chunk_shift / cfg['data']['chunk_length']))
        cfg['data']['chunk_shift'] = eval_chunk_shift

    # Inference-time overrides. Backbone/model size is fixed by the checkpoint
    # weights and is never changed here; only chunk overlap and input resolution
    # can be safely overridden at predict time.
    if mode is not None:
        from feral.presets import infer_chunk_shift
        new_shift = infer_chunk_shift(mode, cfg['data']['chunk_length'])
        if new_shift is not None:
            logger.info("--mode %s: chunk_shift %s -> %s (%.0f%% overlap)",
                        mode, cfg['data']['chunk_shift'], new_shift,
                        100 * (1 - new_shift / cfg['data']['chunk_length']))
            cfg['data']['chunk_shift'] = new_shift
    if resolution is not None:
        logger.info("--resolution %s: resize_to %s -> %s",
                    resolution, cfg['data']['resize_to'], resolution)
        cfg['data']['resize_to'] = resolution
    device = 'cuda'

    # Imported lazily to avoid a circular import (embeddings imports find_videos
    # from this module at import time).
    from feral.embeddings import extract_embeddings_folder

    model, _metadata = load_model_from_checkpoint(cfg, device, checkpoint_path)

    if output is None:
        folder_name = os.path.basename(os.path.normpath(video_folder))
        output = f"embeddings_{folder_name}.npz"
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    emb, ids = extract_embeddings_folder(
        model, cfg, video_folder,
        batch_size=batch_size, num_workers=num_workers,
        pool=cfg['data'].get('embedding_pool', 'mean'),
        save_path=output,
    )
    logger.info("Saved %d embeddings (dim %s) to %s", emb.shape[0], tuple(emb.shape[1:]), output)
    return emb, ids


if __name__ == '__main__':
    from feral.cli import main
    main()
