import json
import os
import shutil
import subprocess

import pytest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from feral.train import main as train_main

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def _build_smoke_cfg(run_name='debug', splits_file=None):
    with open(os.path.join(REPO_ROOT, 'feral', 'default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg['run_name'] = run_name
    cfg['max_batches'] = 1
    cfg.pop('wandb', None)  # disable wandb
    cfg['data']['prefix'] = os.path.join(FIXTURES_DIR, 'videos')
    cfg['data']['splits_file'] = splits_file
    cfg['training']['epochs'] = 1
    cfg['training']['train_bs'] = 1
    cfg['training']['val_bs'] = 1
    cfg['training']['num_workers'] = 0
    cfg['training']['compile'] = False
    cfg['training']['part_warmup'] = 0.0
    cfg['training']['contrastive_num_samples'] = 2
    cfg['training']['contrastive_val_num_samples'] = 2
    return cfg


videos_dir = os.path.join(FIXTURES_DIR, 'videos')
_skip_no_fixtures = pytest.mark.skipif(
    not os.path.isdir(videos_dir) or not os.listdir(videos_dir),
    reason=f"Synthetic fixture videos not found at {videos_dir}. Run: python tests/generate_synthetic_dataset.py",
)

_needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)


@_skip_no_fixtures
def test_autosplit_smoke():
    """Auto-split the fixture folder and run one contrastive epoch end-to-end."""
    cfg = _build_smoke_cfg(run_name='smoke_autosplit')
    train_main(cfg)
    assert os.path.isfile(os.path.join('checkpoints', 'smoke_autosplit_best_checkpoint.pt'))


@_skip_no_fixtures
def test_explicit_splits_smoke(tmp_path):
    """Train/val/test/inference driven by an explicit label-free splits file."""
    videos = sorted(os.listdir(videos_dir))
    splits = {
        'train': videos,
        'val': videos,
        'test': videos,
        'inference': videos[:1],
    }
    splits_path = os.path.join(str(tmp_path), 'splits.json')
    with open(splits_path, 'w') as f:
        json.dump(splits, f)

    cfg = _build_smoke_cfg(run_name='smoke_splits', splits_file=splits_path)
    train_main(cfg)
    assert os.path.isfile(os.path.join('checkpoints', 'smoke_splits_best_checkpoint.pt'))


@_skip_no_fixtures
@pytest.mark.parametrize("resize_to,resize_style", [
    (192, "square"),
    (256, "rectangle"),
    (192, "rectangle"),
])
def test_smoke_resize_variants(resize_to, resize_style):
    """Run a single contrastive iteration under non-default resize configs."""
    cfg = _build_smoke_cfg(run_name=f'smoke_{resize_style}_{resize_to}')
    cfg['data']['resize_to'] = resize_to
    cfg['data']['resize_style'] = resize_style
    train_main(cfg)


@_needs_ffmpeg
def test_smoke_rectangle_nonsquare_video(tmp_path):
    """Full contrastive iteration with an actual non-square video under
    resize_style=rectangle — the real check that rectangle tensors survive the
    whole pipeline (dataset → loader → model)."""
    import cv2
    prefix = os.path.join(str(tmp_path), 'videos')
    os.makedirs(prefix, exist_ok=True)
    video_fn = 'nonsquare.mp4'
    video_path = os.path.join(prefix, video_fn)
    n_frames = 80
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"testsrc=duration={n_frames / 30:.3f}:size=320x240:rate=30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
         video_path],
        check=True, capture_output=True,
    )
    cap = cv2.VideoCapture(video_path)
    cap.release()

    splits_path = os.path.join(str(tmp_path), 'splits.json')
    with open(splits_path, 'w') as f:
        json.dump({'train': [video_fn], 'val': [video_fn]}, f)

    with open(os.path.join(REPO_ROOT, 'feral', 'default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg['run_name'] = 'smoke_rect_nonsquare'
    cfg['max_batches'] = 1
    cfg.pop('wandb', None)
    cfg['data']['prefix'] = prefix
    cfg['data']['splits_file'] = splits_path
    cfg['data']['resize_to'] = 192
    cfg['data']['resize_style'] = 'rectangle'
    cfg['training']['epochs'] = 1
    cfg['training']['train_bs'] = 1
    cfg['training']['val_bs'] = 1
    cfg['training']['num_workers'] = 0
    cfg['training']['compile'] = False
    cfg['training']['part_warmup'] = 0.0
    cfg['training']['contrastive_num_samples'] = 2
    cfg['training']['contrastive_val_num_samples'] = 2
    train_main(cfg)
