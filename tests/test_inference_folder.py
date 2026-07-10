import json
import os
import tempfile

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from feral.train import main as train_main
from feral.inference_folder import run_inference_folder

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
VIDEOS_DIR = os.path.join(FIXTURES_DIR, 'videos')


def _build_smoke_cfg():
    with open(os.path.join(REPO_ROOT, 'feral', 'default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg['run_name'] = 'test_inference'
    cfg['max_batches'] = 1
    cfg.pop('wandb', None)
    cfg['data']['prefix'] = VIDEOS_DIR
    cfg['data']['splits_file'] = None  # auto-split the fixture folder
    cfg['training']['epochs'] = 1
    cfg['training']['train_bs'] = 1
    cfg['training']['val_bs'] = 1
    cfg['training']['num_workers'] = 0
    cfg['training']['compile'] = False
    cfg['training']['part_warmup'] = 0.0
    cfg['training']['contrastive_num_samples'] = 2
    cfg['training']['contrastive_val_num_samples'] = 2
    return cfg


_skip_no_fixtures = pytest.mark.skipif(
    not os.path.isdir(VIDEOS_DIR) or not os.listdir(VIDEOS_DIR),
    reason=f"Synthetic fixture videos not found at {VIDEOS_DIR}. Run: python tests/generate_synthetic_dataset.py",
)

_checkpoint_path = os.path.join(REPO_ROOT, 'checkpoints', 'test_inference_best_checkpoint.pt')


@pytest.fixture(scope="module")
def trained_checkpoint():
    train_main(_build_smoke_cfg())
    return _checkpoint_path


@_skip_no_fixtures
def test_inference_folder_writes_embeddings(trained_checkpoint):
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        output_path = f.name
    try:
        run_inference_folder(
            checkpoint_path=trained_checkpoint,
            video_folder=VIDEOS_DIR,
            output=output_path,
            batch_size=1,
            num_workers=0,
        )

        assert os.path.isfile(output_path)

        data = np.load(output_path, allow_pickle=True)
        assert 'emb' in data and 'files' in data and 'starts' in data
        n = data['emb'].shape[0]
        assert n > 0
        assert data['emb'].ndim == 2          # (N, D) one vector per chunk
        assert len(data['files']) == n
        assert len(data['starts']) == n
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


@_skip_no_fixtures
def test_checkpoint_embeds_training_cfg(trained_checkpoint):
    """The saved checkpoint must carry the full training cfg so downstream
    embedding extraction can read data/model params without default_config.yaml."""
    raw = torch.load(trained_checkpoint, map_location="cpu")
    assert 'cfg' in raw, "checkpoint missing 'cfg' key"
    stored = raw['cfg']
    assert stored['data']['resize_to'] == 256
    assert stored['data']['resize_style'] == 'square'
    assert stored['data']['chunk_length'] == 64
    assert stored['run_name'] == 'test_inference'
    # An unsupervised checkpoint carries no class metadata.
    assert 'class_names' not in raw
    assert 'is_multilabel' not in raw


@_skip_no_fixtures
def test_inference_uses_stored_cfg_not_default(trained_checkpoint, monkeypatch):
    """When the checkpoint has an embedded cfg, embedding extraction must use it
    and must NOT fall back to default_config.yaml. We prove that by patching the
    stored cfg with a distinctive resize_style and asserting the dataset reflects
    it — while stubbing out the model load / forward pass. The patched checkpoint
    is served in-memory (no second multi-GB write) so the test is disk-light."""
    import feral.embeddings as emb_mod
    import feral.inference_folder as inf_mod

    raw = torch.load(trained_checkpoint, map_location="cpu")
    raw['cfg']['data']['resize_style'] = 'rectangle'
    # Serve the patched checkpoint from memory for run_inference_folder's peek.
    monkeypatch.setattr(inf_mod.torch, 'load', lambda *a, **k: raw)
    # Skip building/loading the real backbone — cfg plumbing is what we test.
    monkeypatch.setattr(inf_mod, 'load_model_from_checkpoint',
                        lambda cfg, device, path: (object(), {'cfg': cfg}))

    captured = {}
    real_ctor = emb_mod.ClsDataset

    def spy_ctor(*args, **kwargs):
        captured['resize_to'] = kwargs.get('resize_to')
        captured['resize_style'] = kwargs.get('resize_style')
        return real_ctor(*args, **kwargs)

    monkeypatch.setattr(emb_mod, 'ClsDataset', spy_ctor)

    def boom(*a, **k):
        raise AssertionError("inference should not read default_config when checkpoint has a cfg")
    monkeypatch.setattr(inf_mod, '_load_default_cfg', boom)

    # Stub the actual forward pass — we only care about cfg plumbing here.
    monkeypatch.setattr(emb_mod, 'extract_embeddings',
                        lambda *a, **k: (torch.zeros(0, 4), []))

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        output_path = f.name
    try:
        run_inference_folder(
            checkpoint_path=trained_checkpoint,
            video_folder=VIDEOS_DIR,
            output=output_path,
            batch_size=1,
            num_workers=0,
        )
        assert captured['resize_to'] == 256  # unchanged from training
        assert captured['resize_style'] == 'rectangle'  # picked up from patched cfg
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)
