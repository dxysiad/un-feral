import json
import logging
import random

from torch.utils.data import DataLoader

from feral.dataset import ClsDataset, ContrastiveVideoDataset, collate_fn_inference, collate_fn_contrastive
from feral.utils import resolve_num_workers

logger = logging.getLogger(__name__)


_PARTITION_SPECS = {
    'train':       {'shuffle': True,  'drop_last': True,  'collate_fn': collate_fn_contrastive},
    'val':         {'shuffle': False, 'drop_last': False, 'collate_fn': collate_fn_contrastive},
    'test':        {'shuffle': False, 'drop_last': False, 'collate_fn': collate_fn_contrastive},
    'inference':   {'shuffle': False, 'drop_last': False, 'collate_fn': collate_fn_inference},
}


def resolve_splits(cfg):
    """Return the {partition: [filenames]} split map for an unsupervised run.

    Two sources, in priority order:

    1. ``cfg['data']['splits_file']`` — an optional lightweight JSON listing video
       filenames per partition, e.g.
       ``{"train": [...], "val": [...], "test": [...], "inference": [...]}``.
       No per-frame labels. Only the keys present are used.
    2. Otherwise, auto-split every video found under ``cfg['data']['prefix']`` by
       ``cfg['data']['split_ratios']`` (train/val/test), seeded by ``cfg['seed']``
       for reproducibility. Auto-split produces no ``inference`` partition.

    Partitions that end up empty are omitted from the returned dict.
    """
    prefix = cfg['data']['prefix']
    splits_file = cfg['data'].get('splits_file')

    if splits_file is not None:
        with open(splits_file, 'r') as f:
            raw = json.load(f)
        splits = {p: list(raw[p]) for p in _PARTITION_SPECS if raw.get(p)}
        logger.info("Loaded splits from %s: %s",
                    splits_file, {p: len(v) for p, v in splits.items()})
        return splits

    # Auto-split the video folder.
    from feral.inference_folder import find_videos
    videos = find_videos(prefix)
    rng = random.Random(cfg['seed'])
    rng.shuffle(videos)

    ratios = cfg['data'].get('split_ratios', [0.9, 0.05, 0.05])
    n = len(videos)
    n_train = round(ratios[0] * n)
    n_val = round(ratios[1] * n)
    # Ensure at least one training video; val/test may be empty on tiny folders.
    n_train = max(1, min(n_train, n))
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    splits = {'train': videos[:n_train]}
    if n_val > 0:
        splits['val'] = videos[n_train:n_train + n_val]
    if n_test > 0:
        splits['test'] = videos[n_train + n_val:]
    if n_val == 0 and n_test == 0:
        logger.warning("Auto-split: only %d video(s) — all assigned to train, no val/test held out", n)
    logger.info("Auto-split %d videos (ratios %s): %s",
                n, ratios, {p: len(v) for p, v in splits.items()})
    return splits


def build_datasets_and_loaders(cfg, splits):
    """Build datasets and dataloaders for every partition present in ``splits``.

    - train/val/test -> ``ContrastiveVideoDataset`` (triplets; unsupervised loss).
    - inference -> ``ClsDataset`` (label-free chunk enumeration for embeddings).

    Returns (datasets, loaders): two dicts keyed by partition name. Partitions
    absent/empty in ``splits`` are simply absent from the dicts.
    """
    datasets = {}
    loaders = {}

    train_bs = cfg['training']['train_bs']
    val_bs = cfg['training']['val_bs']
    num_workers = resolve_num_workers(cfg['training']['num_workers'])
    logger.info("DataLoader num_workers=%d (config: %r)", num_workers, cfg['training']['num_workers'])
    persistent_workers = num_workers > 0

    # eval_chunk_shift overrides chunk_shift for the inference partition so
    # embedding extraction can run a denser overlap than training if desired.
    data_kwargs = dict(cfg['data'])
    eval_chunk_shift = data_kwargs.pop('eval_chunk_shift', None)
    data_kwargs.pop('splits_file', None)
    data_kwargs.pop('split_ratios', None)

    train_samples = cfg['training']['contrastive_num_samples']
    val_samples = cfg['training'].get('contrastive_val_num_samples', train_samples)

    for partition, spec in _PARTITION_SPECS.items():
        split = splits.get(partition)
        if not split:
            logger.info("No %s dataset", partition)
            continue

        part_kwargs = dict(data_kwargs)

        if partition == 'inference':
            if eval_chunk_shift is not None:
                logger.info("inference: eval_chunk_shift overrides chunk_shift %s -> %s",
                            part_kwargs.get('chunk_shift'), eval_chunk_shift)
                part_kwargs['chunk_shift'] = eval_chunk_shift
            dataset = ClsDataset(
                partition='inference',
                label_json_dict={'splits': {'inference': split}},
                do_aa=False,
                predict_per_item=cfg['predict_per_item'],
                num_classes=1,
                prefix=part_kwargs['prefix'],
                resize_to=part_kwargs['resize_to'],
                resize_style=part_kwargs.get('resize_style', 'square'),
                chunk_shift=part_kwargs['chunk_shift'],
                chunk_length=part_kwargs['chunk_length'],
                chunk_step=part_kwargs['chunk_step'],
                target_fps=part_kwargs.get('target_fps'),
            )
        else:
            dataset = ContrastiveVideoDataset(
                video_paths=split,
                num_samples=train_samples if partition == 'train' else val_samples,
                chunk_length=part_kwargs['chunk_length'],
                chunk_step=part_kwargs['chunk_step'],
                resize_to=part_kwargs['resize_to'],
                resize_style=part_kwargs.get('resize_style', 'square'),
                do_aa=part_kwargs.get('do_aa', True),
                vid2_max_shift=part_kwargs.get('vid2_max_shift', 8),
                prefix=part_kwargs['prefix'],
                seed=cfg['seed'],
                target_fps=part_kwargs.get('target_fps'),
            )

        loader = DataLoader(
            dataset,
            batch_size=train_bs if partition == 'train' else val_bs,
            shuffle=spec['shuffle'],
            drop_last=spec['drop_last'],
            collate_fn=spec['collate_fn'],
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,
        )

        datasets[partition] = dataset
        loaders[partition] = loader

    return datasets, loaders
