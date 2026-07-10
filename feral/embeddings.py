"""Extract per-clip embeddings from a FeralModel over a folder of videos.

Embedding extraction is inference that taps ``FeralModel.forward_features``
instead of the classification head, so it reuses the folder-inference machinery
(chunk enumeration, collation) from ``inference_folder`` / ``dataset``. The
result is one feature vector per chunk, suitable for dimensionality reduction
(UMAP / t-SNE / PCA) and downstream unsupervised behavior analysis.

Works on any FeralModel — a pretrained backbone built via ``build_model`` or a
model loaded from a checkpoint — so it does not require a trained classifier.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from feral.dataset import ClsDataset, collate_fn_inference
from feral.inference_folder import find_videos, build_inference_labels_json


@torch.no_grad()
def extract_embeddings(model, loader, device, pool="mean", max_batches=None):
    """Tap ``model.forward_features`` over an unlabeled ``(data, names)`` loader.

    Returns ``(emb, ids)``:
      emb : (N, D) float tensor with ``pool='mean'`` (mean over the per-frame
            feature vectors -> one vector per chunk); (N, T, D) with ``pool='none'``.
      ids : list of ``(filename, start_frame_index)`` — one per chunk, in loader order.
    """
    model.eval()
    embs, ids = [], []
    for i, (data, names) in enumerate(tqdm(loader, total=len(loader))):
        data = data.to(device)
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            feats = model.forward_features(data)      # (B, T, D)
        if pool == "mean":
            feats = feats.mean(1)                     # (B, D)
        embs.append(feats.float().cpu())
        # names[b] is the per-frame list; names[b][0] == (fn, start_frame, 0)
        ids.extend((n[0][0], int(n[0][1])) for n in names)
        if max_batches is not None and i + 1 >= max_batches:
            break
    return torch.cat(embs), ids


def extract_embeddings_folder(model, cfg, video_folder, *, batch_size=8,
                              num_workers=4, pool="mean", save_path=None):
    """Build an inference chunk-loader over ``video_folder`` and extract embeddings.

    Reuses the folder-inference machinery; class metadata is irrelevant for the
    inference partition (chunks are enumerated without labels), so dummy values
    are passed. Optionally writes an ``.npz`` (emb, files, starts) to ``save_path``.
    """
    video_filenames = find_videos(video_folder)
    labels_json = build_inference_labels_json(video_filenames)
    dataset = ClsDataset(
        partition='inference', label_json_dict=labels_json, do_aa=False,
        predict_per_item=cfg['predict_per_item'], num_classes=1, prefix=video_folder,
        resize_to=cfg['data']['resize_to'], resize_style=cfg['data'].get('resize_style', 'square'),
        chunk_shift=cfg['data']['chunk_shift'], chunk_length=cfg['data']['chunk_length'],
        chunk_step=cfg['data']['chunk_step'],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn_inference)
    emb, ids = extract_embeddings(model, loader, device='cuda', pool=pool)
    if save_path is not None:
        files  = np.array([f for f, _ in ids])
        starts = np.array([s for _, s in ids])
        np.savez(save_path, emb=emb.numpy(), files=files, starts=starts)
    return emb, ids
