"""Unsupervised behavior discovery from clip embeddings.

Turns the per-clip embeddings produced by ``feral.embeddings`` into discrete
behavior clusters (the B-SOID / MotionMapper recipe: UMAP-reduce -> HDBSCAN) and
renders a per-video ethogram. Consumes ``(emb, ids)`` from
``extract_embeddings_folder`` and needs no labels — this is the unsupervised end
of the pipeline.
"""
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


def _to_numpy(emb):
    if hasattr(emb, "detach"):        # torch tensor
        return emb.detach().cpu().numpy()
    return np.asarray(emb)


def cluster_embeddings(emb, *, cluster_dim=10, min_cluster_size=25,
                       n_neighbors=15, min_dist=0.0, viz=True, seed=0):
    """UMAP-reduce then HDBSCAN-cluster clip embeddings (the B-SOID recipe).

    emb: (N, D) tensor/array. Returns ``(labels, emb2d)``:
      labels : (N,) int cluster id per chunk; -1 = noise/outlier (HDBSCAN).
      emb2d  : (N, 2) UMAP for plotting (None if ``viz=False``).
    """
    import umap
    from sklearn.cluster import HDBSCAN

    X = _to_numpy(emb)
    X_low = umap.UMAP(n_components=cluster_dim, n_neighbors=n_neighbors,
                      min_dist=min_dist, random_state=seed).fit_transform(X)
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X_low)
    emb2d = (umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
                       random_state=seed).fit_transform(X) if viz else None)
    return labels, emb2d


def behavior_ethogram(ids, labels, *, cmap_name="tab20", figsize=None):
    """Per-chunk ethogram: one row per video, chunks in temporal order, colored
    by cluster id (``-1`` noise shown gray). Returns a matplotlib Figure.

    ids   : list of (filename, start_frame) per chunk (from extract_embeddings).
    labels: (N,) cluster id per chunk, aligned with ``ids``.
    """
    labels = np.asarray([int(l) for l in labels])
    rows = defaultdict(list)
    for (fn, start), lab in zip(ids, labels):
        rows[fn].append((int(start), int(lab)))
    videos = sorted(rows)

    # Map cluster ids -> contiguous colormap indices; -1 (noise) -> light gray.
    uniq = sorted(set(labels.tolist()))
    base = plt.get_cmap(cmap_name)
    colors, id_to_idx = [], {}
    if -1 in uniq:
        id_to_idx[-1] = 0
        colors.append((0.83, 0.83, 0.83, 1.0))
    for j, c in enumerate(c for c in uniq if c != -1):
        id_to_idx[c] = len(colors)
        colors.append(base(j % base.N))
    cmap = ListedColormap(colors)
    K = len(colors)

    max_chunks = max(len(v) for v in rows.values())
    fig, ax = plt.subplots(figsize=figsize or (12, 0.5 * len(videos) + 1.5))
    for i, fn in enumerate(videos):
        seq = [id_to_idx[lab] for _, lab in sorted(rows[fn])]
        ax.imshow(np.array(seq)[None, :], aspect="auto",
                  extent=[0, len(seq), i, i + 1], cmap=cmap,
                  vmin=0, vmax=max(K - 1, 1), interpolation="nearest")
    ax.set_ylim(0, len(videos))
    ax.set_xlim(0, max_chunks)
    ax.set_yticks([i + 0.5 for i in range(len(videos))])
    ax.set_yticklabels(videos, fontsize=8)
    ax.set_xlabel("chunk index (time →)")
    #ax.set_title("Behavior ethogram (per-chunk cluster)")
    ax.set_title(" ")
    handles = [mpatches.Patch(color=colors[id_to_idx[c]],
                              label=("noise" if c == -1 else f"cluster {c}"))
               for c in uniq]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7)
    fig.tight_layout()
    return fig
