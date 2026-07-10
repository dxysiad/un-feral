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


def plot_umap_clusters(emb2d, labels, *, cmap_name="tab20", figsize=(8, 7),
                       point_size=6, title="UMAP + HDBSCAN behavior clusters",
                       label_names=None):
    """Scatter the 2-D UMAP embedding colored by integer label.

    emb2d  : (N, 2) UMAP coords (the ``emb2d`` returned by ``cluster_embeddings``).
    labels : (N,) label per chunk. ``-1`` = noise/outlier (drawn gray, underneath
             the real groups) — e.g. HDBSCAN cluster ids. Pass ground-truth class
             ids here instead to color the same UMAP by ground truth.
    label_names : optional {id: name} to label the legend (e.g. behavior names);
                  ids without a name fall back to ``group {id}``. Returns a Figure.
    """
    labels = np.asarray([int(l) for l in labels])
    emb2d = _to_numpy(emb2d)
    uniq = sorted(set(labels.tolist()))
    base = plt.get_cmap(cmap_name)

    def _name(c):
        if label_names is not None and c in label_names:
            return label_names[c]
        return f"group {c}"

    fig, ax = plt.subplots(figsize=figsize)
    if -1 in uniq:  # noise first so real groups sit on top
        m = labels == -1
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=point_size,
                   c=[(0.83, 0.83, 0.83, 1.0)], linewidths=0, label="noise")
    for j, c in enumerate(c for c in uniq if c != -1):
        m = labels == c
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=point_size,
                   color=base(j % base.N), linewidths=0, label=_name(c))

    n_groups = sum(1 for c in uniq if c != -1)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"{title} ({n_groups} groups)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=7, loc="best",
              ncol=1 if n_groups <= 12 else 2)
    fig.tight_layout()
    return fig


def cluster_agreement(labels, gt_labels):
    """External cluster-validity metrics vs. ground-truth labels.

    Both arrays are (N,) aligned per chunk. Returns a dict:
      ari / nmi / homogeneity / completeness / v_measure — standard clustering
        scores that handle a different number of clusters vs. classes;
      purity — fraction of chunks whose HDBSCAN cluster majority matches their
        ground-truth class (over-clustering friendly: many clusters -> one class);
      n_clusters / n_noise / n.
    Noise (-1) is kept as its own cluster for ari/nmi and counts against purity.
    """
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 homogeneity_completeness_v_measure)
    labels = np.asarray([int(l) for l in labels])
    gt = np.asarray([int(g) for g in gt_labels])
    assert labels.shape == gt.shape, "labels and gt_labels must align"
    homo, comp, vmeas = homogeneity_completeness_v_measure(gt, labels)
    purity_hits = 0
    for c in set(labels.tolist()):
        seg = gt[labels == c]
        _, cnts = np.unique(seg, return_counts=True)
        purity_hits += cnts.max()
    return {
        "ari": float(adjusted_rand_score(gt, labels)),
        "nmi": float(normalized_mutual_info_score(gt, labels)),
        "homogeneity": float(homo),
        "completeness": float(comp),
        "v_measure": float(vmeas),
        "purity": purity_hits / len(labels),
        "n_clusters": len(set(labels.tolist()) - {-1}),
        "n_noise": int((labels == -1).sum()),
        "n": int(len(labels)),
    }


def plot_contingency(labels, gt_labels, *, class_names=None, normalize="cluster",
                     cmap="viridis", figsize=None):
    """Heatmap of discovered clusters (rows) vs. ground-truth classes (cols).

    normalize: 'cluster' (row-normalized -> what each cluster maps to),
               'class' (col-normalized -> how each behavior is covered), or
               None (raw counts). ``class_names`` optionally maps class id -> name.
    Returns a matplotlib Figure.
    """
    labels = np.asarray([int(l) for l in labels])
    gt = np.asarray([int(g) for g in gt_labels])
    clusters = sorted(set(labels.tolist()))
    classes = sorted(set(gt.tolist()))
    M = np.zeros((len(clusters), len(classes)))
    for i, c in enumerate(clusters):
        for j, k in enumerate(classes):
            M[i, j] = np.sum((labels == c) & (gt == k))

    if normalize == "cluster":
        disp = M / M.sum(1, keepdims=True).clip(min=1)
    elif normalize == "class":
        disp = M / M.sum(0, keepdims=True).clip(min=1)
    else:
        disp = M

    fig, ax = plt.subplots(
        figsize=figsize or (1.3 * len(classes) + 2, 0.45 * len(clusters) + 2))
    im = ax.imshow(disp, aspect="auto", cmap=cmap, vmin=0,
                   vmax=disp.max() if disp.max() > 0 else 1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([str(class_names.get(k, k)) if class_names else str(k)
                        for k in classes], rotation=45, ha="right")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels(["noise" if c == -1 else f"cluster {c}" for c in clusters])
    ax.set_xlabel("ground-truth behavior")
    ax.set_ylabel("discovered cluster")
    thr = disp.max() * 0.6 if disp.max() > 0 else 0.5
    for i in range(len(clusters)):
        for j in range(len(classes)):
            txt = f"{disp[i, j]:.2f}" if normalize else str(int(M[i, j]))
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if disp[i, j] < thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    suffix = f" ({normalize}-normalized)" if normalize else " (counts)"
    ax.set_title("Cluster ↔ ground-truth contingency" + suffix)
    fig.tight_layout()
    return fig


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
