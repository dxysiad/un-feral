"""
For embedding analysis in jupyter notebooks
"""

import numpy as np
from feral.behavior import cluster_embeddings, plot_umap_clusters, behavior_ethogram
import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def view_npz(filename):
    # Load the file with memory mapping to save RAM
    data = np.load(filename, mmap_mode='r')

    # List the arrays contained within the file
    print("Available arrays:", data.files)

    # Access a specific array and check its shape/type without loading all data
    array_name = data.files[0]
    print("Shape of array:", data[array_name].shape)
    print("Data type:", data[array_name].dtype)

    # View a small slice of the data (e.g., the first 5 items)
    print("First few rows:", data[array_name][:5])

def calc_clusters(embedding_npz):
    data = np.load(embedding_npz, allow_pickle=True)
    emb = data["emb"]                                              # (N, D) one vector per chunk
    ids = list(zip([str(f) for f in data["files"]], [int(s) for s in data["starts"]]))
    print("embeddings:", emb.shape, "| chunks:", len(ids))

    # UMAP-reduce -> HDBSCAN (B-SOID recipe). Raise min_cluster_size for coarser clusters.
    labels, emb2d = cluster_embeddings(emb, min_cluster_size=25, n_neighbors=15, seed=0, viz=True)
    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    print(f"{n_clusters} clusters | noise: {(labels == -1).mean():.1%}")
    return emb, emb2d, ids

# ground-truth label per chunk = majority behavior over that chunk's frames
def chunks_gt(gt_by_video, chunk, vocab, ids):
    gt_chunk, keep = [], []
    for fn, start in ids:
        ann = gt_by_video.get(fn)
        seg = ann[start:start + chunk] if ann is not None else np.array([], int)
        if len(seg) == 0:
            gt_chunk.append(-1); keep.append(False)
        else:
            vals, cnts = np.unique(seg, return_counts=True)
            gt_chunk.append(int(vals[cnts.argmax()])); keep.append(True)
    gt_chunk, keep = np.asarray(gt_chunk), np.asarray(keep)

    print(f"chunks with ground truth: {keep.sum()}/{len(ids)}")
    print("GT behavior counts:", {vocab[k]: int((gt_chunk[keep] == k).sum()) for k in sorted(vocab)})
    return gt_chunk, keep

def scatter_by_gt(coords, y, names, title, xlabel, ylabel, order=(3, 1, 2, 0)):
    """2-D scatter colored by GT behavior. `order` draws the dominant 'other'
    first so the rarer behaviors (attack/mount) sit on top and stay visible."""
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")
    for c in order:
        m = y == c
        ax.scatter(coords[m, 0], coords[m, 1], s=5, color=cmap(c),
                   linewidths=0, alpha=0.5, label=f"{names[c]} (n={m.sum()})")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=9, loc="best")
    fig.tight_layout()
    return fig

# Temporal Proximity Index (TPI) -- SUBTLE (Kwon et al., IJCV 2024), Eqs. 1-2.
#
#   TPI = sum_{i != j} w_ij * p_ij      over k-Means clusters of a 2-D embedding
#
#   p_ij = P(next chunk lands in cluster j | current chunk in cluster i),
#          row-normalized over j != i (self-transitions are excluded).
#   w_ij = softmax_j(1 / d_ij) over j != i, with d_ij = ||c_i - c_j||_2 between centroids.
#
# A good embedding puts clusters that behavior actually flows between close together,
# which makes TPI high. TPI == 1 is chance (transitions unrelated to distance) and
# k is the ceiling. At k = 2 every method scores exactly 2 by construction, so the
# methods only separate at larger k -- that is where to read the plot.


def next_chunk_pairs(ids, keep):
    """Row indices (into the kept subset) of temporally adjacent chunk pairs.

    ids: [(filename, start_frame)] per chunk, in `emb` order. Only pairs within
    the same video and one inference stride apart count as a transition.
    Returns (pairs (M, 2), stride).
    """
    row_of = {i: r for r, i in enumerate(np.flatnonzero(keep))}
    by_video = defaultdict(list)
    for i, (fn, start) in enumerate(ids):
        if i in row_of:
            by_video[fn].append((int(start), row_of[i]))
    for v in by_video.values():
        v.sort()
    deltas = [b[0] - a[0] for v in by_video.values() for a, b in zip(v, v[1:])]
    stride = int(np.bincount(deltas).argmax())
    pairs = np.asarray([(a[1], b[1]) for v in by_video.values()
                        for a, b in zip(v, v[1:]) if b[0] - a[0] == stride])
    return pairs, stride


def tpi(coords, pairs, k, *, seed=0):
    """TPI of a 2-D embedding at k k-Means clusters."""
    coords = np.asarray(coords, dtype=float)
    # softmax(1/d) is not scale-free, and t-SNE spans ~100 units where UMAP spans ~10,
    # so rescale both to unit spread before comparing them.
    coords = coords / coords.std()

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(coords)
    lab, centers = km.labels_, km.cluster_centers_

    T = np.zeros((k, k))
    np.add.at(T, (lab[pairs[:, 0]], lab[pairs[:, 1]]), 1.0)
    np.fill_diagonal(T, 0.0)               # Eq. 1 sums over i != j
    outgoing = T.sum(1)
    P = T / np.where(outgoing > 0, outgoing, 1.0)[:, None]

    with np.errstate(divide="ignore"):
        S = 1.0 / cdist(centers, centers)
    np.fill_diagonal(S, -np.inf)           # Eq. 2 normalizes over j != i
    W = np.exp(S - S.max(1, keepdims=True))
    W /= W.sum(1, keepdims=True)

    live = outgoing > 0                    # clusters nobody ever leaves would score 0
    return float(k * (W * P).sum(1)[live].mean())

def plot_tpi(emb2d, keep, pairs, X_tsne):
    # TPI vs number of clusters for the two embeddings already plotted above.
    # The shuffled control pairs random chunks instead of consecutive ones -- it marks
    # the chance level (TPI ~ 1) that a temporally meaningless embedding would score.

    KS = [2, 4, 8, 16, 32, 64, 128, 256]
    SEEDS = [0, 1, 2]

    rng = np.random.default_rng(0)
    n_kept = int(keep.sum())
    pairs_shuffled = rng.integers(0, n_kept, size=pairs.shape)

    views = {
        "UMAP":               (emb2d[keep], pairs),
        "t-SNE":              (X_tsne,      pairs),
        "UMAP (shuffled)":    (emb2d[keep], pairs_shuffled),
    }

    tpi_scores = {}
    for name, (coords, prs) in views.items():
        tpi_scores[name] = np.array([[tpi(coords, prs, k, seed=s) for s in SEEDS] for k in KS])
        print(name, "done")

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {"UMAP": ("tab:orange", "-"), "t-SNE": ("tab:blue", "-"),
            "UMAP (shuffled)": ("gray", "--")}
    for name, scores in tpi_scores.items():
        color, ls = styles[name]
        ax.errorbar(np.log2(KS), scores.mean(1), yerr=scores.std(1), marker="o",
                    capsize=3, color=color, linestyle=ls, label=name)
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("log2 k  (k-Means clusters)")
    ax.set_ylabel("TPI")
    ax.set_title("Temporal proximity index of the embedding space (higher = better)")
    ax.legend()
    fig.tight_layout()
    #fig.savefig(f"{IMAGES_DIR}/tpi_umap_vs_tsne.png", dpi=150, bbox_inches="tight")
    fig.show()

    print(f"\n{'k':>5} " + " ".join(f"{n:>20}" for n in tpi_scores))
    for i, k in enumerate(KS):
        row = " ".join(f"{tpi_scores[n][i].mean():>14.3f} +-{tpi_scores[n][i].std():5.3f}"
                    for n in tpi_scores)
        print(f"{k:>5} {row}")