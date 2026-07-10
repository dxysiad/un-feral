import torch
import logging

logger = logging.getLogger(__name__)
import random
import os


def get_random_run_name():
    """Generate a random 'size_adjective_animal' run name."""
    sizes = [
        "big", "huge", "giant", "massive", "jumbo", "colossal",
        "mega"
    ]

    adjectives = [
        "beautiful", "graceful", "fluid", "lively", "vibrant", "dynamic", "elegant",
        "spirited", "joyous", "expressive", "fiery", "playful",
        "uplifting", "magnetic", "mesmerizing", "soulful",
        "captivating", "hypnotic", "athletic",
        "sparkling", "sensational"]
    
    cool_animals = [
        "lemur", "platypus", "wombat", "armadillo", "capybara",
        "meerkat", "sloth", "pangolin", "koala", "okapi",
        "yak", "ibis", "cassowary", "toucan", "tapir",
        "gazelle", "lynx", "ocelot", "caracal", "manatee",
        "walrus", "narwhal", "aardvark", "marmot", "porcupine",
        "badger", "jackal", "civet", "quail", "peacock",
        "emu", "sea_otter", "red_panda", "mongoose", "alpaca",
        "reindeer", "ibex", "puffin", "heron", "kookaburra"
    ]
    return '_'.join([
        random.choice(sizes),
        random.choice(adjectives),
        random.choice(cool_animals)
    ])

def check_environment(compile_enabled):
    """Check that the current hardware and software meet FERAL's requirements.
    Call once at startup before any training work begins."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. FERAL requires an NVIDIA GPU with CUDA support."
        )

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    gpu_name = torch.cuda.get_device_name(device)

    # bfloat16 requires compute capability >= 8.0 (Ampere+)
    if capability < (8, 0):
        raise RuntimeError(
            f"GPU '{gpu_name}' has compute capability {capability[0]}.{capability[1]}, "
            f"but FERAL requires >= 8.0 (Ampere or newer) for bfloat16 support."
        )

    # flash attention requires compute capability >= 8.0 and PyTorch >= 2.0
    if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        raise RuntimeError(
            "PyTorch scaled_dot_product_attention is not available. "
            "FERAL requires PyTorch >= 2.0 for flash attention support."
        )
    # FERAL disables math and mem-efficient SDP backends, so flash attention must work.
    # The flash SDP backend requires SM 80+ which we already checked above,
    # but verify it isn't explicitly disabled.
    if hasattr(torch.backends.cuda, 'flash_sdp_enabled') and not torch.backends.cuda.flash_sdp_enabled():
        raise RuntimeError(
            "Flash attention SDP backend is disabled. "
            "FERAL requires flash attention (math and mem-efficient SDP are turned off)."
        )

    if compile_enabled:
        try:
            import triton  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "torch.compile is enabled but triton is not installed. "
                "Run: pip install -r requirements.txt"
            )

    logger.info(
        "Environment OK: GPU='%s', compute capability=%s.%s, bfloat16=yes, flash_attn=yes, compile=%s",
        gpu_name, capability[0], capability[1], "yes" if compile_enabled else "off",
    )


def suggested_num_workers():
    """Suggest a DataLoader worker count from available CPUs (affinity if
    available, else os.cpu_count()). May return None if neither is available."""
    max_num_worker_suggest = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if max_num_worker_suggest is None:
        # os.cpu_count() could return Optional[int]
        # get cpu count first and check None in order to satify mypy check
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            max_num_worker_suggest = cpu_count
    return max_num_worker_suggest


# Hard cap on auto-resolved DataLoader workers. Above ~16, the video-decode
# workers oversubscribe the CPU (decord is itself multithreaded) for little
# throughput gain. The cap also guards against hosts — e.g. shared cloud GPU
# boxes — that expose ALL host logical CPUs to the container, so affinity can
# report a wildly inflated count (a 200+ vCPU host serving a small instance).
MAX_AUTO_NUM_WORKERS = 16


def resolve_num_workers(value, cap=MAX_AUTO_NUM_WORKERS):
    """Resolve a config ``num_workers`` value to a concrete, validated int.

    Auto-detect is requested with ``-1`` (à la scikit-learn's ``n_jobs=-1``),
    which resolves to ``min(cap, CPUs available to this process)``. A
    non-negative int is honored verbatim (``0`` = load in the main process; or
    pin any positive count). This is the single normalization boundary — it
    always returns an int and raises ``ValueError`` on anything else, so no
    downstream code ever sees the sentinel.

    The CPU count comes from :func:`suggested_num_workers`
    (``os.sched_getaffinity``), which respects container cpuset limits — unlike
    ``os.cpu_count()``, which returns the host count.
    """
    if value == -1:
        n = suggested_num_workers() or 1
        return max(1, min(cap, n))
    # bool is an int subclass — reject True/False explicitly.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"num_workers must be -1 (auto) or a non-negative int, got {value!r}"
        )
    return value

def save_model(model, path, metadata):
    """Save the model's state_dict plus `metadata` to `path` via torch.save,
    unwrapping a torch.compile `_orig_mod` wrapper if present."""
    m = model
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    torch.save({
        'state_dict': m.state_dict(),
        **metadata,
    }, path)


