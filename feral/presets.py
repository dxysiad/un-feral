"""Named training/inference presets ("modes") overlaid on default_config.yaml.

A preset is a *sparse* overlay — it only names the keys it changes, and is
deep-merged onto the packaged ``default_config.yaml`` so the base recipe stays
the single source of truth. Exposed on the CLI via ``feral train --mode`` and
``feral infer --mode``.

Modes
-----
lite : smallest V-JEPA 2.1 (ViT-B/384), full fine-tune with 50% chunk overlap.
       Cheapest to train and run.
max  : same backbone as ``default`` (no override). Trains at 66% temporal
       overlap (``chunk_shift`` 21) but extracts embeddings at a denser 80%
       overlap (``eval_chunk_shift`` 12).
rare : lite backbone with the grad-norm clip stabilization knob turned on.
"""

# Sparse overlays, deep-merged onto default_config.yaml. Each preset names ONLY
# the keys that differ from default_config.yaml; everything else is inherited.
PRESETS = {
    # ── lite ── full fine-tune, smallest backbone ─────────────────────────────
    "lite": {
        "backbone": "vjepa2_1_vitb_384",   # smallest V-JEPA 2.1 (384-native; fed at default 256)
        "model": {
            "freeze_encoder_layers": 0,    # full fine-tune
        },
        # resize_to inherits default_config (256) — 384-native backbone runs at
        # 256 via interpolated position embeddings; ~2.25x fewer tokens.
    },

    # ── max ── default backbone + 66% train / 80% inference overlap ────────────
    "max": {
        "data": {
            "chunk_shift": 21,             # 66% TRAIN overlap = chunk_length / 3 (default is 50%)
            "eval_chunk_shift": 12,        # 80% INFERENCE overlap = chunk_length / 5 (denser embeddings)
        },
        # backbone, freeze_encoder_layers, resize_to all inherit default_config.
    },

    # ── rare ── grad-clip stabilization on the lite backbone ───────────────────
    "rare": {
        "backbone": "vjepa2_1_vitb_384",
        "model": {
            "freeze_encoder_layers": 0,
        },
        "training": {
            "grad_clip_norm": 1.0,         # stabilize grad spikes
        },
    },
}

# One-line descriptions for CLI --help / logging.
MODE_HELP = {
    "lite": "smallest V-JEPA 2.1 (ViT-B/384), full fine-tune (cheapest)",
    "max":  "default backbone + 66% train / 80% inference overlap",
    "rare": "lite backbone + grad-clip stabilization",
}


def _deep_merge(base, overlay):
    """Recursively merge ``overlay`` onto ``base``, returning a new dict.

    Nested dicts are merged key-by-key; any non-dict value (including ``None``)
    in ``overlay`` replaces the value in ``base``.
    """
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def apply_mode(cfg, mode):
    """Return ``cfg`` deep-merged with preset ``mode``. Unknown mode -> ValueError."""
    if mode is None:
        return cfg
    if mode not in PRESETS:
        raise ValueError(f"Unknown mode {mode!r}. Choices: {sorted(PRESETS)}")
    return _deep_merge(cfg, PRESETS[mode])


def infer_chunk_shift(mode, chunk_length):
    """Chunk shift (stride) for inference-time overlap under a given mode.

    lite -> 50% overlap (chunk_length / 2); max -> 80% overlap (chunk_length / 5),
    matching ``max``'s ``eval_chunk_shift``. Returns None for modes that don't
    change inference chunking (e.g. ``rare``).
    """
    if mode == "lite":
        return max(1, chunk_length // 2)
    if mode == "max":
        return max(1, chunk_length // 5)
    return None
