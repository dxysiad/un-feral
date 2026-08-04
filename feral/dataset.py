import torchvision
import torchvision.transforms.v2
import os
from decord import VideoReader
import torch
import albumentations as A
import numpy as np
import cv2
import traceback
import random
import logging

logger = logging.getLogger(__name__)

def read_range_video_decord_numpy(path, frames, width=-1, height=-1):
    """Decode the given ``frames`` indices from a video, resizing at decode time.

    Returns a uint8 array of shape (T, H, W, C) — albumentations' native layout.
    """
    vr = VideoReader(path, width=width, height=height)
    return vr.get_batch(frames).asnumpy()


def read_range_video_decord(path, frames, width=-1, height=-1):
    """Decode the given ``frames`` indices from a video, resizing at decode time.

    Returns a uint8 tensor of shape (T, C, H, W).
    """
    video = read_range_video_decord_numpy(path, frames, width=width, height=height)
    return torch.from_numpy(video).permute(0, 3, 1, 2)


def build_contrastive_aug(out_h, out_w):
    return A.Compose([
        A.Affine(rotate=(-180, 180), border_mode=cv2.BORDER_REPLICATE, p=0.95), 
        A.RandomResizedCrop(size=(out_h, out_w), scale=(0.7, 1.0), ratio=(0.9, 1.11), p=0.75),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.75),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(std_range=(0.1, 0.2), per_channel=False, p=0.2),
        A.ToGray(p=0.2)
    ])


def compute_decode_size(orig_w, orig_h, resize_to, resize_style):
    """Target (width, height) for decord decode-time resize, matching
    torchvision `build_resize_transform` output size."""
    if resize_style == "square":
        return resize_to, resize_to
    if resize_style == "rectangle":
        # Match torchvision Resize(int): shorter side -> resize_to, preserve AR.
        if orig_h <= orig_w:
            return round(orig_w * resize_to / orig_h), resize_to
        return resize_to, round(orig_h * resize_to / orig_w)
    raise ValueError(f"resize_style must be 'square' or 'rectangle', got {resize_style!r}")

def get_frame_ids(total_frames, chunk_shift, chunk_length, chunk_step):
        """Split a video of ``total_frames`` into overlapping fixed-size chunks.

        Returns a list of chunks, each a list of ``chunk_length`` frame indices.

        - ``chunk_length``: frames per chunk (the model's temporal window).
        - ``chunk_step``: stride *within* a chunk — pick every Nth frame, so a
          chunk spans ``(chunk_length - 1) * chunk_step + 1`` real frames:
              chunk_step = 1 -- pick every frame        XXXX
              chunk_step = 2 -- pick every other frame  X_X_X_X
              chunk_step = 3 -- pick every third        X__X__X__X
        - ``chunk_shift``: stride *between* consecutive chunks (how far the
          window advances each step). Overlap fraction =
          ``1 - chunk_shift / chunk_length`` (chunk_length 64, chunk_shift 32 ->
          50% overlap; chunk_shift 16 -> 75%).

        A trailing partial window that can't fill ``chunk_length`` is dropped.
        """
        vid_frames = []
        start_ind = 0

        while True:
            last_ind = start_ind + (chunk_length - 1) * chunk_step  + 1
            inds = list(range(start_ind, min(last_ind, total_frames), chunk_step))
            if len(inds) != chunk_length:
                break
            vid_frames.append(inds)
            start_ind = inds[0] + chunk_shift
        return vid_frames

def build_resize_transform(resize_to, resize_style):
    """Construct the torchvision Resize transform for a given `resize_style`.

    - "square":    squish videos to ``(resize_to, resize_to)`` regardless of input aspect ratio.
    - "rectangle": resize so the smallest side becomes ``resize_to``, preserving aspect ratio.
    """
    if resize_style == "square":
        return torchvision.transforms.v2.Resize((resize_to, resize_to), antialias=True)
    if resize_style == "rectangle":
        return torchvision.transforms.v2.Resize(resize_to, antialias=True)
    raise ValueError(f"resize_style must be 'square' or 'rectangle', got {resize_style!r}")


def get_frame_count(path: str):
    """Return the video's frame count via OpenCV, or None if it can't be read."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else None
    finally:
        cap.release()


def get_video_dims(path: str):
    """Return the video's ``(width, height)`` in pixels via OpenCV."""
    cap = cv2.VideoCapture(path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def get_video_fps(path: str):
    """Return the video's frame rate via OpenCV, or None if it can't be read."""
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else None
    finally:
        cap.release()


def fps_decimation_factor(source_fps, target_fps):
    """Frame-stride multiplier that subsamples ``source_fps`` down to ~``target_fps``.

    Returns 1 when ``target_fps`` is None (no fps normalization) or when the
    source rate is unknown/already at-or-below target. Chunks formed with this
    factor cover the same real-time window regardless of the source frame rate.
    We can only decimate, never interpolate, so the factor is clamped to >= 1
    (a source slower than ``target_fps`` is used as-is).
    """
    if target_fps is None:
        return 1
    if source_fps is None or source_fps <= 0:
        logger.warning("Unknown source fps; skipping fps normalization (factor=1)")
        return 1
    return max(1, round(source_fps / target_fps))

class ClsDataset():
    """Label-free chunk enumerator over a set of videos (inference partition only).

    Splits each video into overlapping fixed-size chunks and yields
    ``(video_tensor, names)`` per chunk — the input for embedding extraction.
    """

    def __init__(self, partition, label_json_dict, do_aa, predict_per_item,
                 num_classes, prefix, resize_to, chunk_shift, chunk_length,
                 chunk_step, resize_style="square", target_fps=None, **kwargs):
        """Build the chunk samples for the inference partition and set up transforms.

        ``label_json_dict`` only needs a ``splits`` dict listing filenames per
        partition (no per-frame labels). ``num_classes``/``predict_per_item`` are
        accepted for signature compatibility but unused. Augmentation is never
        applied (inference only).
        """
        assert partition == 'inference', (
            f"ClsDataset only supports the 'inference' partition now, got {partition!r}. "
            "Train/val/test use ContrastiveVideoDataset."
        )
        self.prefix = prefix
        self.partition = partition
        self.predict_per_item = predict_per_item
        self.num_classes = num_classes
        self.json_data = label_json_dict

        self.resize_to = resize_to
        self.resize_style = resize_style
        self.target_fps = target_fps
        self.parse_json(chunk_shift, chunk_length, chunk_step)
        self.aug = None  # no augmentation at inference

        self.norm = torchvision.transforms.v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # vjepa
        self.scale = 0.00392156862745098

    def parse_json(self, chunk_shift, chunk_length, chunk_step):
        """Populate ``self.samples`` with ``(filename, frames)`` chunks.

        For each video in the inference split, splits it into chunks of frame
        indices. Also computes the shared decode size from the first video.
        """
        self.samples = []
        self.decode_size = None

        for fn in self.json_data['splits'][self.partition]:
            pth = os.path.join(self.prefix, fn)
            video_total_frames = get_frame_count(pth)
            if self.decode_size is None:
                orig_w, orig_h = get_video_dims(pth)
                self.decode_size = compute_decode_size(orig_w, orig_h, self.resize_to, self.resize_style)
            # Decimate to ~target_fps by widening the within/between-chunk strides
            # (chunk_length is unchanged, so the model still sees that many frames).
            d = fps_decimation_factor(get_video_fps(pth), self.target_fps)
            frame_ids = get_frame_ids(video_total_frames, chunk_shift * d, chunk_length, chunk_step * d)
            for frames in frame_ids:
                self.samples.append((fn, frames))

    def get_video(self, i):
        """Decode the i-th chunk's frames and return ``(video, names)``.

        ``video`` is the decoded (T, C, H, W) tensor; ``names`` is a list of
        ``(filename, frame_index_in_video, frame_index_in_chunk)`` tuples.
        """
        fn, frames = self.samples[i]
        pth = os.path.join(self.prefix, fn)
        w, h = self.decode_size
        # names are (filename, index of a frame within the video, index of a frame within a chunk)
        return read_range_video_decord(pth, frames, width=w, height=h), [(fn, frames[i], i) for i in range(len(frames))]

    def get_item_simple(self, index):
        """Load, scale and normalize a chunk; return ``(video, names)``."""
        video, names = self.get_video(index)
        outputs = self.norm(video * self.scale)
        return outputs, names

    def __getitem__(self, index):
        """Return the chunk at ``index``, retrying up to 3 random indices on failure."""
        try:
            return self.get_item_simple(index)
        except Exception:
            logger.warning("Error loading index %d:\n%s", index, traceback.format_exc())
            for _ in range(3):
                alt_index = np.random.randint(0, len(self))
                try:
                    return self.get_item_simple(alt_index)
                except Exception:
                    logger.warning("Error loading index %d:\n%s", alt_index, traceback.format_exc())
            raise RuntimeError(f"Failed to load sample after multiple retries.\nLast error:\n{traceback.format_exc()}")

    
    def __len__(self):
        """Number of chunks in the dataset."""
        return len(self.samples)


class ContrastiveVideoDataset():
    """Unsupervised triplet dataset for contrastive / embedding pretraining.

    Each item is a dict of three augmented 64-frame chunks:

        - ``vid1`` and ``vid2``: two *independent* augmentations of the SAME
          chunk (same video, same frames -> a positive pair).
        - ``vid3``: an augmentation of a DIFFERENT, non-overlapping chunk (a
          contrastive / negative view). It is drawn from the anchor's OWN video
          with probability ``neg_same_video_frac`` (so background is constant
          within the triplet and can't be a shortcut), otherwise from any video.

    Unlike ``ClsDataset`` there are no labels. ``__init__`` pre-generates a
    fixed list of ``num_samples`` (primary_chunk, other_chunk) pairs
    (the "generate samples" step); ``__getitem__`` decodes and augments them.
    Call ``resample()`` between epochs to draw fresh pairs.
    """

    def __init__(self, video_paths, num_samples, chunk_length, chunk_step,
                 resize_to, resize_style="square", do_aa=True, prefix="",
                 seed=None, vid2_max_shift=8, target_fps=None,
                 neg_same_video_frac=0.0, **kwargs):
        self.prefix = prefix
        self.video_paths = list(video_paths)
        self.chunk_length = chunk_length
        self.chunk_step = chunk_step
        self.resize_to = resize_to
        self.resize_style = resize_style
        self.num_samples = num_samples
        self.vid2_max_shift = vid2_max_shift
        self.target_fps = target_fps
        self.neg_same_video_frac = neg_same_video_frac
        self.rng = random.Random(seed)

        self.norm = torchvision.transforms.v2.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # vjepa
        self.scale = 0.00392156862745098

        # Per-video frame counts, fps-decimation factor + shared decode size
        # (decode size assumes uniform dims, same assumption ClsDataset makes).
        # With target_fps set, each video is subsampled to ~target_fps so a
        # chunk covers the same real-time window regardless of source fps; the
        # within-chunk stride (and thus real-time span) then varies per video.
        self.fps_factor = {}
        self.frame_counts = {}
        self.decode_size = None
        self.usable_paths = []
        for fn in self.video_paths:
            pth = os.path.join(self.prefix, fn)
            total = get_frame_count(pth)
            d = fps_decimation_factor(get_video_fps(pth), self.target_fps)
            span = (chunk_length - 1) * chunk_step * d + 1
            if total is None or total < span:
                logger.warning("Skipping %s: %s frames, need >= %d", fn, total, span)
                continue
            self.fps_factor[fn] = d
            self.frame_counts[fn] = total
            self.usable_paths.append(fn)
            if self.decode_size is None:
                w, h = get_video_dims(pth)
                self.decode_size = compute_decode_size(w, h, resize_to, resize_style)
        if not self.usable_paths:
            raise ValueError("No videos long enough to form a chunk")
        
        # Built here, not above: the crop needs decode_size so augmented chunks
        # keep the shape ClsDataset produces at inference.
        dec_w, dec_h = self.decode_size
        self.aug = build_contrastive_aug(dec_h, dec_w) if do_aa else None

        self.resample()

    def _step_span(self, fn):
        """Per-video (within-chunk stride, real frames spanned) after fps decimation."""
        step = self.chunk_step * self.fps_factor[fn]
        span = (self.chunk_length - 1) * step + 1
        return step, span

    def _random_chunk_from(self, fn):
        """Pick a random (filename, frame_indices) chunk at a random start within `fn`."""
        step, span = self._step_span(fn)
        start = self.rng.randint(0, self.frame_counts[fn] - span)
        frames = list(range(start, start + span, step))
        return fn, frames

    def _random_chunk(self):
        """Pick a random chunk from a randomly chosen video."""
        return self._random_chunk_from(self.rng.choice(self.usable_paths))

    def _chunk_with_offset(self, fn, start, offset):
        """Return a chunk starting at start + offset"""
        step, span = self._step_span(fn)
        new_start = start + offset
        max_start = self.frame_counts[fn] - span
        new_start = max(0, min(new_start, max_start))
        frames = list(range(new_start, new_start + span, step))
        return fn, frames
    
    def _chunks_overlap(self, chunk_a, chunk_b):
        """Return True if two chunks share any frame indices"""
        fn_a, frames_a = chunk_a
        fn_b, frames_b = chunk_b
        if fn_a != fn_b:
            return False # No overlap
        return len(set(frames_a) & set(frames_b)) > 0

    def resample(self):
        """(Re)generate the fixed list of (primary, other) chunk pairs."""
        self.samples = []
        for _ in range(self.num_samples):
            primary = self._random_chunk()

            # precompute shifted chunk to check vid3 against both
            fn, frames = primary
            prob = self.rng.random()
            if prob >= 0.5:
                # jitter is in target-fps frames -> scale to source frames
                offset = self.rng.randint(-self.vid2_max_shift, self.vid2_max_shift) * self.fps_factor[fn]
            else:
                offset = 0
            shifted = self._chunk_with_offset(fn, frames[0], offset)

            # Draw the negative from the SAME video as the anchor a fraction of the
            # time. When it shares the anchor's background, "same environment?" can no
            # longer separate positive from negative, so the model must key on
            # motion/behavior instead -- this is what stops the embedding clustering
            # by recording environment. The rest of the time keep the corpus-wide draw.
            same_video = self.rng.random() < self.neg_same_video_frac
            draw_neg = (lambda: self._random_chunk_from(fn)) if same_video else self._random_chunk

            # reject any vid3 that overlaps with either vid1 or vid2
            other = draw_neg()
            max_tries = 20
            for _ in range(max_tries):
                if not self._chunks_overlap(other, primary) and not self._chunks_overlap(other, shifted):
                    break
                other = draw_neg()
            else:
                # A short anchor video may have no non-overlapping chunk; fall back to
                # a corpus-wide draw rather than emit a degenerate (overlapping) negative.
                if same_video:
                    other = self._random_chunk()
                else:
                    logger.warning("Could not find non-overlapping vid3 after %d tries", max_tries)

            self.samples.append((primary, shifted, other))

    def _transform(self, video):
        """Augment (optional) a (T,H,W,C) uint8 chunk, then scale/normalize to (T,C,H,W)."""
        if self.aug is not None:
            video = self.aug(images=video)['images']
        video = torch.from_numpy(np.ascontiguousarray(video)).permute(0, 3, 1, 2)
        return self.norm(video * self.scale)

    def _decode(self, chunk):
        """Decode a chunk to a (T,H,W,C) uint8 array — the layout `_transform` augments in."""
        fn, frames = chunk
        w, h = self.decode_size
        return read_range_video_decord_numpy(os.path.join(self.prefix, fn), frames, width=w, height=h)

    def get_item_simple(self, index):
        primary, shifted, other = self.samples[index]

        vid_a = self._decode(primary)   # decoded once...
        vid_b = self._decode(shifted) # same as vid_a, shifted start
        vid_c = self._decode(other)

        return {
            # vid1 & vid2: two independent draws of the augmentation on the
            # SAME source frames (self.aug samples fresh randomness per call).
            "vid1": self._transform(vid_a),
            "vid2": self._transform(vid_b),
            "vid3": self._transform(vid_c),
        }

    def __getitem__(self, index):
        """Return the triplet at ``index``, retrying up to 3 random indices on failure."""
        try:
            return self.get_item_simple(index)
        except Exception:
            logger.warning("Error loading index %d:\n%s", index, traceback.format_exc())
            for _ in range(3):
                alt_index = np.random.randint(0, len(self))
                try:
                    return self.get_item_simple(alt_index)
                except Exception:
                    logger.warning("Error loading index %d:\n%s", alt_index, traceback.format_exc())
            raise RuntimeError(f"Failed to load sample after multiple retries.\nLast error:\n{traceback.format_exc()}")

    def __len__(self):
        return len(self.samples)

def collate_fn_inference(batch):
    """Collate ``(tensor, names)`` items, stacking tensors into a batch."""
    tensors, names = zip(*batch)
    tensors = torch.stack(tensors)
    return tensors, names

def collate_fn_contrastive(batch):
    """Collate triplet dicts into stacked tensors."""
    return {
        'vid1': torch.stack([b['vid1'] for b in batch]),
        'vid2': torch.stack([b['vid2'] for b in batch]),
        'vid3': torch.stack([b['vid3'] for b in batch]),
    }