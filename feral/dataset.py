import torchvision
import os
from decord import VideoReader
import torch
from torchvision.transforms.v2 import TrivialAugmentWide
from torch.nn.functional import one_hot
import numpy as np
import cv2
import json
import traceback
import random
import logging
from feral.utils import get_class_frequencies

logger = logging.getLogger(__name__)

def read_range_video_decord(path, frames, width=-1, height=-1):
    """Decode the given ``frames`` indices from a video, resizing at decode time.

    Returns a uint8 tensor of shape (T, C, H, W).
    """
    vr = VideoReader(path, width=width, height=height)
    video = vr.get_batch(frames).asnumpy()  # (T, H, W, C)
    return torch.from_numpy(video).permute(0, 3, 1, 2)


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

class ClsDataset():
    def __init__(self, partition, label_json_dict, do_aa, predict_per_item,
                 num_classes, prefix, resize_to, chunk_shift, chunk_length,
                 chunk_step, resize_style="square", part_sample=1.0,
                 subsample_keep_rare_threshold=None, **kwargs):
        """Build the chunk samples/labels for a partition and set up transforms.

        Parses the label JSON into ``(filename, frame_ids)`` chunks, configures
        augmentation/normalization, and optionally subsamples the train set to
        ``part_sample`` of its chunks. When ``subsample_keep_rare_threshold`` is
        set, all chunks containing a rare class (class frequency below the
        threshold) are kept and the remainder is filled from common chunks.
        """
        self.prefix = prefix
        self.partition = partition # train / val / test / inference
        self.predict_per_item = predict_per_item
        self.num_classes = num_classes
        self.is_multilabel = None
        self.json_data = label_json_dict
        
        self.resize_to = resize_to
        self.resize_style = resize_style
        self.parse_json(chunk_shift, chunk_length, chunk_step)
        if do_aa and self.partition == "train":
            self.aug = TrivialAugmentWide()
        else:
            self.aug = None

        self.norm = torchvision.transforms.v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # vjepa
        self.scale = 0.00392156862745098

        if part_sample < 1.0 and partition == 'train':
            if subsample_keep_rare_threshold is None: # randomly samples chunks
                logger.info("%s using %.2f%% of chunks", partition, 100 * part_sample)
                new_len = round(part_sample * len(self.samples))
                new_indexes = random.sample(list(range(len(self.samples))), new_len)
                self.samples = [self.samples[i] for i in new_indexes]
                self.labels = [self.labels[i] for i in new_indexes]
            else: 
                # keeps all chunks with a rare class and fills rest with randomly chosen "common" chunks
                all_labels = np.array(self.labels)
                flat_labels = all_labels.reshape(-1) if not self.is_multilabel else all_labels.reshape(-1, all_labels.shape[-1])
                class_freqs = get_class_frequencies(flat_labels, num_classes=self.num_classes)
                rare_classes = np.where(class_freqs < subsample_keep_rare_threshold)[0]
                logger.info("Rare class indexes (<%.2f%%): %s. Class frequencies: %s",
                            subsample_keep_rare_threshold * 100, rare_classes.tolist(), class_freqs)
                
                # Vectorized: find which chunks contain any rare class
                if self.is_multilabel:
                    has_rare = all_labels[:, :, rare_classes].any(axis=(1, 2))
                else:
                    has_rare = np.isin(all_labels, rare_classes).any(axis=1)
                
                # Keep all rare, sample from common to reach target
                rare_idx = np.where(has_rare)[0]
                common_idx = np.where(~has_rare)[0]
                expected_total = round(part_sample * len(self.samples))
                assert len(rare_idx) <= expected_total, f"Found {len(rare_idx)} rare chunks, more than the expected total {expected_total}"
                
                sampled_common = np.random.choice(common_idx, expected_total - len(rare_idx), replace=False)
                final_idx = np.concatenate([rare_idx, sampled_common])
                np.random.shuffle(final_idx)
                logger.info("%s keeping %d rare + %d common = %d chunks",
                            partition, len(rare_idx), len(sampled_common), len(final_idx))
                self.samples = [self.samples[i] for i in final_idx]
                self.labels = [self.labels[i] for i in final_idx]

        if partition != 'inference':
            # sanity-checks dataset composition
            concat_labels = np.array(self.labels)
            if self.is_multilabel:
                cls_cnts = concat_labels.sum(axis=(0, 1))
            else:
                cls_cnts = np.bincount(concat_labels.flatten())
            logger.info("%s class counts: %s", partition, cls_cnts)


    def parse_json(self, chunk_shift, chunk_length, chunk_step):
        """Populate ``self.samples`` and ``self.labels`` from the label JSON.

        For each video in the partition, splits it into chunks of frame indices
        and stores ``(filename, frames)`` samples (plus per-frame labels for
        non-inference partitions). Also computes the shared decode size from the
        first video and sets ``self.is_multilabel``. Asserts the label frame
        count matches the video frame count for non-inference partitions.
        """
        self.samples = []
        self.labels = []
        self.decode_size = None

        for fn in self.json_data['splits'][self.partition]:
            pth = os.path.join(self.prefix, fn)
            video_total_frames = get_frame_count(pth)
            if self.decode_size is None:
                orig_w, orig_h = get_video_dims(pth)
                self.decode_size = compute_decode_size(orig_w, orig_h, self.resize_to, self.resize_style)
            if self.partition != 'inference':
                # checks that video frame count matches label frame count
                json_total_frames = len(self.json_data['labels'][fn])
                assert json_total_frames == video_total_frames, f"Bad json for video {fn}. Video has {video_total_frames} frames, labels have {json_total_frames} frames"
            frame_ids = get_frame_ids(video_total_frames, chunk_shift, chunk_length, chunk_step)
            for frames in frame_ids:
                self.samples.append((fn, frames))
                if self.partition != 'inference':
                    self.labels.append(
                        [self.json_data['labels'][fn][i] for i in frames]
                    )
        if self.partition != 'inference':
            self.is_multilabel = False if len(torch.tensor(self.labels[0]).shape) == 1 else True

    def proc_target(self, target):
        """Convert a chunk's labels to a float tensor, one-hot encoding single-label targets."""
        target = torch.tensor(target).long()
        if len(target.shape) == 1:
            target = one_hot(target, self.num_classes) # one-hot encoding: text to binary numerical arrays
        target = target.float() 
        return target

    def get_video(self, i):
        """Decode the i-th chunk's frames and return ``(video, names)``.

        ``video`` is the decoded (T, C, H, W) tensor; ``names`` is a list of
        ``(filename, frame_index_in_video, frame_index_in_chunk)`` tuples.
        """
        fn, frames = self.samples[i]
        pth = os.path.join(self.prefix, fn)
        w, h = self.decode_size
        # names are (filename as in labels.json, index of a frame within the video, index of a frame within a chunk)
        return read_range_video_decord(pth, frames, width=w, height=h), [(fn, frames[i], i) for i in range(len(frames))]

    def get_item_simple(self, index):
        """Load, augment, scale and normalize a chunk for the given index.

        Returns ``(video, label)`` for train, ``(video, label, names)`` for
        val/test, and ``(video, names)`` for inference.
        """
        video, names = self.get_video(index)
        video = video if self.aug is None else self.aug(video)
        outputs = self.norm(video * self.scale)
        if self.partition != "inference":
            label = self.labels[index]
            label = self.proc_target(label)
        if self.partition == 'train':
            return outputs, label
        elif self.partition == "val" or self.partition == 'test':
            return outputs, label, names
        else:
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
        - ``vid3``: an augmentation of a DIFFERENT chunk, drawn from the same
          or a different video (a contrastive / negative view).

    Unlike ``ClsDataset`` there are no labels. ``__init__`` pre-generates a
    fixed list of ``num_samples`` (primary_chunk, other_chunk) pairs
    (the "generate samples" step); ``__getitem__`` decodes and augments them.
    Call ``resample()`` between epochs to draw fresh pairs.
    """

    def __init__(self, video_paths, num_samples, chunk_length, chunk_step,
                 resize_to, resize_style="square", do_aa=True, prefix="",
                 seed=None, **kwargs):
        self.prefix = prefix
        self.video_paths = list(video_paths)
        self.chunk_length = chunk_length
        self.chunk_step = chunk_step
        self.resize_to = resize_to
        self.resize_style = resize_style
        self.num_samples = num_samples
        self.rng = random.Random(seed)

        self.aug = TrivialAugmentWide() if do_aa else None
        self.norm = torchvision.transforms.v2.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # vjepa
        self.scale = 0.00392156862745098

        # frames a chunk spans in real time (number of 'real' frames in a chunk): (L-1)*step + 1 
        self.span = (chunk_length - 1) * chunk_step + 1

        # Per-video frame counts + shared decode size (assumes uniform dims,
        # same assumption ClsDataset makes).
        self.frame_counts = {}
        self.decode_size = None
        self.usable_paths = []
        for fn in self.video_paths:
            pth = os.path.join(self.prefix, fn)
            total = get_frame_count(pth)
            if total is None or total < self.span:
                logger.warning("Skipping %s: %s frames, need >= %d", fn, total, self.span)
                continue
            self.frame_counts[fn] = total
            self.usable_paths.append(fn)
            if self.decode_size is None:
                w, h = get_video_dims(pth)
                self.decode_size = compute_decode_size(w, h, resize_to, resize_style)
        if not self.usable_paths:
            raise ValueError("No videos long enough to form a chunk")

        self.resample()

    def _random_chunk(self):
        """Pick a random (filename, frame_indices) chunk at a random start."""
        fn = self.rng.choice(self.usable_paths)
        start = self.rng.randint(0, self.frame_counts[fn] - self.span)
        frames = list(range(start, start + self.span, self.chunk_step))
        return fn, frames

    def _chunk_with_offset(self, fn, start, offset):
        """Return a chunk starting at start + offset"""
        new_start = start + offset
        max_start = self.frame_counts[fn] - self.span
        new_start = max(0, min(new_start, max_start))
        frames = list(range(new_start, new_start + self.span, self.chunk_step))
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
            prob = self.rng.random()
            if prob >= 0.5:
                offset = self.rng.randint(-16, 16) # TODO: test different ranges (start w/ 16) / doesn't shift half of the time
            else:
                offset = 0
            fn, frames = primary
            shifted = self._chunk_with_offset(fn, frames[0], offset)
            
            # reject any vid3 that overlaps with either vid1 or vid2
            other = self._random_chunk()
            max_tries = 20
            for _ in range(max_tries):
                if not self._chunks_overlap(other, primary) and \
                not self._chunks_overlap(other, shifted):
                    break
                # TODO: May need to sample from another video if impossible
                other = self._random_chunk()
            else:
                logger.warning("Could not find non-overlapping vid3 after %d tries", max_tries)
            
            self.samples.append((primary, shifted, other))

            """
            # ensure vid3 is a genuinely different chunk (diff video or start)
            while other[0] == primary[0] and other[1][0] == primary[1][0]:
                other = self._random_chunk()
            self.samples.append((primary, other))
            """

    def _transform(self, video):
        """Augment (optional), scale to [0,1], and normalize a (T,C,H,W) chunk."""
        if self.aug is not None:
            video = self.aug(video)
        return self.norm(video * self.scale)

    def _decode(self, chunk):
        fn, frames = chunk
        w, h = self.decode_size
        return read_range_video_decord(os.path.join(self.prefix, fn), frames, width=w, height=h)

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

def collate_fn_val(batch):
    """Collate ``(tensor, target, names)`` items, stacking tensors and targets."""
    tensors, targets, names = zip(*batch)
    tensors = torch.stack(tensors)
    targets = torch.stack(targets)
    return tensors, targets, names

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