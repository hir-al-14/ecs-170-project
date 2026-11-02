"""
Method_Dataset_Loader.py

Stage 3 dataloaders for image classification (MNIST, ORL, CIFAR10).

Data format (confirmed via script_data_loader.py):
    Each dataset is ONE pickle file containing a dict:
        {
            'train': [ {'image': ndarray, 'label': int}, ... ],
            'test' : [ {'image': ndarray, 'label': int}, ... ],
        }

     MNIST : image shape (28, 28)      singlechannel grayscale
     ORL   : image shape (92, 112, 3)  R=G=B, collapsed to 1 channel here
     CIFAR : image shape (32, 32, 3)   full color, kept as 3 channels

Notes:
     Nothing is ever written to disk; pickles are read into RAM once.
     Parsed pickles are cached perfile so train+test construction
      does not parse the same file twice.
     Labels are remapped to contiguous ints starting at 0 so that
      CrossEntropyLoss always works, and num_classes is derived from
      the actual labels present.
     Output tensors are (C, H, W) float32 in [0, 1]  PyTorch CNN convention.
"""

import os
import pickle
import numpy as np
from torch.utils.data import Dataset

# Absolute directory containing this file; lets paths work from any CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))

# Project root is two levels up: stage_3_code -> local_code -> ecs-170-project
_DATA_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', 'data', 'stage_3_data'))

# 1. BASE CLASS
class Dataset_Loader(Dataset):
    """Shared logic for loading, preprocessing, and serving image samples."""

    # classlevel defaults: override these in each subclass
    dataset_source_folder_path = 'data/stage_3_data/'
    file_name = None          # single pickle holding both splits
    num_classes = None        # autocorrected from the data in _load()
    grayscale = False         # True > collapse 3 equal RGB channels to 1

    # Persubclass cache: {absolute_path: parsed_pickle_dict}
    # Declared on each subclass so caches don't collide across datasets.
    _cache = {}

    def __init__(self, split='train', transform=None):
        super().__init__()

        if split not in ('train', 'test'):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.split = split              # MUST be set before _load()
        self.transform = transform

        fpath = os.path.join(self.dataset_source_folder_path, self.file_name)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Data file not found: {fpath}\n"
                f"  CWD is: {os.getcwd()}\n"
                f"  Adjust dataset_source_folder_path to match your layout."
            )

        self._load(fpath)                       # > self.data, self.labels
        self.data = self._preprocess(self.data)  # > (N, C, H, W) float32

    #  load
    def _load(self, fpath):
        """Populate self.data / self.labels for self.split only."""

        # Cache the parsed pickle so train and test share one disk read.
        # Using type(self)._cache keeps each subclass's cache separate.
        cache = type(self)._cache
        if fpath not in cache:
            with open(fpath, 'rb') as f:
                cache[fpath] = pickle.load(f)
        raw = cache[fpath]

        # Fail with a clear message if the structure differs from expectation.
        if not isinstance(raw, dict):
            raise TypeError(
                f"{fpath}: expected a dict with 'train'/'test' keys, "
                f"got {type(raw).__name__}"
            )
        if self.split not in raw:
            raise KeyError(
                f"{fpath}: no '{self.split}' key. Available keys: {list(raw.keys())}"
            )

        items = raw[self.split]
        if not items:
            raise ValueError(f"{fpath}: split '{self.split}' is empty")

        sample = items[0]
        if not isinstance(sample, dict) or 'image' not in sample or 'label' not in sample:
            raise KeyError(
                f"{fpath}: items must be dicts with 'image' and 'label' keys. "
                f"First item looks like: {type(sample).__name__}"
            )

        self.data = np.stack([it['image'] for it in items], axis=0)
        self.labels = np.asarray([int(it['label']) for it in items], dtype=np.int64)

        # Remap labels to contiguous 0..K1 and derive num_classes from reality.
        unique = np.unique(self.labels)
        if unique.min() != 0 or not np.array_equal(unique, np.arange(len(unique))):
            mapping = {int(old): new for new, old in enumerate(unique)}
            self.labels = np.asarray([mapping[int(l)] for l in self.labels],
                                     dtype=np.int64)
        type(self).num_classes = int(len(unique))

    #  preprocess
    def _preprocess(self, arr):
        """Return (N, C, H, W) float32 scaled to [0, 1].

        Robust to channel-first / channel-last / missing-channel layouts:
        the channel axis is whichever trailing-or-leading dim is 1 or 3
        while the other two dims are spatial (>3).
        """
        arr = np.asarray(arr, dtype=np.float32)

        # Case A: (N, H, W) -- no channel axis at all
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=-1)  # -> (N, H, W, 1)
        elif arr.ndim != 4:
            raise ValueError(f"Unexpected image array shape: {arr.shape}")

        # Case B: locate the channel axis among the three trailing dims.
        # Spatial dims are >3; channel dims are 1 or 3.
        d1, d2, d3 = arr.shape[1], arr.shape[2], arr.shape[3]
        cand = [i for i, d in enumerate((d1, d2, d3), start=1) if d in (1, 3)]

        if len(cand) == 1:
            ch_axis = cand[0]  # unambiguous
        elif len(cand) >= 2:
            # Ambiguous (e.g. 1x3 collision) -- prefer an axis of size 3,
            # else the last axis, matching standard NHWC convention.
            threes = [i for i in cand if arr.shape[i] == 3]
            ch_axis = threes[-1] if threes else 3
        else:
            ch_axis = 3  # fall back to NHWC

        # Move detected channel axis to the end -> (N, H, W, C)
        if ch_axis != 3:
            order = [i for i in range(1, 4) if i != ch_axis] + [ch_axis]
            arr = np.transpose(arr, [0] + order)

        # ORL: gray-scale duplicated across R/G/B -> keep a single channel.
        if arr.shape[-1] == 3 and self.grayscale:
            arr = arr[:, :, :, 0:1]

        # NHWC -> NCHW (what nn.Conv2d expects)
        arr = np.transpose(arr, (0, 3, 1, 2))

        # Scale 0-255 -> 0-1 if needed.
        hi = float(arr.max()) if arr.size else 0.0
        if hi > 1.5:
            arr /= 255.0

        return np.ascontiguousarray(arr)

    #  Dataset protocol
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]                    # (C, H, W) float32
        y = int(self.labels[idx])
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __repr__(self):
        c = self.data.shape[1] if self.data is not None else '?'
        return (f"{type(self).__name__}(split={self.split!r}, n={len(self)}, "
                f"C={c}, HW={self.data.shape[2]}x{self.data.shape[3]}, "
                f"classes={type(self).num_classes})")


# 2. MNIST  handwritten digits, singlechannel grayscale
class Dataset_MNIST(Dataset_Loader):
    dataset_source_folder_path = _DATA_ROOT      # was 'data/stage_3_data/'
    file_name = 'MNIST'
    grayscale = False       # images arrive as (28, 28); _preprocess adds channel

# 3. ORL  faces stored as 3 equal RGB channels > collapse to 1
class Dataset_ORL(Dataset_Loader):
    dataset_source_folder_path = _DATA_ROOT
    file_name = 'ORL'
    grayscale = True

# 4. CIFAR10  colored objects, keep all 3 channels
class Dataset_CIFAR(Dataset_Loader):
    dataset_source_folder_path = _DATA_ROOT
    file_name = 'CIFAR'
    grayscale = False