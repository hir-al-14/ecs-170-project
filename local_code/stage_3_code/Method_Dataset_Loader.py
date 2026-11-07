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
     Nothing is ever written to disk; pickles are read into RAM once and cached
      per subclass, so building train+test construction does not parse the same file twice.
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
    _cache = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._cache = {}       # each subclass gets its own cache dict

    def __init__(self, split='train', transform=None):
        super().__init__()

        if split not in ('train', 'test'):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.split = split
        self.transform = transform

        fpath = self._find_file()
        self._load(fpath)                          # -> self.data, self.labels
        self.data = self._preprocess(self.data)    # -> (N, C, H, W) float32

    # locate file
    def _find_file(self):
        candidates = [
            os.path.join(self.dataset_source_folder_path, self.file_name),
            os.path.join(_HERE, 'data', 'stage_3_data', self.file_name),
            os.path.join(os.getcwd(), 'data', 'stage_3_data', self.file_name),
            os.path.join(os.getcwd(), self.file_name),
            os.path.join(_HERE, self.file_name),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(
            f"Could not find data file '{self.file_name}'. Tried:\n  "
            + "\n  ".join(candidates)
            + f"\n  CWD is: {os.getcwd()}"
        )

    # load
    def _load(self, fpath):
        """Populate self.data / self.labels for self.split only."""
        cache = type(self)._cache
        if fpath not in cache:
            with open(fpath, 'rb') as f:
                cache[fpath] = pickle.load(f)
        raw = cache[fpath]

        if not isinstance(raw, dict):
            raise TypeError(
                f"{fpath}: expected a dict with 'train'/'test' keys, "
                f"got {type(raw).__name__}"
            )
        for key in ('train', 'test'):
            if key not in raw:
                raise KeyError(f"{fpath}: no '{key}' key. Available keys: {list(raw.keys())}")

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
        raw_labels = np.asarray([int(it['label']) for it in items], dtype=np.int64)

        # Build ONE label mapping from train+test so both splits agree
        # (e.g. ORL labels may be 1..40; remap to 0..39).
        all_labels = np.asarray(
            [int(it['label']) for s in ('train', 'test') for it in raw[s]],
            dtype=np.int64,
        )
        classes = np.unique(all_labels)
        self.labels = np.searchsorted(classes, raw_labels).astype(np.int64)
        type(self).num_classes = int(len(classes))

    #  preprocess
    def _preprocess(self, arr):
        """Return (N, C, H, W) float32 scaled to [0, 1]."""
        arr = np.asarray(arr, dtype=np.float32)

        # (N, H, W) -> (N, H, W, 1)
        if arr.ndim == 3:
            arr = arr[..., np.newaxis]
        elif arr.ndim != 4:
            raise ValueError(f"Unexpected image array shape: {arr.shape}")

        # Datasets here are channel-last (N, H, W, C). Handle channel-first too.
        if arr.shape[-1] not in (1, 3) and arr.shape[1] in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))

        # ORL: gray-scale duplicated across R/G/B -> keep a single channel.
        if self.grayscale and arr.shape[-1] == 3:
            arr = arr[..., 0:1]

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


# 2. MNIST  handwritten digits, single-channel grayscale
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