"""Sanity check for Stage 3 dataloaders."""

from torch.utils.data import DataLoader
from Method_Dataset_Loader import Dataset_MNIST, Dataset_ORL, Dataset_CIFAR

for cls in (Dataset_MNIST, Dataset_ORL, Dataset_CIFAR):
    for split in ('train', 'test'):
        ds = cls(split=split)
        print(ds)
        x, y = ds[0]
        print(f"   x={tuple(x.shape)} dtype={x.dtype} "
              f"range=[{x.min():.3f},{x.max():.3f}] label={y}")
    dl = DataLoader(cls(split='train'), batch_size=8, shuffle=True)
    xb, yb = next(iter(dl))
    print(f"   batch: X={tuple(xb.shape)} Y={tuple(yb.shape)}\n")