"""Minimal NIfTI file pairing and PyTorch Dataset for HipMRI 2D slices."""

from pathlib import Path

import nibabel as nib
import torch
from torch.utils.data import Dataset


class HipMRI2DDataset(Dataset):
    """Pair case_*.nii.gz images with seg_*.nii.gz masks by filename."""

    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.pairs = []
        for image_path in sorted(self.image_dir.glob("case_*.nii.gz")):
            mask_name = "seg_" + image_path.name[len("case_"):]
            mask_path = self.mask_dir / mask_name
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            self.pairs.append((image_path, mask_path))

        if not self.pairs:
            raise ValueError(f"No case_*.nii.gz images found in {self.image_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = torch.tensor(nib.load(str(image_path)).get_fdata(), dtype=torch.float32)
        mask = torch.tensor(nib.load(str(mask_path)).get_fdata(), dtype=torch.long)
        return image.unsqueeze(0), mask
