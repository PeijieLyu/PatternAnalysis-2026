# 2D Improved U-Net for HipMRI Prostate Segmentation

COMP3710 Pattern Recognition Final Project

- Model: 2D Improved U-Net
- Difficulty: Normal
- Dataset: HipMRI 2D
- Task: Prostate segmentation

## HipMRI 2D Dataset

`dataset.py` provides `HipMRI2DDataset`, a minimal PyTorch Dataset.

### Usage

Run from this project directory in an environment with PyTorch and NiBabel
installed. The example requires access to the dataset on Rangpur.

```python
from pathlib import Path
from dataset import HipMRI2DDataset

base = Path("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
dataset = HipMRI2DDataset(
    image_dir=base / "keras_slices_train",
    mask_dir=base / "keras_slices_seg_train",
)
image, mask = dataset[0]
```

### Pairing and outputs

- Images matching `case_*.nii.gz` are sorted by filename.
- Each image is paired with `seg_*.nii.gz` using the same filename suffix.
- Images are returned as `torch.float32` tensors with shape `[1, H, W]`.
- Masks are returned as `torch.long` tensors with shape `[H, W]`.
- MRI intensity values and valid mask labels are preserved without rescaling.
- Each slice may contain only a subset of labels 0 through 5.

### Input checks

The Dataset rejects missing directories, empty image sets, missing paired
masks, non-2D inputs, and image/mask shape mismatches. Mask values must be
finite integers in the range 0 through 5.

### Validation completed

- Found 11,460 training image/mask pairs on Rangpur.
- Loaded one real training sample: image `[1, 256, 128]`, mask `[256, 128]`.
- Temporary NIfTI tests passed for valid 2D inputs, 3D input rejection,
  and shape mismatch rejection.
- Temporary NIfTI tests passed for valid mask labels and rejection of
  fractional labels, negative labels, labels above 5, NaN, and Inf.

These checks do not constitute a full scan of all training image/mask contents.

Normalization, augmentation, one-hot encoding, class weighting, and training
logic are not implemented.

## Runtime dependencies and inspection tools

`requirements.txt` lists the direct third-party runtime dependencies:
PyTorch, NiBabel, and Matplotlib. Versions are not pinned, and the file does
not specify a CUDA build.

The existing Rangpur environment used for validation reported:
- PyTorch: `2.14.0+cu130`
- NiBabel: `5.4.2`
- Matplotlib: `3.11.2`

These are observed versions, not a separately validated clean installation.

### Run the inspection tools

On Rangpur, activate the existing environment and run from the directory
containing the project scripts:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
HIPMRI_DATA=/home/groups/comp3710/HipMRI_Study_open/keras_slices_data
```

Inspect the first batch:

```bash
python check_dataset.py \
  --image-dir "$HIPMRI_DATA/keras_slices_train" \
  --mask-dir "$HIPMRI_DATA/keras_slices_seg_train" \
  --batch-size 4
```

Check patient IDs across all three splits:

```bash
python check_splits.py \
  --train-dir "$HIPMRI_DATA/keras_slices_train" \
  --val-dir "$HIPMRI_DATA/keras_slices_validate" \
  --test-dir "$HIPMRI_DATA/keras_slices_test"
```

This script uses only the Python standard library and reads filenames,
not image contents. The verified split sizes were 31/4/3 patients and
11,460/660/540 slices, with no shared patient IDs between splits.

Save a side-by-side MRI/mask preview:

```bash
python visualize_sample.py \
  --image-dir "$HIPMRI_DATA/keras_slices_train" \
  --mask-dir "$HIPMRI_DATA/keras_slices_seg_train" \
  --index 0 \
  --output /tmp/hipmri_sample_preview.png
```

Choose a new output filename if the PNG already exists. The visualization
uses a headless backend and displays array orientation without anatomical
reorientation. Batch inspection and visualization read selected samples;
they do not validate every image in the dataset.
