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
masks, non-2D inputs, and image/mask shape mismatches.
MRI values must be finite. Mask values must be finite integers in the
range 0 through 5.

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

## Baseline UNet2D

`modules.py` provides a four-level baseline U-Net built from `DoubleConv`,
`Down`, and `Up` blocks.

- Defaults: 1 input channel, 6 output classes, base channel width 16.
- Default parameter count: 1,962,422.
- Input: `[B, 1, H, W]`, with height and width each at least 16.
- Output: raw logits `[B, 6, H, W]`.
- Downsampling uses max pooling; upsampling uses bilinear interpolation
  to match skip feature dimensions before channel concatenation.
- The output layer is a 1x1 convolution without softmax.

### CPU model smoke checks

Run from the project directory with the Python environment activated:

```bash
python check_model.py
python check_model.py --height 31 --width 17 --base-channels 4 --backward
```

The script uses synthetic inputs and checks output shape and finite values.
With `--backward`, it also checks input and parameter gradients.
It does not update parameters or evaluate segmentation accuracy.

Both commands passed on Rangpur. A real HipMRI sample also produced finite
logits with shape `[1, 6, 256, 128]` in a separate integration check.

The baseline is untrained. GPU compatibility remains unverified; CPU
backward checks on login0 reported a CUDA driver compatibility warning.

## Segmentation metrics and pipeline smoke check

`metrics.py` computes per-class Dice and IoU from a pixel confusion matrix.
Rows represent target classes; columns represent predicted classes.

- Predictions and targets use integer label IDs (`torch.long`).
- Confusion matrices are summed across processed batches before computing
  metrics, rather than averaging per-slice scores.
- A class absent from both prediction and target receives NaN.
- A class present on only one side receives zero.
- Results include all six classes; no class average is computed.

Run a small end-to-end CPU check from the project directory:

```bash
python check_pipeline.py \
  --image-dir /home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train \
  --mask-dir /home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train \
  --batch-size 2 \
  --max-batches 1
```

The script connects Dataset, DataLoader, UNet2D, argmax predictions, and
metrics. It defaults to base channel width 4 and a randomly initialized
model. It does not load a checkpoint or update parameters.

The Rangpur check processed two real slices and counted 65,536 pixels.
This validates pipeline execution on that subset, not segmentation quality
or the contents of the full dataset.

## Rangpur Slurm CUDA diagnostic

`gpu_check.sbatch` requests one A100 GPU for a five-minute diagnostic using
partition `comp3710` and account `comp3710`.

The script uses `$HOME/miniconda3/envs/torch/bin/python`, reports the node,
NVIDIA driver information, and PyTorch CUDA build, then checks a small CUDA
convolution forward/backward operation and finite gradients.

During setup, the partition reported `DefMemPerNode=UNLIMITED` and nodes
reported `RealMemory=1` to Slurm. The script leaves memory at the partition
default instead of requesting a fixed 4G.

### Submit and inspect

Run on Rangpur from the directory containing the script:

```bash
bash -n gpu_check.sbatch
sbatch gpu_check.sbatch
```

Submit once and note the returned job ID. Replace JOB_ID below:

```bash
squeue -j JOB_ID
sacct -j JOB_ID -X --format=JobID,State,ExitCode,NodeList
cat gpu-check-JOB_ID.log
```

Logs are written to the submission directory. The batch job continues
after SSH disconnects.

Successful validation requires COMPLETED, ExitCode 0:0, and the log message
`CUDA convolution forward/backward: PASS`. A pending job's ExitCode 0:0
does not indicate success.

At this setup checkpoint, shell syntax validation passed and diagnostic
job 700650 was pending with reason Priority after its account and memory
request were corrected. GPU runtime compatibility remains unverified.

## Metric regression tests

Run from the project directory in an environment with PyTorch installed:

```bash
python -m unittest -v test_metrics
```

The five test methods cover perfect predictions, confusion matrix
orientation, hand-calculated Dice/IoU, missed and falsely predicted classes,
jointly absent classes, batch accumulation, and invalid label inputs.

All five tests passed on Rangpur using CPU tensors. They use synthetic
labels and do not require HipMRI files or a GPU.
