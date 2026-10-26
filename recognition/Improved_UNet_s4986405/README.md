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
- MRI intensity values are preserved when normalize=False (the default).
Valid mask labels are preserved with either normalization setting.
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

Optional per-slice MRI normalization and six-channel one-hot masks are
available. Optional paired training augmentation is also available.
Class weighting and the full training/validation workflow are not implemented.
A single-step CPU optimizer check is available.

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

## Feasibility Review

**Project:** HipMRI 2D Improved U-Net | **Student:** s4986405 | **Date:** 26 October 2026
**Status:** Draft for tutor consultation; compulsory check-off not yet completed.

### User need and engineering dilemma

An imaging specialist needs an offline prototype that drafts pelvic contours for human review. The investigation asks whether a 2D Improved U-Net reduces prostate base/apex and adjacent-organ boundary errors compared with a basic 2D U-Net enough to justify extra memory and inference time. Results will address 2D architectural changes; inter-slice continuity remains a limitation.

### Scope, model choice and course concepts

Use the supplied HipMRI 2D slices and six label IDs, targeting the Normal difficulty option. Keep the implemented U-Net [1] as baseline. The proposed Improved model will adapt residual context blocks and multi-scale segmentation supervision from Isensee et al. [2]. Their original network processes 3D brain MRI; the 2D adaptation and deviations will be documented. Convolutions, skip/residual connections and overlap-based objectives connect the design to course concepts. Both models will share preprocessing, loss, splits and a matched training budget. Optional per-slice MRI z-score normalization and six-channel one-hot targets
are available. Paired horizontal flips are available for training; shared
training pipeline integration remains to be implemented.

### Preliminary evidence already obtained

Filename audits found 31/4/3 patients and 11,460/660/540 slices in train/validation/test, with no shared patient IDs. One real sample loaded as image [1,256,128] and integer mask [256,128]. The untrained baseline has 1,962,422 parameters and produced finite [1,6,256,128] logits. CPU shape/gradient checks, a two-slice pipeline check and five metric regression tests passed. No trained accuracy or GPU performance is established; diagnostic job 700650 was last observed pending.

### Five testable acceptance criteria (proposed)

1. Preserve patient separation and image/mask alignment through preprocessing, demonstrated by split checks, tests and overlays.
2. Target prostate pooled test Dice >= 0.75; also report all-class Dice/IoU and patient-wise summaries. Confirm primary aggregation with the tutor; freeze it before training.
3. Compare baseline and Improved models on identical held-out patients; analyse 3-5 actual boundary failures with MRI, ground truth, prediction and false-positive/false-negative overlays.
4. Proposed resource limits: peak GPU VRAM <= 12 GiB and median model-only inference <= 100 ms per 256x128 slice at batch size 1, after warm-up and CUDA synchronization. Also report per-case runtime. These limits are unmeasured targets.
5. Record exact dependencies, seeds, configurations and checkpoint selection. Select settings on validation data; use test data only after the experiment protocol is frozen.

### Risks, budget, next experiment and fallback

Key risks are class imbalance, only three test patients, unverified anatomical label-ID mapping/slice ordering, missing 3D context and unresolved GPU compatibility. Verify metadata before prostate/base/apex analysis. Proposed initial budget: two one-hour GPU pilots, then two comparison runs capped at four GPU-hours each. Next: finish check-off, verify CUDA, implement shared preprocessing, then test a tiny training subset before matched experiments. Reduce batch size/channel width if needed. If only baseline is feasible, report the Easy fallback honestly and discuss the revised scope with the tutor.

### References

[1] Ronneberger, Fischer and Brox (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. https://arxiv.org/abs/1505.04597

[2] Isensee et al. (2018). *Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge*. https://arxiv.org/abs/1802.10508

Requirements checked against COMP3710/COMP7710 assessment specification v2.5, Sections 2.2 and 3. This review does not constitute tutor approval.

## Artificial Intelligence Usage Disclosure

### Tools and assistance

OpenAI ChatGPT and Codex assisted with substantial code scaffolding and documentation drafts. Assistance covered NIfTI image/mask pairing; the PyTorch Dataset and its validation checks; baseline U-Net components and assembly; Dice/IoU metrics; inspection, visualization and regression-test scripts; Git commands; Conda/Slurm troubleshooting; assessment-requirement review; literature lookup; and drafting this feasibility review and disclosure. The assistance included proposing experiments and acceptance criteria, which remain plans rather than measured results. This disclosure does not assert use of other AI tools that has not been confirmed.

### Verification performed so far

The student ran the suggested commands in the local WSL repository and on Rangpur and shared the resulting terminal output for review with AI assistance. Verification included:

- Checking Git diffs and maintaining progressive, scoped commits.
- Checking Python or shell syntax for the newly created or modified files.
- Testing NIfTI filename pairing and loading real samples with expected shapes, dtypes and label ranges.
- Testing rejection of invalid dimensions, shape mismatches, invalid mask labels and non-finite MRI values using temporary files.
- Auditing patient IDs across the supplied splits and testing artificial overlap cases.
- Viewing a real MRI/mask side-by-side PNG.
- Checking CPU forward shapes and finite gradients for model blocks and the assembled baseline.
- Checking imports after moving the model to `modules.py`.
- Comparing metric calculations with hand-worked examples, and running five saved metric regression tests successfully.
- Running a two-slice end-to-end CPU pipeline check with a matching pixel count.

### Current limits and responsibility

The baseline has not been trained and the Improved model has not been implemented. The target prostate Dice, GPU memory, inference latency and proposed resource budgets have not been demonstrated. The CUDA diagnostic was last observed pending in Slurm; a submitted job is not evidence of GPU compatibility. The anatomical meaning of label IDs must be checked against dataset metadata before prostate-specific reporting. The full dataset contents have not been exhaustively scanned.

The student remains responsible for reviewing, understanding, correcting and explaining the submitted code, references, experimental design and report. The checks above document observed execution and do not establish that every line has received an independent manual audit. This disclosure must be updated as implementation, verification and actual AI use change. The tutor feasibility check-off remains incomplete.

## Dataset provenance and label semantics

### Data acknowledgement and reference

Data were provided by the HiPMRI Study and accessed through the COMP3710
teaching copy on UQ Rangpur:

`/home/groups/comp3710/HipMRI_Study_open`

The supplied `data_use_agreement.txt` requests acknowledgement of the data
and citation of an applicable dataset-related publication.

Dataset-related reference:
Liu, S., Dowling, J. A., Engstrom, C., Greer, P. B., Crozier, S., and
Chandra, S. S. (2020). *Manipulating Medical Image Translation with
Manifold Disentanglement*.
[Paper](https://arxiv.org/abs/2011.13615).

This reference provides dataset context; its list of anatomical structures
does not establish the numerical label mapping of the course slice files.

### Label metadata audit

The observed training masks contain integer label IDs 0 through 5.
Anatomical names for these IDs remain unverified.

On 26 October 2026, headers were inspected for:
- `seg_004_week_0_slice_0.nii.gz`: shape `[256, 128]`.
- `B006_Week0_SEMANTIC.nii.gz`: shape `[256, 256, 128]`.

Both sampled headers had empty `descrip`, `aux_file`, and `intent_name`
fields, intent `none`, and zero extensions. This finding applies to these
two sampled headers, not an exhaustive metadata audit.

Reports currently use numerical label IDs. Confirm the anatomical mapping
and any course-specific remapping with the tutor or dataset maintainer,
then record the source before reporting prostate-specific results.

### Optional per-slice MRI normalization

`HipMRI2DDataset(..., normalize=True)` enables z-score normalization
independently for each MRI slice. The calculation includes every pixel,
including background, and uses population standard deviation
(`unbiased=False`). Statistics are computed in float64 and the output
image remains float32 with shape `[1, H, W]`.

Constant slices return zero-valued images. Integer masks are unchanged.
The default `normalize=False` preserves the loaded MRI values.
No dataset-wide statistics are fitted or shared between splits.

Enable normalization in the batch inspection script:

```bash
python check_dataset.py \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --batch-size 2 \
  --normalize
```

Rangpur CPU checks passed for ordinary, zero, constant, tiny-value and
large-value synthetic images, including comparison with a NumPy reference.
The default raw values and mask labels were preserved.

One real training slice had normalized mean approximately -4.66e-9 and
population standard deviation 1.0. A normalized batch of two real slices
also passed shape and dtype checks.

Normalization is currently enabled explicitly through the Dataset option
or `check_dataset.py --normalize`. The existing pipeline smoke script
continues to use the default raw-image setting.

### Optional six-channel one-hot masks

The Dataset defaults to `one_hot=False`, returning integer labels with shape
`[H, W]` and dtype `torch.long`. Set `one_hot=True` to return a float32 mask
with shape `[6, H, W]`. Channel k represents numerical label ID k (0 through 5).
All six channels are retained when some classes are absent from a slice.
Anatomical names for these numerical IDs remain unverified.

Each pixel has exactly one active channel. Taking `argmax(dim=0)` on a single
encoded mask recovers the integer labels exactly. MRI values are unaffected
by the encoding option. `normalize=True` and `one_hot=True` can be used together.
Existing mask validation runs before encoding, so invalid labels are rejected.

```python
from dataset import HipMRI2DDataset

dataset = HipMRI2DDataset(image_dir, mask_dir, normalize=True, one_hot=True)
image, target = dataset[0]
```

Run the synthetic regression tests from the project directory:

```bash
python -m unittest -v test_one_hot
```

All three test methods passed on Rangpur CPU. They cover spatial alignment,
optional normalization, fixed channels for absent classes, batching and rejection
of invalid masks. A separate check on two real training slices produced targets
of shape `[2, 6, 256, 128]` with dtype float32. Every pixel had one active channel,
argmax recovered the original masks exactly, and the normalized images matched
those loaded with integer targets. These checks validate encoding correctness;
they do not measure segmentation accuracy.

### Paired training augmentation

Set both `training=True` and `augment=True` to enable a random horizontal
flip with probability 0.5 per sample. Horizontal refers to the last array axis.
The same random decision is used for the MRI image and its integer mask.
Flipping preserves spatial dimensions, MRI values and numerical label IDs;
it requires no spatial interpolation.

The flip runs after input validation and before MRI normalization and one-hot
encoding. It supports either target representation and either normalization
setting. Both new options default to False. Requesting `augment=True` without
`training=True` raises ValueError. Split roles are supplied by the caller;
they are not inferred from directory names. Keep augmentation disabled for
validation, test and inference.

```python
train_dataset = HipMRI2DDataset(
    train_image_dir, train_mask_dir,
    normalize=True, one_hot=True, training=True, augment=True,
)
val_dataset = HipMRI2DDataset(
    val_image_dir, val_mask_dir, normalize=True, one_hot=True,
)
```

Random decisions use the PyTorch RNG. A fixed `torch.manual_seed(...)` makes
the tested single-process sample sequence reproducible. Future training code
must also configure DataLoader shuffling and worker seeds as applicable.

```bash
python -m unittest -v test_augmentation test_one_hot
```

All eight synthetic test methods passed on Rangpur CPU: five augmentation
tests and three one-hot tests. They verify both random branches, image-mask
alignment, the training guard, disabled augmentation using no random draw,
normalized one-hot batches, seed reproducibility, missing classes and invalid
mask rejection. This check does not establish a segmentation accuracy benefit.

### Shared segmentation loss and CPU training-step check

`losses.py` defines `dice_ce_loss(logits, targets)`. This project uses an
unweighted sum of pixel cross-entropy and soft Dice loss as a shared objective
for the planned baseline/improved comparison. This is an implementation choice.

Both tensors have shape `[B, C, H, W]` (C=6 for this project). Logits are raw
floating-point scores; targets are float one-hot masks with exactly one active
channel per pixel. Shape, device, finite values and target validity are checked.

Cross-entropy is averaged over all batch pixels. For each channel, soft Dice
pools predicted probabilities and targets across batch and spatial dimensions:

`Dice_c = (2 * sum(probability_c * target_c) + 1e-5) / (sum(probability_c) + sum(target_c) + 1e-5)`

`Loss = CrossEntropy + 1 - mean(Dice_c)`

Every channel, including numerical label 0, contributes equally to the Dice
average. There are no class weights. Softmax/log-softmax are calculated inside
the loss. Float16/bfloat16 inputs are promoted to float32; float64 logits retain
float64 calculation. The differentiable soft Dice objective is separate from
the hard-label Dice/IoU reporting metrics and their jointly-absent-class NaN policy.

```bash
python -m unittest -v test_losses
python check_training_step.py
```

All six loss test methods passed on Rangpur CPU. Checks cover a hand-calculated
uniform prediction, correct versus incorrect predictions, large finite logits
and gradients, joint channel permutations, invalid inputs and a U-Net optimizer
update on synthetic data.

The separate training-step smoke check loaded two real training slices with
normalization, paired augmentation and one-hot targets, then ran a base-width-4
U-Net and one Adam step on CPU. Image shape was `[2, 1, 256, 128]`; targets were
`[2, 6, 256, 128]`. One observed pre-update loss was approximately 3.03997.
Finite gradients, finite updated parameters and a parameter change were verified.

The script defaults to the Rangpur teaching dataset; use `--data-root` to select
another directory containing the training image and mask folders. It saves no
checkpoint and reports no segmentation accuracy. CPU backward emitted a CUDA
driver warning, but both checks completed successfully. GPU runtime compatibility
remains unverified.
