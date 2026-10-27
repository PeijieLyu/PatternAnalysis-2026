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
Baseline training/validation loops, checkpoint/history saving and a single-step
CPU optimizer check are available. ImprovedUNet2D is implemented and its
architecture tests passed on CPU. Training, prediction and checkpoint checking
support both architectures, with CPU integration tests passed. Numeric-label
evaluation exports pooled, patient, case and slice metrics with timing records.
A fixed full-training comparison protocol and Slurm runner are prepared.
Class weighting remains unimplemented.

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

Limited CPU and GPU training smoke runs have completed; no final trained
baseline has been evaluated. Bounded full-model GPU checks for both architectures
passed (see the latest real-MRI evidence below). CPU backward checks on login0
reported a CUDA driver compatibility warning.

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

After its account and memory request were corrected, diagnostic job 700650
completed on a100-1 with state COMPLETED and ExitCode 0:0. Its log reported
`CUDA convolution forward/backward: PASS`. This verifies the small CUDA
diagnostic on the allocated node. Later full-model GPU smoke checks for both
architectures also passed, as recorded in the latest evidence below.

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

Use the supplied HipMRI 2D slices and six label IDs, targeting the Normal difficulty option. Keep the implemented U-Net [1] as baseline. The implemented Improved model adapts residual context blocks and multi-scale segmentation outputs from Isensee et al. [2]. Their original network processes 3D brain MRI; the 2D adaptation and deviations are documented below. Eight CPU architecture tests and the shared-CLI CPU integration checks passed; real-data Improved-model CPU/GPU smoke runs also passed; full training and comparative experiments remain pending. Convolutions, skip/residual connections and overlap-based objectives connect the design to course concepts. Both models will share preprocessing, loss, splits and a matched training budget. Optional per-slice MRI z-score normalization and six-channel one-hot targets
are available. Paired horizontal flips and baseline training/validation loops
are implemented, together with checkpoint and history saving. The Improved
model class and its training/prediction CLI integration are implemented and CPU-tested.

### Preliminary evidence already obtained

Filename audits found 31/4/3 patients and 11,460/660/540 slices in train/validation/test, with no shared patient IDs. One real sample loaded as image [1,256,128] and integer mask [256,128]. The default-width baseline has 1,962,422 parameters; its initial untrained forward check produced finite [1,6,256,128] logits. CPU shape/gradient checks, a two-slice pipeline check and five metric regression tests passed. No final trained accuracy or inference-speed result is established. Diagnostic job 700650 completed on an A100; full-model GPU smoke jobs 704509 and 704574 later completed successfully, with bounded memory observations recorded below.

### Five testable acceptance criteria (proposed)

1. Preserve patient separation and image/mask alignment through preprocessing, demonstrated by split checks, tests and overlays.
2. Target prostate pooled test Dice >= 0.75; also report all-class Dice/IoU and patient-wise summaries. Confirm primary aggregation with the tutor; freeze it before training.
3. Compare baseline and Improved models on identical held-out patients; analyse 3-5 actual boundary failures with MRI, ground truth, prediction and false-positive/false-negative overlays.
4. Proposed resource limits: peak GPU VRAM <= 12 GiB and median model-only inference <= 100 ms per 256x128 slice at batch size 1, after warm-up and CUDA synchronization. Also report per-case runtime. These remain proposed full-experiment targets; bounded allocator-memory observations are now available below, while GPU inference latency remains unmeasured.
5. Record exact dependencies, seeds, configurations and checkpoint selection. Select settings on validation data; use test data only after the experiment protocol is frozen.

### Risks, budget, next experiment and fallback

Key risks are class imbalance, only three test patients, unverified anatomical label-ID mapping/slice ordering, missing 3D context and unverified long-run resource use and final model quality. Verify metadata before prostate/base/apex analysis. Proposed initial budget: two one-hour GPU pilots, then two comparison runs capped at four GPU-hours each. Next: finish check-off, confirm label anatomy, run the frozen v1 full-training/validation comparison using the shared preprocessing. Both models have passed bounded real-MRI GPU smoke checks. Reduce batch size/channel width if needed. If only baseline is feasible, report the Easy fallback honestly and discuss the revised scope with the tutor.

### References

[1] Ronneberger, Fischer and Brox (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. https://arxiv.org/abs/1505.04597

[2] Isensee et al. (2018). *Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge*. https://arxiv.org/abs/1802.10508

Requirements checked against COMP3710/COMP7710 assessment specification v2.5, Sections 2.2 and 3. This review does not constitute tutor approval.

## Artificial Intelligence Usage Disclosure

### Tools and assistance

OpenAI ChatGPT and Codex assisted with substantial code scaffolding and documentation drafts. Assistance covered NIfTI image/mask pairing; the PyTorch Dataset and its validation checks; baseline U-Net components and assembly; Dice/IoU metrics; inspection, visualization and regression-test scripts; Git commands; Conda/Slurm troubleshooting; assessment-requirement review; literature lookup; and drafting this feasibility review and disclosure. Subsequent assistance also covered optional MRI z-score normalization, six-channel one-hot targets, paired training flips, the Dice/CE loss, training/validation loops, checkpoint and history persistence, training-curve plotting, and checkpoint-based prediction/export, and full-model GPU smoke-job/profiling code, native-shape batching, and the paper-inspired 2D Improved model architecture and the shared architecture-selection/checkpoint-loading paths, split evaluation, patient/case metric aggregation, separated inference timing, and the matched experiment protocol/Slurm orchestration with their tests. The assistance included proposing experiments and acceptance criteria, which remain plans rather than measured results. This disclosure does not assert use of other AI tools that has not been confirmed.

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
- Verifying optional normalization and one-hot targets on synthetic inputs and real slices, including exact mask recovery.
- Running paired-augmentation/one-hot tests (8), loss tests (6), and checkpoint/training-loop tests (9) successfully on CPU.
- Running a real-data CPU optimizer step and limited two-epoch training/validation check, then checking saved history consistency and model reloading.
- Running all five history-plot tests successfully and rendering the saved smoke history to a PNG reviewed with AI assistance.
- Confirming Slurm job 700650 completed with exit code 0:0 and its CUDA convolution forward/backward check passed on an allocated A100 node.
- Running all five prediction tests successfully, exporting a real validation prediction, and reviewing its aligned numeric-label/TP/FP/FN panels with AI assistance.
- Running three memory-report unit tests plus three training-loop tests successfully on CPU and checking the Slurm script syntax. GPU smoke job 704367 later raised a heterogeneous-slice batching error in epoch 2.
- Running six native-shape batching tests plus three training-loop tests successfully on Rangpur CPU, then confirming replacement GPU smoke job 704509 completed successfully after the original heterogeneous-shape failure.
- Running all eight Improved-model architecture tests successfully on Rangpur CPU, including shared-loss backward/optimizer checks and baseline compatibility; the default Improved model reported 3,187,746 parameters.
- Running all 20 shared-model-selection, prediction, training-loop and checkpoint tests successfully on Rangpur CPU, including synthetic training/save/reload/prediction for both architectures and old baseline checkpoint compatibility.
- Running an Improved-model real-MRI CPU smoke epoch and checking saved curves, history consistency and checkpoint reloading.
- Confirming baseline GPU job 704509 and Improved GPU job 704574 both completed with ExitCode 0:0, successful checkpoint checks and final smoke PASS markers, then reviewing their bounded allocator-memory profiles with AI assistance.
- Running all 13 evaluation/metric regression tests successfully on Rangpur CPU and exporting a two-slice real validation smoke evaluation from Improved checkpoint 704574, with patient separation and partial scope retained.
- Running all eight comparison-protocol/orchestration tests on Rangpur, checking the Slurm array script syntax, and inspecting partition/QoS/account wall-time records. The orchestration tests stub training children and do not establish that the formal 20-epoch jobs ran.

### Current limits and responsibility

Both architectures passed limited real-MRI GPU training/validation, checkpoint and allocator-profile checks. The Improved model also passed a real-MRI CPU smoke run. The full-training protocol is prepared and its orchestration tests passed; formal 20-epoch jobs have not yet been launched at this checkpoint. No fully trained model or comparative held-out result has been evaluated. Short-run tensor/allocator peaks are recorded below; the proposed long-run resource budget, whole-device memory target, GPU inference latency and target prostate Dice remain unverified. The evaluation tool passed synthetic CPU tests and a two-slice real validation export; those results are smoke checks. The anatomical meaning of label IDs must be checked against dataset metadata before prostate-specific reporting. The full dataset contents have not been exhaustively scanned.

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
driver warning, but both CPU checks completed successfully. A subsequent Slurm
CUDA convolution test passed. Bounded GPU training checks for both models
have since passed; see the latest real-MRI evidence below.

### Baseline training and validation loops

`train.py` now connects normalized MRI slices, one-hot masks, the baseline
U-Net, the shared Dice/CE objective, Adam and per-class Dice/IoU reporting.
Before creating the loaders, it uses `check_splits.collect_patients` to reject
train/validation patient overlap. It uses the supplied training and validation
folders and does not open the test split.

Training shuffles samples and enables parameter updates. `--augment` optionally
enables paired training flips. Validation disables augmentation and autograd and
sets the model to evaluation mode. It preserves parameters, running buffers and
existing gradients. The default loader configuration uses zero worker processes;
the global PyTorch RNG and the separate shuffle generator use `--seed` (3710 by
default).

`run_epoch` reports a sample-weighted average of per-batch objective values.
The soft Dice term is pooled within each batch, so this average is not a soft
Dice objective recomputed over the whole split. Hard-label metrics are computed
from one confusion matrix accumulated over all processed pixels. Jointly absent
classes retain NaN, and results use numerical label IDs.

Run a bounded CPU check from the project directory:

```bash
python -m unittest -v test_train
python -u train.py \
  --data-root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
  --output-dir "$HOME/hipmri_runs/baseline_smoke_$(date +%Y%m%d_%H%M%S)" \
  --epochs 1 --batch-size 2 --base-channels 4 --device cpu --augment \
  --max-train-batches 1 --max-val-batches 1
```

The defaults are one epoch, batch size 2, base width 4, CPU, learning rate 1e-3,
one batch per phase and no augmentation. A batch limit of 0 processes every
batch in that phase. Nonzero limits print PARTIAL RUN: their metrics describe
only the processed subset and are not final evaluation results. This version
requires a new --output-dir and saves checkpoints and epoch histories.

All three training-loop test methods passed on Rangpur CPU. They verify parameter
updates and the batch limit, validation with no gradients or model-state changes,
pooled confusion counts including an incomplete final batch, and rejection of
empty loaders and invalid limits.

The initial loop-only smoke run verified 31 training patients and 4 validation patients
with no overlap. Each phase processed two slices and 65,536 pixels. One observed
training loss was 2.931334 and validation loss was 2.962813. Validation label 5
had NaN Dice/IoU because it was absent from both predictions and targets in that
batch. The run ended with `Training/validation run: PASS`. This was one CPU
training batch followed by one validation batch; no trained checkpoint or final
accuracy result was produced. The separate Slurm CUDA diagnostic has since
passed. Later full-model GPU smoke checks for both architectures passed;
see the latest real-MRI evidence below.

### Checkpoints and epoch histories

Training now requires `--output-dir` naming a new run directory. An existing
directory is rejected to protect earlier runs. Each completed epoch writes:

- `last.pt`: the latest model and optimizer checkpoint.
- `best.pt`: the first epoch with the lowest validation loss so far; ties retain
  the earlier checkpoint. Training loss and test data do not select it.
- `config.json`: run arguments, model and preprocessing settings, loss settings,
  patient lists/counts, source-file SHA-256 hashes and Python/PyTorch versions.
- `history.json` and `history.csv`: losses, processed counts and per-class
  Dice/IoU for each epoch. JSON also retains confusion matrices.

Checkpoints contain model and Adam state dictionaries, architecture and channel
configuration, epoch, best validation loss/epoch, history, configuration, and
PyTorch/shuffle RNG states (plus CUDA RNG states when running on CUDA). Tensor
snapshots are stored on CPU and loaded with `weights_only=True`. Files are
replaced atomically individually; this is not a multi-file transaction.

Jointly absent metric values are stored as JSON null and empty CSV fields.
Non-finite loss values are rejected. `partial_run` records whether either phase
was batch-limited; such checkpoints are execution checks, not final models.

```bash
python -m unittest -v test_training_io test_train
python check_checkpoint.py --run-dir /path/to/run
```

The six checkpoint test methods cover model output equality after reloading,
restored Adam producing the same next update, validation-only best selection,
JSON/CSV handling, protection of existing run directories, invalid losses/epoch
numbers and retained RNG metadata. Together with the three loop tests, the
combined command runs nine tests; all nine passed on Rangpur CPU.

A two-epoch Rangpur CPU smoke run processed one two-slice training batch and one
two-slice validation batch per epoch. The latest and best checkpoints both
corresponded to epoch 2; its validation loss was approximately 2.959447.
`check_checkpoint.py` confirmed two history entries, agreement between JSON,
CSV row counts and the latest checkpoint, minimum-loss selection, and strict
model loading with finite output. Both history consistency and checkpoint
reload checks passed. The saved run was marked `partial_run=True`.

The CLI currently starts fresh runs; automatic resume is not yet implemented.
The saved optimizer/RNG states provide the information for adding resume support.
The single-step script `check_training_step.py` still saves no checkpoint.
Model weight extensions (`*.pt`, `*.pth`, `*.ckpt`) and the project's `/runs/`
directory are ignored by Git. Runtime outputs should remain outside source
control. No final segmentation accuracy is established. Bounded full-model
GPU training/profile checks have since passed for both architectures; see the
latest real-MRI evidence below.

### Training history plots

`plot_history.py` reads the saved `config.json` and `history.json` and exports
`training_curves.png`. It displays train/validation loss and separate panels
for train/validation per-class Dice and IoU. Colors consistently represent
numerical label IDs 0 through 5; no anatomical mapping or class-average metric
is inferred. Metric axes use the full 0-1 score range.

Curves use raw recorded epoch values without smoothing. JSON null scores are
converted to missing plot points, preserving gaps instead of inventing zeros.
A label undefined for every plotted epoch is annotated in that panel.
The title includes architecture, base channel width and an explicit PARTIAL RUN
marker whenever the saved configuration reports a batch-limited run.

After a new training loop completes, `train.py` automatically creates the PNG
in its run directory. Source-file hashes for new runs include `plot_history.py`.
Existing runs can be plotted separately without loading weights or retraining:

```bash
python -m unittest -v test_plot_history
python plot_history.py --run-dir /path/to/run
```

The output defaults to the run directory. To preserve existing figures, the
script refuses to overwrite an output file. Use `--output /path/to/new.png`
when another export is needed. Matplotlib uses the non-interactive Agg backend.

All five plot test methods passed on Rangpur: raw values and null handling,
invalid histories, missing run-scope metadata, PNG rendering for one/two epochs,
and existing-output protection. A PNG was also rendered from the real two-epoch
CPU smoke history and visually reviewed. Titles, axes and legend were readable;
validation label 5 was correctly marked undefined throughout. These two epochs
each used one batch per phase; the figure demonstrates the plotting workflow
and does not establish convergence or final segmentation quality.

### Completed CUDA diagnostic

On 27 October 2026, Slurm accounting and the saved diagnostic log were checked
for job 700650. The observed results were:

| Item | Observed result |
| --- | --- |
| Slurm state / exit code | COMPLETED / 0:0 |
| Allocated node | a100-1 |
| GPU | NVIDIA A100-PCIE-40GB |
| NVIDIA driver | 590.48.01 |
| NVIDIA-SMI CUDA version field | 13.1 |
| PyTorch / PyTorch CUDA build | 2.14.0+cu130 / 13.0 |
| CUDA availability | True |
| Convolution forward/backward and finite gradients | PASS |

The CPU login-node warning does not describe this allocated node's successful
diagnostic. The result establishes this small CUDA operation, not U-Net GPU
training, peak VRAM, inference latency or segmentation quality. The five-minute
allocation has ended; subsequent GPU work must obtain a new Slurm allocation.

### Checkpoint-based prediction and aligned exports

`predict.py` loads a baseline UNet2D checkpoint with its saved channel width and
normalization setting, sets evaluation mode, disables autograd and augmentation,
and predicts one image/mask pair selected by `--index`. Targets remain integer
label IDs for metrics. The current script requires paired masks for its
qualitative/metric export and supports the baseline architecture only.

An existing output directory is rejected. A new output directory receives:

- `prediction.png`: MRI, ground truth, prediction and a selected-label overlap
  panel. Green denotes TP, red FP and blue FN for the requested numeric label.
- `prediction.nii.gz`: uint8 numerical labels with the input image's 2D shape
  and affine. The stored array grid is preserved without transposing/resizing.
- `prediction.json`: checkpoint epoch/source, partial-run flag, preprocessing,
  confusion matrix, per-class Dice/IoU and selected-label TP/FP/FN counts.
  Jointly absent scores are recorded as null.

`--label-id` explicitly selects a numerical ID from 0 through 5. Anatomical
names remain unverified. A partial checkpoint is refused by default;
`--allow-partial` explicitly permits a smoke check and both the figure and JSON
retain that status. It does not turn the checkpoint into a final trained model.

```bash
python -m unittest -v test_predict
python predict.py \
  --checkpoint /path/to/run/best.pt \
  --image-dir /path/to/validation_images \
  --mask-dir /path/to/validation_masks \
  --output-dir /path/to/new_prediction \
  --index 0 --label-id 1 --device cpu --allow-partial
```

All five prediction test methods passed on Rangpur CPU. They check checkpoint
metadata/partial-run guards, saved channel width and inference without gradients,
a hand-calculated TP/FP/FN map, NIfTI shape/affine/dtype and score consistency,
PNG rendering, and protection of existing prediction outputs.

A separate real validation check used the epoch-2, base-width-4 partial checkpoint
on `case_036_week_0_slice_0.nii.gz`. The prediction shape was `[256, 128]` and
normalization was per-slice z-score. For numerical label 1, the observed counts
were TP=0, FP=0, FN=25191. Prediction export completed and the downloaded PNG
was visually reviewed: panels, numeric color scale, partial-checkpoint title
and blue FN overlay were consistent with the reported result. The prediction
panel was dominated by label 4. This smoke checkpoint has only two limited CPU
updates and provides no valid evidence of useful segmentation quality.

NIfTI inputs/outputs (`*.nii`, `*.nii.gz`) are ignored by Git alongside model
weights. Runtime predictions remain outside the source repository. The held-out
test split was not used in this prediction check.

### Full U-Net GPU smoke job and allocator profiling

`gpu_train_smoke.sbatch` requests one A100 and one CPU from partition/account
`comp3710` for at most five minutes. It runs the complete baseline U-Net at
base width 16 and batch size 2 for two epochs, each limited to two training
batches and one validation batch. Training uses normalization, one-hot targets,
paired augmentation and the shared Dice/CE objective. The test split is unused.
This bounded run is an execution/resource check, not a final training experiment.

The script must be submitted with `sbatch`; it rejects direct execution without
a Slurm job allocation. Its run directory is
`$HOME/hipmri_runs/unet_gpu_smoke_JOB_ID` and its submission-directory log is
`unet-gpu-smoke-JOB_ID.log`. After training it checks checkpoint/history
consistency and prints the GPU profile. Only successful completion of all
commands prints `U-Net GPU smoke check: PASS`.

For CUDA runs, `train.py` resets peak statistics before model creation and
`gpu_profile.py` synchronizes CUDA before saving `gpu_profile.json`. The report
contains peak allocated/reserved bytes and GiB, device capacity/name, allocator
backend, node/job ID, model configuration, processed sample counts and run scope.
Timing includes preparation, checkpoint I/O and plotting; it is not inference
latency. CPU runs do not invoke the CUDA profiling path.

These are PyTorch tensor/allocator memory statistics, not whole-device memory
from NVIDIA-SMI. No resource-target result is claimed before a real GPU run is
reviewed. API references:
[peak allocated](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_allocated.html),
[peak reserved](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_reserved.html).

```bash
python -m unittest -v test_gpu_profile test_train
bash -n gpu_train_smoke.sbatch
sbatch gpu_train_smoke.sbatch
```

Submit once and use the returned job ID to inspect progress:

```bash
sacct -j JOB_ID -X --format=JobID,State,ExitCode,NodeList
cat unet-gpu-smoke-JOB_ID.log
```

All six CPU tests passed: three memory-accounting tests (known byte/GiB values
and invalid counters/times) and three existing training-loop tests. Shell syntax
validation also passed. Job 704367 subsequently saved epoch 1 but raised a
DataLoader stacking error in epoch 2: image tensors [1, 256, 128] and
[1, 256, 144] occurred in the same batch. It did not reach the final
checkpoint/history checks, GPU profile export or smoke-check PASS marker.
After the native-shape batching fix described below, replacement job 704509
completed with state COMPLETED, ExitCode 0:0 on a100-5 and the final GPU smoke
PASS marker. Its bounded profile is recorded in the latest evidence below.
The earlier job 700650 validated only a small CUDA convolution.

The previously transcribed real-prediction FN count was corrected from 23919
to 25191 against the terminal output; TP=0 and FP=0 were unchanged. GPU job logs
matching `unet-gpu-smoke-*.log` are ignored by Git.

### Native-shape batches for heterogeneous MRI slices

The training data include different native image widths; the previously inspected
256-by-128 slice is an example, not a fixed size for the entire dataset.
`ShapeBatchSampler` reads image header shapes and groups indices by (height,
width). Every batch contains one shape. It retains the final, possibly smaller
batch of each group and covers every sample once in a full epoch.

Images and masks retain their native pixel grid. No resizing, cropping or padding
is introduced. Dataset checks still reject invalid labels and mismatched masks.
Training shuffles indices within each shape and then shuffles batch order using
the recorded generator. Validation uses deterministic groups and order.

The requested batch size is an upper bound; small groups can produce smaller
batches. Hard Dice/IoU still use the confusion matrix pooled over all original
pixels. Reported epoch loss remains the sample-weighted mean of batch losses.
The batch construction must be held fixed between baseline and Improved-model
comparisons. Short runs with batch limits cover only a subset and do not establish
segmentation quality, regardless of their reported metric values.

`config.json` records the batching strategy, per-split shape/sample counts and
that final batches are retained. Source hashes include `shape_batching.py`.
No changes to the model, loss or metric functions are required for this fix.

```bash
python -m unittest -v test_shape_batching test_train
```

All nine test methods passed on Rangpur CPU on 27 October 2026. Six new tests
cover complete index coverage with incomplete batches, unchanged image pixels
and one-hot masks, reproducible shuffling, invalid batch sizes/non-2D images,
continued rejection of mismatched masks, and pooled pixel accounting across
three native shapes (all 7,680 fixture pixels). The three existing training-loop
tests also passed, including parameter updates and validation-state isolation.
This establishes CPU regression checks, not successful GPU execution.

Original job 704367 and its existing artifacts are retained. Replacement job
704509 uses the same bounded baseline settings and its own job-specific log/run
directory. It subsequently completed successfully with a GPU profile and
checkpoint checks, as recorded below. No final segmentation result is claimed.

### ImprovedUNet2D architecture and CPU verification

`modules.py` now also provides `ImprovedUNet2D`. The existing `UNet2D` baseline
class remains available. This implementation adapts the architectural ideas in
Isensee et al. [2] to single-channel, six-class 2D HipMRI slices:
[original paper, architecture in section 2.1](https://arxiv.org/pdf/1802.10508).

The encoder uses widths b, 2b, 4b, 8b and 16b. Its context blocks add an identity
skip around two pre-activation 3-by-3 convolutions with Dropout2d (default 0.3)
between them. Pre-activation uses affine InstanceNorm2d and LeakyReLU (0.01).
Stride-2 convolutions connect encoder levels. Each decoder stage resizes features
to the exact skip size with nearest interpolation, reduces channels, concatenates
the skip, and applies 3-by-3 then 1-by-1 localization convolutions.

Three segmentation heads at quarter, half and full decoder resolution are
aligned to the input grid and summed as raw logits. The shared Dice/CE objective
is applied to that sum; no separate auxiliary losses are attached. `forward_heads`
returns the three aligned maps for inspection. Default b=16 gives 3,187,746
parameters, versus 1,962,422 for the baseline at the same width. These counts
were observed in CPU tests, not inferred from GPU memory usage.

```python
import torch
from modules import ImprovedUNet2D

model = ImprovedUNet2D(base_channels=16).eval()
with torch.no_grad():
    logits = model(torch.randn(1, 1, 256, 128))  # [1, 6, 256, 128]
```

Height and width must each be at least 32. This conservative guard keeps multiple
spatial values at the deepest InstanceNorm layer. Exact skip-size interpolation
also supports odd dimensions; input channel counts are checked explicitly.

This is a 2D adaptation, not an exact reproduction of the paper's 3D brain-MRI
system. Current choices include whole-slice processing, per-slice all-pixel
z-score normalization, the shared Dice-plus-cross-entropy loss and paired flips.
The original 3D preprocessing, full augmentation/training schedule and test-time
ensemble procedure are not reproduced. No superiority over the baseline is
claimed before matched experiments.

```bash
python -m unittest -v test_improved_model
```

All eight tests passed on Rangpur CPU on 27 October 2026. They cover the default
MRI output shape and finite logits; odd dimensions/configurable channels;
residual identity and input gradients; contribution of all three raw-logit
heads; finite shared-loss gradients and an optimizer update; deterministic eval
without dropout RNG consumption; invalid input/configuration rejection; and
strict state reloading/output equivalence of the existing baseline.

`train.py`, `predict.py` and `check_checkpoint.py` now support both architectures
through the shared factory described below. CPU integration tests use synthetic
data. Subsequent Improved-model real-MRI CPU and GPU smoke runs passed;
its bounded GPU profile is recorded below. Full training and held-out evaluation
remain pending. Architecture/CLI tests and the Improved GPU job used a separate
scratch directory from baseline GPU job 704509. Label anatomy and the tutor
check-off remain unresolved.

### Shared model selection and checkpoint consumers

`model_factory.py` constructs the explicit supported architectures `UNet2D`
and `ImprovedUNet2D`. Training defaults to `UNet2D`, preserving the previous CLI
behavior. Select the other model using `--architecture ImprovedUNet2D`; both use
the same dataset, native-shape batches, augmentation option, loss, metrics and
recorder. The existing CLI base-width default is 4. Improved-model dropout is
currently fixed at 0.3 in training and is recorded in `model_config`.

For example, a bounded CPU check with a new output directory is:

```bash
python train.py \
  --data-root /path/to/keras_slices_data \
  --output-dir /path/to/new_improved_smoke_run \
  --architecture ImprovedUNet2D \
  --epochs 1 --batch-size 2 --base-channels 4 --device cpu --augment \
  --max-train-batches 1 --max-val-batches 1
```

This example is an execution check, not a completed experiment. Positive batch
limits mark the saved run as partial. The default CLI still starts a new run;
checkpoint restore for prediction does not implement training resume.

Checkpoints retain their architecture and exact model configuration. Prediction
and checkpoint inspection restore those fields through the same factory, using
strict state loading. Unknown architectures, invalid channel/dropout settings,
conflicting nested/top-level model metadata and incompatible state dictionaries
are rejected. Early baseline checkpoints with explicit top-level architecture
and width remain supported even when nested config fields were omitted.

`check_checkpoint.py` checks saved history/selection consistency and reloads the
saved model with a finite 32-by-32 CPU probe suitable for both architectures.
Prediction figures print the saved architecture; `prediction.json` also records
it and the model configuration. Prediction keeps the existing partial-checkpoint
guard and numerical-label reporting. Source hashes include `model_factory.py`.

```bash
python -m unittest -v test_model_factory test_predict test_train test_training_io
```

All 20 tests passed on Rangpur CPU on 27 October 2026: six new factory/integration
tests, five prediction tests, three training-loop tests and six checkpoint tests.
The new tests cover configured model construction, metadata/configuration guards,
strict weight compatibility, old baseline prediction and the full CLI path for
both models. Each CLI path trains for one epoch on tiny synthetic NIfTI fixtures
with two native sizes, saves checkpoints and curves, checks the run and exports
a validation prediction. The baseline test omits `--architecture` to verify the
old default. Each fixture split contributes two samples and 2,560 original pixels.

These CPU results validate integration and regression behavior. They do not
establish real-data Improved-model learning, GPU memory use, inference speed or
segmentation accuracy by themselves. Subsequent real-MRI checks are recorded
below. Baseline GPU job 704509 completed successfully using a separate scratch
directory; Improved GPU job 704574 also completed successfully.

### Verified real-MRI smoke runs for both architectures

On 27 October 2026, both bounded GPU jobs completed with Slurm state COMPLETED,
ExitCode 0:0, successful history/checkpoint reloading and their final smoke PASS
markers. Each used one NVIDIA A100-PCIE-40GB, native-shape batches, base width 16,
requested batch size 2, paired training flips and the shared normalization/loss.
Each completed two epochs limited to two training batches and one validation
batch per epoch: four train samples and two validation samples in each epoch.
Both saved `partial_run=true`; neither used the held-out test split.

| Observation | Baseline UNet2D | ImprovedUNet2D |
|---|---:|---:|
| Slurm job | 704509 | 704574 |
| Node | a100-5 | a100-9 |
| Peak allocated bytes | 126316032 | 289378304 |
| Peak allocated GiB | 0.117641 | 0.269505 |
| Peak reserved bytes | 159383552 | 329252864 |
| Peak reserved GiB | 0.148438 | 0.306641 |
| Observed run wall time, seconds | 25.32 | 25.48 |

Both reported PyTorch 2.14.0+cu130, PyTorch CUDA build 13.0 and the native
allocator backend. Improved-model dropout was 0.3. Memory observations are
PyTorch tensor/allocator peaks, not whole-device NVIDIA-SMI measurements. Wall
time includes data preparation, checkpoint I/O and plotting; it is not model-only
inference latency. These are single short jobs on different A100 nodes, without
repeat timing trials. Their close wall times and small validation losses do not
establish speed or segmentation-quality superiority. Resource measurements must
be repeated for the actual training/evaluation configuration.

Saved GPU runs are outside the source repository:
- `$HOME/hipmri_runs/unet_gpu_smoke_704509`
- `$HOME/hipmri_runs/improved_gpu_smoke_704574`

The baseline ran from `$HOME/hipmri_commit6_check`; the Improved model ran from
`$HOME/hipmri_improved_checks`. Their saved source hashes/configuration records
preserve the distinct runtime snapshots. Original failed job 704367 and its
existing artifacts were retained after the native-shape batching fix.

`gpu_improved_smoke.sbatch` requests the same A100/CPU/count/time settings as the
baseline script, with explicit `--architecture ImprovedUNet2D`. It uses a unique
`improved_gpu_smoke_JOB_ID` run directory and `improved-gpu-smoke-JOB_ID.log` log.
Its five-minute limit, fail-on-error behavior, checkpoint check and final PASS
marker bound and verify the diagnostic. Shell syntax validation passed before
job 704574 was submitted. Both GPU log patterns are ignored by Git.

The Improved model also passed a separate CPU real-MRI smoke run at base width 4
and seed 3710, saved under
`$HOME/hipmri_runs/improved_cpu_smoke_20261027_170503`. One epoch processed one
train batch and one validation batch, each with two samples / 65,536 pixels.
Training/validation, history consistency and checkpoint reload reported PASS;
curves were saved and the checkpoint retained `partial_run=true`.

Image-header grouping reported 11,400 training slices of shape [256,128] and
60 of shape [256,144], plus 660 validation slices of shape [256,128]. Patient
separation passed for 31 training and four validation patients. These header
counts do not constitute an exhaustive image/mask contents audit. Full training,
held-out/patient-level final evaluation, GPU model-only latency, anatomical label mapping
and compulsory tutor check-off remain to be completed.

### Split evaluation, patient metrics and inference timing

`evaluate.py` restores either architecture from its checkpoint and evaluates
numerical labels 0-5 at native slice sizes, with inference batch size 1. It uses
the saved normalization setting and disables training augmentation and autograd.
Confusion counts are pooled over original pixels independently at global,
patient and patient/week-case levels. It does not average slice Dice scores.
A case means one patient/week acquisition; patient results combine all included
weeks. CSV rows include TP/FP/FN, target/prediction counts and Dice/IoU for each
numerical label. No label ID is assumed to be the prostate.

Input checks compare evaluation patients with the checkpoint's training and
validation IDs. Validation patient IDs and total slice count must match the saved
split metadata. Duplicate patient/week/slice identifiers are rejected. Validation
smoke checks can explicitly allow partial checkpoints or limit selected samples;
these remain marked in JSON and every CSV. Test mode requires a non-partial
checkpoint, no sample limit, and patients disjoint from both training and
validation. The actual test split must remain unused until the experiment
protocol is frozen. Test-mode regression fixtures use temporary synthetic data.

Each selected shape is warmed up three times by default. Model-forward timing
excludes image loading, preprocessing, input transfer, argmax and metrics. CUDA
is synchronized before and after the measured forward call, consistent with
[PyTorch's synchronization API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.synchronize.html).
One forward is timed per selected slice; medians are reported overall and by
native shape. This is a stated sampling protocol, not repeated-trial latency
confidence estimation. GPU timing and repeated trials remain necessary for the
resource comparison.

Per-case processing time separately includes slice I/O, preprocessing, transfers,
forward execution, argmax and confusion accumulation; it excludes shape warmup
and result export. Patient processing time sums its included cases. Checkpoint
SHA256 is checked before and after evaluation to detect concurrent modification.
Saved checkpoint provenance and current evaluator/dependency source hashes are
recorded. Evaluation always requires a new output directory.

Output files are:

- `evaluation.json`: configuration, scope, provenance, confusion matrices,
  per-class results and timing details.
- `pooled_metrics.csv`, `patients_metrics.csv`, `cases_metrics.csv` and
  `slice_metrics.csv`: numeric-label metrics and scope flags.
- `case_times.csv`: processing time for each included patient/week case.

Absent classes receive JSON null and empty CSV metric cells when absent from
both prediction and target. Missed or falsely predicted classes receive zero
overlap as appropriate. The existing output directory is preserved on refusal.

```bash
python -m unittest -v test_evaluate test_metrics
python evaluate.py \
  --checkpoint /path/to/run/best.pt \
  --data-root /path/to/keras_slices_data \
  --split validate --output-dir /path/to/new_evaluation --device cpu \
  --max-samples 2 --allow-partial
```

All 13 test methods passed on Rangpur CPU on 27 October 2026: eight new evaluation
tests plus five existing metric tests. Coverage includes hand-calculated global
and patient scores across different native sizes, absent-class serialization,
subset/partial/test guards, patient/count/filename checks, non-finite output,
checkpoint modification and preserving existing outputs.

A subsequent real validation check used the partial Improved GPU checkpoint
`$HOME/hipmri_runs/improved_gpu_smoke_704574/best.pt`. It processed only 2 of 660
validation slices, representing one patient and one case. Patient separation and
evaluation export passed, with `smoke_only=true`. Outputs were saved under
`$HOME/hipmri_runs/improved_eval_smoke_20261027_173139`.

Numerical label 4 was absent from both predictions and targets, so its overlap
metrics were undefined. Other two-slice scores do not establish generalization,
prostate accuracy or superiority over the baseline. The observed CPU median
model-forward time was about 133.56 ms over just two timed forwards after warmup;
it is a pipeline check, not a GPU latency result. Full validation/test evaluation
on fully trained checkpoints and repeated GPU timing remain pending.

### Frozen full-training comparison protocol v1

`comparison_protocol.json` fixes the initial comparison before full training.
Protocol ID is `hipmri-2d-comparison-v1`, dated 27 October 2026. The two models
share the following configuration; Improved-model dropout remains 0.3 as part
of its architecture. This compares complete model designs, not individual
architectural ablations.

| Setting | Both models |
|---|---|
| Architecture runs | UNet2D and ImprovedUNet2D |
| Epochs | 20 complete epochs each |
| Base channels / batch size | 16 / 2 |
| Optimizer / learning rate | Adam / 0.001, constant |
| Training seed | 3710 |
| Data | 11,460 training slices / 31 patients; 660 validation slices / 4 patients |
| Preprocessing | Per-slice all-pixel z-score; six-class one-hot targets |
| Training augmentation | Paired last-axis flips with probability 0.5 |
| Loss | Shared Dice plus cross-entropy, unchanged weights/smoothing |
| Batch limits | Both 0, meaning all train/validation batches |
| Checkpoint selection | Minimum full-validation loss; earliest epoch on ties |
| After-training evaluation | Three complete validation passes at batch size 1 |
| Timing warmup | Three forward calls per selected shape before measurement |
| Requested resources | One A100 and one CPU per model, at most four hours each |

Twenty epochs is the fixed initial training budget, not a claim of convergence
or target accuracy. A seed is recorded but bitwise deterministic CUDA training
is not claimed. Three validation/timing passes reuse the same selected weights;
they are not three independent training seeds. Anatomical label mapping must
come from the tutor/dataset metadata and be recorded independently of observed
class scores before final held-out reporting. Tutor check-off remains pending.

`run_comparison.py` validates the protocol, expected train/validation slice and
patient counts, paired masks and patient separation. It never inspects the real
test split. Each task creates a new output directory, saves the exact protocol,
full source-file SHA256 records, source Git commit and a filename/byte-size data
manifest. The manifest is not a hash of image/mask contents. Runtime-file changes
between stages cause failure rather than producing a completed comparison.

The runner launches the existing training CLI, verifies all 20 history epochs
and full sample counts, checks checkpoint consistency, then evaluates the best
checkpoint on the complete validation split three times. It checks that the
checkpoint hash and pooled confusion counts remain consistent between passes.
Output layout is:

- `training/`: checkpoints, epoch histories, training curves and GPU profile.
- `validation_1/`, `validation_2/`, `validation_3/`: complete numeric-label
  evaluation exports and timing records.
- `protocol.json`, `data_manifest.json`, `experiment_provenance.json`:
  configuration, input manifest and stage/status records.
- `comparison_result.json`: completed-task summary, best epoch, training GPU
  profile, validation/patient metrics and timing medians across passes.

Only a successful full task prints `Full training and validation comparison
task: PASS`. A time limit or failed stage does not count as a completed protocol,
even if some checkpoints survive. Automatic training resume is not implemented.
Review Slurm accounting together with the final marker and completed summary.
These formal jobs have not yet been launched at this documentation checkpoint.

`comparison_train.sbatch` uses array task 0 for the baseline and task 1 for the
Improved model. The array requests identical initial resources; each task has
its own output and `comparison-train-ARRAY_JOB_ID_TASK_ID.log`. It explicitly
inherits the environment and requires `HIPMRI_SOURCE_COMMIT` to name the full
40-character commit SHA used for the source snapshot. Array IDs/log placeholders
follow [Slurm's job-array documentation](https://slurm.schedmd.com/job_array.html).
Logs are ignored by Git and weights/data remain outside the source repository.

Before submission, commit the protocol/code, export this project tree from that
exact Git revision using `git archive`, and extract it into a fresh snapshot
directory on Rangpur. Run the array from that directory with the matching
`HIPMRI_SOURCE_COMMIT`. Later development continues in the normal checkout or
scratch directory, preserving the queued/running source snapshot. `--dry-run`
prints planned commands without reading data, launching children or creating
outputs; it still checks that all required runtime files exist.

```bash
python -m unittest -v test_comparison
bash -n comparison_train.sbatch
```

All eight orchestration tests passed locally and on Rangpur. They cover matched
full-split plans for both models, rejecting invalid scope/budgets, dry-run
isolation, mocked complete execution and provenance, changed source, incomplete
sample counts, existing output preservation and preflight data errors. Training
and evaluation child processes are stubbed in these tests; the earlier real-MRI
smoke checks establish the underlying CLI execution separately. Bash syntax
validation also passed.

Observed scheduler records on 27 October 2026 showed partition `comp3710` UP,
`MaxTime=UNLIMITED`, default time 03:00:00, and partition QoS `comp3710` MaxWall
12:00:00. The user's account-association output had no explicit MaxWall value.
The script explicitly requests 04:00:00, below the observed partition QoS limit.
Actual scheduler acceptance, queue delay and whether 20 epochs fit this budget
remain unverified until the formal jobs are submitted and complete.
