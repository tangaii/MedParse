# MedParse

**MedParse: A Task-Specialized Vision-Language Framework for Medical Image Parsing**

This repository contains the public inference and downstream-component fitting
code accompanying the FLARE 2026 Task 3 paper.

## 1. Install

Requirements: Python 3.10 or later. Install the package and test dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

## 2. Prepare Data

Challenge images and labels are not distributed here. Obtain authorized data
and model files under their respective terms. Inference input is an unlabeled
JSONL manifest with one image per record:

```json
{"uid":"case-001","task_type":"classification","dataset":"fundus","prompt":"...","images":["/path/to/image.png"]}
```

Set the task_type field to one of these values: classification,
multi_label_classification, detection, or regression. Validate and normalize
a manifest with:

```bash
python scripts/prepare_data.py \
  --input raw_manifest.jsonl \
  --output prepared.jsonl \
  --image-root /path/to/images
```

This command validates records and image references; it does not extract model
features. Keep labeled targets separate from inference inputs. For Detection,
convert labeled box records to normalized targets with:

```bash
python tools/prepare_detection_targets.py \
  --input detection_labels.jsonl \
  --output detection_targets.json \
  --image-root /path/to/images
```

## 3. Provide Model Assets

Download or provide authorized external assets; they are not included in the
repository. The base model is `google/medgemma-1.5-4b-it`. Supply the primary
and regression LoRA adapters separately, plus the fitted task assets required
for the tasks being run. See [`checkpoints/README.md`](checkpoints/README.md)
for exact filenames, schemas, and producers.

The public training CLI fits downstream components from compatible labeled
feature caches. The public generic adapter helper does not reproduce the final
competition LoRA adapters, and data-derived route/template/candidate assets
may require authorized external preparation. Do not treat fitting public
heads alone as a reconstruction of all final competition assets. Store model
weights, fitted assets, private reference data, and feature caches outside the
repository.

## 4. Extract Features and Fit Components

Extract base-model image-token features when they match the component's
expected input schema:

```bash
python tools/extract_features.py \
  --input prepared.jsonl \
  --base /path/to/medgemma \
  --output features.npz \
  --device cuda:0
```

Detection uses a separate primary-adapter feature cache. Its preprocessing is
fixed to 896-pixel images and feature batch size 8; retain the same settings
and model software stack for reproducible replay:

```bash
python tools/extract_detection_features.py \
  --input detection.jsonl \
  --base /path/to/medgemma \
  --adapter /path/to/primary-adapter \
  --output detection_features.npz \
  --device cuda:0
```

Fit a component using a feature archive and its matching labels/targets. The
component names and required NPZ keys are documented by the CLI and checkpoint
contract:

```bash
python train.py --help
python train.py --component detection-head \
  --features detection_features.npz \
  --targets detection_targets.json \
  --output /path/to/checkpoints/spatial_query_decoder.pt \
  --device cuda:0
```

## 5. Run Inference

First validate the input/output contract without loading models:

```bash
python scripts/prepare_smoke_data.py --output-dir .smoke
python inference.py \
  --input .smoke/input.jsonl \
  --output .smoke/predictions.jsonl \
  --dry-run \
  --audit-json .smoke/audit.json
```

Run model-backed inference with external model and checkpoint paths:

```bash
python inference.py \
  --input prepared.jsonl \
  --output predictions.jsonl \
  --checkpoint-dir /path/to/checkpoints \
  --base /path/to/medgemma \
  --adapter /path/to/primary-adapter \
  --reg-adapter /path/to/regression-adapter \
  --device cuda:0 \
  --audit-json run-audit.json
```

The runtime checks record identity, task, output validity, finite boxes, and
the regression range. Only load assets required by the tasks in the input.

## 6. Evaluate

Compare canonical predictions with a labeled reference file:

```bash
python evaluate.py \
  --reference labeled_reference.jsonl \
  --predictions predictions.jsonl \
  --output metrics.json
```

This local evaluator reports task metrics for diagnostics. It is not the
organizers' official scorer.

## Paper Results

On the organizer-reported FLARE 2026 Task 3 hidden test, MedParse ranked
second overall, with classification balanced accuracy 0.8204, multi-label F1
0.5366, Detection F1 0.7267, and regression MAE 15.1703.

## Citation

```bibtex
@misc{medparse2026,
  author = {Luan, Xiao and Tang, Ai and Yuan, Wei and Zhang, Fuyou},
  title  = {MedParse: A Task-Specialized Vision-Language Framework for
            Medical Image Parsing},
  year   = {2026},
  note   = {FLARE 2026 Task 3 paper and software},
  url    = {https://github.com/tangaii/MedParse}
}
```

## License

The repository code is released under the MIT License. External models,
adapters, dependencies, and datasets remain subject to their own terms.
