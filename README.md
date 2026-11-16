# MedParse

**MedParse: A Task-Specialized Vision-Language Framework for Medical Image Parsing**

MedParse is a four-task medical image parsing pipeline built on a shared
MedGemma vision-language backbone. One image and one task question are mapped
to a classification option, a label set, image-space boxes, or a bounded scalar.
This repository is the public implementation accompanying the paper.

External checkpoint names and schemas are documented in
[checkpoints/README.md](checkpoints/README.md).

## Environments and Requirements

| Component | Setting |
| --- | --- |
| OS | Linux x86_64 |
| Python | 3.12.3 |
| Development accelerator | PPU-ZW810E, 96 GiB |
| Qualification accelerator | NVIDIA A10 |
| CUDA | 12.9 |
| Dependencies | `requirements.txt`, `pyproject.toml` |

Install the runtime and development dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install -e ".[dev]"
```

One local qualification replay used an NVIDIA A10 for 1,783 inputs and
completed in 3,802 s (about 63.4 min). Peak host-sampled GPU memory was
21,797 MiB (about 21.3 GiB); peak resident memory was 19,554,471,936 bytes
(about 18.2 GiB). This local replay is distinct from the organizer-reported
runtime in the official leaderboard below.

## Dataset

FLARE challenge data and images are obtained by the user under the organizer's
access and licensing terms; they are not included here. The public inference
input is unlabeled JSONL (a JSON list is also accepted), with one local image
per row:

```json
{"uid":"case-001","task_type":"classification","dataset":"fundus","prompt":"...","images":["/path/to/image.png"]}
```

Supported task types are `classification`, `multi_label_classification`,
`detection`, and `regression`. Each row has a unique UID, a dataset/source,
one question or prompt, and exactly one image. Training labels use a separate
contract. Inputs must not include answer, target, label, reference, or
prediction fields; detection predictions use original-image `[x1,y1,x2,y2]`
boxes. `scripts/prepare_data.py` performs manifest validation and normalization.

The paper reports the following task-specific labeled development resources.
Rows are question records and unique references are manifest image-reference
counts; neither is a patient count.

| Task | Split | Rows | Unique refs | Modality (rows) |
| --- | --- | ---: | ---: | --- |
| CLS | Train | 16,971 | 23,389 | Digital Camera 596; Dermoscopy 2,860; Endoscopy 64; Fundus 1,380; Mammography 3,583; Microscopy 722; Retinography 1,113; Ultrasound 2,447; X-ray 4,206 |
| CLS | Public validation | 2,387 | 4,975 | Digital Camera 149; Dermoscopy 716; Endoscopy 16; Mammography 910; Retinography 279; Ultrasound 317 |
| MLC | Train | 7,134 | 5,499 | X-ray 7,134 |
| MLC | Public validation | 970 | 970 | X-ray 970 |
| DET | Train | 5,885 | 3,310 | Ultrasound 5,885 |
| DET | Public validation | 175 | 175 | Ultrasound 175 |
| REG | Train | 5,852 | 3,277 | Ultrasound 5,150; X-ray 702 |
| REG | Public validation | -- | -- | No public labeled regression manifest reported |

These are task-specific resources, not one homogeneous cohort. Counts are not
patient counts. Labeled public validation data support development where
available; no public labeled regression validation manifest is reported here.
Challenge images and labels are not redistributed.

The BUS-UCLM/BUSI ultrasound data form a separate 175-case labeled cohort for
the Detection study. This cohort is not the organizer's hidden Detection split
and is not part of the official hidden-test results.

## Preprocessing

Classification, multi-label classification, and Detection use EXIF-corrected
RGB images resized to `896 x 896` with BICUBIC resampling. Regression combines
the original image with overlapping aspect-ratio crops or four quadrant crops
for near-square images. Its 960-dimensional geometry descriptor uses grayscale,
BILINEAR resizing, intensity scaling, and reflected-boundary Sobel derivatives.

Prepare an unlabeled local manifest with:

```bash
python scripts/prepare_data.py \
  --input raw_manifest.jsonl \
  --output prepared.jsonl \
  --image-root /path/to/images
```

Manifest preparation validates and normalizes references; it is not source
image preprocessing and does not create learned assets.

## Method Overview

The implementation shares a MedGemma representation interface and adds four
task-specific predictors. The mapping below links each paper component to its
primary implementation.

| Component | Function | Primary code |
| --- | --- | --- |
| Shared interface | Prompting, generation, decoder states, and image tokens | `src/medical_parsing/models/backbone.py` |
| Task-routed classification | Semantic image-token head or generation route | `src/medical_parsing/tasks/classification.py` |
| Evidence-guided set decoding | Candidate refinement and cardinality-aware F1 decoding | `src/medical_parsing/tasks/multilabel.py` |
| Spatial query decoding | Box prediction from image tokens and decoder states | `src/medical_parsing/tasks/detection.py` |
| Retrieval-refined quantile regression | Visual, retrieval, residual, and quantile refinement to `[0,100]` | `src/medical_parsing/tasks/regression.py` |

## Training

`train.py` fits downstream components from user-owned labeled arrays or feature
caches. The public components are:

| Component | Input | Output |
| --- | --- | --- |
| `classification-head` | token NPZ + semantic-label JSONL | `classification_heads.pt` |
| `detection-head` | image-token/query-state NPZ + normalized targets | `spatial_query_decoder.pt` |
| `multilabel-selector-ranker` | selector/ranker features and targets | selector and ranker CatBoost files |
| `multilabel-probability-models` | features + `[N,10,4]` targets | `multilabel_probability_models.joblib` |
| `multilabel-residual-head` | tokens, row features, probabilities, targets | `multilabel_residual_head.pt` |
| `regression-visual-estimator` | visual features + scalar targets | `regression_visual_model.joblib` |
| `regression-reference` | visual/geometry features + targets, UIDs, groups | `regression_reference.joblib` |
| `regression-residuals` | UID-aligned residual values | `regression_residuals.npz` |
| `regression-quantile-head` | tokens, geometry, scalar targets | `regression_quantile_head.pt` |

Fit a component using its CLI name and feature/target inputs:

```bash
python train.py --component <component> [component options]
```

Detection fitting defaults are 30 epochs, batch size 256, AdamW with learning
rate `1e-3` and weight decay `1e-4`, 5% warmup, and cosine decay. See
`python train.py --help` and [checkpoints/README.md](checkpoints/README.md) for
all component options, serialized keys, and feature producers. The generic
`src/medical_parsing/training/adapters.py` helper does not claim to reproduce
the final competition LoRA adapters.

## Inference

Run the contract smoke test without external models or checkpoints:

```bash
python scripts/prepare_smoke_data.py --output-dir .smoke
python inference.py \
  --input .smoke/input.jsonl \
  --output .smoke/predictions.jsonl \
  --dry-run \
  --audit-json .smoke/audit.json
```

For model-backed inference, pass the external base model, adapters, and
checkpoint directory explicitly:

```bash
python inference.py \
  --input prepared.jsonl \
  --output predictions.jsonl \
  --checkpoint-dir /path/to/external/checkpoints \
  --base /path/to/medgemma \
  --adapter /path/to/primary-task-adapter \
  --reg-adapter /path/to/regression-task-adapter \
  --device cuda:0 \
  --audit-json run-audit.json
```

The required assets are listed in [checkpoints/README.md](checkpoints/README.md).
Inference validates UID order, task identity, legal labels, finite boxes, and
the `[0, 100]` regression range.

## Evaluation

Evaluate canonical predictions locally with:

```bash
python evaluate.py \
  --reference labeled_reference.jsonl \
  --predictions predictions.jsonl \
  --output metrics.json
```

The evaluator reports classification accuracy; multi-label exact match,
micro-F1, and sample-F1; Detection F1 at IoU 0.5; and regression MAE, RMSE,
and bias. It is a local diagnostic evaluator, not the organizer's official
scorer.

## Results

### Official FLARE 2026 Task 3 results

The organizer evaluated the submitted system on the hidden test set. MedParse,
submitted as `lightbridge`, ranked second overall. The full six-team
leaderboard reported in the paper is reproduced here; runtime is the
organizer-reported value in seconds.

| Team | CLS BA | MLC F1 | DET F1 | REG MAE | Runtime (s) | Overall rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hexapus | 0.8585 | 0.5310 | 0.8633 | 18.5286 | 6162 | 1 |
| **lightbridge (MedParse)** | **0.8204** | **0.5366** | **0.7267** | **15.1703** | **5659** | **2** |
| AIMHI | 0.8313 | 0.4694 | 0.6900 | 19.8822 | 1606 | 3 |
| LIST-SEU | 0.8507 | 0.4809 | 0.5633 | 17.3954 | 3774 | 4 |
| MedCat | 0.7446 | 0.4663 | 0.6400 | 18.5188 | 1450 | 5 |
| MedGemma1.5-FLARE | 0.7502 | 0.4430 | 0.6300 | 19.0979 | 1033 | 6 |

Higher BA/F1 and lower MAE/runtime are better. Official challenge scores are
reported by the organizers; `evaluate.py` is a local evaluator and is not the
organizer's scorer.

### Labeled development diagnostics

The paper also reports local comparisons on labeled development data. These
diagnostics use different task-specific protocols and must not be interpreted
as official hidden-test results or as one uniform four-task test set.

| Task | Evaluation unit | Comparison | Result |
| --- | --- | --- | --- |
| CLS | 5,151 records / 2,275 image groups | Direct vs. routed | BA 0.644420 vs. 0.764283; paired delta +0.119863 (95% CI 0.090796 to 0.150539) |
| MLC | 2,605 records / 1,266 image groups | Direct vs. candidate refinement | Micro-F1 0.551832 vs. 0.562904; paired delta +0.011072 (95% CI 0.004984 to 0.017257) |
| MLC | Same grouped OOF records | Non-GFM control vs. GFM | Micro-F1 0.544443 vs. 0.551820; paired delta +0.007377 (95% CI 0.003663 to 0.010966) |
| MLC | Same grouped OOF records | GFM vs. residual surrogate | Micro-F1 0.551820 vs. 0.548789; paired delta -0.003031 (95% CI -0.005126 to -0.000781) |
| REG | 561 source OOF rows | G4; Q50; Q75 | MAE 13.246975; 12.852755; 12.754508. Q75-Q50 delta -0.098247 (95% CI -0.251303 to 0.056902) |
| DET | 59-case holdout | Generated coordinates vs. spatial query | F1 at IoU 0.5: 0.158730 vs. 0.464000; paired delta +0.305270 (95% CI 0.173565 to 0.433353) |

The CLS comparison uses a different prompt in the historical direct-prediction
configuration, so it is not an isolated routing ablation. The MLC candidate
refinement uses fold-local ExtraTrees models; its probability and residual
comparisons use Logistic/Ridge surrogate components, not the production
CatBoost selector/ranker and neural residual head. The full residual-surrogate
score does not establish an improvement over direct generation.

For Detection, 116 public development cases were used for configuration
selection and 59 cases formed the separate holdout. The combined 175-case
public cohort includes those 116 development cases: spatial-query F1 was
0.453552 and generated-coordinate baseline F1 was 0.200542 on that combined
cohort. The 175-case result is descriptive and is not an independent holdout
estimate. Both public-cohort results are separate from the official hidden-test
Detection F1 of 0.7267.

## Reproducibility Notes

Reproducing fitted predictions requires authorized FLARE data, the
`google/medgemma-1.5-4b-it` base model, both PEFT adapters, data-derived
manifests, and fitted task assets. These are external and are not committed.
The generic adapter-training helper does not reproduce the final competition
adapters. The checkpoint README documents required files and schemas; official
scores require the organizer's evaluator.

## Repository Structure

```text
configs/default.yaml      Runtime defaults and reference contracts
checkpoints/README.md     External asset names and schemas
src/medical_parsing/      Package implementation
  data/                   Manifest preparation
  evaluation/             Local metrics
  inference/              Four-task orchestration
  models/                 Backbone and neural modules
  tasks/                  Task implementations
  training/               Component fitting
tests/                    Contract and behavior tests
scripts/                  Manifest and smoke-data utilities
tools/                    Feature and target preparation utilities
inference.py              Public inference CLI
train.py                  Public fitting CLI
evaluate.py               Local evaluation CLI
```

## Contributing

Open an issue for a reproducibility problem or a narrowly scoped bug. Before a
pull request, run `pytest -q` and the smoke-test commands. Do not commit
challenge data, model weights, adapters, fitted checkpoints, private keys, or
local machine paths.

## Citation

For the paper, cite the manuscript as follows (publication and DOI metadata
are not asserted here):

```bibtex
@misc{luan2026medparse,
  author = {Luan, Xiao and Tang, Ai and Yuan, Wei and Zhang, Fuyou},
  title   = {MedParse: A Task-Specialized Vision-Language Framework for
             Medical Image Parsing},
  year   = {2026},
  note   = {FLARE 2026 Task 3 manuscript}
}
```

For the software release, use:

```bibtex
@software{medparse2026software,
  author  = {Luan, Xiao and Tang, Ai and Yuan, Wei and Zhang, Fuyou},
  title   = {MedParse: A Task-Specialized Vision-Language Framework for
             Medical Image Parsing},
  version = {0.1.0},
  url     = {https://github.com/tangaii/MedParse}
}
```

Machine-readable software citation metadata is provided in
[CITATION.cff](CITATION.cff).

## Acknowledgement

We thank the FLARE 2026 organizers and data contributors and acknowledge the
MedGemma model and the open-source software used in this work.

## License

The original research code is released under the MIT License; see
[LICENSE](LICENSE). External model, adapter, dependency, and dataset terms
remain applicable.
