# MedParse

**MedParse: A Task-Specialized Vision-Language Framework for Medical Image Parsing**

MedParse combines a shared MedGemma representation with task-specific
predictors for classification, multi-label prediction, detection, and
regression. This repository accompanies the FLARE 2026 Task 3 paper.

External checkpoint names and schemas are documented in
[checkpoints/README.md](checkpoints/README.md).

## Installation

Install the runtime and optional test dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Dataset

Challenge data are not included; obtain them under the organizers' access and
licensing terms. Inference accepts an unlabeled JSONL manifest, for example:

```json
{"uid":"case-001","task_type":"classification","dataset":"fundus","prompt":"...","images":["/path/to/image.png"]}
```

Supported tasks are `classification`, `multi_label_classification`,
`detection`, and `regression`. `scripts/prepare_data.py` validates and
normalizes manifests; it does not preprocess images or create learned assets.

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

These are task-specific resources, not one homogeneous cohort. Rows are
question records and unique references are image identifiers, not patient
counts. No public labeled regression validation manifest is reported.

The BUS-UCLM/BUSI ultrasound data form a separate 175-case labeled cohort for
the Detection study. This cohort is not the organizer's hidden Detection split
and is not part of the official hidden-test results.

## Preprocessing and Method

Classification, multi-label classification, and Detection use EXIF-corrected
RGB images resized to `896 x 896` with BICUBIC resampling. Regression combines
the original image with overlapping aspect-ratio crops or four quadrant crops
for near-square images. Its 960-dimensional geometry descriptor uses grayscale,
BILINEAR resizing, intensity scaling, and reflected-boundary Sobel derivatives.

To validate and normalize a manifest:

```bash
python scripts/prepare_data.py \
  --input raw_manifest.jsonl \
  --output prepared.jsonl \
  --image-root /path/to/images
```

The implementation of each paper component is located here:

| Component | Function | Primary code |
| --- | --- | --- |
| Shared interface | Prompting, generation, decoder states, and image tokens | `src/medical_parsing/models/backbone.py` |
| Task-routed classification | Semantic image-token head or generation route | `src/medical_parsing/tasks/classification.py` |
| Evidence-guided set decoding | Candidate refinement and cardinality-aware F1 decoding | `src/medical_parsing/tasks/multilabel.py` |
| Spatial query decoding | Box prediction from image tokens and decoder states | `src/medical_parsing/tasks/detection.py` |
| Retrieval-refined quantile regression | Visual, retrieval, residual, and quantile refinement to `[0,100]` | `src/medical_parsing/tasks/regression.py` |

## Fitting Components

`train.py` fits supported downstream components from user-provided labeled
features. Consult its help and the checkpoint contract for component inputs and
asset schemas. The generic adapter trainer does not reproduce the final
competition LoRA adapters.

```bash
python train.py --component <component> [component options]
```

## Inference

Run a contract smoke test without external models or checkpoints:

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

## Contributing

Open an issue for a reproducibility problem or a narrowly scoped bug. Before a
pull request, run `pytest -q` and the smoke-test commands. Do not commit
challenge data, model weights, adapters, fitted checkpoints, private keys, or
local machine paths.

## Citation

```bibtex
@misc{medparse2026,
  author = {Luan, Xiao and Tang, Ai and Yuan, Wei and Zhang, Fuyou},
  title   = {MedParse: A Task-Specialized Vision-Language Framework for
             Medical Image Parsing},
  year    = {2026},
  note    = {FLARE 2026 Task 3 paper and software},
  url     = {https://github.com/tangaii/MedParse}
}
```

Machine-readable citation metadata is provided in
[CITATION.cff](CITATION.cff).

## License

The original research code is released under the MIT License; see
[LICENSE](LICENSE). External model, adapter, dependency, and dataset terms
remain applicable.
