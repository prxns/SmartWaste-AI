# SmartWaste AI

**CNN-Based Waste Classification with Mamdani Fuzzy Logic Decision Support for Intelligent Waste Segregation.**

A local Streamlit application that classifies individual waste images into the six original TrashNet categories and explains a numerical recycling suitability assessment. The classifier is ImageNet MobileNetV2; the decision engine is a separate, genuine NumPy Mamdani system. Contamination is entered by the user, never inferred by the CNN.

## Windows quick start

Install **Python 3.11, 64-bit** without removing your existing Python. Open a terminal in this repository:

```bat
setup.bat
run.bat
```

`setup.bat` creates `.venv` and installs pinned dependencies. `run.bat` activates that environment and starts the application at the local URL printed by Streamlit, normally http://localhost:8501. No accounts, database, API keys or paid services are needed.

If the repository already contains `models/smartwaste.keras`, classification works immediately after setup. Otherwise activate the environment and train:

```bat
.venv\Scripts\activate
python -m scripts.download_dataset
python -m scripts.train_model --epochs 12
streamlit run app.py
```

Only the resized TrashNet ZIP is downloaded, with a 150 MB limit. If automatic download is blocked, download [dataset-resized.zip](https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip) and run:

```bat
python -m scripts.download_dataset --archive C:\Downloads\dataset-resized.zip
```

Training uses a CPU-friendly 160×160 input, batch size 32, seed 42, frozen ImageNet MobileNetV2, global average pooling, dropout 0.3 and six-class softmax. Adam starts at 0.001; inverse-frequency class weights apply only to training. Up to 12 epochs run, with early stopping (patience 4), validation-loss checkpointing and learning-rate reduction (patience 2). Training duration depends on your CPU. Internet access is required once for packages, the dataset and ImageNet weights.

## Application

| Section | Working functionality |
|---|---|
| Dashboard | Actual session totals, category distribution, suitability summary, history, CSV export and clear history |
| Waste Classification | Bounded JPEG/PNG decoding, preview, CNN prediction, top three softmax scores, manual contamination and fuzzy decision |
| Fuzzy Logic Laboratory | Independent sliders, membership degrees, nine firing strengths, clipped/aggregated output and centroid |
| Model Evaluation | Actual held-out metrics, class report, confusion matrix, training curves and class distribution |
| About & Methodology | Architecture, assumptions, methodology and limitations |

Uploads stay in memory; the application does not permanently save photographs. Limits are 10 MB and 24 megapixels. A missing model produces an explanation and exact commands, never mock predictions. Scores are uncalibrated softmax outputs, not guaranteed correctness. Results apply to six material classes, not mixed-waste scene detection.

## Dataset and reproducibility

Source: [TrashNet by Gary Thung and Mindy Yang](https://github.com/garythung/trashnet), resized archive only. The source describes 2,527 images; the downloader computes the actual validated unique-image distribution and archive byte count in `results/dataset.json`. It inspects all ZIP members before extraction, rejects traversal, symlinks, Windows device paths, encrypted archives and excessive extracted sizes, and validates every image and the six category folders before installation.

Exact decoded duplicates are removed before a deterministic stratified approximately 70/15/15 train/validation/test split. The real archive contains identical images assigned different labels; all members of those ambiguous groups are excluded rather than selecting an arbitrary label. Raw counts, modelling counts and every excluded path/hash/reason are recorded in dataset metadata. Each retained image's relative path and content hash are recorded in `results/splits.csv`. Only training images are augmented. Test data never selects epochs, weights or hyperparameters. `mobilenet_v2.preprocess_input` is used for both training and inference; normalization is not applied twice.

The pipeline saves the best Keras checkpoint, original class mapping, configuration, model hashes, training history, held-out predictions, macro precision/recall/F1, per-class report, confusion matrix and accuracy/loss plots. Re-evaluate the saved test split with:

```bat
python -m scripts.evaluate_model
```

## Fuzzy inference and a specification limitation

Illustrative material potentials in `config.py`: cardboard 80, glass 85, metal 90, paper 75, plastic 60, trash 15. These values are normalized academic assumptions, not measured recycling rates. Contamination is manually set from 0 to 100.

All three universes use Low trapezoid `[0,0,25,50]`, Medium triangle `[25,50,75]` and High trapezoid `[50,75,100,100]`. The nine rules use minimum AND and clipping, maximum aggregation, and centroid integration on 1,001 output samples. The engine calculates the exact area and first moment of the sampled piecewise-linear aggregate. Empty aggregates raise a clear error.

| Material potential | Low contamination | Medium contamination | High contamination |
|---|---|---|---|
| Low | Low | Low | Low |
| Medium | Medium | Medium | Low |
| High | High | Medium | Low |

**The requested global monotonicity guarantees conflict with these exact shapes and inference operators.** For potential 15, contamination 25 produces 19.4444 while contamination 26 produces 19.6316. For contamination 100, raising potential 38→39 reduces the score about 21.9119→21.6899. Asymmetric shoulders move their centroid when clipped. We retain the specified Mamdani system and expose two explicitly expected-failing regression tests; these are known unmet properties, not passing guarantees. See `docs/FUZZY_VALIDATION.md`. Imposing monotonicity requires an approved change to the membership/rule design or postprocessing, which would change the specified system.

## Validation

```bat
python -m pytest -q
python -m unittest discover -s tests -v
```

Tests cover membership endpoints, analytic centroids, min/max inference, range, explicit monotonicity counterexamples, image validation, safe extraction, deterministic disjoint splits, model serialization, trained-artifact provenance, every application page and interactive state. TensorFlow/UI/model-specific tests skip only when their dependencies or artifact are unavailable. The two global monotonicity tests are expected failures with documented mathematical counterexamples.

GitHub Actions installs the pinned environment, runs Pytest, checks a real Streamlit health endpoint, and on the first main-branch build trains the missing CNN on CPU. After successful evaluation and tests, it generates the report and commits real model/results into this existing repository. It uses ordinary fast-forward pushes, never force-pushes. Artifacts are also retained on the workflow run. If repository policy blocks Actions or bot writes, run the same commands locally; no training metrics are invented as a fallback.

## Documentation and structure

```text
app.py / config.py               Streamlit UI and configurable assumptions
smartwaste/                     CNN, fuzzy engine, preprocessing, charts, dataset, evaluation
scripts/                        Safe download, training, evaluation, report generation
models/                         Keras checkpoint, labels, configuration and provenance
results/                        Actual metrics, dataset distribution, splits and plots
assets/                         Theme, icon and generated academic figures
docs/SmartWaste_AI_Project_Report.md
docs/SmartWaste_AI_Project_Report.docx
docs/VIVA_PREPARATION.md
docs/PROJECT_ARCHITECTURE.md
docs/REFERENCES.md
tests/                          Unit and integration checks
setup.bat / run.bat              Windows setup and launch
```

Regenerate the Word report from its Markdown source and current real artifacts:

```bat
python -m scripts.build_report
```

Student, roll number, supervisor and institution fields intentionally contain bracketed placeholders. Fill them before submission. Do not describe the fuzzy suitability score as a real recycling rate. Dataset and third-party attribution are retained in `docs/REFERENCES.md` and `docs/THIRD_PARTY_NOTICES.md`.
