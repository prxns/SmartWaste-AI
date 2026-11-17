# Validation record

## Local execution on 18 November 2026

Python 3.11.9 is installed on Windows. A project virtual environment was created without modifying the system Python. External DNS/network access prevented package installation, the resized dataset download and normal Git HTTPS transport. The connected GitHub integration remained available for repository operations.

`python -m unittest discover -s tests -q` executed **35 tests: 26 passed, 7 skipped, and 2 expected failures**. The seven skips require TensorFlow, Streamlit/Plotly, or a trained checkpoint that is not available locally. The two expected failures document the mathematical monotonicity conflict in the exact prescribed Mamdani system. They are known unmet requirements, not passing global-property tests.

`python -m compileall -q app.py config.py smartwaste scripts tests` completed successfully. The required `python -m pytest -q` command was attempted locally and could not run because Pytest could not be installed. The training and Streamlit startup commands were also attempted; missing TensorFlow/Streamlit dependencies prevent local execution. No local empirical CNN metrics are claimed.

The Word report was generated with python-docx using the bundled document runtime. The supplied renderer could not find bundled LibreOffice, and Word COM access was denied by the environment. Therefore local page-layout rendering has not been verified. The repository workflow includes independent report rendering when its execution environment permits.

## Repository validation

The `Validate and prepare demonstration` GitHub workflow is configured to install the complete pinned environment, run Pytest and verify a real Streamlit health endpoint. On an initial main-branch run with no model, it downloads/validates the resized dataset, trains ImageNet MobileNetV2, evaluates the held-out split, runs the suite again, exercises real classification in a browser and creates actual screenshots and an updated report. These are configured checks; their execution status must be verified on the workflow run before they are claimed as completed.

Actual training metrics, when available, are recorded in `results/metrics.json` with the exact model and split hashes. Dataset archive size and source are in `results/dataset.json`. The report builder incorporates only measured artifacts whose model hash matches the checkpoint.
