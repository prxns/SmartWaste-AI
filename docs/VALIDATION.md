# Validation record

## Local execution on 18 November 2026

Python 3.11.9 is installed on Windows. A project virtual environment was created without modifying the system Python. External DNS/network access prevented package installation, the resized dataset download and normal Git HTTPS transport. The connected GitHub integration remained available for repository operations.

`python -m unittest discover -s tests -q` executed **36 tests: 27 passed, 7 skipped, and 2 expected failures** after the duplicate-handling regression was added. The seven skips require TensorFlow, Streamlit/Plotly, or a trained checkpoint that is not available locally. The two expected failures document the mathematical monotonicity conflict in the exact prescribed Mamdani system. They are known unmet requirements, not passing global-property tests.

`python -m compileall -q app.py config.py smartwaste scripts tests` completed successfully. The required `python -m pytest -q` command was attempted locally and could not run because Pytest could not be installed. The training and Streamlit startup commands were also attempted; missing TensorFlow/Streamlit dependencies prevent local execution. No local empirical CNN metrics are claimed.

The Word report was generated with python-docx using the bundled document runtime. The supplied renderer could not find bundled LibreOffice, and Word COM access was denied by the environment. Therefore local page-layout rendering has not been verified. The repository workflow includes independent report rendering when its execution environment permits.

## Repository validation

The `Validate and prepare demonstration` GitHub workflow is configured to install the complete pinned environment, run Pytest and verify a real Streamlit health endpoint. On an initial main-branch run with no model, it downloads/validates the resized dataset, trains ImageNet MobileNetV2, evaluates the held-out split, runs the suite again, exercises real classification in a browser and creates actual screenshots and an updated report. These are configured checks; their execution status must be verified on the workflow run before they are claimed as completed.

Actual training metrics, when available, are recorded in `results/metrics.json` with the exact model and split hashes. Dataset archive size and source are in `results/dataset.json`. The report builder incorporates only measured artifacts whose model hash matches the checkpoint.

## Actual training and test evidence

Run `44157444636` downloaded and inspected the resized TrashNet archive: **42,834,870 compressed bytes** and 43,288,847 uncompressed bytes. The six original folders contain 2,527 images. After removing exact duplicates and all members of conflicting-label duplicate groups, **2,521 unique unambiguous images** remain. All exclusions are recorded in the actual dataset metadata.

The seeded 160-pixel MobileNetV2 head trained for 12 epochs with a frozen ImageNet base, batch size 32 and class weights. Validation loss selected the checkpoint; the held-out test set contains **379 images**. Actual metrics are accuracy **0.8153034300791556**, macro precision **0.7772985522055932**, macro recall **0.7821352743452845** and macro F1 **0.7769867768489016**.

The full post-training Pytest suite completed with **34 passed, 2 xfailed and 6 subtests passed**, with no dependency or model skips. All five Streamlit sections, navigation, laboratory interaction, session history and the real model's load/hash/inference checks passed. The Streamlit server health endpoint also passed in CI.

The first archive validation correctly halted on contradictory image labels. A dedicated regression now verifies that every member of an ambiguous group is excluded without inventing a preferred label. A later browser-capture step failed because pinned Playwright did not support `ubuntu-latest`, which had moved to Ubuntu 26.04. The trained and tested model was preserved as an Actions artifact. Recovery uses that checkpoint on an explicitly pinned Ubuntu 24.04 runner; it does not repeat training or alter the held-out metrics.

Local Git object writes are also restricted. Commits and ordinary fast-forward updates are made through the connected GitHub integration. No force push, repository visibility change, credential file or raw dataset is committed.

## Browser and document verification

Run `44161441051` completed successfully on Ubuntu 24.04. It recovered the already-trained checkpoint without retraining, passed all post-training tests again, uploaded a genuine held-out image through the running Streamlit UI, obtained a real prediction, changed contamination using the slider, and captured the classifier, populated dashboard, fuzzy laboratory and evaluation page. The generated Word report incorporated the measured metrics and real screenshots and rendered successfully to PDF and page images for layout review.

The first browser interaction attempted the End key, which the slider did not implement. The corrected check uses the supported arrow keys and passed. Page review identified an inherited decorative title border and a sparse final reference page; the report generator removes the border and starts references on a new page. Model evaluation remains unchanged.

## Final delivery verification

Run `44164749880` completed successfully with the real model and measured evaluation artifacts. A checkpoint-restoration check removed the local CI copy, downloaded the published **9,718,514-byte** Keras file, verified SHA-256 and compared its bytes with the original. Pytest reported **34 passed, 2 xfailed and 6 subtests passed**. There were no model or dependency skips. The actual Streamlit health check passed.

The browser capture now waits for Streamlit's documented `notRunning` state and for loading skeletons to disappear. Real predictions, manual contamination changes, the populated dashboard, laboratory charts and evaluation page rendered successfully. All **19 pages** of the final Word report were visually checked: the first 16 matched the already-reviewed rendered pages byte-for-byte, and the three updated screenshot pages were reviewed separately. Title borders, tables, figures, references and screenshots are readable without document-layout clipping.

The final runtime pins Keras 3.15.1, matching the environment used for the supplied model, as well as TensorFlow 2.20. The one-time artifact recovery and report-refresh configuration is removed. The published model's SHA-256 is `753e4de8261cef6ea0dbaa19e018471f5638454a0e5772dba00262b58baeb6da`. The complete GitHub repository includes the checkpoint. `setup.bat` can restore and verify it if a working copy omits the binary.

## Windows workspace delivery

On 18 November 2026, the exact trained checkpoint was also restored to the existing Windows working folder as `models/smartwaste.keras`. Its 9,718,514-byte size and SHA-256 match the published training manifest. The Keras archive passed its ZIP integrity check; embedded metadata confirms Keras 3.15.1 and the saved checkpoint timestamp. Temporary transfer files were kept outside the main project history, and the transfer workflow and branch were removed after verified receipt.

The remaining local environment limitation is package installation: the isolated virtual environment exists, but this session cannot resolve the package-index host, so TensorFlow, Streamlit, Plotly and Pytest cannot be installed here. The model and full application have already passed inference, UI and startup checks in GitHub CI. Run `setup.bat` and then `run.bat` from a normal Windows terminal with internet access to complete installation and launch locally.

## UI refinement validation — 18 November 2026

The clipped page eyebrow was caused by custom main-container top padding of 2.2rem, reduced to 1rem on small screens, overriding Streamlit's native 6rem toolbar clearance. Both overrides are removed. Shared page headers now use explicit line heights without fixed heights or hidden overflow. Sidebar branding, active navigation, cards, charts, the classifier empty state, image/result panels and the methodology layout are refined consistently.

[Windows browser validation](https://github.com/prxns/SmartWaste-AI/actions/runs/44239270551) passed setup.bat, dependency checks, the complete suite and the real run.bat browser flow. Pytest reported **34 passed, 2 xfailed and 6 subtests passed**. [Linux validation](https://github.com/prxns/SmartWaste-AI/actions/runs/44239270812) also passed the suite, checkpoint restoration and Streamlit startup.

All five pages were tested at **1920×1080, 1366×768 and 1280×720**, with device scale 1 and normal browser zoom. Fifteen page/viewport combinations and three classified-result layouts passed checks for horizontal overflow, toolbar/header clearance and sidebar clipping. Seven additional overview checks passed. Actual screenshots are in `assets/screenshots/`; the full responsive captures are retained with the Windows run.

The browser uploaded a real held-out image, used the shipped CNN, changed contamination to 100, confirmed the displayed value and Low suitability, and downloaded a CSV containing the matching session record. Each keyboard change waits for Streamlit's rendered update. The existing unit suite also verifies laboratory interaction and session-history clearing.

Checksums confirm that the trained model, its manifest and metrics, model loader module, fuzzy engine, configuration, existing expected-failure tests, requirements and Windows launch scripts are unchanged. The two prescribed fuzzy monotonicity conflicts remain documented. No retraining, runtime dependency, database, desktop packaging or alternate frontend was introduced.
