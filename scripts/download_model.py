"""Restore the real published checkpoint, with a strict SHA-256 check."""
import hashlib
import json
import os
import tempfile
from urllib.request import urlopen
from config import MODEL_DIR, MODEL_PATH

MODEL_REF = os.environ.get("SMARTWASTE_MODEL_REF", "main")
MODEL_URL = f"https://raw.githubusercontent.com/prxns/SmartWaste-AI/{MODEL_REF}/models/smartwaste.keras"


def main():
    manifest = MODEL_DIR / "model_manifest.json"
    if not manifest.is_file():
        raise ValueError("Model manifest is missing. Download the complete repository or train the model.")
    expected = json.loads(manifest.read_text(encoding="utf-8"))["model_sha256"]
    MODEL_DIR.mkdir(exist_ok=True)
    if MODEL_PATH.is_file():
        if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != expected:
            raise ValueError("Existing checkpoint does not match its manifest. Preserve it and check your repository version.")
        print("Published checkpoint is already present and verified.")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=MODEL_DIR, suffix=".tmp", delete=False) as output:
            temporary = output.name
            digest = hashlib.sha256()
            total = 0
            with urlopen(MODEL_URL, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 50 * 1024 * 1024:
                        raise ValueError("Checkpoint exceeds the 50 MiB limit.")
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Downloaded checkpoint differs from the manifest. Update the complete repository and retry.")
        os.replace(temporary, MODEL_PATH)
        temporary = None
        print(f"Downloaded and SHA-256 verified the real trained checkpoint ({total:,} bytes).")
    finally:
        if temporary is not None:
            os.unlink(temporary)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Checkpoint download failed: {error}")
