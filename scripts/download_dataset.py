"""Download only the resized archive, inspect, validate, then install safely."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile, BadZipFile
from config import DATA_DIR, DATASET_URL, RESULTS_DIR
from smartwaste.dataset import inventory

MAX_ARCHIVE_BYTES = 150 * 1024 * 1024
MAX_EXTRACTED_BYTES = 500 * 1024 * 1024


def safe_extract(archive, destination):
    """Validate ALL members before creating ANY files; reject Windows path tricks."""
    destination = Path(destination).resolve()
    with ZipFile(archive) as zipped:
        members = zipped.infolist()
        if not members or len(members) > 10000:
            raise ValueError("Archive has an unreasonable number of entries.")
        if sum(m.file_size for m in members) > MAX_EXTRACTED_BYTES:
            raise ValueError("Archive exceeds the extracted-size limit.")
        targets = set()
        for member in members:
            # ZipInfo normalizes backslashes on Windows; inspect the original
            # archive name so unsafe entries cannot hide behind normalization.
            name = member.orig_filename
            path = PurePosixPath(name)
            mode = member.external_attr >> 16
            if ("\\" in name or ":" in name or path.is_absolute() or ".." in path.parts
                    or stat.S_ISLNK(mode) or member.flag_bits & 1):
                raise ValueError(f"Unsafe ZIP entry: {name}")
            if any(p.endswith((".", " ")) or p.split(".")[0].upper() in
                   {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))} for p in path.parts):
                raise ValueError("Unsafe Windows ZIP filename.")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or str(target).casefold() in targets:
                raise ValueError("Duplicate or escaping ZIP path.")
            targets.add(str(target).casefold())
        print(f"Archive inspected: {len(members)} entries, {sum(m.file_size for m in members):,} uncompressed bytes")
        for member in members:
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded resized ZIP.")
    parser.add_argument("--destination", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    if args.destination.exists():
        frame = inventory(args.destination)
        print("Existing dataset validated:", frame.category.value_counts().sort_index().to_dict())
        return
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.destination.parent, prefix="trashnet-") as temporary:
        temporary = Path(temporary)
        archive = args.archive
        if archive is None:
            archive = temporary / "dataset-resized.zip"
            with urlopen(DATASET_URL, timeout=60) as response, archive.open("wb") as output:
                count = 0
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > MAX_ARCHIVE_BYTES:
                        raise ValueError("Download exceeds the resized archive limit.")
                    output.write(chunk)
        if archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive is too large. Use dataset-resized.zip only.")
        import hashlib
        archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
        extracted = temporary / "extracted"
        safe_extract(archive, extracted)
        dataset = extracted / "dataset-resized"
        frame = inventory(dataset)
        # Install only after every image and category has passed validation.
        shutil.move(str(dataset), str(args.destination))
        RESULTS_DIR.mkdir(exist_ok=True)
        metadata = {"source": DATASET_URL, "archive_bytes": archive.stat().st_size,
                    "archive_sha256": archive_hash, "unique_images": len(frame),
                    "class_distribution": frame.category.value_counts().sort_index().to_dict(),
                    "raw_class_distribution": frame.attrs["raw_class_distribution"],
                    "excluded_images": frame.attrs["excluded_images"],
                    "attribution": "Gary Thung and Mindy Yang, TrashNet"}
        (RESULTS_DIR / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, BadZipFile) as error:
        raise SystemExit(f"Dataset preparation failed: {error}")
