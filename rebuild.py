"""Rebuild the exact Final archive locally using the Python standard library."""
import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BLOCK_SIZE = 4 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New ZIP file to create")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        candidate = Path(scratch) / "candidate.zip"
        with zipfile.ZipFile(candidate, "x") as archive:
            for name, spec in sorted(manifest["members"].items()):
                source = ROOT / "submission" / name
                if not source.is_file() or source.stat().st_size != spec["bytes"] or sha256(source) != spec["sha256"]:
                    parser.exit(1, f"Missing or changed archive member: {name}\n")
                info = zipfile.ZipInfo(name, date_time=(2026, 11, 19, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                with source.open("rb") as src, archive.open(info, "w") as dst:
                    shutil.copyfileobj(src, dst, BLOCK_SIZE)
        digest = sha256(candidate)
        if digest != manifest["sha256"] or candidate.stat().st_size != manifest["bytes"]:
            parser.exit(1, "Rebuilt archive differs from the recorded Final artifact.\n")
        try:
            args.output.hardlink_to(candidate)
        except FileExistsError:
            parser.exit(1, f"Output already exists: {args.output}\n")
        print(f"{digest}  {args.output.name}")


if __name__ == "__main__":
    main()
