#!/usr/bin/env python3
"""Download a pinned KiTS23 ZIP and extract only the fixed evaluation assets."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/source.json"
MANIFEST = ROOT / "manifests/cases_200.jsonl"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_inputs(source_path=SOURCE, manifest_path=MANIFEST):
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    manifest_bytes = Path(manifest_path).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != source["cases_manifest_sha256"]:
        raise ValueError("Fixed cases manifest SHA-256 mismatch")
    cases = [json.loads(line) for line in manifest_bytes.decode().splitlines() if line.strip()]
    if not cases or len({row["sample_id"] for row in cases}) != len(cases):
        raise ValueError("Empty manifest or duplicate sample IDs")
    assets = {}

    def add_asset(member, checksum, folder):
        path = PurePosixPath(member)
        if (len(path.parts) != 3 or path.parts[:2] != ("kits23", folder)
                or path.suffix != ".png" or "\\" in member or ".." in path.parts
                or not re.fullmatch(r"[0-9a-f]{64}", checksum)):
            raise ValueError("Invalid ZIP member or checksum in manifest")
        if member in assets and assets[member] != checksum:
            raise ValueError("Conflicting asset checksums")
        assets[member] = checksum

    for row in cases:
        if row["split"] != "train":
            raise ValueError("This fixed candidate manifest must contain train only")
        add_asset(row["image"], row["image_sha256"], "train")
        for target in [*row["targets"], *([row["kidney"]] if row.get("kidney") else [])]:
            add_asset(target["mask"], target["mask_sha256"], "train_mask")
    return source, cases, assets


def destination(root, member):
    root = Path(root).resolve()
    target = root / member
    # Refuse symlinks out of the requested data directory, including existing files.
    if not target.resolve().is_relative_to(root):
        raise ValueError("Asset path escapes extraction directory")
    return target


def missing_assets(extracted, assets):
    return [member for member, checksum in assets.items()
            if not destination(extracted, member).is_file()
            or sha256_file(destination(extracted, member)) != checksum]


def download_archive(source, directory):
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError as error:
        raise ValueError("Download dependency missing. Run: bash setup.sh") from error
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {source['repo_id']}/{source['filename']} "
          f"({source['size_bytes'] / 1e9:.2f} GB, resumable).", flush=True)
    try:
        return Path(hf_hub_download(
            repo_id=source["repo_id"], repo_type=source["repo_type"],
            filename=source["filename"], revision=source["revision"],
            local_dir=directory,
        ))
    except GatedRepoError:
        raise ValueError("KiTS23 source requires access: open " + source["url"]
                         + ", accept/request access, then set HF_TOKEN for that account.") from None
    except HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        raise ValueError(f"Hugging Face HTTP {status}; check dataset access, HF_TOKEN and network.") from None
    except Exception:
        # Do not copy signed CDN URLs or credentials into logs.
        raise ValueError("Download failed. Check network/proxy/HF_TOKEN and rerun; "
                         "Hugging Face keeps partial downloads for resumption.") from None


def prepare(data_dir, archive=None, verify_only=False, source_path=SOURCE, manifest_path=MANIFEST):
    source, cases, assets = load_inputs(source_path, manifest_path)
    data_dir = Path(data_dir).resolve()
    extracted = data_dir / "extracted"
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_assets(extracted, assets)
    archive_verified_now = False
    if missing and verify_only:
        raise ValueError(f"{len(missing)} missing/corrupt assets; run data preparation first")
    if missing:
        archive = Path(archive).resolve() if archive else download_archive(source, data_dir / "downloads")
        if not archive.is_file() or archive.stat().st_size != source["size_bytes"]:
            raise ValueError("Archive missing or wrong size; keep partial download and rerun, "
                             "or supply the complete original ZIP with --archive")
        print("Checking archive SHA-256 (reads the complete ZIP once)...", flush=True)
        if sha256_file(archive) != source["sha256"]:
            raise ValueError("Archive SHA-256 mismatch; refusing to extract")
        archive_verified_now = True
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
            if len(names) != len(set(names)):
                raise ValueError("ZIP contains duplicate member names")
            for index, member in enumerate(missing, 1):
                # Read only allowlisted members; never extractall a remote archive.
                info = zipped.getinfo(member)
                if info.file_size > 32 * 1024 * 1024:
                    raise ValueError("Unexpectedly large PNG in ZIP")
                raw = zipped.read(member)
                if (not raw.startswith(b"\x89PNG\r\n\x1a\n")
                        or hashlib.sha256(raw).hexdigest() != assets[member]):
                    raise ValueError("PNG SHA-256 mismatch: " + member)
                target = destination(extracted, member)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".png.tmp")
                temporary.write_bytes(raw)
                os.replace(temporary, target)
                if index % 100 == 0 or index == len(missing):
                    print(f"Extracted {index}/{len(missing)} required assets", flush=True)
    # Every asset has now been freshly verified (existing file or ZIP bytes).
    prepared = []
    for case in cases:
        row = json.loads(json.dumps(case))
        row["image"] = "extracted/" + case["image"]
        for target in [*row["targets"], *([row["kidney"]] if row.get("kidney") else [])]:
            target["mask"] = str(destination(extracted, target["mask"]))
        prepared.append(row)
    jobs = data_dir / "cases.jsonl"
    temporary = jobs.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared), encoding="utf-8")
    os.replace(temporary, jobs)
    metadata = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source": source, "source_lock_sha256": sha256_file(source_path),
        "prepared_cases_sha256": sha256_file(jobs),
        "num_slices": len(cases), "num_case_keys": len({row['case_id'] for row in cases}),
        "groups": dict(Counter(row["group"] for row in cases)),
        "assets_verified": len(assets), "archive_verified_this_invocation": archive_verified_now,
        "script_sha256": sha256_file(__file__),
    }
    write_json(data_dir / "prepared.json", metadata)
    print(f"Ready: {len(cases)} slices, {len(assets)} verified assets; {jobs}", flush=True)
    return jobs, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--archive", type=Path, help="Use an existing original kits23.zip; no network")
    parser.add_argument("--verify-only", action="store_true", help="Validate prepared files without downloading")
    args = parser.parse_args()
    prepare(args.data_dir, args.archive, args.verify_only)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"Data preparation error: {error}", file=sys.stderr)
        sys.exit(2)
