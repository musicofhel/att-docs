#!/usr/bin/env python3
"""Download helper for Nie/Katyal/Engel (2023) binocular rivalry SSVEP dataset.

Dataset: "An Accumulating Neural Signal Underlying Binocular Rivalry Dynamics"
    84 subjects, 34-channel EEG at 360 Hz (preprocessed), 12 x 120s rivalry epochs
    DOI: 10.13020/9sy5-a716
    Repository: https://conservancy.umn.edu/handle/11299/257166
    License: CC BY-NC 3.0 US

The dataset is hosted as a multi-part zip archive on the UMN Data Repository (DRUM):
    eegdata.zip   (~4.0 GB)
    eegdata.z01   (~4.0 GB)
    eegdata.z02   (~3.66 GB)
    paper_figures.zip

All three eegdata parts must be downloaded and placed in the same directory.
Then extract with: unzip eegdata.zip (which reads the .z01/.z02 parts automatically).

IMPORTANT: The UMN DRUM repository does not provide a stable direct-download API.
Files must be downloaded via the web interface or by discovering the bitstream URLs
from the repository page. The placeholder URLs below need to be replaced with actual
bitstream URLs obtained from:
    https://conservancy.umn.edu/handle/11299/257166

To find the real URLs:
    1. Visit the repository page above
    2. Click on each file (eegdata.zip, eegdata.z01, eegdata.z02)
    3. Copy the download URL (typically of the form:
       https://conservancy.umn.edu/bitstreams/<uuid>/download)
    4. Replace the PLACEHOLDER entries below

Alternatively, download manually through a browser and place the files in
the output directory, then run this script with --extract-only.

Usage:
    python scripts/download_eeg.py --output data/eeg/rivalry_ssvep
    python scripts/download_eeg.py --output data/eeg/rivalry_ssvep --extract-only
    python scripts/download_eeg.py --output data/eeg/rivalry_ssvep --subjects 1-5
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download manifest
# ---------------------------------------------------------------------------
# PLACEHOLDER URLs -- replace with actual bitstream UUIDs from the DRUM page.
# To get the real URLs:
#   1. Go to https://conservancy.umn.edu/handle/11299/257166
#   2. Click each file's download link
#   3. The redirect will reveal a URL like:
#      https://conservancy.umn.edu/bitstreams/<uuid>/download
#
# The MD5 checksums below are also placeholders. After downloading once,
# compute them with: md5sum eegdata.zip eegdata.z01 eegdata.z02
# and update here for future verification.

DOWNLOAD_MANIFEST = [
    {
        "filename": "eegdata.zip",
        "url": "PLACEHOLDER_URL_eegdata_zip",
        "md5": "PLACEHOLDER_MD5",
        "size_gb": 4.0,
        "description": "Multi-part zip archive (part 1 of 3)",
    },
    {
        "filename": "eegdata.z01",
        "url": "PLACEHOLDER_URL_eegdata_z01",
        "md5": "PLACEHOLDER_MD5",
        "size_gb": 4.0,
        "description": "Multi-part zip archive (part 2 of 3)",
    },
    {
        "filename": "eegdata.z02",
        "url": "PLACEHOLDER_URL_eegdata_z02",
        "md5": "PLACEHOLDER_MD5",
        "size_gb": 3.66,
        "description": "Multi-part zip archive (part 3 of 3)",
    },
]

TOTAL_SIZE_GB = sum(f["size_gb"] for f in DOWNLOAD_MANIFEST)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md5_file(path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, description: str = "") -> None:
    """Download a file with progress reporting."""
    if url.startswith("PLACEHOLDER"):
        log.error(
            "Cannot download %s: URL is a placeholder. "
            "See script header for instructions on obtaining real URLs from "
            "https://conservancy.umn.edu/handle/11299/257166",
            dest.name,
        )
        raise RuntimeError(f"Placeholder URL for {dest.name}")

    desc = f" ({description})" if description else ""
    log.info("Downloading %s%s ...", dest.name, desc)
    log.info("  URL: %s", url)
    log.info("  Dest: %s", dest)

    # Use urllib for simplicity; for large files, wget/curl may be preferable
    try:
        urllib.request.urlretrieve(url, str(dest), _reporthook)
        print()  # newline after progress
    except Exception as e:
        log.error("Download failed for %s: %s", dest.name, e)
        raise


def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    """Progress callback for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  [{bar}] {pct:5.1f}% ({mb:.0f}/{total_mb:.0f} MB)", end="")
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r  Downloaded {mb:.0f} MB", end="")


def verify_checksum(path: Path, expected_md5: str) -> bool:
    """Verify MD5 checksum. Returns True if match or if expected is placeholder."""
    if expected_md5.startswith("PLACEHOLDER"):
        log.warning(
            "Skipping MD5 verification for %s (placeholder checksum). "
            "Compute with: md5sum %s",
            path.name,
            path,
        )
        return True

    log.info("Verifying MD5 for %s ...", path.name)
    actual = md5_file(path)
    if actual == expected_md5:
        log.info("  MD5 OK: %s", actual)
        return True
    else:
        log.error(
            "  MD5 MISMATCH for %s: expected %s, got %s",
            path.name, expected_md5, actual,
        )
        return False


def extract_multipart_zip(zip_dir: Path, output_dir: Path) -> None:
    """Extract the multi-part zip archive.

    The .z01 and .z02 files must be in the same directory as eegdata.zip.
    Standard `unzip` handles multi-part archives when all parts are co-located.
    """
    zip_path = zip_dir / "eegdata.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"eegdata.zip not found in {zip_dir}")

    # Check that all parts are present
    for entry in DOWNLOAD_MANIFEST:
        part = zip_dir / entry["filename"]
        if not part.exists():
            raise FileNotFoundError(
                f"Missing archive part: {part}. "
                f"All three parts (eegdata.zip, .z01, .z02) are required."
            )

    log.info("Extracting multi-part archive to %s ...", output_dir)
    log.info("  This may take several minutes for ~11.66 GB of data.")

    # Use unzip (handles multi-part .zip/.z01/.z02 archives)
    # The -o flag overwrites without prompting
    try:
        subprocess.run(
            ["unzip", "-o", str(zip_path), "-d", str(output_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("Extraction complete.")
    except FileNotFoundError:
        log.error(
            "unzip not found. Install with: sudo apt install unzip "
            "(or use 7z: 7z x %s -o%s)",
            zip_path, output_dir,
        )
        raise
    except subprocess.CalledProcessError as e:
        log.error("Extraction failed: %s", e.stderr)
        raise


def list_subjects(data_dir: Path) -> list[str]:
    """List extracted subject directories."""
    if not data_dir.exists():
        return []
    subjects = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "Epochs").exists():
            subjects.append(p.name)
    return subjects


def parse_subject_range(spec: str, total: int = 84) -> list[int]:
    """Parse subject range specification.

    Examples:
        "all"     -> [1, 2, ..., 84]
        "1-5"     -> [1, 2, 3, 4, 5]
        "1,3,5"   -> [1, 3, 5]
        "10-20"   -> [10, 11, ..., 20]
    """
    if spec.lower() == "all":
        return list(range(1, total + 1))

    subjects = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            subjects.extend(range(int(start), int(end) + 1))
        else:
            subjects.append(int(part))
    return sorted(set(subjects))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Nie/Katyal/Engel (2023) binocular rivalry SSVEP dataset "
            "from the UMN Data Repository."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_eeg.py --output data/eeg/rivalry_ssvep\n"
            "  python scripts/download_eeg.py --output data/eeg/rivalry_ssvep --extract-only\n"
            "  python scripts/download_eeg.py --output data/eeg/rivalry_ssvep --verify-only\n"
            "\n"
            "NOTE: Download URLs are currently placeholders. See script header\n"
            "for instructions on obtaining real URLs from the UMN repository.\n"
            "For manual download, visit:\n"
            "  https://conservancy.umn.edu/handle/11299/257166"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/eeg/rivalry_ssvep"),
        help="Output directory for extracted data (default: data/eeg/rivalry_ssvep)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help=(
            "Directory to store downloaded zip files "
            "(default: <output>/../_downloads)"
        ),
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="all",
        help=(
            'Subject range to keep after extraction. '
            'Examples: "all", "1-5", "1,3,5,10-20". '
            "Note: the full archive must be downloaded regardless; "
            "this flag only controls which subjects are retained."
        ),
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Skip download, just extract existing zip files.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify MD5 checksums of downloaded files.",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep zip files after extraction (default: delete to save space).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_subjects",
        help="List already-extracted subject directories and exit.",
    )

    args = parser.parse_args()
    output_dir = args.output.resolve()

    # List mode
    if args.list_subjects:
        subjects = list_subjects(output_dir)
        if subjects:
            print(f"Found {len(subjects)} subject directories in {output_dir}:")
            for s in subjects:
                print(f"  {s}")
        else:
            print(f"No subject directories found in {output_dir}")
        return

    # Setup directories
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = args.download_dir or (output_dir.parent / "_downloads")
    download_dir.mkdir(parents=True, exist_ok=True)

    # Verify-only mode
    if args.verify_only:
        all_ok = True
        for entry in DOWNLOAD_MANIFEST:
            path = download_dir / entry["filename"]
            if not path.exists():
                log.warning("File not found: %s", path)
                all_ok = False
                continue
            if not verify_checksum(path, entry["md5"]):
                all_ok = False
        if all_ok:
            log.info("All checksums verified.")
        else:
            log.error("Some files missing or checksum mismatch.")
            sys.exit(1)
        return

    # Download phase
    if not args.extract_only:
        log.info("=" * 60)
        log.info("Nie/Katyal/Engel (2023) Binocular Rivalry SSVEP Dataset")
        log.info("Total download size: ~%.1f GB (3 zip parts)", TOTAL_SIZE_GB)
        log.info("Repository: https://conservancy.umn.edu/handle/11299/257166")
        log.info("=" * 60)

        for entry in DOWNLOAD_MANIFEST:
            dest = download_dir / entry["filename"]

            if dest.exists():
                log.info(
                    "%s already exists (%.1f GB), skipping download.",
                    dest.name,
                    dest.stat().st_size / (1024**3),
                )
                verify_checksum(dest, entry["md5"])
                continue

            download_file(entry["url"], dest, entry["description"])
            verify_checksum(dest, entry["md5"])

    # Extraction phase
    log.info("")
    log.info("Extracting archive ...")
    extract_multipart_zip(download_dir, output_dir)

    # List what was extracted
    subjects = list_subjects(output_dir)
    log.info("Extracted %d subject directories.", len(subjects))

    # Filter subjects if requested
    if args.subjects.lower() != "all":
        wanted_indices = parse_subject_range(args.subjects, len(subjects))
        log.info("Retaining subjects: %s", wanted_indices)
        # Subject directories are named like "Sucharit - 012516_3629"
        # We map by sorted order (1-indexed)
        for i, subj_name in enumerate(subjects):
            if (i + 1) not in wanted_indices:
                subj_path = output_dir / subj_name
                log.info("  Removing subject %d: %s", i + 1, subj_name)
                shutil.rmtree(subj_path)

        subjects = list_subjects(output_dir)
        log.info("Retained %d subject directories.", len(subjects))

    # Cleanup zips
    if not args.keep_zips and not args.extract_only:
        log.info("Removing downloaded zip files to save space ...")
        for entry in DOWNLOAD_MANIFEST:
            path = download_dir / entry["filename"]
            if path.exists():
                path.unlink()
                log.info("  Removed %s", path.name)
        # Remove download dir if empty
        if download_dir.exists() and not any(download_dir.iterdir()):
            download_dir.rmdir()

    log.info("")
    log.info("Done. Subject directories are in: %s", output_dir)
    log.info("Run the batch pipeline with:")
    log.info("  python scripts/batch_eeg.py --data-dir %s --output-dir results/batch_eeg", output_dir)


if __name__ == "__main__":
    main()
