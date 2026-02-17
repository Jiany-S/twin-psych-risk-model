"""Download WESAD with kagglehub and copy a local subject subset."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DEFAULT_SUBJECTS = [
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
]
KAGGLE_DATASET = "orvile/wesad-wearable-stress-affect-detection-dataset"


def _normalize_subject_id(text: str) -> str:
    value = str(text).strip()
    if value.lower().startswith("s"):
        value = value[1:]
    if not value.isdigit():
        raise ValueError(f"Invalid subject id '{text}'. Expected format like S2.")
    return f"S{int(value)}"


def _discover_subject_dirs(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    pattern = re.compile(r"^S(\d+)$", re.IGNORECASE)
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        subject_id = f"S{int(match.group(1))}"
        has_payload = any(path.glob("*.pkl")) or any(path.glob("*.csv"))
        if has_payload and subject_id not in found:
            found[subject_id] = path
    return found


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a local WESAD subject subset.")
    parser.add_argument(
        "--subjects",
        type=str,
        default=",".join(DEFAULT_SUBJECTS),
        help="Comma-separated subject IDs, e.g. S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S13,S14,S15,S16,S17",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/wesad_subset",
        help="Output directory for copied subset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    subjects = [_normalize_subject_id(x) for x in args.subjects.split(",") if x.strip()]

    output_dir = Path(args.output)
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Re-run with --force to overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download WESAD. Install dependencies from requirements.txt."
        ) from exc

    source_root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    subject_dirs = _discover_subject_dirs(source_root)
    if not subject_dirs:
        raise FileNotFoundError(
            f"No subject folders found under downloaded dataset path: {source_root}"
        )

    missing = [s for s in subjects if s not in subject_dirs]
    if missing:
        available = ", ".join(sorted(subject_dirs.keys()))
        raise ValueError(
            f"Requested subjects not found: {missing}. Available subjects: {available}"
        )

    copied: list[str] = []
    for subject in subjects:
        src = subject_dirs[subject]
        dst = output_dir / subject
        shutil.copytree(src, dst)
        copied.append(subject)

    print(f"WESAD subset created at: {output_dir.resolve()}")
    print(f"Subjects copied ({len(copied)}): {', '.join(copied)}")


if __name__ == "__main__":
    main()
