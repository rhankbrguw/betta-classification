import logging
import sys
from pathlib import Path

# Configuration
DEFAULT_TARGETS: list[Path] = [Path("dataset"), Path("dataset_clean")]
SUPPORTED_EXTS: set[str]    = {".jpg", ".jpeg", ".png", ".webp"}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Core
def rename_class(class_dir: Path) -> int:
    """
    Rename all image files in a class directory to zero-padded sequential names.

    Uses a two-pass rename (original → temp → final) to avoid collisions
    when new names overlap with existing filenames in the same directory.

    Args:
        class_dir: Path to the class subdirectory.

    Returns:
        Number of files successfully renamed.
    """
    images = sorted(
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )

    if not images:
        logger.warning("[%s] No supported images found — skipping.", class_dir.name)
        return 0

    # Pass 1: temp names to prevent mid-rename collisions
    temp_files: list[tuple[Path, str]] = []
    for idx, img in enumerate(images):
        temp_path = class_dir / f"_tmp_{idx:06d}{img.suffix.lower()}"
        img.rename(temp_path)
        temp_files.append((temp_path, img.suffix.lower()))

    # Pass 2: final zero-padded sequential names
    for final_idx, (temp_path, suffix) in enumerate(temp_files, start=1):
        temp_path.rename(class_dir / f"{final_idx:06d}{suffix}")

    return len(temp_files)


def process_dataset(dataset_dir: Path) -> int:
    """
    Rename all class subdirectories within a dataset directory.

    Args:
        dataset_dir: Root dataset directory containing class subdirectories.

    Returns:
        Total number of files renamed across all classes.
    """
    if not dataset_dir.exists():
        logger.warning("Directory '%s' not found — skipping.", dataset_dir)
        return 0

    class_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    if not class_dirs:
        logger.warning("No class subdirectories found in '%s'.", dataset_dir)
        return 0

    logger.info("[%s] Found %d class(es).", dataset_dir.name, len(class_dirs))
    total = 0
    for class_dir in class_dirs:
        count = rename_class(class_dir)
        logger.info("  [%s/%s] %d file(s) renamed.", dataset_dir.name, class_dir.name, count)
        total += count

    return total


def main() -> None:
    # Accept explicit targets from CLI args, fall back to defaults
    targets = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_TARGETS

    logger.info("Targets: %s", [str(t) for t in targets])

    grand_total = 0
    for target in targets:
        grand_total += process_dataset(target)

    logger.info("Rename complete. Total files processed: %d.", grand_total)


if __name__ == "__main__":
    main()
