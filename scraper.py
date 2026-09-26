import logging
from pathlib import Path

import imagehash
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from PIL import Image, UnidentifiedImageError

# Configuration
DATASET_ROOT: Path = Path("dataset")

# Keywords 
CATEGORIES: dict[str, list[str]] = {
    "Double_Tail": [
        # Target: Double Tail HalfMoon (DTHM) 
        "double tail halfmoon betta fish lateral view",
        "DTHM betta fish side profile show quality",
        "double tail halfmoon betta full body photography",
        "betta splendens DTHM 180 degree caudal spread lateral",
        "double tail halfmoon betta fish white background",
        "DTHM betta split caudal peduncle side view",
        "ikan cupang double tail halfmoon tampak samping",
    ],
    "Plakat": [
        # Target: standard Plakat 
        "plakat betta fish side view short fin",
        "plakat betta splendens male full body lateral",
        "betta plakat traditional short tail side profile",
        "plakat betta fish photography lateral pose",
        "wild type plakat betta fish side view",
        "plakat betta fighting fish full body lateral",
        "ikan cupang plakat ekor pendek tampak samping",
    ],
    "Veiltail": [
        # Target: standard Veiltail 
        "veiltail betta fish side view drooping tail",
        "veil tail betta splendens full body lateral",
        "veiltail betta fish long flowing caudal side profile",
        "betta splendens veiltail male lateral photography",
        "veiltail betta fish full body white background",
        "common veiltail betta fish side view",
        "ikan cupang veiltail ekor panjang tampak samping",
    ],
}

CRAWL_PER_KEYWORD: int   = 150
MIN_SHORT_SIDE: int       = 224   # Resolution floor matches MobileNetV3 input size
PHASH_THRESHOLD: int      = 8     # Hamming distance; <8 treated as duplicate
SUPPORTED_EXTS: set[str]  = {".jpg", ".jpeg", ".png", ".webp"}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Filters
def compute_phash(path: Path) -> imagehash.ImageHash | None:
    """Compute perceptual hash for near-duplicate detection."""
    try:
        with Image.open(path) as img:
            return imagehash.phash(img.convert("RGB"))
    except (UnidentifiedImageError, OSError):
        return None


def is_below_resolution_floor(path: Path) -> bool:
    """Return True if the image's shortest side is below MIN_SHORT_SIDE."""
    try:
        with Image.open(path) as img:
            return min(img.size) < MIN_SHORT_SIDE
    except (UnidentifiedImageError, OSError):
        return True  # Corrupt file — treat as invalid


def deduplicate(folder: Path) -> tuple[int, int]:
    """
    Remove low-resolution and near-duplicate images from a directory.

    Returns:
        (removed, kept): count of deleted and retained files.
    """
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    seen_hashes: list[imagehash.ImageHash] = []
    removed = 0

    for file in files:
        if is_below_resolution_floor(file):
            file.unlink()
            removed += 1
            continue

        h = compute_phash(file)
        if h is None:
            file.unlink()
            removed += 1
            continue

        if any(abs(h - seen) < PHASH_THRESHOLD for seen in seen_hashes):
            file.unlink()
            removed += 1
        else:
            seen_hashes.append(h)

    return removed, len(seen_hashes)

# Crawling
def crawl_keyword(keyword: str, dest: Path, max_num: int) -> None:
    """Crawl a single keyword across Bing and Google."""
    engines = [("Bing", BingImageCrawler), ("Google", GoogleImageCrawler)]
    for engine_name, crawler_cls in engines:
        logger.info("  [%s] %r", engine_name, keyword)
        try:
            crawler = crawler_cls(
                storage={"root_dir": str(dest)},
                log_level=logging.WARNING,
            )
            crawler.crawl(keyword=keyword, max_num=max_num, file_idx_offset="auto")
        except Exception as exc:
            logger.warning("  [%s] Crawl failed: %s", engine_name, exc)


def main() -> None:
    logger.info("Starting data acquisition pipeline")

    results: dict[str, int] = {}

    for label, keywords in CATEGORIES.items():
        dest = DATASET_ROOT / label
        dest.mkdir(parents=True, exist_ok=True)

        logger.info("[%s] — %d keywords", label, len(keywords))
        for keyword in keywords:
            crawl_keyword(keyword, dest, CRAWL_PER_KEYWORD)

        removed, kept = deduplicate(dest)
        logger.info("[%s] Dedup: -%d removed, %d retained", label, removed, kept)
        results[label] = kept

    total = sum(results.values())
    col_w = max(len(k) for k in results)
    print("\n── Dataset Summary " + "─" * 30)
    for label, count in results.items():
        warning = "  ⚠ insufficient" if count < 150 else ""
        print(f"  {label:<{col_w}}  {count:>4} images{warning}")
    print(f"  {'Total':<{col_w}}  {total:>4} images")
    print("─" * 49)
    print("Next step: run remove_bg.py for feature isolation.")


if __name__ == "__main__":
    main()
