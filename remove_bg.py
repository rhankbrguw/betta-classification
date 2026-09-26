import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from rembg import new_session, remove
from tqdm import tqdm

# Configuration
INPUT_DIR: Path  = Path("dataset")
OUTPUT_DIR: Path = Path("dataset_clean")
REMBG_MODEL: str = "u2net"           # Explicit model; avoid implicit default drift
SUPPORTED_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Processing
def process_image(input_path: Path, output_path: Path, session) -> bool:
    """
    Remove background and composite subject onto a black RGB canvas.

    The black canvas (pixel value 0) ensures zero gradient contribution from
    background regions during convolution, preventing texture-based shortcuts
    in the classifier.

    Args:
        input_path:  Source image path.
        output_path: Destination path for the processed PNG.
        session:     Pre-initialized rembg ONNX inference session.

    Returns:
        True if processing succeeded, False on any failure.
    """
    try:
        with Image.open(input_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            subject_rgba: Image.Image = remove(img, session=session)

            black_canvas = Image.new("RGB", subject_rgba.size, (0, 0, 0))

            if subject_rgba.mode == "RGBA":
                alpha_mask = subject_rgba.split()[3]
                black_canvas.paste(subject_rgba, mask=alpha_mask)
            else:
                black_canvas.paste(subject_rgba)

            # PNG — lossless; preserves sharp fin edges critical for morphology
            black_canvas.save(output_path, format="PNG")
            return True

    except UnidentifiedImageError:
        logger.error("Corrupt image dropped: %s", input_path.name)
        return False
    except Exception as exc:
        logger.error("Inference failed on %s: %s", input_path.name, exc)
        return False


def main() -> None:
    if not INPUT_DIR.exists():
        logger.critical("Input directory '%s' not found. Aborting.", INPUT_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    logger.info("Found %d class(es) to process.", len(class_dirs))

    # Initialize ONNX session once — eliminates per-image model load overhead
    logger.info("Initializing %s ONNX session...", REMBG_MODEL)
    session = new_session(REMBG_MODEL)

    for cls_dir in sorted(class_dirs):
        out_cls_dir = OUTPUT_DIR / cls_dir.name
        out_cls_dir.mkdir(exist_ok=True)

        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
        logger.info("[%s] Processing %d image(s).", cls_dir.name, len(images))

        success = 0
        for img_path in tqdm(images, desc=cls_dir.name, unit="img"):
            out_path = out_cls_dir / f"{img_path.stem}.png"

            if out_path.exists():
                # Idempotency: skip already-processed files on re-runs
                success += 1
                continue

            if process_image(img_path, out_path, session):
                success += 1

        logger.info(
            "[%s] Done: %d/%d retained.", cls_dir.name, success, len(images)
        )

    logger.info("Pipeline complete. Next step: run rename.py then proceed to Colab.")


if __name__ == "__main__":
    main()
