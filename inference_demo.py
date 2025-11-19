# inference_demo.py
import argparse
from pathlib import Path

from weights.loader import load_yolov11_model


# Available YOLOv11 variants used in this repository.
# The larger YOLOv11-X model is not included here because the corresponding
# weight file exceeds the public GitHub file size limit. It can be shared
# separately upon reasonable request to the corresponding authors.
VARIANTS = ["n", "s", "m", "l"]

# Supported image file extensions for test images.
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_test_image(project_root: Path, test_id: int) -> Path:
    """
    Locate a test image named `test_<id>.<ext>` inside the `test_images` folder.

    The function searches for files whose stem matches `test_<id>` and whose
    extension is one of ALLOWED_EXTS. If multiple files share the same stem
    with different extensions, the first one in sorted order is used.
    """
    test_dir = project_root / "test_images"
    if not test_dir.exists():
        raise FileNotFoundError(f"test_images directory not found: {test_dir}")

    prefix = f"test_{test_id}"
    candidates = [
        p
        for p in test_dir.iterdir()
        if p.is_file()
        and p.stem == prefix
        and p.suffix.lower() in ALLOWED_EXTS
    ]

    if not candidates:
        # If files with the correct stem exist but use unsupported extensions,
        # raise a more informative error to help debugging.
        raw_candidates = [p for p in test_dir.iterdir() if p.is_file() and p.stem == prefix]
        if raw_candidates:
            raise FileNotFoundError(
                f"Found files with name '{prefix}' but unsupported extension. "
                f"Supported: {', '.join(sorted(ALLOWED_EXTS))}"
            )
        raise FileNotFoundError(
            f"No image found for pattern '{prefix}.<ext>' in {test_dir}. "
            f"Supported extensions: {', '.join(sorted(ALLOWED_EXTS))}"
        )

    # If multiple candidates exist (e.g., test_1.jpg and test_1.png),
    # use the first one in lexicographic order to keep the behavior deterministic.
    return sorted(candidates)[0]


def run_inference(test_id: int, device: str = "cuda"):
    """
    Run inference for all available YOLOv11 variants on a single test image.

    For an input image `test_<id>.<ext>` in `test_images/`,
    this function runs each model variant in VARIANTS and saves the
    visualization outputs to:

        results/test_<id>_<variant>.<ext>
    """
    project_root = Path(__file__).resolve().parent

    # Locate the input test image.
    image_path = find_test_image(project_root, test_id)
    print(f"Input image: {image_path}")

    # Output directory for result images.
    results_root = project_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    for v in VARIANTS:
        print(f"\n[Variant {v}] Loading model...")
        model = load_yolov11_model(variant=v, device=device)
        print(f"[Variant {v}] Model loaded. Running inference...")

        # Output file name: test_<id>_<variant>.<original_extension>
        save_name = f"test_{test_id}_{v}{image_path.suffix}"
        save_path = results_root / save_name

        # Run YOLO inference.
        results = model(str(image_path))

        # Save the image with segmentation masks overlaid.
        results[0].plot(
            save=True,
            filename=str(save_path),
            line_width=0,
            font_size=2,
            boxes=True,
            conf=False,
            probs=False,
        )

        print(f"[Variant {v}] Saved output to: {save_path}")

    print("\nAll variants finished. Check the 'results' folder.")


def main():
    parser = argparse.ArgumentParser(
        description="SpineEndo-YOLOv11: run all model variants on a test image."
    )
    parser.add_argument(
        "--test_id",
        type=int,
        default=1,
        help="Test image index (1–4). Uses test_images/test_<id>.<ext> (default: 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference: 'cuda' or 'cpu' (default: 'cuda').",
    )

    args = parser.parse_args()

    if not 1 <= args.test_id <= 4:
        raise ValueError("test_id must be between 1 and 4 (inclusive).")

    run_inference(test_id=args.test_id, device=args.device)


if __name__ == "__main__":
    main()
