"""
SpineEndo-YOLOv11 — real-time instance segmentation inference.

Run the released spine-endoscopy YOLO-V11 segmentation models on the bundled
demo frames, or on YOUR OWN spine-endoscopy images or video.

Examples
--------
# 1) Bundled demo frame (variant l):
python inference_demo.py

# 2) Your own image:
python inference_demo.py --source path/to/your_frame.jpg

# 3) Your own endoscopy VIDEO (an annotated video is written to results/):
python inference_demo.py --source path/to/your_video.mp4

# 4) A folder of images:
python inference_demo.py --source path/to/folder

# 5) A specific variant, or all variants, on CPU:
python inference_demo.py --source your_video.mp4 --variant m --device cpu
python inference_demo.py --source your_video.mp4 --variant all

Model weights for variants n, s, m, l are provided in `weights/` and are loaded
automatically by the bundled loader. The larger YOLO-V11 x model exceeds GitHub's
file-size limit and can be requested from the corresponding author.
"""

import argparse
from pathlib import Path
import importlib.machinery
import importlib.util

VARIANTS = ["n", "s", "m", "l"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def load_loader_module():
    """Load the compiled loader (weights/loader.pyc) that prepares the models."""
    project_root = Path(__file__).resolve().parent
    pyc_path = project_root / "weights" / "loader.pyc"
    if not pyc_path.exists():
        raise FileNotFoundError(f"Compiled loader not found: {pyc_path}")
    module_name = "weights.loader"
    loader = importlib.machinery.SourcelessFileLoader(module_name, str(pyc_path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def resolve_source(project_root: Path, source: str, test_id: int) -> str:
    """Return the inference source: user-provided path, or a bundled demo frame."""
    if source:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"--source not found: {p}")
        return str(p)
    # Fall back to a bundled demo image: test_images/test_<id>.<ext>
    test_dir = project_root / "test_images"
    if not test_dir.exists():
        raise FileNotFoundError(f"test_images directory not found: {test_dir}")
    cands = sorted(
        p for p in test_dir.iterdir()
        if p.is_file() and p.stem == f"test_{test_id}" and p.suffix.lower() in IMAGE_EXTS
    )
    if not cands:
        raise FileNotFoundError(
            f"No demo image 'test_{test_id}.<ext>' in {test_dir}. "
            f"Provide your own input with --source instead."
        )
    return str(cands[0])


def run(source: str, variants, device: str, conf: float):
    """Run inference for the selected variant(s) on `source` (image / video / folder)."""
    project_root = Path(__file__).resolve().parent
    loader_module = load_loader_module()
    results_root = project_root / "results"

    for v in variants:
        print(f"\n[Variant {v}] Loading model ...")
        model = loader_module.load_yolov11_model(variant=v, device=device)
        print(f"[Variant {v}] Running inference on: {source}")
        # Ultralytics handles images, folders, and videos natively.
        # For a video input, an annotated video is written automatically.
        model.predict(
            source=source,
            device=device,
            conf=conf,
            save=True,
            project=str(results_root),
            name=v,
            exist_ok=True,
            line_width=2,
            show_conf=False,
            show_boxes=True,
            verbose=True,
        )
        print(f"[Variant {v}] Output saved to: {results_root / v}")

    print("\nDone. Annotated results (images or video) are in the 'results/' folder.")


def main():
    parser = argparse.ArgumentParser(
        description=("SpineEndo-YOLOv11: real-time instance segmentation on your own "
                     "spine-endoscopy images or video.")
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help=("Path to YOUR image, video, or a folder of images "
              "(images: .jpg/.png/.tif ...; videos: .mp4/.avi/.mov ...). "
              "If omitted, a bundled demo frame is used."),
    )
    parser.add_argument(
        "--variant", type=str, default="l",
        help="Model variant: n, s, m, l, or 'all' (default: l).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="'cuda' or 'cpu' (default: cuda).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--test_id", type=int, default=1,
        help="Index of the bundled demo image when --source is not given (1-4, default: 1).",
    )
    args = parser.parse_args()

    variant = args.variant.lower()
    if variant == "all":
        variants = VARIANTS
    elif variant in VARIANTS:
        variants = [variant]
    else:
        raise ValueError(f"--variant must be one of {VARIANTS} or 'all'.")

    project_root = Path(__file__).resolve().parent
    source = resolve_source(project_root, args.source, args.test_id)
    run(source, variants, args.device, args.conf)


if __name__ == "__main__":
    main()
