# inference_demo.py
import argparse
from pathlib import Path

from weights.loader import load_yolov11_model


VARIANTS = ["n", "s", "m", "l"]
# 허용할 이미지 확장자들
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_test_image(project_root: Path, test_id: int) -> Path:
    """
    test_images 폴더에서 test_<id>.* 형태의 이미지를 찾는다.
    확장자는 ALLOWED_EXTS 중 하나.
    """
    test_dir = project_root / "test_images"
    if not test_dir.exists():
        raise FileNotFoundError(f"test_images directory not found: {test_dir}")

    prefix = f"test_{test_id}"
    candidates = [
        p for p in test_dir.iterdir()
        if p.is_file()
        and p.stem == prefix
        and p.suffix.lower() in ALLOWED_EXTS
    ]

    if not candidates:
        # prefix만 맞는 파일이 있는지 한 번 더 확인해서 디버깅 도움
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

    # 같은 이름으로 여러 확장자가 있어도 첫 번째 하나만 사용
    return sorted(candidates)[0]


def run_inference(test_id: int, device: str = "cuda"):
    """
    단일 테스트 이미지(test_<id>.*)에 대해
    YOLOv11 모델 variant 별로 추론을 수행하고,
    results/test_<id>_<variant>.<ext> 에 결과 이미지를 저장.
    """
    project_root = Path(__file__).resolve().parent

    # 입력 이미지 찾기
    image_path = find_test_image(project_root, test_id)
    print(f"Input image: {image_path}")

    # 결과 저장 루트
    results_root = project_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    for v in VARIANTS:
        print(f"\n[Variant {v}] Loading encrypted model...")
        model = load_yolov11_model(variant=v, device=device)
        print(f"[Variant {v}] Model loaded. Running inference...")

        # 저장 파일 이름: test_<id>_<variant>.<원본확장자>
        save_name = f"test_{test_id}_{v}{image_path.suffix}"
        save_path = results_root / save_name

        # YOLO 추론
        results = model(str(image_path))

        # segmentation mask가 그려진 이미지를 저장
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
        help="Test image index (1~4). Uses test_images/test_<id>.<ext> (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference: 'cuda' or 'cpu' (default: cuda)",
    )

    args = parser.parse_args()

    if not 1 <= args.test_id <= 4:
        raise ValueError("test_id must be between 1 and 4 (inclusive).")

    run_inference(test_id=args.test_id, device=args.device)


if __name__ == "__main__":
    main()
