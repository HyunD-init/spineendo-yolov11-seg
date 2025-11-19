# SpineEndo-YOLO-V11-Seg

Official inference demo for:

> Jung HJ, Nam NE, Lee C, et al.  
> **Real-time instance segmentation of spine endoscopy images using a YOLO-V11 deep convolutional neural network**.  
> *PLOS ONE*, 2025. (in press)

<p align="center">
  <img src="image1.jpg" width="350">
  <img src="image2.png" width="350">
</p>



This repository provides an inference pipeline for running the proposed
YOLO-V11-based instance segmentation models on sample spine endoscopy images
or user-provided images.

The full clinical dataset cannot be shared publicly due to patient privacy and
institutional regulations. Instead, we provide a small de-identified sample in
`test_images/` and pre-packaged trained model weights that can be used directly
with the provided scripts.

---

## Repository structure

```text
spineendo-yolov11-seg/
├── inference_demo.py        # Main script for running inference
├── requirements.txt         # Python dependencies
├── test_images/             # De-identified sample endoscopy frames
│   ├── test_1.jpg
│   ├── test_2.jpg
│   ├── test_3.jpg
│   └── test_4.jpg
└── weights/
    ├── __init__.py
    ├── __pycache__/
    │   └── loader.cpython-312.pyc   # Compiled loader for the model weights
    ├── best_n.pt.enc                
    ├── best_n.pt.enc.meta.txt
    ├── best_s.pt.enc
    ├── best_s.pt.enc.meta.txt
    ├── best_m.pt.enc
    ├── best_m.pt.enc.meta.txt
    ├── best_l.pt.enc
    ├── best_l.pt.enc.meta.txt
    # The YOLO-V11 x weights are not hosted here due to GitHub file size limits.
    # They can be shared separately upon reasonable request to the
    # corresponding authors of the paper.
````


## Installation

Tested environment:

* Python 3.12
* PyTorch (CUDA-enabled GPU recommended for speed, but CPU also works)
* Windows 11 / Linux (e.g., Ubuntu 22.04)

1. Clone the repository:

```bash
git clone https://github.com/HyunD-init/spineendo-yolov11-seg.git
cd spineendo-yolov11-seg
```

2. Create and activate a virtual environment (optional but recommended):

```bash
# Example using conda
conda create -n spineendo python=3.12 -y
conda activate spineendo
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick start: run inference on sample images

The script `inference_demo.py` runs the YOLOv11 variants
(`n`, `s`, `m`, `l`) on a single test image and saves the
outputs into the `results/` directory.

> The larger YOLO-V11 x model is not included here because its weight file
> exceeds the public GitHub file size limit. It can be shared separately
> upon reasonable request to the corresponding authors.

### Run on `test_1` (default)

```bash
python inference_demo.py --test_id 1 --device cuda
```

Example console output (truncated):

```text
Input image: .../test_images/test_1.jpg

[Variant n] Loading model...
[Variant n] Model loaded. Running inference...

image 1/1 .../test_1.jpg: 640x640 2 Instruments, 2 Bones, 1 LF, 1 SoftTissue, 1 Vessel, 2 Duras, 4 Fats, 5.9ms
Speed: 6.9ms preprocess, 5.9ms inference, 100.7ms postprocess per image at shape (1, 3, 640, 640)
[Variant n] Saved output to: .../results/test_1_n.jpg

[Variant s] Loading model...
[Variant s] Model loaded. Running inference...
...
[Variant l] Saved output to: .../results/test_1_l.jpg

All variants finished. Check the 'results' folder.
```

The `Speed:` line is printed by the Ultralytics YOLO framework and reports
per-image preprocessing, inference, and postprocessing times.
This allows users to directly measure and compare inference latency on their
own hardware.

After the script finishes, you will find files like:

* `results/test_1_n.jpg`
* `results/test_1_s.jpg`
* `results/test_1_m.jpg`
* `results/test_1_l.jpg`

Each image shows the instance segmentation masks produced by the
corresponding YOLO-V11 variant.

### Choose a different test image

Images are expected to be named `test_<id>.<ext>` inside `test_images/`,
where `<ext>` can be `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, or `.tiff`.

For example:

```bash
python inference_demo.py --test_id 3 --device cpu
```

This will look for `test_images/test_3.*`, run all variants on that image, and
save `results/test_3_n.*`, ..., `results/test_3_l.*`.

---

## Using your own images

To test the model on your own spine endoscopy frames:

1. Place your image in the `test_images/` folder.
2. Rename it to follow the same pattern, for example:

   * `test_5.png`
3. Run:

```bash
python inference_demo.py --test_id 5 --device cuda
```

The script will automatically detect `test_5.*` with supported extensions,
run all model variants, and save the outputs into the `results` folder.

As with the sample images, the console output will include a `Speed:` line for
each run, allowing you to measure inference time for your own images and
hardware configuration.

---

## Reproducibility and custom training

Our models are based on the standard YOLO-V11 segmentation architecture
(“YOLOv11-seg”) implemented in the Ultralytics YOLO framework.

* **Reproducing or adapting training on custom datasets**
  Researchers who wish to train or fine-tune YOLO-V11-seg models on their own
  datasets can do so using the official Ultralytics YOLO-V11 segmentation
  implementation and documentation (e.g., by downloading the YOLO-V11-seg
  model from the official website and following the standard training
  procedures on a custom dataset with segmentation masks).

  The network architecture and training strategy used in this study follow
  standard YOLO-V11-seg practices, as described in the Methods section of
  the paper. Therefore, training a YOLOv11-seg model on a similarly prepared
  dataset should yield comparable behavior and allow independent validation
  of the approach.

* **Reproducing inference behavior and runtime**
  This repository focuses on the inference pipeline:

  * it provides our trained models in a packaged format,
  * exposes a simple script (`inference_demo.py`) to run all variants on
    a given image,
  * and prints detailed timing information (preprocess / inference /
    postprocess) for each run.

  By running the provided script on either the sample frames or their own
  images, users can:

  * qualitatively inspect the instance segmentation outputs, and
  * quantitatively assess inference latency on their own hardware.

Together with the hyperparameters and implementation details described in the
paper, this repository is intended to support both qualitative and quantitative
reproducibility of the main findings.

---

## Data availability

The original clinical videos used in the study contain potentially identifiable
patient information and cannot be shared publicly due to institutional and IRB
restrictions.

This repository provides:

* a small de-identified subset of sample frames in `test_images/`, and
* trained YOLO-V11 models in a packaged format that can be used with the
  provided inference scripts.

Access to the full dataset may be requested from the corresponding institution’s
data access committee, as described in the Data Availability Statement of the
paper.

---

## Citation

If you use this code or these models in your research, please cite:

```text
Jung HJ, Nam NE, Lee C, et al.
Real-time instance segmentation of spine endoscopy images using a YOLO-V11
deep convolutional neural network. PLOS ONE. 2025.
```

(I will replace this with the final citation/DOI once the paper is published.)

````
