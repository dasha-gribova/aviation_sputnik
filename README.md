# Aviation Sputnik: Aircraft Detection on Aerial and Satellite Imagery

Aviation Sputnik is a computer vision pipeline for detecting aircraft on
high-resolution aerial and satellite imagery. It combines a custom-trained
YOLO detector with a sliding-window (slicing) inference strategy,
multiprocessing execution, and global non-maximum suppression to reliably
detect small objects in very large images.

## Overview

Detecting aircraft on satellite imagery is challenging:

- Images are large (thousands of pixels per side), while aircraft occupy only a small fraction of the frame.
- Feeding a full-resolution tile into a detector either exceeds memory limits or forces downscaling that destroys small objects.
- Dense parking areas produce many closely spaced objects that must be separated cleanly.

This project addresses these challenges with a SAHI-style (Slicing Aided
Hyper Inference) pipeline: the image is scanned with overlapping windows,
each window is processed by a YOLO model, and per-window detections are
merged into a single global set of bounding boxes using non-maximum
suppression.

## Key Features

- Custom-trained YOLO model (Ultralytics) for aircraft detection.
- Sliding-window inference with configurable window size and overlap (`calc_scan_areas`).
- Multiprocessing producer-consumer architecture for parallel window processing.
- Global non-maximum suppression (`torchvision.ops.nms`) to merge duplicate detections from overlapping windows.
- Coordinate remapping from window-local frames back to the full-image coordinate system.
- Interactive visualization with OpenCV: total detection count and per-object confidence scores.

## How It Works

1. **Scan area generation.** The input image (or a scan-area bounding box) is divided into overlapping windows by `calc_scan_areas` with configurable window size (default 800x800 px) and overlap (default 20 percent per axis).
2. **Parallel inference.** A producer process scans the image directory and feeds windows into a queue; worker processes run the YOLO model on each window.
3. **Coordinate remapping.** Window-local detection coordinates are shifted back to the global image frame.
4. **Global NMS.** Detections from all windows are merged with `torchvision.ops.nms` to remove duplicates produced by overlapping windows.
5. **Visualization.** Final boxes and confidence scores are drawn on a downscaled copy of the image (maximum display width 1280 px) and shown in an OpenCV window titled `SAHI YOLO Global Detections`. Press `q` to stop.

Pipeline diagram:

```text
Large input image (satellite / aerial tile)
        |
        v
Sliding-window generator (calc_scan_areas: size, overlap)
        |
        v
Multiprocessing producer-consumer: YOLO inference per window
        |
        v
Coordinate remapping to the global image frame
        |
        v
Global NMS (torchvision.ops.nms)
        |
        v
Final detections (bounding boxes + confidence scores)
```

## Project Structure

```text
aviation_sputnik/
├── detect_plane_ground.py    # Inference pipeline: slicing + multiprocessing + NMS + visualization
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes weights, virtual environment, caches, screenshots
└── README.md
```

The trained model weights (`best(2).pt`) are not stored in the repository;
see the Model Weights section below.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/dasha-gribova/aviation_sputnik.git
cd aviation_sputnik
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Weights

The trained YOLO weights (`best(2).pt`) exceed GitHub file-size limits and
are distributed via GitHub Releases.

- Open the Releases page of the repository and download `best(2).pt` from the latest release (tag `v1.0`), or use the command line:

```bash
wget -O "best(2).pt" https://github.com/dasha-gribova/aviation_sputnik/releases/download/v1.0/best%282%29.pt
```

- Place the file in the project root before running inference.

## Usage

Run the pipeline on the configured directory of images:

```bash
python detect_plane_ground.py
```

The script processes every image window by window and displays the merged
result. The overlay shows the total detection count and the confidence of
each detection.

## Configuration

Key parameters of `calc_scan_areas` and the visualization loop in
`detect_plane_ground.py`:

| Parameter        | Default     | Description                                              |
|------------------|-------------|----------------------------------------------------------|
| `window_w_h`     | `(800, 800)`| Sliding window size in pixels                            |
| `overlay`        | `(0.2, 0.2)`| Overlap fraction between neighboring windows, per axis   |
| `force_int`      | `True`      | Round window coordinates to integers                     |
| `max_disp_width` | `1280`      | Maximum width of the visualization window in pixels      |

Detection confidence and NMS IoU thresholds are defined in the script near
the inference and post-processing calls.

## Results

Qualitative evaluation on satellite tiles of public airfields:

- Aircraft are detected on stands, taxiways, and runways, including densely packed parking areas.
- Typical confidence for clearly visible aircraft is 0.8-0.9; smaller or partially occluded objects are detected with confidence 0.5-0.8.
- Representative frames yield 45-60 detections per scene with no visible duplicate boxes after global NMS.

## Technical Notes

- **Why slicing.** Full-frame inference on 4K+ tiles either runs out of memory or requires downscaling at which aircraft become a few pixels wide. Slicing preserves native resolution.
- **Why overlap.** Objects cut by a window border are fully visible in at least one neighboring window, which prevents missed detections at tile edges.
- **Why global NMS.** Overlapping windows detect the same object multiple times; NMS in the global coordinate frame collapses these duplicates into a single box.
- **Why producer-consumer multiprocessing.** Image loading and preprocessing run in a separate process from inference, keeping the compute pipeline saturated through a shared queue.

## Requirements

- Python 3.10+
- ultralytics
- torch
- torchvision
- opencv-python
- numpy

Exact versions are listed in `requirements.txt`.

