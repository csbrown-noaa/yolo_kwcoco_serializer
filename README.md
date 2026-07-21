# YOLO to KWCOCO Serializer

A lightweight, stateful adapter that bridges the gap between the [Ultralytics](https://github.com/ultralytics/ultralytics) prediction engine and the [KWCOCO](https://github.com/Kitware/kwcoco) dataset standard.

This library consumes streaming `Results` objects directly from Ultralytics YOLO and compiles them into a highly-compliant, portable JSON manifest. It handles the mathematical mapping of bounding boxes, intelligently groups sequential video frames into KWCOCO video entities, and provides an injection point for custom probability distributions.

## Features

* **Zero-Boilerplate Video Tracking:** Automatically detects when predictions are coming from a video file or stream and increments frame counters internally. No manual sequence tracking required.
* **Portable Pathing:** Strips absolute directory paths, strictly using leaf filenames as IDs to ensure dataset manifests remain portable across Docker containers and cloud environments.
* **Duck-Typed Architecture:** Built to orchestrate predictions organically. Pre-stubbed for safe routing of Segmentation masks, Pose keypoints, and Classification probabilities.
* **Soft Score Injection:** Conditionally captures and serializes custom `soft_scores` into the strictly compliant KWCOCO `"prob"` schema if attached to the `Results` object (useful for hierarchical or probabilistic modeling).

## Installation

Install directly via pip from your local repository:

```bash
# Core installation
pip install .

# Install with Ultralytics included (for full end-to-end inference)
pip install .[ultralytics]
```

## Usage

The serializer acts as a stateful bucket. You initialize it with your class map, pour your YOLO predictions into it, and then save the manifest to disk.

### Basic Image and Video Streaming

Because the `add_result()` method checks the path of each incoming frame, you can use the exact same script to process folders of standalone images or multi-hour video streams.

```python
import os
from ultralytics import YOLO
from yolo2kwcoco.kwcoco_serializer import Yolo2KwcocoSerializer

# 1. Load your trained model
model = YOLO("yolov8n.pt")

# 2. Initialize the Serializer with the model's category map
serializer = Yolo2KwcocoSerializer(categories=model.names)

# 3. Run streaming inference (works for directories, .mp4, RTSP, etc.)
# stream=True is highly recommended to prevent memory bloat on large datasets/videos
results_stream = model.predict(source="path/to/data", stream=True)

# 4. Pipe the results directly into the serializer
for result in results_stream:
    serializer.add_result(result)

# 5. Export the compliant KWCOCO manifest
serializer.save("output_manifest.json")
```

### Custom Probability Injection

If you are using a custom YOLO predictor that attaches a full class-probability distribution to the detection results (such as a hierarchical model), the serializer will automatically detect it and map it to the strict KWCOCO `"prob"` array.

Simply ensure your custom predictor attaches a numpy array of shape `(N_detections, N_classes)` or a PyTorch tensor to the `soft_scores` attribute of the Ultralytics `Results` object before passing it to the serializer.

```python
# Inside your custom Predictor's postprocess() method:
res = Results(orig_img, path=img_path, names=self.model.names, boxes=final_pred)

# Attach custom probabilities
res.soft_scores = computed_distribution_array 
results.append(res)
```

## Testing

This library is self-documenting and utilizes Python's built-in `doctest` module to verify the integrity of the serializer logic without needing to download heavy model weights or instantiate PyTorch.

To run the test suite, simply execute the file as a module:

```bash
python -m doctest -v kwcoco_serializer.py
```

## Contributing

Currently, the serializer natively supports Spatial Bounding Boxes (`result.boxes`).

The internal `add_result` pipeline utilizes a duck-typing architecture that is pre-stubbed out for Segmentation, Pose, and Classification. Contributions to fill in these `NotImplementedError` stubs are highly welcome.

# Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and
Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project
code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the
Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub
project will be governed by all applicable Federal law. Any reference to specific commercial products,
processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or
imply their endorsement, recommendation or favoring by the Department of Commerce. The Department
of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to
imply endorsement of any commercial product or activity by DOC or the United States Government.

