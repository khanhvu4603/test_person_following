# Robot Vision Demo (Webcam) — Face Auth (VGG16+PCA), Person (MobileNet-SSD), Hand (MediaPipe)

This is a **webcam** demo that mirrors the paper's perception stack:
- **Face Auth**: VGG16 used as a **feature extractor** on cropped faces + **PCA** (and a small KNN) for admin identity.
  - In the paper, VGG16 did detection + PCA identity. For a quick webcam test, we keep the **auth idea** (VGG16+PCA) but use a standard face **detector** (OpenCV Res10 SSD) to crop faces.
- **Person Detection**: **MobileNet-SSD (Caffe)**, keep **class=person**.
- **Hand**: **MediaPipe Hands** with a simple finger-count gesture mapper to discrete commands (START/STOP/LEFT/RIGHT/FORWARD/BACKWARD) like the paper.

> This matches the *spirit* and the **modules** used in the paper, adapted for a fast local demo on a PC webcam (no depth).

## 1) Setup

**Python 3.9+** recommended. Create a venv and install deps:
```bash
pip install -r requirements.txt
```

### Download models into `robot_vision_demo/models/`

- **MobileNet-SSD (Caffe)**
  - `MobileNetSSD_deploy.prototxt` (deploy text)
  - `MobileNetSSD_deploy.caffemodel` (weights)
  - Common mirrors:
    - https://github.com/chuanqi305/MobileNet-SSD (original)
    - https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/ (prototxt)
    - Many tutorials also host the same two files.
- **Face Detector (OpenCV DNN Res10)**
  - `deploy.prototxt`
  - `res10_300x300_ssd_iter_140000.caffemodel`
  - From OpenCV's face detection examples (DNN).

Place all four files under:
```
robot_vision_demo/models/
```

## 2) Enroll the admin face

Run enrollment (collects about 40 samples of your face — largest detected face in view). Look at the camera and move slightly for variation.

```bash
python main.py --enroll --label admin
```

This creates:
- `data/face_pca.pkl` — PCA projection (whitened)
- `data/face_cls_knn.pkl` — a small KNN classifier
- `data/face_mean.npy` — mean vector

## 3) Run the real-time demo

```bash
python main.py
```

Keys: `q` quit, `h` help in console.

Overlays show:
- **Face**: green box + auth status `ADMIN (conf)` if recognized (distance-based acceptance), otherwise `UNKNOWN`.
- **Person**: orange boxes for `person` detections (MobileNet-SSD).
- **Hand**: landmarks and a simple **gesture** label; the script prints a stabilized gesture in the overlay.

## 4) How this maps to the paper

- **Face Auth**: paper describes **VGG16** + **PCA**. We use VGG16's **fc1** (4096-D) as embedding → **PCA (128-D)** → **KNN** + distance threshold for acceptance.
- **Person**: paper selected **MobileNet-SSD** for speed and multi-scale. We use the same architecture via OpenCV DNN with COCO/VOC-style weights and keep **class=15 (person)**.
- **Hand**: paper uses **MediaPipe Hands**; we do the same. We convert landmarks to a simple **gesture** (finger-count heuristic) mapping to the paper's discrete commands.

## 5) Notes & Tips

- This webcam demo **does not use depth**, so **follow/avoid** control is not included — it focuses on **perception** as requested.
- For face auth, if you get false accepts, increase the acceptance threshold in code (`thresh` in `predict_face_label`), or use more enrollment samples.
- If you plan to move to the robot later, you can feed person bbox + (eventual) depth into your controller, and map hand gestures to discrete motion commands.

## 6) Troubleshooting

- `FileNotFoundError` for models: make sure the four model files are placed under `robot_vision_demo/models/` exactly as named.
- Low FPS? Reduce webcam resolution (e.g., 640x480), or set `--fps` when opening capture if your camera supports it.
- If TensorFlow crashes on CPU AVX issues, try a different TF build or use `tensorflow-cpu` specifically.

——
Enjoy testing!