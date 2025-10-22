# main_v2.py — Hướng dẫn sử dụng & tối ưu

README này mô tả cách chuẩn bị môi trường, convert MobileNetV2 sang ONNX, enroll dữ liệu, chạy nhận dạng, và *chỉnh các thông số trực tiếp trong code* (không cần truyền cờ `--`), theo đúng bản `main_v2.py` bạn đang dùng.

> **Tóm tắt chạy nhanh**
> ```bash
> # 1) Convert MobileNetV2 sang ONNX (một lần)
> python - <<'PY'
> import tf2onnx, tensorflow as tf
> from tensorflow.keras.applications import MobileNetV2
> m = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))
> spec = (tf.TensorSpec((None,224,224,3), tf.float32, name="input"),)
> model_proto, _ = tf2onnx.convert.from_keras(m, input_signature=spec, opset=13, output_path="mb2_gap.onnx")
> PY
>
> # 2) Enroll (tạo dữ liệu nhận dạng)
> python main_v2.py --enroll --label admin
>
> # 3) Chạy thường (không cần truyền các cờ tối ưu)
> python main_v2.py
> ```

---
## 0) 

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
test_person_following/models/
```
## 1) Chuẩn bị thư viện & mô hình

### Thư viện Python cần có
```bash
pip install opencv-python onnxruntime insightface numpy scikit-learn joblib tf2onnx tensorflow
# (server không cần UI): pip install opencv-python-headless
# Nếu dùng Orange Pi, vẫn để CPUExecutionProvider (ONNX Runtime CPU).
```

> *InsightFace dùng cho phần khuôn mặt (detector + embedding). ONNX Runtime dùng cho MobileNetV2 (body).*

### Person detector (MobileNet-SSD, Caffe)
Đặt file trong thư mục `models/` (cùng cấp `main_v2.py`):
- `models/MobileNetSSD_deploy.prototxt`
- `models/MobileNetSSD_deploy.caffemodel`

Nếu chưa có, bạn có thể tìm bộ *MobileNetSSD_deploy* phổ biến trên GitHub / OpenCV Zoo.

### MobileNetV2 (body) dạng ONNX
Tạo file `mb2_gap.onnx` bằng đoạn code ở phần *Tóm tắt chạy nhanh* phía trên.
- Input: **(1,224,224,3)** NHWC, float32, chuẩn hóa theo Keras: `x/127.5 - 1.0`
- Output: vector 1280-D (GAP). Code sẽ ghép thêm HSV-hist (48-D) ⇒ 1328-D.

> **Chú ý**: Convert ONNX chỉ cần làm **một lần** (trừ khi bạn muốn thay model).

---

## 2) Enroll dữ liệu (tạo template lưu trong thư mục `data/`)

Chạy:
```bash
python main_v2.py --enroll --label admin
```
- Nhấn **`b`** để thêm mẫu **body-only** (lấy người lớn nhất khung).
- Nhấn **`q`** để kết thúc.
- Sau khi enroll, thư mục `data/` sẽ có:
  - `face_templates.npy`, `face_centroid.npy` (bắt buộc)
  - `body_pca.pkl`, `body_cls_knn.pkl`, `body_mean.npy`, `body_admin_centroid.npy` (nếu thu đủ mẫu body)

> Khuyến nghị thu **≥ 40–60** mẫu face ở nhiều góc/ánh sáng để ổn định.

---

## 3) Chạy nhận dạng

```bash
python main_v2.py
```
- Mặc định camera `--cam 0` trong code. Nếu muốn dùng camera khác, sửa biến `CAM_INDEX` trong phần **CONFIG** của file `main_v2.py`.

---

## 4) Chỉnh thông số trực tiếp trong code (không cần `--`)

Mở `main_v2.py`, ở phần **CONFIG** đầu file, bạn có thể chỉnh các biến sau:

```python
# Camera
CAM_INDEX      = 0

# File ONNX MobileNetV2 (body)
MB2_ONNX_PATH  = "mb2_gap.onnx"

# Temporal memory hyperparams
EMA_ALPHA       = 0.7
CONFIRM_FRAMES  = 5
MARGIN_DELTA    = 0.06
PERSIST_FRAMES  = 10
IOU_STICKY      = 0.35

# Ngưỡng/siêu tham số nhận dạng
ACCEPT_FACE     = 0.26   # ngưỡng chấp nhận face (cosine mix)
ACCEPT_BODY     = 0.26   # ngưỡng chấp nhận body
PERSON_CONF     = 0.4    # ngưỡng detector người (SSD)
PCA_DIM_BODY    = 192    # số chiều PCA cho body

# --- Các tham số tối ưu cố định trong code ---
USE_ROI         = True   # ưu tiên detect face theo ROI quanh person_box
SKIP_K          = 3      # số khung bỏ qua giữa các lần detect (skip-frames)
ROI_EXPAND      = 0.30   # biên nới ROI (0.30 ≈ 1.3x)
FACE_ON         = 2      # số khung liên tiếp có mặt để chuyển sang MODE_FACE_DOMINANT
FACE_OFF        = 4      # số khung liên tiếp mất mặt để về MODE_BODY_DOMINANT

# Hiển thị HUD
NO_VIS          = False  # True để tắt rectangle/putText/imshow (FPS cao hơn)

# FaceAnalysis pack & det_size
FACE_PACK_NAME  = 'buffalo_l'     # có thể dùng 'buffalo_sc' để nhẹ hơn
FACE_DET_SIZE   = (384, 384)      # 320x320 nhanh hơn nếu mặt không quá nhỏ
```

### Giải thích nhanh từng thông số

- **USE_ROI**: Khi đã có `person_box` đang bám, chỉ detect face trong `ROI = expand(person_box, ROI_EXPAND)` ⇒ giảm pixel phải xử lý ⇒ tăng FPS.
- **SKIP_K**: Mỗi `K` khung mới detect lại; các khung giữa dùng **tracker** hình học để bám người ⇒ giảm tần suất gọi các mô-đun nặng.
- **ROI_EXPAND**: Mức nới ROI quanh `person_box`. 0.30 ≈ 1.3× kích thước người (đủ để không mất mặt khi người hơi chuyển động).
- **FACE_ON / FACE_OFF**: Hysteresis cho **gating**:
  - Chuyển sang **FACE_DOMINANT** sau khi thấy mặt liên tiếp `FACE_ON` khung.
  - Chuyển về **BODY_DOMINANT** sau khi **không thấy** mặt `FACE_OFF` khung.
- **NO_VIS**: Tắt vẽ giúp FPS tăng, đặc biệt trên SBC (Orange Pi).
- **FACE_PACK_NAME / FACE_DET_SIZE**:
  - `buffalo_sc` nhẹ hơn `buffalo_l` (đề xuất nếu bạn muốn thêm FPS).
  - `(320,320)` giúp rớt chi phí detect ~20–35% so với `(384,384)` (nếu mặt đủ lớn).

> **Mẹo**: Với Orange Pi 5 Plus, ONNX Runtime nên dùng **CPUExecutionProvider**, giữ thread hợp lý (mặc định code đã OK).

---

## 5) Cơ chế gating & “k=1 khi ADMIN”

- **Gating** hai mode:
  - `MODE_FACE_DOMINANT`: chỉ xử lý **Face**; **không** tính Body ⇒ tiết kiệm.
  - `MODE_BODY_DOMINANT`: chỉ xử lý **Body**; **không** gọi Face ⇒ tiết kiệm.
- **Streak + hysteresis** (`FACE_ON`, `FACE_OFF`) tránh nhấp nháy mode khi mặt chập chờn.
- **k=1 khi ADMIN**: Sau khi hệ thống đã xác nhận ADMIN (ổn định trong vài khung), chỉ tính **embedding 1 khuôn mặt** gắn với ADMIN (ưu tiên IOU cao nhất với `admin_track["face_box"]`). Khi đông người, điều này tiết kiệm đáng kể so với tính 2–5 mặt mỗi khung.

Kết quả: trung bình **FPS tăng rõ** mà tính ổn định vẫn đảm bảo.

---

## 6) Quy trình hoạt động khuyến nghị

1. **Enroll** đủ mẫu face/body (đa góc, nhiều ánh sáng).
2. **Chạy** với `USE_ROI=True`, `SKIP_K=3`, `ROI_EXPAND=0.30`.
3. Xem FPS; nếu còn nặng:
   - Giảm `FACE_DET_SIZE` xuống `(320,320)`.
   - Chuyển `FACE_PACK_NAME='buffalo_sc'`.
   - Đặt `NO_VIS=True`.
4. (Tùy chọn) Bật “gating” mạnh hơn: tăng `FACE_OFF` nếu hay mất mặt chập chờn.

---

## 7) Troubleshooting

- **`FileNotFoundError: ONNX model not found: mb2_gap.onnx`**  
  → Bạn chưa convert hoặc đặt sai đường dẫn. Sửa `MB2_ONNX_PATH` trong CONFIG hoặc convert lại.

- **Thiếu MobileNet-SSD (Caffe)**  
  → Đảm bảo có `models/MobileNetSSD_deploy.prototxt` và `models/MobileNetSSD_deploy.caffemodel` đúng đường dẫn.

- **`AttributeError: cv2.TrackerCSRT_create`**  
  → OpenCV bản không có *contrib*. Code đã *fallback* (không crash). Nếu muốn tracker, cài `opencv-contrib-python`.  

- **FPS vẫn thấp**  
  → Tắt HUD (`NO_VIS=True`), giảm `FACE_DET_SIZE`, dùng `buffalo_sc`, tăng `SKIP_K`, hoặc dùng gating. Trên SBC, tránh chạy quá nhiều luồng nền.

---

## 8) Cấu trúc thư mục chuẩn

```
project/
├─ main_v2.py
├─ mb2_gap.onnx
├─ models/
│  ├─ MobileNetSSD_deploy.prototxt
│  └─ MobileNetSSD_deploy.caffemodel
└─ data/
   ├─ face_templates.npy
   ├─ face_centroid.npy
   ├─ body_pca.pkl
   ├─ body_cls_knn.pkl
   ├─ body_mean.npy
   └─ body_admin_centroid.npy
```

---

## 9) Gợi ý benchmark nhanh

- Bật/ tắt từng tính năng và quan sát FPS (in ra bằng `cv2.getTickFrequency()` hoặc thêm bộ đếm thời gian).
- Thử các bộ tham số:
  - **Nhẹ**: `buffalo_sc`, `FACE_DET_SIZE=(320,320)`, `NO_VIS=True`.
  - **Gating mạnh**: `FACE_ON=2`, `FACE_OFF=6`, `SKIP_K=4`.
  - **Khung đông người**: “k=1 khi ADMIN” sẽ cho lợi ích lớn.

---

Chúc bạn chạy mượt! Nếu muốn mình tạo sẵn *script benchmark* FPS hoặc *profile* từng khối (face/body/person), mình có thể cung cấp thêm.