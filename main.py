# robot_vision_demo/main.py
# ------------------------------------------------------------
# Demo pipeline (webcam) inspired by the paper's perception stack:
# - Face Auth: VGG16 (feature extractor) + PCA (recognizer)
# - Person Detection: MobileNet-SSD (OpenCV DNN) -> keep only 'person' class
# - Hand: MediaPipe Hands -> simple gesture mapping to discrete commands
#
# Notes:
# - In the paper, VGG16 was used as a face detector head + PCA identity. For a fast webcam demo,
#   we use a standard face DETECTOR (OpenCV DNN) and keep VGG16+PCA for AUTH (identity) on cropped faces.
# - Person detector uses MobileNet-SSD Caffe model (pretrained). You need to download weights (see README).
# - This script supports two modes for face auth: (A) Enroll admin, (B) Run recognition.
#   See keyboard help once running (hit 'h').
#
# Tested with: Python 3.9+, OpenCV 4.7+, TensorFlow 2.11+, scikit-learn, mediapipe 0.10+.
# ------------------------------------------------------------

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import deque

# Face embedding (VGG16) + PCA
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib

# MediaPipe Hands
import mediapipe as mp

# ------------ Config Paths ------------
HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
DATA = HERE / "data"
DATA.mkdir(exist_ok=True, parents=True)
(MODELS).mkdir(exist_ok=True, parents=True)

# MobileNet-SSD (OpenCV DNN - Caffe)
# Download from README links and place here:
MOBILENET_PROTOTXT = MODELS / "MobileNetSSD_deploy.prototxt"
MOBILENET_WEIGHTS  = MODELS / "MobileNetSSD_deploy.caffemodel"

# Face detector (OpenCV DNN - Res10)
FACE_PROTO = MODELS / "res10_300x300_ssd_iter_140000.caffemodel"
FACE_TXT   = MODELS / "deploy.prototxt"

# PCA/Classifier save paths
PCA_PATH   = DATA / "face_pca.pkl"
CLS_PATH   = DATA / "face_cls_knn.pkl"
MEAN_PATH  = DATA / "face_mean.npy"

# ------------ Utils ------------

def load_opencv_face_detector():
    if not (FACE_PROTO.exists() and FACE_TXT.exists()):
        raise FileNotFoundError(
            f"Missing face detector model files:\n{FACE_TXT}\n{FACE_PROTO}\n"
            "See README for download links."
        )
    net = cv2.dnn.readNetFromCaffe(str(FACE_TXT), str(FACE_PROTO))
    return net

def detect_faces_dnn(net, frame, conf_thresh=0.6):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    scores = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
                scores.append(float(confidence))
    return boxes, scores

def load_person_detector():
    if not (MOBILENET_PROTOTXT.exists() and MOBILENET_WEIGHTS.exists()):
        raise FileNotFoundError(
            f"Missing MobileNet-SSD model files:\n{MOBILENET_PROTOTXT}\n{MOBILENET_WEIGHTS}\n"
            "See README for download links."
        )
    net = cv2.dnn.readNetFromCaffe(str(MOBILENET_PROTOTXT), str(MOBILENET_WEIGHTS))
    return net

def detect_persons_dnn(net, frame, conf_thresh=0.4):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300,300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    scores = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        class_id = int(detections[0,0,i,1])
        # 'person' = class 15 in standard MobileNet-SSD (VOC-style)
        if class_id == 15 and confidence > conf_thresh:
            box = detections[0,0,i,3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
                scores.append(float(confidence))
    return boxes, scores

def build_vgg16_feature_extractor():
    # Use VGG16 up to 'fc1' as a high-level descriptor (4096-D), then we'll PCA it.
    base = VGG16(weights="imagenet", include_top=True)
    # We'll keep the model and extract 'fc1' layer output via a sub-model:
    from tensorflow.keras.models import Model
    feat_model = Model(inputs=base.input, outputs=base.get_layer("fc1").output)
    return feat_model

def crop_and_preprocess_face(frame, box, target_size=(224,224)):
    (x1, y1, x2, y2) = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    roi = cv2.resize(roi, target_size)
    arr = img_to_array(roi)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def compute_face_embedding(vgg_model, frame, box):
    arr = crop_and_preprocess_face(frame, box)
    if arr is None:
        return None
    feat = vgg_model.predict(arr, verbose=0)  # shape (1, 4096)
    return feat.flatten()

def enroll_collect_samples(vcap, face_net, vgg_model, label, num_samples=30, conf_thresh=0.7):
    print(f"[ENROLL] Collecting {num_samples} face samples for label='{label}'...")
    feats = []
    count = 0
    while True:
        ret, frame = vcap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        boxes, scores = detect_faces_dnn(face_net, frame, conf_thresh=conf_thresh)
        # choose the biggest box as the current subject
        sel = None
        if boxes:
            areas = [(i, (b[2]-b[0])*(b[3]-b[1])) for i,b in enumerate(boxes)]
            sel = max(areas, key=lambda x: x[1])[0]
            box = boxes[sel]
            emb = compute_face_embedding(vgg_model, frame, box)
            if emb is not None:
                feats.append(emb)
                count += 1
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
                cv2.putText(frame, f"ENROLL {count}/{num_samples}", (box[0], max(20, box[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, "Press 'q' to abort", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Enroll Admin Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= num_samples:
            break
    try:
        cv2.destroyWindow("Enroll Admin Face")
    except:
        pass
    return np.array(feats)  # shape (N,4096)

def train_pca_and_classifier(X, label, n_components=128, k=3):
    # Centering
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    Z = pca.fit_transform(Xc)
    # Simple KNN classifier with only one class + negative buffer (we will add impostor online)
    # For a single-class enrollment, KNN still works for distance-to-centroid thresholding.
    knn = KNeighborsClassifier(n_neighbors=min(k, len(Z)))
    y = np.array([label]*len(Z))
    knn.fit(Z, y)
    joblib.dump(pca, PCA_PATH)
    joblib.dump(knn, CLS_PATH)
    np.save(MEAN_PATH, mean)
    print(f"[ENROLL] Saved PCA -> {PCA_PATH}\n[ENROLL] Saved KNN -> {CLS_PATH}\n[ENROLL] Saved mean -> {MEAN_PATH}")

def load_pca_and_classifier():
    if not (PCA_PATH.exists() and CLS_PATH.exists() and MEAN_PATH.exists()):
        return None, None, None
    pca = joblib.load(PCA_PATH)
    knn = joblib.load(CLS_PATH)
    mean = np.load(MEAN_PATH)
    return pca, knn, mean

def predict_face_label(emb, pca, knn, mean, thresh=0.8):
    # Project and classify; we also compute distance to nearest neighbor to reject impostors.
    Z = pca.transform((emb - mean).reshape(1, -1))
    label = knn.predict(Z)[0]
    # distance-based acceptance:
    dists, idxs = knn.kneighbors(Z, n_neighbors=1, return_distance=True)
    d = float(dists[0,0])
    # Convert to a pseudo-confidence (smaller dist -> higher conf)
    conf = np.exp(-d)
    is_accept = (conf >= thresh)
    return label, conf, is_accept, d

# --------- Hand / Gesture mapping ----------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_extended_fingers(landmarks, handedness_label):
    # Simple heuristic: count fingers extended based on y-coordinates (for upright camera).
    # Thumb uses x depending on left/right.
    tips = [4, 8, 12, 16, 20]
    cnt = 0
    # Convert to list of (x,y)
    pts = [(lm.x, lm.y) for lm in landmarks.landmark]
    # Thumb
    if handedness_label == "Right":
        if pts[4][0] < pts[3][0]:  # thumb open to the left for right hand
            cnt += 1
    else:
        if pts[4][0] > pts[3][0]:  # left hand
            cnt += 1
    # Other fingers: tip above pip -> extended (y smaller means higher in image)
    for tid, pip in zip([8,12,16,20],[6,10,14,18]):
        if pts[tid][1] < pts[pip][1]:
            cnt += 1
    return cnt

def gesture_from_fingers(n):
    # Map simple gestures -> commands (match paper's discrete set)
    # 0: STOP, 1: TURN LEFT, 2: TURN RIGHT, 3: FORWARD, 4: BACKWARD, 5: START
    if n == 0: return "STOP"
    if n == 1: return "TURN_LEFT"
    if n == 2: return "TURN_RIGHT"
    if n == 3: return "FORWARD"
    if n == 4: return "BACKWARD"
    if n >= 5: return "START"
    return "UNKNOWN"

# --------- Main Loop ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    parser.add_argument("--enroll", action="store_true", help="Enroll admin face first")
    parser.add_argument("--label", type=str, default="admin", help="Label for admin user")
    parser.add_argument("--fps", type=int, default=0, help="Target camera FPS (0=default)")
    parser.add_argument("--face_conf", type=float, default=0.7, help="Face DET confidence threshold")
    parser.add_argument("--person_conf", type=float, default=0.4, help="Person DET confidence threshold")
    parser.add_argument("--pca_dim", type=int, default=128, help="PCA embedding dimension")
    args = parser.parse_args()

    # Video capture
    cap = cv2.VideoCapture(args.cam)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    # Load models
    print("[INIT] Loading models...")
    face_net = load_opencv_face_detector()
    person_net = load_person_detector()
    vgg_model = build_vgg16_feature_extractor()

    # Enroll mode
    if args.enroll:
        feats = enroll_collect_samples(cap, face_net, vgg_model, args.label, num_samples=40, conf_thresh=args.face_conf)
        if len(feats) < 5:
            print("[ENROLL] Not enough samples collected. Aborting.")
            return
        train_pca_and_classifier(feats, args.label, n_components=args.pca_dim, k=3)
        print("[ENROLL] Done. Now run without --enroll to authenticate.")
        return

    # Load PCA/Classifier (must exist)
    pca, knn, mean = load_pca_and_classifier()
    if pca is None:
        print("[WARN] No enrolled admin found. Run with --enroll first to register admin face.")
        print("       Running without face auth (detector still runs).")

    # Init MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    # Smoothing for gesture outputs
    gesture_hist = deque(maxlen=8)

    print("[RUN] Press 'h' for help, 'q' to quit.")
    last_fps_t = time.time()
    frame_count = 0

    admin_status = "UNKNOWN"
    admin_conf   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ---------- Face: detect + (optional) auth ----------
        try:
            face_boxes, face_scores = detect_faces_dnn(face_net, frame, conf_thresh=args.face_conf)
        except Exception as e:
            face_boxes, face_scores = [], []

        if len(face_boxes) > 0:
            # Choose the largest for auth display
            areas = [(i, (b[2]-b[0])*(b[3]-b[1])) for i,b in enumerate(face_boxes)]
            sel = max(areas, key=lambda x: x[1])[0]
            fb = face_boxes[sel]
            cv2.rectangle(frame, (fb[0], fb[1]), (fb[2], fb[3]), (0,255,0), 2)
            cv2.putText(frame, f"face:{face_scores[sel]:.2f}", (fb[0], max(20, fb[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if pca is not None:
                emb = compute_face_embedding(vgg_model, frame, fb)
                if emb is not None:
                    label, conf, accepted, dist = predict_face_label(emb, pca, knn, mean, thresh=0.8)
                    admin_status = f"{label if accepted else 'UNKNOWN'}"
                    admin_conf   = conf
                    txt = f"{admin_status} ({conf:.2f})"
                    color = (0,255,0) if accepted else (0,165,255)
                    cv2.putText(frame, txt, (fb[0], fb[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ---------- Person detection (MobileNet-SSD) ----------
        try:
            pboxes, pscores = detect_persons_dnn(person_net, frame, conf_thresh=args.person_conf)
        except Exception as e:
            pboxes, pscores = [], []

        for (pb, sc) in zip(pboxes, pscores):
            cv2.rectangle(frame, (pb[0], pb[1]), (pb[2], pb[3]), (255,140,0), 2)
            cv2.putText(frame, f"person:{sc:.2f}", (pb[0], max(20, pb[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,140,0), 2)

        # ---------- Hand: MediaPipe + simple gesture ----------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        gesture = "NONE"
        if result.multi_hand_landmarks and result.multi_handedness:
            for hlm, hd in zip(result.multi_hand_landmarks, result.multi_handedness):
                mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
                handed_label = hd.classification[0].label  # "Left"/"Right"
                n = count_extended_fingers(hlm, handed_label)
                gesture = gesture_from_fingers(n)
                # draw gesture near wrist
                wrist = hlm.landmark[0]
                px, py = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, f"{handed_label}:{gesture}", (px, max(20, py-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                gesture_hist.append(gesture)

        # ---------- HUD ----------
        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10.0 / (now - last_fps_t)
            last_fps_t = now
            cv2.putText(frame, f"FPS ~ {fps:.1f}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame, f"ADMIN: {admin_status} ({admin_conf:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50) if admin_status!='UNKNOWN' else (0,165,255), 2)

        cv2.putText(frame, "Keys: h-help  q-quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Perception Demo (FaceAuth + Person + Hand)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('h'):
            print("----- Help -----")
            print("q: Quit")
            print("h: Show this help")
            print("----------------")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()