# main_v2.py
# ------------------------------------------------------------
# ArcFace face + BODY embeddings (MobileNetV2 GAP ONNX + HSV color histogram)
# + body-only enroll hotkey ('b') for back/side views
# + temporal memory (EMA + hysteresis + margin + persistence)
# + ROI-first face detect, adaptive skip-frames with tracker (CSRT/KCF/MOSSE),
#   crop/resize face on person ROI
# + Mode gating: FACE_DOMINANT / BODY_DOMINANT with streak hysteresis
# + k=1 face when ADMIN (only compute embedding for the admin face)
# ------------------------------------------------------------

import cv2
import numpy as np
import argparse
from pathlib import Path
import time
import joblib
import onnxruntime as ort

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

# ============================================================
#                C O N F I G   (chỉnh tại đây)
# ============================================================
# Camera
CAM_INDEX      = 0

# File ONNX MobileNetV2 (body)
MB2_ONNX_PATH  = "mb2_gap.onnx"

# Detector người (MobileNet-SSD Caffe) kỳ vọng đặt trong ./models
# models/MobileNetSSD_deploy.prototxt + models/MobileNetSSD_deploy.caffemodel

# Temporal memory hyperparams
EMA_ALPHA       = 0.7
CONFIRM_FRAMES  = 5
MARGIN_DELTA    = 0.06
PERSIST_FRAMES  = 10
IOU_STICKY      = 0.35

# Ngưỡng/siêu tham số nhận dạng
ACCEPT_FACE     = 0.26
ACCEPT_BODY     = 0.26
PERSON_CONF     = 0.4
PCA_DIM_BODY    = 192

# --- Các tham số bạn yêu cầu cố định trong code ---
USE_ROI         = True      # thay --use_roi
SKIP_K          = 3         
ROI_EXPAND      = 0.30     
FACE_ON         = 2         
FACE_OFF        = 4        

# Hiển thị HUD (rectangle/putText/imshow)
NO_VIS          = False     # đặt True nếu muốn tắt vẽ để tăng FPS

# FaceAnalysis pack & det_size (có thể đổi sang 'buffalo_sc' và 320 để nhanh hơn)
FACE_PACK_NAME  = 'buffalo_sc' # buffalo_l or buffalo_sc (nhẹ hơn)
FACE_DET_SIZE   = (384, 384)  # (320,320) hay (384,384)

# ============================================================

# ------------------------------
# Paths
# ------------------------------
HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
DATA   = HERE / "data"
MODELS.mkdir(exist_ok=True, parents=True)
DATA.mkdir(exist_ok=True, parents=True)

MOBILENET_PROTOTXT = MODELS / "MobileNetSSD_deploy.prototxt"
MOBILENET_WEIGHTS  = MODELS / "MobileNetSSD_deploy.caffemodel"

FACE_TEMPL_PATH = DATA / "face_templates.npy"
FACE_CENT_PATH  = DATA / "face_centroid.npy"

BODY_PCA_PATH  = DATA / "body_pca.pkl"
BODY_CLS_PATH  = DATA / "body_cls_knn.pkl"
BODY_MEAN_PATH = DATA / "body_mean.npy"
BODY_CENT_PATH = DATA / "body_admin_centroid.npy"

# ------------------------------
# Utils: detector / tracker
# ------------------------------
def load_person_det():
    if not (MOBILENET_PROTOTXT.exists() and MOBILENET_WEIGHTS.exists()):
        raise FileNotFoundError("Missing MobileNet-SSD (prototxt+caffemodel) under models/")
    net = cv2.dnn.readNetFromCaffe(str(MOBILENET_PROTOTXT), str(MOBILENET_WEIGHTS))
    # Optional: tăng tốc nếu OpenCV build kèm OpenVINO/CUDA:
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)  # OpenVINO
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def detect_persons(net, frame, conf_thresh=PERSON_CONF):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300,300), 127.5)
    net.setInput(blob)
    det = net.forward()
    boxes, scores = [], []
    for i in range(det.shape[2]):
        conf = det[0,0,i,2]; cls = int(det[0,0,i,1])
        if cls==15 and conf>conf_thresh:
            box = det[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w-1,x2), min(h-1,y2)
            if x2>x1 and y2>y1:
                boxes.append((x1,y1,x2,y2)); scores.append(float(conf))
    return boxes, scores

def build_face_app(det_size=FACE_DET_SIZE, providers=None):
    if FaceAnalysis is None:
        raise ImportError("insightface chưa được cài. pip install insightface onnxruntime")
    app = FaceAnalysis(name=FACE_PACK_NAME, providers=providers or ['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=det_size)
    return app

def detect_faces_arcface(app, frame_bgr, conf_thresh=0.5, offset_xy=(0,0)):
    """offset_xy: cộng vào bbox để quy về toạ độ full-frame khi detect trên ROI."""
    ox, oy = offset_xy
    faces = app.get(frame_bgr)
    out = []
    for f in faces:
        x1,y1,x2,y2 = map(int, f.bbox[:4])
        if getattr(f, 'det_score', 1.0) < conf_thresh:
            continue
        feat = f.normed_embedding
        if feat is None or feat.size == 0:
            continue
        out.append(dict(
            box=(x1+ox, y1+oy, x2+ox, y2+oy),
            kps=f.kps,
            feat=feat.astype(np.float32)
        ))
    return out

# ------------------------------
# MobileNetV2 ONNX (body embedding)
# ------------------------------
def mb2_preprocess_keras_style(x_uint8):
    x = x_uint8.astype(np.float32)
    x = x / 127.5 - 1.0
    return x

def build_mb2_onnx(path=MB2_ONNX_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"ONNX model not found: {path}. Hãy export bằng tf2onnx như hướng dẫn.")
    so = ort.SessionOptions()
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    return sess

def expand(box, shape, m=0.20):
    x1,y1,x2,y2 = box
    H,W = shape[:2]; w=x2-x1; h=y2-y1
    x1 = max(0, int(x1 - m*w)); y1 = max(0, int(y1 - m*h))
    x2 = min(W-1, int(x2 + m*w)); y2 = min(H-1, int(y2 + m*h))
    return (x1,y1,x2,y2)

def body_arr(frame, box):
    x1,y1,x2,y2 = expand(box, frame.shape, 0.20)
    roi = frame[y1:y2, x1:x2]
    if roi.size==0: return None
    roi224 = cv2.resize(roi, (224,224))
    roi_rgb = cv2.cvtColor(roi224, cv2.COLOR_BGR2RGB)
    arr = mb2_preprocess_keras_style(roi_rgb)[None,...]  # (1,224,224,3) float32
    return roi224, arr

def hsv_histogram(roi_bgr, bins=16):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    histH = cv2.calcHist([hsv],[0],None,[bins],[0,180]).flatten()
    histS = cv2.calcHist([hsv],[1],None,[bins],[0,256]).flatten()
    histV = cv2.calcHist([hsv],[2],None,[bins],[0,256]).flatten()
    h = np.concatenate([histH, histS, histV]).astype(np.float32)
    h /= (np.linalg.norm(h)+1e-8)
    return h

def body_feature_onnx(frame, box, ort_session):
    out = body_arr(frame, box)
    if out is None: return None
    roi224, arr = out
    inp_name = ort_session.get_inputs()[0].name
    emb = ort_session.run(None, {inp_name: arr.astype(np.float32)})[0].reshape(-1).astype(np.float32)  # (1280,)
    col = hsv_histogram(roi224, bins=16)                                                                # (48,)
    feat = np.concatenate([emb, col], axis=0).astype(np.float32)                                       # (1328,)
    feat /= (np.linalg.norm(feat) + 1e-8)
    return feat

# ------------------------------
# Simple body head (PCA + KNN + centroid)
# ------------------------------
def train_pca_knn(X, label="admin", n_components=PCA_DIM_BODY, k=3):
    mean = X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    Z = pca.fit_transform(X - mean)
    knn = KNeighborsClassifier(n_neighbors=min(k,len(Z)))
    y = np.array([label]*len(Z))
    knn.fit(Z,y)
    cent = Z.mean(axis=0)
    return pca, knn, mean, cent

def predict_body_conf(feat, pca, knn, mean, centroid=None):
    Z = pca.transform((feat-mean).reshape(1,-1))
    dists,_ = knn.kneighbors(Z, 1, return_distance=True)
    d=float(dists[0,0])
    conf=np.exp(-d)
    if centroid is not None:
        z=Z[0]
        cos=float(np.dot(z,centroid)/(np.linalg.norm(z)*np.linalg.norm(centroid)+1e-8))
        cos01 = max(0.0, min(1.0, (cos+1)/2))
        conf = 0.5*conf + 0.5*cos01
    return conf

# ------------------------------
# Misc helpers
# ------------------------------
def center_of(box):
    x1,y1,x2,y2 = box
    return ( (x1+x2)//2, (y1+y2)//2 )

def pick_person_for_face(fbox, pboxes):
    if not pboxes: return None
    fcx,fcy = center_of(fbox)
    containing = []
    for i,(x1,y1,x2,y2) in enumerate(pboxes):
        if x1<=fcx<=x2 and y1<=fcy<=y2:
            containing.append((i,(x2-x1)*(y2-y1)))
    if containing:
        containing.sort(key=lambda t:-t[1])
        return pboxes[containing[0][0]]
    def cdist(pb):
        pcx,pcy = center_of(pb)
        return abs(pcx-fcx) + abs(pcy-fcy)
    return min(pboxes, key=cdist)

def bbox_iou(a, b):
    if a is None or b is None: return 0.0
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih; ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6
    return inter/ua

def ema(prev, new, alpha=EMA_ALPHA):
    return alpha*prev + (1-alpha)*new

# Robust tracker ctor: CSRT -> KCF -> MOSSE (legacy & non-legacy)
def _get_ctor(path):
    cur = cv2
    for name in path.split('.'):
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur

def _create_tracker():
    candidates = [
        "legacy.TrackerCSRT_create",  "TrackerCSRT_create",
        "legacy.TrackerKCF_create",   "TrackerKCF_create",
        "legacy.TrackerMOSSE_create", "TrackerMOSSE_create",
    ]
    for c in candidates:
        ctor = _get_ctor(c)
        if callable(ctor):
            try:
                return ctor()
            except Exception:
                continue
    return None

def _rect2tuple(rect):
    x,y,w,h = rect
    return (int(x), int(y), int(x+w), int(y+h))

def _tuple2rect(box):
    x1,y1,x2,y2 = box
    return (int(x1), int(y1), int(x2-x1), int(y2-y1))

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # Chỉ giữ 2 tham số CLI theo yêu cầu
    ap = argparse.ArgumentParser()
    ap.add_argument("--enroll", action="store_true")
    ap.add_argument("--label", type=str, default="admin")
    args = ap.parse_args()

    cap = cv2.VideoCapture(CAM_INDEX)

    # Face (ArcFace)
    face_app = build_face_app(det_size=FACE_DET_SIZE)  # có thể đổi FACE_PACK_NAME/DET_SIZE trong config

    # Person detector
    person_net = load_person_det()

    # Body ONNX
    mb2_sess = build_mb2_onnx(MB2_ONNX_PATH)

    # -------- ENROLL ----------
    if args.enroll:
        print("[ENROLL] ArcFace(face) + BODY(ONNX MB2+HSV). Press 'b' to add BODY-only sample (largest person). 'q' to finish.")
        E = []; B = []
        while True:
            ok,frame = cap.read()
            if not ok: break
            frame=cv2.flip(frame,1)

            faces = detect_faces_arcface(face_app, frame, conf_thresh=0.5)
            pboxes,_ = detect_persons(person_net, frame, PERSON_CONF)

            if faces:
                faces_sorted = sorted(faces, key=lambda f:(f["box"][2]-f["box"][0])*(f["box"][3]-f["box"][1]), reverse=True)
                for f in faces_sorted:
                    E.append(f["feat"])
                    x1,y1,x2,y2 = f["box"]
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    pb = pick_person_for_face(f["box"], pboxes) if pboxes else None
                    if pb is not None:
                        feat = body_feature_onnx(frame, pb, mb2_sess)
                        if feat is not None:
                            B.append(feat)
                            cv2.rectangle(frame,(pb[0],pb[1]),(pb[2],pb[3]),(255,140,0),2)

            # HUD
            cv2.putText(frame, f"ENROLL: face {len(E)} | body {len(B)}  ('b' body-only, 'q' finish)", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

            # Show pboxes
            if pboxes:
                for pb in pboxes:
                    cv2.rectangle(frame,(pb[0],pb[1]),(pb[2],pb[3]),(200,200,200),1)

            cv2.imshow("ENROLL (ArcFace + MB2(ONNX)+HSV)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('b'):
                if pboxes:
                    areas = [(i,(pb[2]-pb[0])*(pb[3]-pb[1])) for i,pb in enumerate(pboxes)]
                    j,_ = max(areas, key=lambda t:t[1])
                    feat = body_feature_onnx(frame, pboxes[j], mb2_sess)
                    if feat is not None:
                        B.append(feat)

        cv2.destroyAllWindows()

        # Save face
        E = np.array(E, dtype=np.float32)
        if len(E) < 10:
            print("[ENROLL] Not enough FACE samples (need >=10; recommend 40–60)."); return
        centroid = E.mean(axis=0); centroid /= (np.linalg.norm(centroid)+1e-8)
        np.save(FACE_TEMPL_PATH, E); np.save(FACE_CENT_PATH, centroid)

        # Save body
        if len(B) >= 10:
            B = np.array(B, dtype=np.float32)
            bpca,bknn,bmean,bcent = train_pca_knn(B, args.label, PCA_DIM_BODY, 3)
            joblib.dump(bpca, BODY_PCA_PATH); joblib.dump(bknn, BODY_CLS_PATH)
            np.save(BODY_MEAN_PATH, bmean); np.save(BODY_CENT_PATH, bcent)
            print("[ENROLL] Saved BODY models.")
        else:
            print("[ENROLL] Not enough BODY samples; body auth will be weaker.")
        print("[ENROLL] Saved FACE templates + centroid in", DATA)
        return

    # -------- LOAD DB ----------
    if not (FACE_TEMPL_PATH.exists() and FACE_CENT_PATH.exists()):
        print("[WARN] No FACE templates; run with --enroll."); face_db=None
    else:
        face_db = (np.load(FACE_TEMPL_PATH), np.load(FACE_CENT_PATH))

    if not (BODY_PCA_PATH.exists() and BODY_CLS_PATH.exists() and BODY_MEAN_PATH.exists()):
        print("[WARN] No BODY models; run with --enroll."); body_models=None
    else:
        body_models = (joblib.load(BODY_PCA_PATH), joblib.load(BODY_CLS_PATH), np.load(BODY_MEAN_PATH), np.load(BODY_CENT_PATH) if BODY_CENT_PATH.exists() else None)

    print(f"[RUN] ArcFace + cosine; BODY = MB2(ONNX)+HSV; temporal + ROI/Skip; MODE gating.")
    print(f"      USE_ROI={USE_ROI} | SKIP_K={SKIP_K} | ROI_EXPAND={ROI_EXPAND} | FACE_ON={FACE_ON} | FACE_OFF={FACE_OFF}")

    admin_track = {"face_box": None, "person_box": None, "score_ema": 0.0, "frames_seen": 0, "frames_missing": 0}

    # Tracker cho person_box giữa các lần detect
    tracker = None
    tracker_box_xywh = None

    # --- Gating mode & streak ---
    MODE_FACE_DOMINANT = 1
    MODE_BODY_DOMINANT = 2
    mode = MODE_BODY_DOMINANT  # khởi đầu: ưu tiên body
    seen_face_streak = 0
    no_face_streak   = 0

    frame_idx = 0
    refresh_every = max(1, int(SKIP_K))   # detect lại mỗi K khung
    last_face_emb = None
    last_face_box = None

    while True:
        ok,frame = cap.read()
        if not ok: break
        frame=cv2.flip(frame,1)
        H,W = frame.shape[:2]
        frame_idx += 1

        ready_admin = (admin_track["frames_seen"] >= CONFIRM_FRAMES and admin_track["score_ema"] > 0.5 and admin_track["person_box"] is not None)
        do_refresh = (frame_idx % refresh_every == 0) or (not ready_admin)

        pboxes = []
        faces  = []

        # -------- MODE GATING: quyết định gọi mô-đun nào --------
        if mode == MODE_FACE_DOMINANT:
            # ƯU TIÊN FACE — không tính body để tiết kiệm
            if ready_admin and not do_refresh:
                if tracker is None and admin_track["person_box"] is not None:
                    tracker = _create_tracker()
                    if tracker is not None:
                        tracker_box_xywh = tuple(_tuple2rect(admin_track["person_box"]))
                        tracker.init(frame, tracker_box_xywh)
                    else:
                        do_refresh = True
                ok_trk = False
                if tracker is not None:
                    ok_trk, rect = tracker.update(frame)
                    if ok_trk:
                        tracker_box_xywh = rect
                        tracked_box = _rect2tuple(rect)
                        pboxes = [tracked_box]
                    else:
                        do_refresh = True
                else:
                    do_refresh = True
                # Face ROI-first
                if USE_ROI and not do_refresh and pboxes:
                    rb = expand(pboxes[0], frame.shape, m=ROI_EXPAND)
                    rx1,ry1,rx2,ry2 = rb
                    roi = frame[ry1:ry2, rx1:rx2]
                    if roi.size > 0:
                        faces = detect_faces_arcface(face_app, roi, conf_thresh=0.5, offset_xy=(rx1,ry1))
            else:
                # refresh: hạn chế chạy person full-frame
                if admin_track["person_box"] is not None and USE_ROI:
                    rb = expand(admin_track["person_box"], frame.shape, m=ROI_EXPAND)
                    rx1,ry1,rx2,ry2 = rb
                    roi = frame[ry1:ry2, rx1:rx2]
                    if roi.size > 0:
                        faces = detect_faces_arcface(face_app, roi, conf_thresh=0.5, offset_xy=(rx1,ry1))
                    pboxes = [admin_track["person_box"]]
                else:
                    pboxes,_ = detect_persons(person_net, frame, PERSON_CONF)
                    if pboxes:
                        rb = expand(pboxes[0], frame.shape, m=ROI_EXPAND)
                        rx1,ry1,rx2,ry2 = rb
                        roi = frame[ry1:ry2, rx1:rx2]
                        if roi.size > 0:
                            faces = detect_faces_arcface(face_app, roi, conf_thresh=0.5, offset_xy=(rx1,ry1))
                    else:
                        faces = detect_faces_arcface(face_app, frame, conf_thresh=0.5)

        else:  # MODE_BODY_DOMINANT
            # ƯU TIÊN BODY — không gọi face để tiết kiệm
            if ready_admin and not do_refresh and admin_track["person_box"] is not None:
                if tracker is None:
                    tracker = _create_tracker()
                    if tracker is not None:
                        tracker_box_xywh = tuple(_tuple2rect(admin_track["person_box"]))
                        tracker.init(frame, tracker_box_xywh)
                    else:
                        do_refresh = True
                ok_trk = False
                if tracker is not None:
                    ok_trk, rect = tracker.update(frame)
                    if ok_trk:
                        tracker_box_xywh = rect
                        tracked_box = _rect2tuple(rect)
                        pboxes = [tracked_box]
                    else:
                        do_refresh = True
                else:
                    do_refresh = True
            else:
                pboxes,_ = detect_persons(person_net, frame, PERSON_CONF)
            faces = []  # cứng: không detect mặt ở BODY_DOMINANT

        # ---- Face score (k=1 khi ADMIN) ----
        face_scores = []
        if face_db is not None and faces:
            templates, centroid = face_db
            templates_T = templates.T if templates.size else templates
            centroid = centroid.astype(np.float32)

            admin_active = (admin_track["frames_seen"] >= CONFIRM_FRAMES and admin_track["face_box"] is not None)
            if admin_active:
                faces = sorted(faces, key=lambda f: bbox_iou(f["box"], admin_track["face_box"]), reverse=True)[:1]
            else:
                faces = sorted(faces, key=lambda f:(f["box"][2]-f["box"][0])*(f["box"][3]-f["box"][1]), reverse=True)[:2]

            for f in faces:
                e = f["feat"]
                # tái dùng embedding nếu IOU lớn và khung xen kẽ
                if last_face_emb is not None and last_face_box is not None and (frame_idx % 2 == 0):
                    if bbox_iou(f["box"], last_face_box) >= 0.7:
                        e_use = last_face_emb
                    else:
                        e_use = e
                else:
                    e_use = e
                cs = float(np.max(e_use @ templates_T)) if templates.size else 0.0
                cc = float(e_use @ centroid)
                face_scores.append(0.5*cs + 0.5*cc)
            if faces:
                last_face_emb = faces[0]["feat"]
                last_face_box = faces[0]["box"]
            else:
                last_face_emb = None
                last_face_box = None
        else:
            face_scores = [0.0]*len(faces)

        # ---- Body conf (ONNX) ----
        body_confs = []
        if body_models is not None and pboxes:
            if mode == MODE_FACE_DOMINANT:
                body_confs = [0.0]*len(pboxes)
            else:
                bpca,bknn,bmean,bcent = body_models
                for pb in pboxes:
                    feat = body_feature_onnx(frame, pb, mb2_sess)
                    if feat is None:
                        body_confs.append(0.0)
                    else:
                        body_confs.append(float(predict_body_conf(feat, bpca, bknn, bmean, centroid=bcent)))
        else:
            body_confs = [0.0]*len(pboxes)

        # ---- Ghép candidate face+person ----
        candidates = []
        used_person = set()
        for i, f in enumerate(faces):
            fb = f["box"]
            pb = pick_person_for_face(fb, pboxes) if pboxes else None
            j = pboxes.index(pb) if (pb in pboxes) else None
            if j is not None: used_person.add(j)
            fconf = face_scores[i] if i < len(face_scores) else 0.0
            bconf = body_confs[j] if (j is not None and j < len(body_confs)) else 0.0
            score = max(fconf, 0.95*bconf)
            candidates.append(dict(face_box=fb, person_box=pb, face_conf=fconf, body_conf=bconf, score=score))

        for j, pb in enumerate(pboxes):
            if j in used_person: continue
            bconf = body_confs[j] if j < len(body_confs) else 0.0
            score = 0.95*bconf
            candidates.append(dict(face_box=None, person_box=pb, face_conf=0.0, body_conf=bconf, score=score))

        best = max(candidates, key=lambda c: c["score"]) if candidates else None
        is_admin = False

        if best is None:
            admin_track["frames_missing"] += 1
            if admin_track["frames_missing"] > PERSIST_FRAMES:
                admin_track.update({"face_box":None,"person_box":None,"score_ema":0.0,"frames_seen":0})
        else:
            still_visible = (bbox_iou(admin_track["person_box"], best["person_box"]) > IOU_STICKY)
            if admin_track["frames_seen"] == 0 and admin_track["person_box"] is None:
                admin_track["score_ema"] = best["score"]
            else:
                admin_track["score_ema"] = ema(admin_track["score_ema"], best["score"])
            challenger_better = (best["score"] > admin_track["score_ema"] + MARGIN_DELTA)

            if admin_track["person_box"] is None or (not still_visible and challenger_better):
                admin_track.update({
                    "face_box": best["face_box"],
                    "person_box": best["person_box"],
                    "score_ema": best["score"],
                    "frames_seen": 1,
                    "frames_missing": 0
                })
                # re-init tracker theo box mới
                if admin_track["person_box"] is not None:
                    tracker = _create_tracker()
                    if tracker is not None:
                        tracker_box_xywh = tuple(_tuple2rect(admin_track["person_box"]))
                        tracker.init(frame, tracker_box_xywh)
            else:
                admin_track["frames_seen"] += 1
                admin_track["frames_missing"] = 0
                if bbox_iou(admin_track["person_box"], best["person_box"]) > IOU_STICKY:
                    admin_track["person_box"] = best["person_box"]
                    if best["face_box"] is not None:
                        admin_track["face_box"] = best["face_box"]

            face_ok = (best["face_conf"] >= ACCEPT_FACE)
            body_ok = (best["body_conf"] >= ACCEPT_BODY)
            is_admin = (admin_track["frames_seen"] >= CONFIRM_FRAMES) and (face_ok or body_ok)

        # ---- Cập nhật streaks & chuyển MODE (hysteresis) ----
        has_face_now = (len(faces) > 0)
        if has_face_now:
            seen_face_streak += 1
            no_face_streak = 0
        else:
            seen_face_streak = 0
            no_face_streak += 1

        if mode == MODE_BODY_DOMINANT and seen_face_streak >= FACE_ON:
            mode = MODE_FACE_DOMINANT
        elif mode == MODE_FACE_DOMINANT and no_face_streak >= FACE_OFF:
            mode = MODE_BODY_DOMINANT

        # ----------------- HUD (có thể tắt bằng NO_VIS) -----------------
        if not NO_VIS:
            for idx, pb in enumerate(pboxes):
                color = (50,220,50) if (is_admin and admin_track["person_box"] is not None and bbox_iou(pb, admin_track["person_box"]) > IOU_STICKY) else (255,140,0)
                cv2.rectangle(frame, (pb[0],pb[1]), (pb[2],pb[3]), color, 2)
                if is_admin and admin_track["person_box"] is not None and bbox_iou(pb, admin_track["person_box"]) > IOU_STICKY:
                    cv2.putText(frame, "ADMIN", (pb[0], max(20, pb[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,220,50), 2)

            for f in faces:
                x1,y1,x2,y2 = f["box"]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                if is_admin and admin_track["face_box"] is not None and bbox_iou(f["box"], admin_track["face_box"]) > IOU_STICKY:
                    cv2.putText(frame, "ADMIN", (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            if best is None:
                face_best = 0.0; body_best = 0.0
            else:
                face_best = best["face_conf"]; body_best = best["body_conf"]
            status = f"{'ADMIN' if is_admin else 'UNKNOWN'} | MODE: {'FACE' if mode==MODE_FACE_DOMINANT else 'BODY'}"
            cv2.putText(frame, f"{status} (face {face_best:.2f} | body {body_best:.2f} | EMA {admin_track['score_ema']:.2f})",
                        (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50) if is_admin else (0,165,255), 2)

            cv2.imshow("ADMIN (ArcFace + MB2(ONNX)+HSV) + Temporal + ROI/Skip + MODE Gating", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
        else:
            if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
