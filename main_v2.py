# main_v2.py
# ------------------------------------------------------------
# ArcFace face + BODY embeddings (MobileNetV2 GAP ONNX + HSV color histogram)
# + body-only enroll hotkey ('b') for back/side views
# + temporal memory (EMA + hysteresis + margin + persistence)
# + ROI-first face detect, adaptive skip-frames with tracker (CSRT/KCF/MOSSE),
#   crop/resize face on person ROI
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

# ------------------------------
# Hyperparams (temporal memory)
# ------------------------------
EMA_ALPHA       = 0.7
CONFIRM_FRAMES  = 5
MARGIN_DELTA    = 0.06
PERSIST_FRAMES  = 10
IOU_STICKY      = 0.35

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

# ------------------------------------------------------------
# Utils: detector / tracker
# ------------------------------------------------------------
def load_person_det():
    if not (MOBILENET_PROTOTXT.exists() and MOBILENET_WEIGHTS.exists()):
        raise FileNotFoundError("Missing MobileNet-SSD (prototxt+caffemodel) under models/")
    net = cv2.dnn.readNetFromCaffe(str(MOBILENET_PROTOTXT), str(MOBILENET_WEIGHTS))
    # (Optional) tăng tốc nếu OpenCV build kèm OpenVINO/CUDA:
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)  # OpenVINO
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def detect_persons(net, frame, conf_thresh=0.4):
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

def build_face_app(det_size=(384,384), providers=None):
    if FaceAnalysis is None:
        raise ImportError("insightface chưa được cài. pip install insightface onnxruntime")
    app = FaceAnalysis(name='buffalo_l', providers=providers or ['CPUExecutionProvider'])
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
    """
    Chuẩn Keras MobileNetV2: scale về [-1,1].
    Input: uint8 RGB (H,W,3) hoặc (N,H,W,3). Output float32 (N,H,W,3)
    """
    x = x_uint8.astype(np.float32)
    x = x / 127.5 - 1.0
    return x

def build_mb2_onnx(path="mb2_gap.onnx"):
    if not Path(path).exists():
        raise FileNotFoundError(f"ONNX model not found: {path}. Hãy export bằng tf2onnx như hướng dẫn.")
    so = ort.SessionOptions()
    # Bạn có thể tinh chỉnh thread nếu muốn:
    # so.intra_op_num_threads = 0
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
    # Tên input lấy từ model (thường là "input")
    inp_name = ort_session.get_inputs()[0].name
    emb = ort_session.run(None, {inp_name: arr.astype(np.float32)})[0].reshape(-1).astype(np.float32)  # (1280,)
    col = hsv_histogram(roi224, bins=16)                                                                # (48,)
    feat = np.concatenate([emb, col], axis=0).astype(np.float32)                                       # (1328,)
    feat /= (np.linalg.norm(feat) + 1e-8)
    return feat

# ------------------------------
# Simple body head (PCA + KNN + centroid)
# ------------------------------
def train_pca_knn(X, label="admin", n_components=192, k=3):
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
    # cv2 tracker trả (x,y,w,h) -> (x1,y1,x2,y2)
    x,y,w,h = rect
    return (int(x), int(y), int(x+w), int(y+h))

def _tuple2rect(box):
    # (x1,y1,x2,y2) -> (x,y,w,h)
    x1,y1,x2,y2 = box
    return (int(x1), int(y1), int(x2-x1), int(y2-y1))

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--enroll", action="store_true")
    ap.add_argument("--label", type=str, default="admin")
    ap.add_argument("--accept_face", type=float, default=0.26)   # cosine
    ap.add_argument("--accept_body", type=float, default=0.26)
    ap.add_argument("--person_conf", type=float, default=0.4)
    ap.add_argument("--pca_dim_body", type=int, default=192)
    # --- NEW switches ---
    ap.add_argument("--use_roi", action="store_true", default=True,
                    help="Ưu tiên chạy face detector trên ROI quanh admin person_box")
    ap.add_argument("--roi_expand", type=float, default=0.30,
                    help="Hệ số nới biên ROI (0.30 ≈ 1.3x)")
    ap.add_argument("--skip_k", type=int, default=3,
                    help="Số khung bỏ qua detector giữa 2 lần detect (>=1)")
    ap.add_argument("--no_vis", action="store_true",
                    help="Tắt vẽ rectangle/putText/imshow để tăng FPS")
    ap.add_argument("--mb2_onnx", type=str, default="mb2_gap.onnx",
                    help="Đường dẫn ONNX MobileNetV2 (include_top=False, pooling='avg')")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)

    # Face (ArcFace)
    face_app = build_face_app(det_size=(384,384))  # giảm chi phí detector face

    # Person detector
    person_net = load_person_det()

    # Body ONNX
    mb2_sess = build_mb2_onnx(args.mb2_onnx)

    # -------- ENROLL ----------
    if args.enroll:
        print("[ENROLL] ArcFace(face) + BODY(ONNX MB2+HSV). Press 'b' to add BODY-only sample (largest person). 'q' to finish.")
        E = []; B = []
        while True:
            ok,frame = cap.read()
            if not ok: break
            frame=cv2.flip(frame,1)

            faces = detect_faces_arcface(face_app, frame, conf_thresh=0.5)
            pboxes,_ = detect_persons(person_net, frame, args.person_conf)

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
            bpca,bknn,bmean,bcent = train_pca_knn(B, args.label, args.pca_dim_body, 3)
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

    print("[RUN] ArcFace + cosine; BODY = MB2(ONNX)+HSV; temporal memory + ROI-first + skip-frames. q=quit.")

    admin_track = {"face_box": None, "person_box": None, "score_ema": 0.0, "frames_seen": 0, "frames_missing": 0}

    # Tracker cho person_box giữa các lần detect
    tracker = None
    tracker_box_xywh = None

    frame_idx = 0
    refresh_every = max(1, int(args.skip_k))   # detect lại mỗi K khung
    last_face_emb = None
    last_face_box = None

    while True:
        ok,frame = cap.read()
        if not ok: break
        frame=cv2.flip(frame,1)
        H,W = frame.shape[:2]
        frame_idx += 1

        # ----- quyết định có skip detectors không -----
        ready_admin = (admin_track["frames_seen"] >= CONFIRM_FRAMES and admin_track["score_ema"] > 0.5 and admin_track["person_box"] is not None)
        do_refresh = (frame_idx % refresh_every == 0) or (not ready_admin)

        pboxes = []
        faces  = []

        if ready_admin and not do_refresh:
            # --- SKIP DETECTORS: dùng tracker để cập nhật person_box ---
            if tracker is None and admin_track["person_box"] is not None:
                # khởi tạo tracker từ box hiện có
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
                    # tracker mất: buộc refresh ngay
                    do_refresh = True
            else:
                do_refresh = True

            # ROI-first face detect khi có box theo-vết
            if args.use_roi and not do_refresh and pboxes:
                rb = expand(pboxes[0], frame.shape, m=args.roi_expand)
                (rx1,ry1,rx2,ry2) = rb
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size > 0:
                    faces = detect_faces_arcface(face_app, roi, conf_thresh=0.5, offset_xy=(rx1,ry1))

        if do_refresh:
            # --- REFRESH DETECTORS: chạy full-frame person, face theo chế độ ---
            pboxes,_ = detect_persons(person_net, frame, args.person_conf)

            if args.use_roi and admin_track["person_box"] is not None:
                # Ưu tiên ROI quanh admin để tăng tốc & ổn định
                rb = expand(admin_track["person_box"], frame.shape, m=args.roi_expand)
                (rx1,ry1,rx2,ry2) = rb
                roi = frame[ry1:ry2, rx1:rx2]
                if roi.size > 0:
                    faces = detect_faces_arcface(face_app, roi, conf_thresh=0.5, offset_xy=(rx1,ry1))
            else:
                faces = detect_faces_arcface(face_app, frame, conf_thresh=0.5)

            # đồng bộ tracker theo box mới (nếu có best ở vòng trước)
            if admin_track["person_box"] is not None:
                tracker = _create_tracker()
                if tracker is not None:
                    tracker_box_xywh = tuple(_tuple2rect(admin_track["person_box"]))
                    tracker.init(frame, tracker_box_xywh)

        # ---- Face score ----
        face_scores = []
        if face_db is not None and faces:
            templates, centroid = face_db
            templates_T = templates.T if templates.size else templates
            centroid = centroid.astype(np.float32)
            # chỉ lấy top-2 mặt lớn nhất để giảm chi phí (thường đủ cho 1 admin)
            faces = sorted(faces, key=lambda f:(f["box"][2]-f["box"][0])*(f["box"][3]-f["box"][1]), reverse=True)[:2]
            for f in faces:
                e = f["feat"]
                # dùng lại embedding khung trước nếu IOU lớn và khung xen kẽ
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
            # cache embedding/box đầu tiên (mặt lớn nhất)
            last_face_emb = faces[0]["feat"]
            last_face_box = faces[0]["box"] if faces else None
        else:
            face_scores = [0.0]*len(faces)

        # ---- Body conf (ONNX) ----
        body_confs = []
        if body_models is not None and pboxes:
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

            face_ok = (best["face_conf"] >= args.accept_face)
            body_ok = (best["body_conf"] >= args.accept_body)
            is_admin = (admin_track["frames_seen"] >= CONFIRM_FRAMES) and (face_ok or body_ok)

        # ----------------- HUD (có thể tắt bằng --no_vis) -----------------
        if not args.no_vis:
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
            status = "ADMIN" if is_admin else "UNKNOWN"
            cv2.putText(frame, f"STATUS: {status} (face {face_best:.2f} | body {body_best:.2f} | EMA {admin_track['score_ema']:.2f})",
                        (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50) if is_admin else (0,165,255), 2)

            cv2.imshow("ADMIN (ArcFace + MB2(ONNX)+HSV) + Temporal + ROI/Skip", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
        else:
            if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
