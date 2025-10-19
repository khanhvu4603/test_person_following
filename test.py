# robot_vision_demo/main.py
# ------------------------------------------------------------
# Webcam perception demo (ADMIN-only labeling, best-match + temporal memory):
# - Face Auth: VGG16(fc1) + PCA/KNN  → per-face confidence
# - Person: MobileNet-SSD (OpenCV DNN)
# - Body Auth: MobileNetV2(GAP) + PCA/KNN → per-person confidence
# - MediaPipe Hands: disabled
#
# ADMIN được chọn theo:
#   score = max(conf_face, 0.9*conf_body)  (best-match toàn cục trong frame)
# Sau đó áp dụng bộ nhớ theo thời gian:
#   - EMA làm mượt score
#   - Hysteresis: cần số khung liên tiếp để “đăng quang”
#   - Margin: chỉ đổi ADMIN nếu người mới hơn ADMIN hiện tại ≥ MARGIN_DELTA
#   - Persist: mất dấu ngắn hạn vẫn giữ ADMIN
# ------------------------------------------------------------

import cv2
import numpy as np
import argparse
from pathlib import Path

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_pre
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mb2_pre
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib

# ---------------- Hyper-params for temporal memory ----------------
EMA_ALPHA       = 0.7   # hệ số làm mượt điểm (0..1); 0.7 → mượt vừa phải
CONFIRM_FRAMES  = 5     # cần >= n khung liên tiếp để “lên ADMIN”
MARGIN_DELTA    = 0.05  # người thách thức phải hơn ADMIN hiện tại ít nhất margin này
PERSIST_FRAMES  = 8     # mất dấu <= n khung vẫn giữ ADMIN
IOU_STICKY      = 0.30  # IoU tối thiểu để coi là “cùng người” và cập nhật bbox

# ---------------- Paths ----------------
HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
DATA   = HERE / "data"
MODELS.mkdir(exist_ok=True, parents=True)
DATA.mkdir(exist_ok=True, parents=True)

# Caffe models
MOBILENET_PROTOTXT = MODELS / "MobileNetSSD_deploy.prototxt"
MOBILENET_WEIGHTS  = MODELS / "MobileNetSSD_deploy.caffemodel"
FACE_TXT   = MODELS / "deploy.prototxt"
FACE_PROTO = MODELS / "res10_300x300_ssd_iter_140000.caffemodel"

# Save files
FACE_PCA_PATH  = DATA / "face_pca.pkl"
FACE_CLS_PATH  = DATA / "face_cls_knn.pkl"
FACE_MEAN_PATH = DATA / "face_mean.npy"
FACE_CENT_PATH = DATA / "face_admin_centroid.npy"

BODY_PCA_PATH  = DATA / "body_pca.pkl"
BODY_CLS_PATH  = DATA / "body_cls_knn.pkl"
BODY_MEAN_PATH = DATA / "body_mean.npy"
BODY_CENT_PATH = DATA / "body_admin_centroid.npy"

# --------------- Loaders ---------------
def load_face_det():
    if not (FACE_TXT.exists() and FACE_PROTO.exists()):
        raise FileNotFoundError("Missing face detector (deploy.prototxt, res10_*.caffemodel) under models/")
    return cv2.dnn.readNetFromCaffe(str(FACE_TXT), str(FACE_PROTO))

def load_person_det():
    if not (MOBILENET_PROTOTXT.exists() and MOBILENET_WEIGHTS.exists()):
        raise FileNotFoundError("Missing MobileNet-SSD (prototxt+caffemodel) under models/")
    return cv2.dnn.readNetFromCaffe(str(MOBILENET_PROTOTXT), str(MOBILENET_WEIGHTS))

def detect_faces(net, frame, conf_thresh=0.6):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)
    det = net.forward()
    boxes, scores = [], []
    for i in range(det.shape[2]):
        conf = det[0,0,i,2]
        if conf > conf_thresh:
            box = det[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w-1,x2), min(h-1,y2)
            if x2>x1 and y2>y1:
                boxes.append((x1,y1,x2,y2)); scores.append(float(conf))
    return boxes, scores

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

def build_vgg_fc1():
    base = VGG16(weights="imagenet", include_top=True)
    return Model(inputs=base.input, outputs=base.get_layer("fc1").output)  # (None,4096)

def build_mb2_gap():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")  # (None,1280)

# ---------------- Utils ----------------
def expand(box, shape, m=0.20):
    x1,y1,x2,y2 = box
    H,W = shape[:2]; w=x2-x1; h=y2-y1
    x1 = max(0, int(x1 - m*w)); y1 = max(0, int(y1 - m*h))
    x2 = min(W-1, int(x2 + m*w)); y2 = min(H-1, int(y2 + m*h))
    return (x1,y1,x2,y2)

def face_arr(frame, box):
    x1,y1,x2,y2 = expand(box, frame.shape, 0.20)
    roi = frame[y1:y2, x1:x2]
    if roi.size==0: return None
    roi = cv2.resize(roi, (224,224))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    arr = img_to_array(roi)[None,...]
    return vgg_pre(arr)

def body_arr(frame, box):
    x1,y1,x2,y2 = expand(box, frame.shape, 0.20)
    roi = frame[y1:y2, x1:x2]
    if roi.size==0: return None
    roi = cv2.resize(roi, (224,224))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    arr = img_to_array(roi)[None,...]
    return mb2_pre(arr)

def center_of(box):
    x1,y1,x2,y2 = box
    return ( (x1+x2)//2, (y1+y2)//2 )

def pick_person_for_face(fbox, pboxes):
    """Chọn bbox người chứa tâm mặt; nếu không có thì gần tâm mặt nhất."""
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

def train_pca_knn(X, label="admin", n_components=128, k=3):
    mean = X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    Z = pca.fit_transform(X - mean)
    knn = KNeighborsClassifier(n_neighbors=min(k,len(Z)))
    y = np.array([label]*len(Z))
    knn.fit(Z,y)
    cent = Z.mean(axis=0)
    return pca, knn, mean, cent

def predict_label(emb, pca, knn, mean, centroid=None):
    Z = pca.transform((emb-mean).reshape(1,-1))
    dists,_ = knn.kneighbors(Z, 1, return_distance=True)
    d=float(dists[0,0])
    conf=np.exp(-d)   # ánh xạ khoảng cách về (0..1) — phụ thuộc dữ liệu
    if centroid is not None:
        z=Z[0]
        cos=float(np.dot(z,centroid)/(np.linalg.norm(z)*np.linalg.norm(centroid)+1e-8))
        cos01 = max(0.0, min(1.0, (cos+1)/2))
        conf = 0.5*conf + 0.5*cos01
    return conf

# ---------- IoU & EMA ----------
def bbox_iou(a, b):
    if a is None or b is None: return 0.0
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1)
    inter=iw*ih; ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6
    return inter/ua

def ema(prev, new, alpha=EMA_ALPHA):
    return alpha*prev + (1-alpha)*new

# ---------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--enroll", action="store_true")
    ap.add_argument("--label", type=str, default="admin")
    ap.add_argument("--accept_face", type=float, default=0.23)  # bạn có thể hạ 0.25 khi test
    ap.add_argument("--accept_body", type=float, default=0.23)
    ap.add_argument("--face_conf", type=float, default=0.7)
    ap.add_argument("--person_conf", type=float, default=0.4)
    ap.add_argument("--pca_dim_face", type=int, default=128)
    ap.add_argument("--pca_dim_body", type=int, default=128)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)

    face_net   = load_face_det()
    person_net = load_person_det()
    vgg = build_vgg_fc1()
    mb2 = build_mb2_gap()

    # --------- ENROLL ---------
    if args.enroll:
        print("[ENROLL] Collecting FACE + BODY embeddings for", args.label)
        F,B=[],[]
        while True:
            ok,frame = cap.read()
            if not ok: break
            frame=cv2.flip(frame,1)
            fboxes,_ = detect_faces(face_net, frame, args.face_conf)
            pboxes,_ = detect_persons(person_net, frame, args.person_conf)
            if fboxes:
                # thu tất cả mặt trong khung để đa dạng
                for fb in fboxes:
                    arr = face_arr(frame, fb)
                    if arr is not None:
                        f = vgg.predict(arr, verbose=0).flatten(); F.append(f)
                        cv2.rectangle(frame,(fb[0],fb[1]),(fb[2],fb[3]),(0,255,0),2)
                # ghép mỗi mặt với một bbox người
                for fb in fboxes:
                    cand = pick_person_for_face(fb, pboxes) if pboxes else None
                    if cand is not None:
                        arrb = body_arr(frame, cand)
                        if arrb is not None:
                            b = mb2.predict(arrb, verbose=0).flatten(); B.append(b)
                            cv2.rectangle(frame,(cand[0],cand[1]),(cand[2],cand[3]),(255,140,0),2)
            cv2.putText(frame, f"ENROLL: face {len(F)} | body {len(B)}  (q to finish)", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.imshow("ENROLL", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
        cv2.destroyAllWindows()
        F=np.array(F); B=np.array(B)
        if len(F)<10 or len(B)<10:
            print("[ENROLL] Not enough samples (need >=10 each; recommend 40–60)."); return
        fpca,fknn,fmean,fcent = train_pca_knn(F, args.label, args.pca_dim_face, 3)
        bpca,bknn,bmean,bcent = train_pca_knn(B, args.label, args.pca_dim_body, 3)
        joblib.dump(fpca, FACE_PCA_PATH); joblib.dump(fknn, FACE_CLS_PATH); np.save(FACE_MEAN_PATH, fmean); np.save(FACE_CENT_PATH, fcent)
        joblib.dump(bpca, BODY_PCA_PATH); joblib.dump(bknn, BODY_CLS_PATH); np.save(BODY_MEAN_PATH, bmean); np.save(BODY_CENT_PATH, bcent)
        print("[ENROLL] Saved face/body models in", DATA)
        return

    # --------- INFERENCE ---------
    if not (FACE_PCA_PATH.exists() and FACE_CLS_PATH.exists() and FACE_MEAN_PATH.exists()):
        print("[WARN] No FACE models; run with --enroll."); face_models=None
    else:
        face_models = (joblib.load(FACE_PCA_PATH), joblib.load(FACE_CLS_PATH), np.load(FACE_MEAN_PATH), np.load(FACE_CENT_PATH) if FACE_CENT_PATH.exists() else None)

    if not (BODY_PCA_PATH.exists() and BODY_CLS_PATH.exists() and BODY_MEAN_PATH.exists()):
        print("[WARN] No BODY models; run with --enroll."); body_models=None
    else:
        body_models = (joblib.load(BODY_PCA_PATH), joblib.load(BODY_CLS_PATH), np.load(BODY_MEAN_PATH), np.load(BODY_CENT_PATH) if BODY_CENT_PATH.exists() else None)

    print("[RUN] ADMIN is selected by highest similarity (face/body) with temporal memory. q=quit.")

    # ---- Trạng thái theo thời gian (temporal memory) ----
    admin_track = {
        "face_box": None,
        "person_box": None,
        "score_ema": 0.0,
        "frames_seen": 0,      # số khung đã xác nhận liên tiếp
        "frames_missing": 0    # số khung mất dấu liên tiếp
    }

    while True:
        ok,frame = cap.read()
        if not ok: break
        frame=cv2.flip(frame,1)
        H,W=frame.shape[:2]

        fboxes,_ = detect_faces(face_net, frame, args.face_conf)
        pboxes,_ = detect_persons(person_net, frame, args.person_conf)

        # per-face conf
        face_confs = []
        if face_models is not None:
            fpca,fknn,fmean,fcent = face_models
            for fb in fboxes:
                arr = face_arr(frame, fb)
                if arr is None:
                    face_confs.append(0.0)
                else:
                    femb = vgg.predict(arr, verbose=0).flatten()
                    face_confs.append(float(predict_label(femb, fpca, fknn, fmean, centroid=fcent)))
        else:
            face_confs = [0.0]*len(fboxes)

        # per-person conf
        body_confs = []
        if body_models is not None:
            bpca,bknn,bmean,bcent = body_models
            for pb in pboxes:
                arrb = body_arr(frame, pb)
                if arrb is None:
                    body_confs.append(0.0)
                else:
                    bemb = mb2.predict(arrb, verbose=0).flatten()
                    body_confs.append(float(predict_label(bemb, bpca, bknn, bmean, centroid=bcent)))
        else:
            body_confs = [0.0]*len(pboxes)

        # Build paired candidates (face→best person) + body-only
        candidates = []
        used_person = set()
        for i, fb in enumerate(fboxes):
            pb = pick_person_for_face(fb, pboxes) if pboxes else None
            j = pboxes.index(pb) if (pb in pboxes) else None
            if j is not None: used_person.add(j)
            fconf = face_confs[i] if i < len(face_confs) else 0.0
            bconf = body_confs[j] if (j is not None and j < len(body_confs)) else 0.0
            score = max(fconf, 0.9*bconf)
            candidates.append(dict(face_box=fb, person_box=pb, face_conf=fconf, body_conf=bconf, score=score))

        for j, pb in enumerate(pboxes):
            if j in used_person: continue
            bconf = body_confs[j] if j < len(body_confs) else 0.0
            score = 0.9*bconf
            candidates.append(dict(face_box=None, person_box=pb, face_conf=0.0, body_conf=bconf, score=score))

        # --------- Temporal decision (EMA + hysteresis + margin) ----------
        best = max(candidates, key=lambda c: c["score"]) if candidates else None
        is_admin = False

        if best is None:
            # không ai trong khung
            admin_track["frames_missing"] += 1
            if admin_track["frames_missing"] > PERSIST_FRAMES:
                admin_track.update({"face_box":None,"person_box":None,"score_ema":0.0,"frames_seen":0})
        else:
            # tính IoU với ADMIN hiện tại (nếu có)
            still_visible = bbox_iou(admin_track["person_box"], best["person_box"]) > IOU_STICKY
            # cập nhật EMA (nếu chưa có score_ema, lấy score hiện tại)
            if admin_track["frames_seen"] == 0 and admin_track["person_box"] is None:
                admin_track["score_ema"] = best["score"]
            else:
                admin_track["score_ema"] = ema(admin_track["score_ema"], best["score"])

            challenger_better = (best["score"] > admin_track["score_ema"] + MARGIN_DELTA)

            if admin_track["person_box"] is None or (not still_visible and challenger_better):
                # khởi tạo hoặc đổi ADMIN (khi hơn đủ margin)
                admin_track.update({
                    "face_box": best["face_box"],
                    "person_box": best["person_box"],
                    "score_ema": best["score"],
                    "frames_seen": 1,
                    "frames_missing": 0
                })
            else:
                # duy trì ADMIN hiện tại
                admin_track["frames_seen"] += 1
                admin_track["frames_missing"] = 0
                # nếu bbox mới cùng người (IoU đủ), cập nhật để bám sát hơn
                if bbox_iou(admin_track["person_box"], best["person_box"]) > IOU_STICKY:
                    admin_track["person_box"] = best["person_box"]
                    if best["face_box"] is not None:
                        admin_track["face_box"] = best["face_box"]

            # chỉ công bố ADMIN khi đủ số khung xác nhận + vượt ngưỡng nhận dạng
            face_ok = (best["face_conf"] >= args.accept_face)
            body_ok = (best["body_conf"] >= args.accept_body)
            is_admin = (admin_track["frames_seen"] >= CONFIRM_FRAMES) and (face_ok or body_ok)

        # --------- Vẽ kết quả ----------
        # vẽ người: tất cả có bbox; chỉ ADMIN có chữ
        for pb in pboxes:
            color = (50,220,50) if (is_admin and admin_track["person_box"] is not None and bbox_iou(pb, admin_track["person_box"]) > IOU_STICKY) else (255,140,0)
            cv2.rectangle(frame, (pb[0],pb[1]), (pb[2],pb[3]), color, 2)
            if is_admin and bbox_iou(pb, admin_track["person_box"]) > IOU_STICKY:
                cv2.putText(frame, "ADMIN", (pb[0], max(20, pb[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,220,50), 2)

        # vẽ mặt: tất cả face box; chỉ ADMIN có chữ
        for fb in fboxes:
            cv2.rectangle(frame,(fb[0],fb[1]),(fb[2],fb[3]),(0,255,0),2)
            if is_admin and admin_track["face_box"] is not None and bbox_iou(fb, admin_track["face_box"]) > IOU_STICKY:
                cv2.putText(frame, "ADMIN", (fb[0], max(20, fb[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # HUD (hiển thị best trong khung hiện tại để theo dõi tuning)
        if best is None:
            face_best = 0.0; body_best = 0.0; score_show = 0.0
        else:
            face_best = best["face_conf"]; body_best = best["body_conf"]; score_show = best["score"]
        status = "ADMIN" if is_admin else "UNKNOWN"
        cv2.putText(frame, f"STATUS: {status} (best face {face_best:.2f} | best body {body_best:.2f} | EMA {admin_track['score_ema']:.2f})",
                    (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50) if is_admin else (0,165,255), 2)

        cv2.imshow("ADMIN best-match + Temporal Memory (Face+Body, Person)", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
