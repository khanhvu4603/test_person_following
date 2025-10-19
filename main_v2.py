
# robot_vision_demo/main.py
# ------------------------------------------------------------
# Webcam perception demo (ADMIN-only labeling):
# - Face Auth: VGG16(fc1) + PCA/KNN
# - Person: MobileNet-SSD (OpenCV DNN)
# - Body Auth: MobileNetV2(GAP) + PCA/KNN
# - MediaPipe Hands: DISABLED (commented out per request)
#
# Behavior:
# * Only the ADMIN gets text label "ADMIN" on both the FACE box and the chosen PERSON box.
# * Other people still have person boxes (different color) but NO text.
# * Matching of FACE->PERSON uses: (1) person that CONTAINS face-center; else (2) person NEAREST to face-center.
# * Margins: face=0.20, body=0.20 for more stable embeddings.
# * Default accept thresholds: face=0.55, body=0.60 (tune later).
# ------------------------------------------------------------

import cv2
import numpy as np
import argparse
import time
from pathlib import Path

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_pre
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mb2_pre
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib

# -------------- Paths -----------------
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

# -------------- Loaders ---------------
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

# -------------- Utils -----------------
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
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # BGR->RGB (IMPORTANT)
    arr = img_to_array(roi)[None,...]
    return vgg_pre(arr)

def body_arr(frame, box):
    x1,y1,x2,y2 = expand(box, frame.shape, 0.20)
    roi = frame[y1:y2, x1:x2]
    if roi.size==0: return None
    roi = cv2.resize(roi, (224,224))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # BGR->RGB
    arr = img_to_array(roi)[None,...]
    return mb2_pre(arr)

def pick_person_for_face(fbox, pboxes):
    """Pick the person box for this face:
       (1) person that CONTAINS face-center; if multiple, choose the larger
       (2) else person NEAREST to face-center
    """
    if not pboxes: return None
    fx1,fy1,fx2,fy2 = fbox
    fcx, fcy = (fx1+fx2)//2, (fy1+fy2)//2
    containing = []
    for i,(x1,y1,x2,y2) in enumerate(pboxes):
        if x1<=fcx<=x2 and y1<=fcy<=y2:
            containing.append((i,(x2-x1)*(y2-y1)))
    if containing:
        containing.sort(key=lambda t:-t[1])
        return pboxes[containing[0][0]]
    # nearest center
    def center_dist(pb):
        x1,y1,x2,y2 = pb
        pcx,pcy = (x1+x2)//2, (y1+y2)//2
        return abs(pcx-fcx) + abs(pcy-fcy)
    return min(pboxes, key=center_dist)

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
    lab = knn.predict(Z)[0]
    dists,_ = knn.kneighbors(Z, 1, return_distance=True)
    d=float(dists[0,0]); conf=np.exp(-d)
    if centroid is not None:
        z=Z[0]; cos=float(np.dot(z,centroid)/(np.linalg.norm(z)*np.linalg.norm(centroid)+1e-8))
        conf = 0.5*conf + 0.5*max(0.0, min(1.0, (cos+1)/2))
    return lab, conf

# -------------- Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--enroll", action="store_true")
    ap.add_argument("--label", type=str, default="admin")
    ap.add_argument("--accept_face", type=float, default=0.55)
    ap.add_argument("--accept_body", type=float, default=0.60)
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

    # ------------- ENROLL (collect face+body) -------------
    if args.enroll:
        print("[ENROLL] Collecting FACE + BODY embeddings for", args.label)
        F,B=[],[]
        while True:
            ok,frame = cap.read()
            if not ok: break
            frame=cv2.flip(frame,1)
            fboxes,fscores = detect_faces(face_net, frame, args.face_conf)
            pboxes,pscores = detect_persons(person_net, frame, args.person_conf)

            if fboxes:
                # largest face
                idx = np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in fboxes])
                fb = fboxes[idx]
                arr = face_arr(frame, fb)
                if arr is not None:
                    f=vgg.predict(arr, verbose=0).flatten(); F.append(f)
                    cv2.rectangle(frame,(fb[0],fb[1]),(fb[2],fb[3]),(0,255,0),2)

                # match person to this face
                cand = pick_person_for_face(fb, pboxes) if pboxes else None
                if cand is not None:
                    arrb = body_arr(frame, cand)
                    if arrb is not None:
                        b=mb2.predict(arrb, verbose=0).flatten(); B.append(b)
                        cv2.rectangle(frame,(cand[0],cand[1]),(cand[2],cand[3]),(255,140,0),2)

            cv2.putText(frame, f"ENROLL: face {len(F)} | body {len(B)}  (press 'q' to finish)",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.imshow("ENROLL", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break

        cv2.destroyAllWindows()
        F=np.array(F); B=np.array(B)
        if len(F)<10 or len(B)<10:
            print("[ENROLL] Not enough samples (need >=10 each; recommend 40–60).")
            return
        fpca,fknn,fmean,fcent = train_pca_knn(F, args.label, args.pca_dim_face, 3)
        bpca,bknn,bmean,bcent = train_pca_knn(B, args.label, args.pca_dim_body, 3)
        joblib.dump(fpca, FACE_PCA_PATH); joblib.dump(fknn, FACE_CLS_PATH); np.save(FACE_MEAN_PATH, fmean); np.save(FACE_CENT_PATH, fcent)
        joblib.dump(bpca, BODY_PCA_PATH); joblib.dump(bknn, BODY_CLS_PATH); np.save(BODY_MEAN_PATH, bmean); np.save(BODY_CENT_PATH, bcent)
        print("[ENROLL] Saved face/body models in", DATA)
        return

    # ------------- INFERENCE -------------
    # Load models
    if not (FACE_PCA_PATH.exists() and FACE_CLS_PATH.exists() and FACE_MEAN_PATH.exists()):
        print("[WARN] No FACE models; run with --enroll.") 
        face_models=None
    else:
        face_models = (joblib.load(FACE_PCA_PATH), joblib.load(FACE_CLS_PATH), np.load(FACE_MEAN_PATH), np.load(FACE_CENT_PATH) if FACE_CENT_PATH.exists() else None)

    if not (BODY_PCA_PATH.exists() and BODY_CLS_PATH.exists() and BODY_MEAN_PATH.exists()):
        print("[WARN] No BODY models; run with --enroll.") 
        body_models=None
    else:
        body_models = (joblib.load(BODY_PCA_PATH), joblib.load(BODY_CLS_PATH), np.load(BODY_MEAN_PATH), np.load(BODY_CENT_PATH) if BODY_CENT_PATH.exists() else None)

    print("[RUN] q=quit. Only ADMIN is labeled. Others: boxes only, no text.")

    while True:
        ok,frame = cap.read()
        if not ok: break
        frame=cv2.flip(frame,1)
        H,W=frame.shape[:2]

        # Detect
        fboxes,fscores = detect_faces(face_net, frame, args.face_conf)
        pboxes,pscores = detect_persons(person_net, frame, args.person_conf)

        # Choose one FACE (largest) for admin candidate
        admin_face_box=None; face_conf=0.0
        if fboxes:
            idx = np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in fboxes])
            admin_face_box = fboxes[idx]
            # draw face box (always, green) — text only if ADMIN confirmed later
            cv2.rectangle(frame,(admin_face_box[0],admin_face_box[1]),(admin_face_box[2],admin_face_box[3]),(0,255,0),2)
            if face_models is not None:
                fpca,fknn,fmean,fcent = face_models
                arr = face_arr(frame, admin_face_box)
                if arr is not None:
                    femb = vgg.predict(arr, verbose=0).flatten()
                    _, face_conf = predict_label(femb, fpca, fknn, fmean, centroid=fcent)

        # Pick the PERSON that corresponds to this face (if any)
        admin_pb = None; body_conf = 0.0
        if pboxes:
            if admin_face_box is not None:
                admin_pb = pick_person_for_face(admin_face_box, pboxes)
            else:
                # no face: pick center-most person
                admin_pb = min(pboxes, key=lambda bb: abs((bb[0]+bb[2])//2 - W//2))
            # compute body conf only if face conf is weak
            if body_models is not None and (face_conf < args.accept_face):
                bpca,bknn,bmean,bcent = body_models
                arrb = body_arr(frame, admin_pb)
                if arrb is not None:
                    bemb = mb2.predict(arrb, verbose=0).flatten()
                    _, body_conf = predict_label(bemb, bpca, bknn, bmean, centroid=bcent)

        # Decision: ADMIN only if max(face, 0.9*body) passes min threshold
        conf_final = max(face_conf, 0.9*body_conf)
        is_admin = conf_final >= min(args.accept_face, args.accept_body)
        
        # Print similarity comparison with enrolled admin
        if face_conf > 0 or body_conf > 0:
            print(f"🔍 COMPARING WITH ENROLLED ADMIN:")
            print(f"   Face similarity: {face_conf:.3f} (threshold: {args.accept_face})")
            print(f"   Body similarity: {body_conf:.3f} (threshold: {args.accept_body})")
            print(f"   Final score: {conf_final:.3f} → {'✅ ADMIN' if is_admin else '❌ UNKNOWN'}")
            print("-" * 50)

        # Draw PERSON boxes: all people get boxes (orange), only admin gets label
        for pb in pboxes:
            color = (50,220,50) if (is_admin and admin_pb is not None and pb==admin_pb) else (255,140,0)
            cv2.rectangle(frame, (pb[0],pb[1]), (pb[2],pb[3]), color, 2)
            if is_admin and admin_pb is not None and pb==admin_pb:
                cv2.putText(frame, "ADMIN", (pb[0], max(20, pb[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,220,50), 2)

        # FACE text only if ADMIN
        if is_admin and admin_face_box is not None:
            cv2.putText(frame, "ADMIN", (admin_face_box[0], max(20, admin_face_box[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # HUD
        status = "ADMIN" if is_admin else "UNKNOWN"
        cv2.putText(frame, f"STATUS: {status} (face {face_conf:.2f} | body {body_conf:.2f})", (10,35),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7, (50,220,50) if is_admin else (0,165,255),2)

        cv2.imshow("ADMIN-only Label Demo (Face+Body, Person)", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()