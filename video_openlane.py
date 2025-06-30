#!/usr/bin/env python3
# =============================================================
# lane_video_side_by_side_fixed_scale.py
#
# • pickle  → 시간순(18자리 타임스탬프) 정렬
# • 왼쪽  : 2-D 투영 (OpenCV BGR)
# • 오른쪽: 3-D 플롯 (Matplotlib RGB)
# • 3-D 축  : 고정 스케일, ego (0,0,0) 표시
# =============================================================

# ───────────── 사용자 설정 ──────────────
PICKLE_DIR   = "/home/moon/LATR/experiments/openlane_pred_json"
OUTPUT_VIDEO = "./video/output_side_by_side.mp4"
FPS          = 5
IMG_ROOT     = "/data/openlane/images"        # file_path가 상대경로일 경우 붙일 루트
# 3-D 축 범위 [min, max]  (단위: m)
X_RANGE = (-15, 15)    # 전방 (+)·후방(-)
Y_RANGE = (0, 30)      # 오른쪽 (+)·왼쪽(-)
Z_RANGE = (-15, 15)    # 위 (+)·아래(-)
# ──────────────────────────────────────

import os, glob, pickle, cv2, re, sys, numpy as np, matplotlib
matplotlib.use("Agg")                         # GUI 없는 백엔드
import matplotlib.pyplot as plt
from pathlib import Path
try:
    import torch
    def to_np(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
except ImportError:
    torch = None
    def to_np(x): return np.asarray(x)

# ────────────── 단일 RGB 팔레트(0-1) ──────────────
BASE_COLORS = {
    0:(.5,.5,.5), 1:(1,1,1),   2:(.78,.78,.78), 3:(1,1,.70),
    4:(.78,.78,.70), 5:(.86,1,.78), 6:(.78,1,.86), 7:(1,1,0),
    8:(.8,.8,0), 9:(1,.7,0), 10:(1,.55,0), 11:(.8,.9,.3),
    12:(.5,1,.6), 13:(.39,.78,.63), 14:(1,0,0), 20:(0,0,1), 21:(0,0,1)
}
def rgb_to_bgr_uint(rgb): r,g,b = [int(c*255) for c in rgb]; return (b,g,r)
COL_3D = BASE_COLORS                                           # Matplotlib RGB
COL_2D = {k: rgb_to_bgr_uint(v) for k,v in BASE_COLORS.items()}# OpenCV BGR
CATEGORY_NAMES = {
    0:'invalid',1:'white-dash',2:'white-solid',3:'double-white-dash',
    4:'double-white-solid',5:'white-ldash-rsolid',6:'white-lsolid-rdash',
    7:'yellow-dash',8:'yellow-solid',9:'double-yellow-dash',10:'double-yellow-solid',
    11:'yellow-ldash-rsolid',12:'yellow-lsolid-rdash',13:'fishbone',
    14:'others',20:'roadedge',21:'roadedge'
}

# ───────────── 주요 함수 ──────────────
def project_xyz_to_image(xyz_list, extr, intr):
    cam_rep = np.linalg.inv(np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]]))
    R_vg = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    R_gc = np.array([[1,0,0],[0,0,1],[0,-1,0]])
    ext = extr.copy()
    ext[:3,:3] = np.linalg.inv(R_vg) @ ext[:3,:3] @ R_vg @ R_gc
    ext[:2,3] = 0
    inv_ext = np.linalg.inv(ext)
    uv_all=[]
    for lane in xyz_list:
        if lane.size == 0:
            uv_all.append(np.empty((0,2)))
            continue
        pts_h = np.hstack([lane, np.ones((len(lane),1))])
        cam = (inv_ext @ pts_h.T).T[:,:3]
        uv_h = (intr @ cam.T).T
        uv_all.append(uv_h[:,:2] / uv_h[:,2:3])
    return uv_all

def overlay_2d(img, uv_lanes, cats):
    legend = set()
    for lane, c in zip(uv_lanes, cats):
        col = COL_2D.get(c, (128,128,128)); legend.add(c)
        for p, q in zip(lane[:-1], lane[1:]):
            cv2.line(img, tuple(np.round(p).astype(int)),
                          tuple(np.round(q).astype(int)), col, 5)
    x0, y0 = 10, 10
    for i, c in enumerate(sorted(legend)):
        cv2.rectangle(img, (x0,y0+i*60), (x0+40,y0+40+i*60), COL_2D[c], -1)
        cv2.putText(img, CATEGORY_NAMES.get(c, str(c)),
                    (x0+50, y0+35+i*60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,0,0), 3)
    return img

def render_3d(lanes_xyz, cats, size):
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    # 1) 그리드 끄기
    # ax.grid(True, color='k', alpha=0.2, lw=0.5)

    # # 2) 눈금과 눈금 레이블 전부 제거
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    # 3) 축(스파인)과 배경 패널도 모두 숨기기
    ax.set_axis_off()
    for lane, c in zip(lanes_xyz, cats):
        if lane.size == 0:
            continue
        ax.plot(lane[:,0], lane[:,1], lane[:,2],
                lw=2, color=COL_3D.get(c, (0.5,0.5,0.5)))
    ax.scatter(0,0,0, c='r', s=100, label='ego')
    ax.set_xlim(X_RANGE); ax.set_ylim(Y_RANGE); ax.set_zlim(Z_RANGE)
    ax.view_init(30,-90)
    ax.set_box_aspect((X_RANGE[1]-X_RANGE[0],
                       Y_RANGE[1]-Y_RANGE[0],
                       Z_RANGE[1]-Z_RANGE[0]))
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def ts_key(p):             # pickle → 18자리 타임스탬프
    try:
        with open(p, 'rb') as f:
            d = pickle.load(f)
        fp = d["file_path"]
        fp = fp[0] if isinstance(fp, (list,tuple)) else fp
        m = re.search(r'(\d{18})\.jpg$', fp)
        return int(m.group(1)) if m else -1
    except Exception:
        return -1

# ──────────────────────────────────────
def main():
    pkls = sorted(glob.glob(f"{PICKLE_DIR}/*.pickle"), key=ts_key)
    if not pkls:
        print("⚠️  pickle 없음")
        return

    vw = None
    first = True
    for pkl in pkls:
        pred = pickle.load(open(pkl, 'rb'))
        img_path = pred["file_path"]
        img_path = img_path[0] if isinstance(img_path, (list,tuple)) else img_path
        if not os.path.isabs(img_path):
            img_path = os.path.join(IMG_ROOT, img_path)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        ext = np.asarray(to_np(pred["extrinsic"]))
        intr = np.asarray(to_np(pred["intrinsic"]))
        lanes_xyz = [np.asarray(to_np(x)) for x in pred["pred_laneLines"]]
        cats = [int(np.argmax(to_np(p))) for p in pred["pred_laneLines_prob"]]

        uv = project_xyz_to_image(lanes_xyz, ext, intr)
        img2d = overlay_2d(img.copy(), uv, cats)
        img3d = render_3d(lanes_xyz, cats, (w, h))
        frame = np.hstack([img2d, img3d])

        if first:
            h0, w0 = frame.shape[:2]
            vw = cv2.VideoWriter(
                OUTPUT_VIDEO,
                cv2.VideoWriter_fourcc(*'mp4v'),
                FPS,
                (w0, h0)
            )
            print("⋯ writing", OUTPUT_VIDEO, (w0, h0))
            first = False
        vw.write(frame)

    if vw:
        vw.release()
        print("✅ saved:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
