#!/usr/bin/env python3
# =============================================================
# video_openlane_GT.py
#
# • pickle → 시간순(18자리 타임스탬프) 정렬
# • 2×2 그리드:
#     ┌───────────────────────┬───────────────────────┐
#     │ Prediction 2D (full)  │ Prediction 3D (half)  │
#     ├───────────────────────┼───────────────────────┤
#     │ GT 2D (full)          │ GT 3D (half)          │
# • 3-D 축: 고정 스케일, ego (0,0,0) 표시
# =============================================================

# ───────────── 사용자 설정 ──────────────
PICKLE_DIR   = "/home/moon/LATR/experiments/openlane_pred_json"
OUTPUT_VIDEO = "./video/output_2D_3D_GT.mp4"
FPS          = 5

IMG_ROOT     = "/data/openlane/images"        # file_path가 상대경로일 경우 붙일 루트
GT_ROOT      = "/data/openlane/lane3d_1000"   # 이미지 상대 경로와 동일 구조

# 3-D 축 범위 [min, max] (단위: m)
X_RANGE = (0, 30)
Y_RANGE = (-20, 20)
Z_RANGE = (-10, 10)

X_RANGE_PRED = Y_RANGE
Y_RANGE_PRED = X_RANGE
Z_RANGE_PRED = Z_RANGE
# ──────────────────────────────────────────

import os
import glob
import pickle
import json
import cv2
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Torch 텐서 지원 (선택)
try:
    import torch
    def to_np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
except ImportError:
    torch = None
    def to_np(x):
        return np.asarray(x)

# ────────────── 색상 팔레트 ──────────────
BASE_COLORS = {
    0:(.5,.5,.5),   1:(1,1,1),    2:(.78,.78,.78), 3:(1,1,.70),
    4:(.78,.78,.70),5:(.86,1,.78),6:(.78,1,.86), 7:(1,1,0),
    8:(.8,.8,0),    9:(1,.7,0),   10:(1,.55,0),   11:(.8,.9,.3),
    12:(.5,1,.6),   13:(.39,.78,.63),14:(1,0,0),   20:(0,0,1),
    21:(0,0,1)
}
def rgb_to_bgr_uint(rgb):
    r,g,b = [int(c*255) for c in rgb]
    return (b,g,r)

COL_2D = {k: rgb_to_bgr_uint(v) for k,v in BASE_COLORS.items()}
COL_3D = BASE_COLORS

CATEGORY_NAMES = {
    0:'invalid',1:'white-dash',2:'white-solid',3:'double-white-dash',
    4:'double-white-solid',5:'white-ldash-rsolid',6:'white-lsolid-rdash',
    7:'yellow-dash',8:'yellow-solid',9:'double-yellow-dash',
    10:'double-yellow-solid',11:'yellow-ldash-rsolid',
    12:'yellow-lsolid-rdash',13:'fishbone',14:'others',
    20:'roadedge',21:'roadedge'
}
# ──────────────────────────────────────────

def overlay_2d(img, uv_lanes, cats):
    """
    이미지 위에 2D 차선(uv 좌표)과 범례를 그려 반환.
    img    : (H, W, 3) BGR 이미지
    uv_lanes: List of (N_i,2) UV 점
    cats   : List of category IDs
    """
    legend = set()
    for lane_uv, c in zip(uv_lanes, cats):
        color = COL_2D.get(c, (128,128,128))
        legend.add(c)
        for p, q in zip(lane_uv[:-1], lane_uv[1:]):
            pt1 = tuple(np.round(p).astype(int))
            pt2 = tuple(np.round(q).astype(int))
            cv2.line(img, pt1, pt2, color, 5)

    x0, y0 = 10, 10
    for i, c in enumerate(sorted(legend)):
        cv2.rectangle(img,
                      (x0, y0 + 60*i),
                      (x0 + 40, y0 + 40 + 60*i),
                      COL_2D[c], -1)
        cv2.putText(img,
                    CATEGORY_NAMES.get(c, str(c)),
                    (x0 + 50, y0 + 35 + 60*i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,0,0),
                    3)
    return img



def project_xyz_to_image_3D(xyz_list, extrinsic, intrinsic):
    """
    3D 좌표 리스트를 카메라 이미지 상 UV 좌표로 투영.
    xyz_list : List of (N_i,3) 점
    extrinsic: (4,4) 외부 파라미터
    intrinsic: (3,3) 내부 파라미터
    """
    cam_rep = np.linalg.inv(np.array([
        [0,  0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0,  0, 0, 1]
    ], dtype=float))

    uv_all = []
    for xyz in xyz_list:
        if xyz.size == 0:
            uv_all.append(np.empty((0,2)))
            continue
        pts_h = np.hstack([xyz, np.ones((xyz.shape[0],1))])  # (N,4)
        cam_h = (cam_rep @ pts_h.T).T                      # (N,4)
        cam   = cam_h[:,:3]                                # (N,3)

        uv_h = (intrinsic @ cam.T).T    # (N,3)
        uv   = uv_h[:,:2] / uv_h[:,2:3]
        uv_all.append(uv)
    return uv_all

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

def render_3d(lanes_xyz, cats, size):
    """
    3D 차선을 Matplotlib로 렌더링해 BGR 이미지 반환.
    lanes_xyz: List of (N_i,3) 점
    cats     : List of category IDs
    size     : (width, height) 출력 픽셀 크기
    """
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax  = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    # ax.set_axis_off()

    for lane, c in zip(lanes_xyz, cats):
        if lane.size == 0:
            continue
        ax.plot(lane[:,0], lane[:,1], lane[:,2],
                lw=2, color=COL_3D.get(c, (0.5,0.5,0.5)))
    ax.scatter(0, 0, 0, c='r', s=130)

    ax.set_xlim(X_RANGE_PRED)
    ax.set_ylim(Y_RANGE_PRED)
    ax.set_zlim(Z_RANGE_PRED)

    ax.set_box_aspect((X_RANGE_PRED[1]-X_RANGE_PRED[0],
                       Y_RANGE_PRED[1]-Y_RANGE_PRED[0],
                       Z_RANGE_PRED[1]-Z_RANGE_PRED[0]))
    ax.dist = 0   # 원하는 카메라 거리 (기본값은 Matplotlib 버전마다 다릅니다)

    fig.tight_layout(pad=0)
    
    ax.view_init(30, 270)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)



def render_3d_GT(lanes_xyz, cats, size):
    """
    3D 차선을 Matplotlib로 렌더링해 BGR 이미지 반환.
    lanes_xyz: List of (N_i,3) 점
    cats     : List of category IDs
    size     : (width, height) 출력 픽셀 크기
    """
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax  = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    # ax.set_axis_off()

    for lane, c in zip(lanes_xyz, cats):
        if lane.size == 0:
            continue
        ax.plot(lane[:,0], lane[:,1], lane[:,2],
                lw=2, color=COL_3D.get(c, (0.5,0.5,0.5)))
    ax.scatter(0, 0, 0, c='r', s=130)

    ax.set_xlim(X_RANGE)
    ax.set_ylim(Y_RANGE)
    ax.set_zlim(Z_RANGE)

    ax.set_box_aspect((X_RANGE[1]-X_RANGE[0],
                       Y_RANGE[1]-Y_RANGE[0],
                       Z_RANGE[1]-Z_RANGE[0]))
    ax.dist = 0   # 원하는 카메라 거리 (기본값은 Matplotlib 버전마다 다릅니다)
    
    fig.tight_layout(pad=0)

    ax.view_init(30, 180)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def ts_key(p):
    """pickle 파일명에서 18자리 타임스탬프로 정렬 키 생성."""
    try:
        d  = pickle.load(open(p,'rb'))
        fp = d["file_path"][0] if isinstance(d["file_path"], (list,tuple)) else d["file_path"]
        m  = re.search(r'(\d{18})\.jpg$', fp)
        return int(m.group(1)) if m else -1
    except:
        return -1

def main():
    pkls = sorted(glob.glob(f"{PICKLE_DIR}/*.pickle"), key=ts_key)
    if not pkls:
        print("⚠️  pickle 없음")
        return

    writer = None
    first = True
    scale_3d = 0.7  # 3D 패널 너비 비율

    for pkl in pkls:
        pred = pickle.load(open(pkl,'rb'))
        rel  = pred["file_path"][0] if isinstance(pred["file_path"], (list,tuple)) else pred["file_path"]
        img_path = rel if os.path.isabs(rel) else os.path.join(IMG_ROOT, rel)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # ─ Prediction 2D & 3D ─
        ext_pr   = to_np(pred["extrinsic"])
        intr_pr  = to_np(pred["intrinsic"])
        lanes_pr = [to_np(x) for x in pred["pred_laneLines"]]
        cats_pr  = [int(np.argmax(to_np(p))) for p in pred["pred_laneLines_prob"]]

        uv_pr    = project_xyz_to_image(lanes_pr, ext_pr, intr_pr)
        img2d_pr = overlay_2d(img.copy(), uv_pr, cats_pr)
        img3d_pr = render_3d(lanes_pr, cats_pr, (w, h))
        # 3D 축소
        img3d_pr = cv2.resize(img3d_pr, (int(w*scale_3d), h))

        # ─ GT 2D & 3D ─
        gt_file = os.path.join(GT_ROOT, rel).replace(".jpg", ".json")
        with open(gt_file, 'r') as f:
            gt = json.load(f)

        ext_gt   = np.array(gt.get("extrinsic", list(gt.values())[0]))
        intr_gt  = np.array(gt.get("intrinsic", list(gt.values())[1]))
        raw_lanes= gt.get("lanes", list(gt.values())[2] if len(gt.values())>=3 else [])
        lanes_xyz_gt, cats_gt = [], []
        for ln in raw_lanes:
            xyz = np.array(ln["xyz"])
            if xyz.ndim == 2 and xyz.shape[0] == 3:
                xyz = xyz.T
            lanes_xyz_gt.append(xyz)
            c = ln.get("category", 0)
            if isinstance(c, (list,tuple)): c = c[0]
            cats_gt.append(int(c))

        uv_gt    = project_xyz_to_image_3D(lanes_xyz_gt, ext_gt, intr_gt)
        img2d_gt = overlay_2d(img.copy(), uv_gt, cats_gt)
        img3d_gt = render_3d_GT(lanes_xyz_gt, cats_gt, (w, h))
        img3d_gt = cv2.resize(img3d_gt, (int(w*scale_3d), h))

        # ─ 2×2 Grid 조합 ─
        row1  = np.hstack([img2d_pr, img3d_pr])
        row2  = np.hstack([img2d_gt, img3d_gt])
        frame = np.vstack([row1, row2])

        if first:
            H, W = frame.shape[:2]
            writer = cv2.VideoWriter(
                OUTPUT_VIDEO,
                cv2.VideoWriter_fourcc(*'mp4v'),
                FPS,
                (W, H)
            )
            print("⋯ writing", OUTPUT_VIDEO, (W, H))
            first = False

        writer.write(frame)

    if writer:
        writer.release()
        print("✅ saved:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
