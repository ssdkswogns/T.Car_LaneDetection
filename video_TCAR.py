import os
import glob
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# — 수정할 부분 —
PICKLE_DIR    = "/home/moon/code/LATR/TCAR_pred_json_image_2"
OUTPUT_VIDEO  = "output_2.avi"
FPS           = 5
FRAME_SIZE    = None  # (width, height) — 첫 프레임 로드 후 자동 설정

# 이전에 정의한 유틸 함수들
def project_xyz_to_image(xyz_list, extrinsic, intrinsic):
    # (위에서 작성하신 것과 동일)
    cam_rep = np.linalg.inv(np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]]))
    R_vg   = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    R_gc   = np.array([[1,0,0],[0,0,1],[0,-1,0]])
    extrinsic = extrinsic.copy()
    extrinsic[:3,:3] = np.linalg.inv(R_vg) @ extrinsic[:3,:3] @ R_vg @ R_gc
    extrinsic[0:2,3] = 0
    uv_all = []
    for xyz in xyz_list:
        if xyz.shape[0]==0:
            uv_all.append(np.empty((0,2))); continue
        xyz_h = np.hstack([xyz, np.ones((xyz.shape[0],1))])
        cam_xyz_h = (np.linalg.inv(extrinsic) @ xyz_h.T).T[:,:3]
        uv_h = (intrinsic @ cam_xyz_h.T).T
        uv_all.append(uv_h[:,:2] / uv_h[:,2:3])
    return uv_all

# 카테고리별 색상 (BGR 순서)
CATEGORY_COLORS = {
    0: (128, 128, 128),     # unknown - gray
    1: (255, 255, 255),     # white-dash - white
    2: (200, 200, 200),     # white-solid - light gray
    3: (255, 255, 180),     # double-white-dash - off white
    4: (200, 200, 180),     # double-white-solid - grayish white
    5: (220, 255, 200),     # white-ldash-rsolid - pale green
    6: (200, 255, 220),     # white-lsolid-rdash - pale cyan
    7: (0, 255, 255),       # yellow-dash - yellow
    8: (0, 200, 200),       # yellow-solid - dark yellow
    9: (0, 180, 255),       # double-yellow-dash - orange
    10: (0, 140, 255),      # double-yellow-solid - dark orange
    11: (0, 160, 200),      # yellow-ldash-rsolid - cyan yellow
    12: (0, 200, 160),      # yellow-lsolid-rdash - teal yellow
    13: (100, 200, 160),      # yellow-lsolid-rdash - teal yellow
    14: (0, 0, 255),        # right-curbside - red
    20: (255, 0, 0),        # left-curbside - blue
}
# 카테고리별 이름
CATEGORY_NAMES = {
    0: 'invalid',
    1: 'white-dash',
    2: 'white-solid',
    3: 'double-white-dash',
    4: 'double-white-solid',
    5: 'white-ldash-rsolid',
    6: 'white-lsolid-rdash',
    7: 'yellow-dash',
    8: 'yellow-solid',
    9: 'double-yellow-dash',
    10: 'double-yellow-solid',
    11: 'yellow-ldash-rsolid',
    12: 'yellow-lsolid-rdash',
    13: 'fishbone',
    14: 'others',
    20: 'roadedge',
}

           

extrinsic = np.array([[0.9999558812277093, -4.412690247759514e-05, -0.009393276900625495, 1.5442298691351966], 
[6.784166429129727e-05, 0.9999968115246617, 0.0025243490286893143, -0.024168247502349863], 
[0.009393135558690344, -0.002524874913047423, 0.9999526958866851, 2.1160897275662878], 
[0.0, 0.0, 0.0, 1.0]])
intrinsic = np.array([[2079.511010136365, 0.0, 953.9357583878485], 
[0.0, 2079.511010136365, 661.581335767017], 
[0.0, 0.0, 1.0]])

def visualize_lanes_on_image(image, projected_lanes, categories):
    legend = set()
    for lane, cat in zip(projected_lanes, categories):
        c = int(cat)
        color = CATEGORY_COLORS.get(c,(128,128,128))
        legend.add(c)
        for i in range(len(lane)-1):
            p1 = tuple(np.round(lane[i]).astype(int))
            p2 = tuple(np.round(lane[i+1]).astype(int))
            cv2.line(image, p1, p2, color, 5)
    # 범례
    x0,y0=10,10
    for idx,c in enumerate(sorted(legend)):
        col = CATEGORY_COLORS[c]
        name= CATEGORY_NAMES[c]
        y=y0+idx*60
        cv2.rectangle(image,(x0,y),(x0+40,y+40),col,-1)
        cv2.putText(image,name,(x0+45,y+35),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),3)
    return image

# — 메인 루프 —
pickle_paths = glob.glob(os.path.join(PICKLE_DIR, "*.pickle"))

# 2) 파일명에서 숫자를 뽑아 정수로 변환하는 함수
def extract_idx(path):
    # basename: 'TCAR_prediction_loader_123.pickle'
    stem = Path(path).stem  # 'TCAR_prediction_loader_123'
    # 뒤의 숫자 부분만 뽑아서 int로
    num_str = stem.split("_")[-1]
    return int(num_str)

# 3) 숫자 순으로 정렬
pickle_paths = sorted(pickle_paths, key=extract_idx)

video_writer = None

for pk in pickle_paths:
    # 1) pickle 로드
    with open(pk, "rb") as f:
        pred = pickle.load(f)
    # 2) 이미지 읽기
    raw_file = pred["file_path"][0]  # 리스트의 첫 요소
    img = cv2.imread(raw_file)
    if img is None:
        print(f"WARNING: cannot read {raw_file}")
        continue
    # 3) 프레임 크기 결정
    if FRAME_SIZE is None:
        h, w = img.shape[:2]
        FRAME_SIZE = (w*2, h)  # 横に2つ並べるので幅×2
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, FRAME_SIZE)
    # 4) 투영

    lanes_xyz  = [np.array(x) for x in pred["pred_laneLines"]]
    lanes_cat  = [np.argmax(p) for p in pred["pred_laneLines_prob"]]
    lanes_uv   = project_xyz_to_image(lanes_xyz, extrinsic, intrinsic)
    overlaid   = visualize_lanes_on_image(img.copy(), lanes_uv, lanes_cat)
    # 5) 좌·우 합치기
    combined = np.hstack([img, overlaid])
    video_writer.write(combined)
if video_writer is not None:
    video_writer.release()
print("Saved video to", OUTPUT_VIDEO)
