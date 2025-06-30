<br />
<p align="center">
  
  <h3 align="center"><strong>Optimized version of LATR using Tensor RT</strong></h3>
  <h4 align="center"><strong>(LATR: 3D Lane Detection from Monocular Images with Transformer)</strong></h4>
  
</p>
# LATR-Based Visualization Toolkit

본 프로젝트는 **LATR(Lane-Aware Transformer)** 공식 레포를 기반으로 기능을 확장·개선한 시각화 도구입니다.  
주요 기능은 다음 두 가지입니다.

1. **OpenLane-V1 예측 결과 시각화**  
2. **자체 수집 2-D 이미지(“TCAR” 데이터셋) 상의 투영(projection)**  

This is the TensorRT conversion repo of [LATR: 3D Lane Detection from Monocular Images with Transformer](https://arxiv.org/abs/2308.04583).

## 사용 방법 - 핵심 설정

## How to convert LATR to tensorRT engine?
First, you need to download trt plugin file from mmdeploy repository. This allows to convert custom ops in mmcv and mmdetection.

Then, you need to make a minor change in 

## For other setup
To set up the required packages and data, please refer to the [official LATR repository](https://github.com/JMoonr/LATR).

## TRT Evaluation
You can download the [pretrained models](#pretrained-models) to `./pretrained_models` directory and refer to the [eval guide](./docs/train_eval.md#evaluation) for evaluation.

## Acknowledgment

This library is inspired by [OpenLane](https://github.com/OpenDriveLab/PersFormer_3DLane), [GenLaneNet](https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection), [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [SparseInst](https://github.com/hustvl/SparseInst), [ONCE](https://github.com/once-3dlanes/once_3dlanes_benchmark) and many other related works, we thank them for sharing the code and datasets.


## Citation
If you find LATR is useful for your research, please consider citing the paper:

```tex
@article{luo2023latr,
  title={LATR: 3D Lane Detection from Monocular Images with Transformer},
  author={Luo, Yueru and Zheng, Chaoda and Yan, Xu and Kun, Tang and Zheng, Chao and Cui, Shuguang and Li, Zhen},
  journal={arXiv preprint arXiv:2308.04583},
  year={2023}
}
```
| 설정 파일 | 항목 | 값 | 동작 |
|-----------|------|----|------|
| `config/_base_/base_res101_bs16exp100.pf` | `evaluate_case` | `""` (빈 문자열) | **OpenLane-V1** 예측 결과를 시각화합니다. |
| | | `"TCAR"` | **TCAR 데이터셋**(자체 2-D 영상)에 예측 결과를 투영합니다. |

> **Tip:** 위 설정만 변경해도 원하는 데이터 타입에 맞게 시각화 방식을 전환할 수 있습니다.

---

## 핵심 코드 수정 지점

### 1. `config/_base_/base_res101_bs16exp100.pf`
- `evaluate_case` 값을 **빈 문자열(`""`)** 또는 **`"TCAR"`** 로 지정하여 시각화 모드를 선택합니다.

### 2. `runner.py`
- `def get_calibration_matrix()` 내부에서 **카메라 내·외부 파라미터**를 직접 수정해야 합니다.
  - **Intrinsic(내부)** 행렬: 초점 거리, 주점(cx, cy) 등
  - **Extrinsic(외부)** 행렬: 회전(R) · 변환(t) 값  
- 프로젝트 루트의 `calib/` 폴더(또는 사용자가 지정한 경로)에 보정 파일을 두고,
  `get_calibration_matrix()`에서 해당 파일을 불러오도록 구현해 두었습니다.  
  필요 시 파일 경로 또는 파싱 로직을 맞춰 주세요.

---
