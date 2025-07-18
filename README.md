<br />
<p align="center">
  
  <h2 align="center"><strong>Optimized LATR with TensorRT</strong></h2>
  <h4 align="center"><strong>LATR: 3D Lane Detection from Monocular Images with Transformer</strong></h4>
  
</p>
# LATR-Based Visualization Toolkit

This project extends and improves the official **LATR (Lane-Aware Transformer)** repository with enhanced visualization tools and TensorRT optimization.

## Key Features

1. **Visualization of OpenLane-V1 Prediction Results**
2. **Projection of Predictions onto Custom 2D Images ("TCAR" Dataset)**

This repository provides TensorRT conversion for [LATR: 3D Lane Detection from Monocular Images with Transformer](https://arxiv.org/abs/2308.04583).

---

## Quick Start

### 1. TensorRT Conversion
- Download the TensorRT plugin file from the mmdeploy repository. This is required to convert custom ops in mmcv and mmdetection.
- Make minor changes as needed (see official mmdeploy documentation for details).

### 2. Setup
- For package and data setup, refer to the [official LATR repository](https://github.com/JMoonr/LATR).

### 3. Evaluation
- Download [pretrained models](#pretrained-models) to the `./pretrained_models` directory.
- See the [evaluation guide](./docs/train_eval.md#evaluation) for instructions.

---

## Configuration

| Config File | Option | Value | Description |
|-------------|--------|-------|-------------|
| `config/_base_/base_res101_bs16exp100.pf` | `evaluate_case` | `""` (empty string) | Visualize **OpenLane-V1** prediction results |
|             |        | `"TCAR"`           | Project predictions onto **TCAR** (custom 2D images) |

> **Tip:** Change only the above setting to switch visualization mode for your data type.

---

## Key Code Modification Points

### 1. `config/_base_/base_res101_bs16exp100.pf`
- Set the `evaluate_case` value to either **empty string (`""`)** or **`"TCAR"`** to select the visualization mode.

### 2. `runner.py`
- In `def get_calibration_matrix()`, you must manually set the **camera intrinsic and extrinsic parameters**:
  - **Intrinsic matrix:** Focal length, principal point (cx, cy), etc.
  - **Extrinsic matrix:** Rotation (R) and translation (t) values
- Place your calibration file in the `calib/` folder at the project root (or your specified path).
- The function `get_calibration_matrix()` is implemented to load the calibration file. Adjust the file path or parsing logic as needed.

---

## Acknowledgment

This library is inspired by [OpenLane](https://github.com/OpenDriveLab/PersFormer_3DLane), [GenLaneNet](https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection), [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [SparseInst](https://github.com/hustvl/SparseInst), [ONCE](https://github.com/once-3dlanes/once_3dlanes_benchmark), and many other related works. We thank them for sharing their code and datasets.

---

## Citation
If you find LATR useful for your research, please consider citing the following paper:

```tex
@article{luo2023latr,
  title={LATR: 3D Lane Detection from Monocular Images with Transformer},
  author={Luo, Yueru and Zheng, Chaoda and Yan, Xu and Kun, Tang and Zheng, Chao and Cui, Shuguang and Li, Zhen},
  journal={arXiv preprint arXiv:2308.04583},
  year={2023}
}
```
