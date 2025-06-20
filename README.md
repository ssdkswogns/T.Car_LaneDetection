<br />
<p align="center">
  
  <h3 align="center"><strong>Optimized version of LATR using Tensor RT</strong></h3>
  <h4 align="center"><strong>(LATR: 3D Lane Detection from Monocular Images with Transformer)</strong></h4>
  
</p>


This is the TensorRT conversion repo of [LATR: 3D Lane Detection from Monocular Images with Transformer](https://arxiv.org/abs/2308.04583).

![fig2](/assets/fig2.png)  

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