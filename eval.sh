#!/usr/bin/env bash
# Evaluation script for LATR model using 4 GPUs

# 사용할 GPU 디바이스 지정
export CUDA_VISIBLE_DEVICES='0'

# 분산 평가 실행
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    main.py \
    --config config/release_iccv/latr_1000_baseline_lite.py \
    --save_pred \
    --cfg-options \
        evaluate=true \
        eval_ckpt=pretrained_models/openlane_lite.pth
