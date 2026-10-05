#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m rollout.server \
    --model ${GDRIVE_LOCAL}/model/Qwen3.5-4B \
    --max-reside 128 \
    --offload \
    --port 8000
