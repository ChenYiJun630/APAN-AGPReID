checkpoint_path="/workspace/ReID/AGPReID/APAN-AGPReID/logs/CARGO/b64_g4_lr0.008_lambda0.001_a0.01_b1.2/delta1.0_deltaSec0.8_M32"

CUDA_VISIBLE_DEVICES=0 \
python3 tools/train_net.py \
  --config-file "${checkpoint_path}/config.yaml" \
  --eval-only \
  MODEL.WEIGHTS "${checkpoint_path}/model_best.pth" \
  OUTPUT_DIR "${checkpoint_path}_test"