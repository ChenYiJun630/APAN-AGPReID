checkpoint_path="XXX"

CUDA_VISIBLE_DEVICES=0 \
python3 tools/train_net.py \
  --config-file "${checkpoint_path}/config.yaml" \
  --eval-only \
  MODEL.WEIGHTS "${checkpoint_path}/model_best.pth" \
  OUTPUT_DIR "${checkpoint_path}_test"