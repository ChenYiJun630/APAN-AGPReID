export CUDA_VISIBLE_DEVICES=0,1,2,3
CONFIG_FILE=./configs/AGReIDv2/apan.yml
delta1=0.8
delta2=1.0
beta=1.2
alpha=0.01
topk=16
lambda=0.001
M=$((topk + 16))
OUTPUT_DIR="XXX"

python3 tools/train_net.py \
  --config-file ${CONFIG_FILE} \
  --num-gpus 4 \
  SOLVER.IMS_PER_BATCH 64 \
  MODEL.BACKBONE.TOPK "${topk}" \
  MODEL.BACKBONE.DELTA1 "${delta1}" \
  MODEL.BACKBONE.DELTA2 "${delta2}" \
  MODEL.LOSSES.CE.VIEW_LAMBDA "${lambda}" \
  MODEL.LOSSES.ADM.BETA ${beta} \
  MODEL.LOSSES.ADM.ALPHA ${alpha} \
  OUTPUT_DIR ${OUTPUT_DIR}

