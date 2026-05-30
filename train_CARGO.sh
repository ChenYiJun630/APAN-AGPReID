export CUDA_VISIBLE_DEVICES=0,1,2,3
CONFIG_FILE=./configs/CARGO/apan.yml
delta1=1.0
delta2=0.8
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

