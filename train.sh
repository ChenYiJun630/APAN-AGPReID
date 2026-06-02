# CARGO
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 tools/train_net.py \
--config-file ./configs/CARGO/apan.yml \
--num-gpus 4 SOLVER.IMS_PER_BATCH 64

