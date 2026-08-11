# CARGO
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --config-file ./configs/CARGO/apan.yml --num-gpus 4 SOLVER.IMS_PER_BATCH 64

# AGReIDv2
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --config-file ./configs/AGReIDv2/apan.yml --num-gpus 4 SOLVER.IMS_PER_BATCH 64

# AGReID
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/AGReID/apan.yml --num-gpus 1 SOLVER.IMS_PER_BATCH 128

# LAGPeR
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/LAGPeR/apan.yml --num-gpus 1 SOLVER.IMS_PER_BATCH 64

