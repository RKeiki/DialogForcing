cd /home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation

# export MASTER_ADDR=<master节点IP或hostname>
# export MASTER_PORT=29500
export NNODES=4
export NODE_RANK=3
export NUM_GPUS=8
export NUM_CPUS=96
export VENV_PATH="/home/liujingqi/wanAR/OmniForcing/LTX-2/.venv"

bash scripts/train_stage1_bidirectional_dmd.sh configs/stage1_bidirectional_dmd.yaml --no_visualize