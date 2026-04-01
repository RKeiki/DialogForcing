cd /home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation

# export MASTER_ADDR=<teacher节点IP或hostname>
# export MASTER_PORT=29500
export NNODES=2
export NODE_RANK=0
export NUM_GPUS=8
export NUM_CPUS=96
export VENV_PATH="/home/liujingqi/wanAR/OmniForcing/LTX-2/.venv"

bash scripts/train_stage1_bidirectional_dmd.sh configs/stage1_bidirectional_dmd_split_teacher_worker.yaml --no_visualize
