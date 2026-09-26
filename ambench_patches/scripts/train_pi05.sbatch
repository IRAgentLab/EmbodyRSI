#!/bin/bash
#SBATCH --propagate=NONE
#SBATCH --job-name=pi05
#SBATCH --partition=gpuA800
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/liaodl/ambench/logs/%x_%j.out
#SBATCH --error=/public/home/liaodl/ambench/logs/%x_%j.err

# Usage:
#   sbatch --export=CONFIG=<config_name>,EXP=<exp_name>,STEPS=<n>,BATCH=<n> train_pi05.sbatch
# Defaults below reproduce the AM-Bench MT-FT protocol (40k steps, batch 32).

set -x
source /public/home/liaodl/ambench/env/hpc_env.sh
cd $OPENPI_SRC

CONFIG=${CONFIG:-pi05_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative}
EXP=${EXP:-mt_ft}
STEPS=${STEPS:-40000}
BATCH=${BATCH:-32}

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_MODE=disabled
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python scripts/train.py $CONFIG --keep-period 1000000 \
  --exp-name=$EXP \
  --num-train-steps=$STEPS \
  --batch-size=$BATCH \
  --overwrite \
  --no-wandb-enabled
