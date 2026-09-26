#!/bin/bash
#SBATCH --propagate=NONE
#SBATCH --job-name=train_mt
#SBATCH --partition=gpuA800
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/liaodl/ambench/logs/%x_%j.out
#SBATCH --error=/public/home/liaodl/ambench/logs/%x_%j.err
# 多任务从头训练（π₀.₅ 基座起步）：先在 GPU 节点上算归一化统计（CPU 分区连不上校园代理，LeRobot 会去查 HF 仓库），再训练。
#   sbatch --export=ALL,CONFIG=<cfg>,REPO=<am_bench/ 下仓库名>,EXP=<exp>,STEPS=40000,BATCH=32 env/train_mt.sbatch
set -uo pipefail
A=/public/home/liaodl/ambench
source $A/env/hpc_env.sh
cd $OPENPI_SRC
CONFIG=${CONFIG:?}; REPO=${REPO:?}; EXP=${EXP:?}; STEPS=${STEPS:-40000}; BATCH=${BATCH:-32}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 WANDB_MODE=disabled
echo "=== node $(hostname) cfg=$CONFIG repo=$REPO exp=$EXP steps=$STEPS batch=$BATCH $(date +%F\ %T)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
NS=$OPENPI_SRC/assets/$CONFIG/am_bench/$REPO/norm_stats.json
if [ ! -f "$NS" ]; then
  echo "--- 计算归一化统计 ($CONFIG) ---"; T0=$SECONDS
  # LeRobot 初始化时会经校园代理(login5)查一次 HF 仓库；登录节点过载时代理不通，这里每 5 分钟重试，最多 3 小时
  for k in $(seq 1 36); do
    python scripts/compute_norm_stats.py --config-name $CONFIG 2>&1 | tail -3
    [ -f "$NS" ] && break
    echo "--- norm_stats 第 $k 次未成功（多半是代理不通），5 分钟后重试 $(date +%T)"; sleep 300
  done
  [ -f "$NS" ] || { echo "!! norm_stats 未生成: $NS"; exit 1; }
  echo "--- norm_stats 生成 $NS  $((SECONDS-T0))s"
else
  echo "--- 复用已有 norm_stats: $NS"
fi
python scripts/train.py $CONFIG --keep-period 1000000 --exp-name=$EXP --num-train-steps=$STEPS --batch-size=$BATCH --overwrite --no-wandb-enabled
RC=$?; echo "=== train RC=$RC $(date +%F\ %T)"; exit $RC
