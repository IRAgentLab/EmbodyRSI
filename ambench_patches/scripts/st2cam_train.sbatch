#!/bin/bash
#SBATCH --propagate=NONE
#SBATCH --job-name=st2cam
#SBATCH --partition=gpuA800
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu2
#SBATCH --cpus-per-task=24
#SBATCH --time=8:00:00
#SBATCH --output=/public/home/liaodl/ambench/logs/%x_%j.out
#SBATCH --error=/public/home/liaodl/ambench/logs/%x_%j.err
# ---- 线程预算：按申请的核数算，不按节点 64 核开（公用节点）----
NT=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-${SLURM_JOB_CPUS_PER_NODE:-8}}}
export OMP_NUM_THREADS=$(( NT/2 > 4 ? NT/2 : 4 )) MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export KIT_THREADS=$(( NT*2/3 > 8 ? NT*2/3 : 8 ))      # Isaac carb.tasking 线程池；Isaac Lab 自己按 64核/8卡=8 配，低于 8 会卡死
( while sleep 120; do echo "[threads] $(date +%T) 本用户总线程=$(ps -u $USER -o nlwp= | awk '{s+=$1} END{print s}') 本作业 OMP=$OMP_NUM_THREADS KIT=$KIT_THREADS top3=$(ps -u $USER -o nlwp=,comm= --sort=-nlwp | head -3 | tr -s " " | tr "\n" ";")"; done ) &
THRPID=$!
trap "kill $THRPID 2>/dev/null" EXIT
# 双相机 ST 探针：从 mt7_no_ndt 起步，include_base_image=True 的配置，跑 STEPS 步（默认 10k）。
#   sbatch --export=ALL,CFG=pi05_am_bench_st2cam_peginhole,REPO=st2cam_PegInHole,EXP=st2cam_PegInHole env/st2cam_train.sbatch


set -uo pipefail
A=/public/home/liaodl/ambench
source $A/env/hpc_env.sh
CFG=${CFG:?}
REPO=${REPO:?}          # am_bench/ 下的仓库目录名，用来定位 norm_stats
EXP=${EXP:-${REPO}_m7}
STEPS=${STEPS:-10000}
FROM=${FROM:-$A/src/ambench/ext/openpi/checkpoints/pi05_am_bench_mt7_no_ndt_20hz_h50_ee_local_relative/mt7_no_ndt/39999}
cd $A/src/ambench/ext/openpi
echo "=== node $(hostname)  cfg=$CFG  repo=$REPO  exp=$EXP  steps=$STEPS  from=$FROM  $(date +%F\ %T)"
nvidia-smi --query-gpu=index,name --format=csv,noheader
NS=$A/src/ambench/ext/openpi/assets/$CFG/am_bench/$REPO/norm_stats.json
if [ ! -f "$NS" ]; then
  echo "--- 计算归一化统计 ($CFG) ---"; T0=$SECONDS
  python scripts/compute_norm_stats.py --config-name $CFG 2>&1 | tail -5
  [ -f "$NS" ] || { echo "!! norm_stats 未生成: $NS"; exit 1; }
  echo "--- norm_stats 生成 $NS  $((SECONDS-T0))s"
else
  echo "--- 复用已有 norm_stats: $NS"
fi
T1=$SECONDS
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py $CFG --keep-period ${KEEP:-1000000} \
  --exp-name=$EXP \
  --weight-loader.params-path "$FROM/params" \
  --num-train-steps=$STEPS --batch-size=8 --overwrite --no-wandb-enabled
RC=$?
echo "=== RC=$RC  train $((SECONDS-T1))s  $(date +%F\ %T)"
ls $A/src/ambench/ext/openpi/checkpoints/$CFG/$EXP/
exit $RC
