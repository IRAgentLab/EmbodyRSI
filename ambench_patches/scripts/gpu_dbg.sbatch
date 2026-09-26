#!/bin/bash
#SBATCH --propagate=NONE
#SBATCH --job-name=gpu_dbg
#SBATCH --partition=gpuA800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/public/home/liaodl/ambench/logs/%x_%j.out
#SBATCH --error=/public/home/liaodl/ambench/logs/%x_%j.err
# 通用 Isaac GPU 调试作业：sbatch --job-name=xxx --export=ALL,SCRIPT='env/xxx.py',ARGS='...' env/gpu_dbg.sbatch
A=/public/home/liaodl/ambench
NT=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$(( NT/2 > 4 ? NT/2 : 4 )) MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export KIT_THREADS=$(( NT*2/3 > 8 ? NT*2/3 : 8 ))
echo "=== node $(hostname)  $(date '+%F %T')"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d " ")
if [ "$DRV" != "${EXPECT_DRV:-545.23.06}" ]; then echo "!! driver $DRV mismatch"; echo "$(date '+%F %T') $(hostname -s) driver=$DRV" >> $A/logs/bad_isaac_nodes.txt; exit 75; fi
cd $A
bash $A/env/isaac_py.sh $A/$SCRIPT $ARGS --kit_args="--/plugins/carb.tasking.plugin/threadCount=$KIT_THREADS"
RC=$?; echo "=== RC=$RC $(date +%FT%T)"; exit $RC
