#!/bin/bash
#SBATCH --propagate=NONE
#SBATCH --job-name=cpu_run
#SBATCH --partition=cpuXeon6458
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/public/home/liaodl/ambench/logs/%x_%j.out
#SBATCH --error=/public/home/liaodl/ambench/logs/%x_%j.err
# 通用 CPU 分析作业：轨迹/日志解析这类几百 MB JSON 的活不在登录节点跑。
# 用法：sbatch --job-name=xxx --export=ALL,CMD='python ... args' env/cpu_run.sbatch
set -uo pipefail
cd /public/home/liaodl/ambench
export PATH=/public/home/liaodl/ambench/env/convert/bin:$PATH
echo "=== $(hostname)  $(date '+%F %T')  CMD=$CMD"
eval "$CMD"; RC=$?
echo "=== RC=$RC  $(date +%FT%T)"; exit $RC
