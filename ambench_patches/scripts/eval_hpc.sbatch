#!/bin/bash
#SBATCH --propagate=NONE
#SBATCH --job-name=eval
#SBATCH --partition=gpuA800
#SBATCH --gres=gpu:1        # 策略服务和 Isaac Sim 合用一张卡：


                            # 实测 Isaac Sim 只吃 2.7 GB，JAX 那 45 GB 是预分配不是需求
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00   # 30 次 rollout 在长 episode 任务上会超过 2 小时（实测 3 个任务卡在 2:00:05）
#SBATCH --output=/public/home/liaodl/ambench/logs/eval_%x_%j.out
#SBATCH --error=/public/home/liaodl/ambench/logs/eval_%x_%j.err
# ---- 线程预算：按申请的核数算，不按节点 64 核开（公用节点）----
NT=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-${SLURM_JOB_CPUS_PER_NODE:-8}}}
export OMP_NUM_THREADS=$(( NT/2 > 4 ? NT/2 : 4 )) MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export KIT_THREADS=$(( NT*2/3 > 8 ? NT*2/3 : 8 ))      # Isaac carb.tasking 线程池；Isaac Lab 自己按 64核/8卡=8 配，低于 8 会卡死
( while sleep 120; do echo "[threads] $(date +%T) 本用户总线程=$(ps -u $USER -o nlwp= | awk '{s+=$1} END{print s}') 本作业 OMP=$OMP_NUM_THREADS KIT=$KIT_THREADS top3=$(ps -u $USER -o nlwp=,comm= --sort=-nlwp | head -3 | tr -s " " | tr "\n" ";")"; done ) &
THRPID=$!
trap "kill $THRPID 2>/dev/null" EXIT
#
# AM-Bench 策略评测（HPC 版）—— 打通后 A10 就可以退租。
# 一张卡起 openpi 策略服务（conda 环境），一张卡跑 Isaac Sim 评测（apptainer 容器）。
# 两者同节点、共享网络命名空间，走 localhost:8000。
#
#   sbatch --export=ALL,CKPT=<路径>,TASK=<名>,ROLLOUTS=<n> env/eval_hpc.sbatch
set -uo pipefail
A=/public/home/liaodl/ambench
CFG=${CFG:-pi05_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative}
CKPT=${CKPT:-none}   # EXPERT=1 时不需要
TASK=${TASK:-PressButton}
ROLLOUTS=${ROLLOUTS:-6}
NENV=${NENV:-1}
# 端口按作业号派生：多个评测作业可能落到同一节点，写死 8000 会让第二个
# 以 'address already in use' 秒退（实测 4 个作业因此失败）。
PORT=${PORT:-$(( 18000 + SLURM_JOB_ID % 10000 ))}

# 12 个任务全部列全，prompt 必须和录制时一字不差（它是语言条件）
declare -A PROMPT=(
 [PressButton]="press the button"
 [PegInHole]="insert the peg into the hole"
 [RotateValve]="rotate the valve clockwise"
 [PushSlider]="push the slider to the right"
 [PullLever]="pull the lever upwards"
 [OpenDoor]="open the door"
 [NDT]="contact the inspection point and hold"
 [LemonHarvesting]="pick the yellow lemon and place it in the bin"
 [FrameAssembly]="pick up the frame and install it on the four posts"
 [CabinetPickPlace]="open the cabinet sliding door, pick up the red can, and place it on top of the cabinet"
 [WipeWindow]="clear the red stain on the window"
 [TossBall]="toss the ball into the bin"
)
P=${PROMPT[$TASK]:?未知任务}
FULL="${TASK}-Am-FAHexa-Abs-PID-Direct-v0"
OUT=$A/outputs/eval_hpc/${SLURM_JOB_ID}_${TASK}
mkdir -p "$OUT"

echo "=== node $(hostname)  GPUs:"; nvidia-smi --query-gpu=index,name --format=csv,noheader
# 驱动守卫：容器图形库来自 545.23.06；不匹配的节点上 Isaac Sim 会挂死耗满 --time
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d " ")
[ "$DRV" != "${EXPECT_DRV:-545.23.06}" ] && {
  echo "!! 驱动不匹配 $DRV，本节点不可用于 Isaac Sim，立即退出"; exit 75; }
echo "=== 驱动匹配 ($DRV)"
echo "=== ckpt=$CKPT"; echo "=== task=$FULL rollouts=$ROLLOUTS prompt=\"$P\""

# ---- 1) 策略服务（GPU 0，conda openpi 环境）----
source $A/env/hpc_env.sh
cd $OPENPI_SRC
if [ "${EXPERT:-0}" = "1" ]; then
  echo "=== 专家在环模式：不起策略服务、不预热（HOLD=${HOLD:-0}）"
  SPID=""
else
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  nohup python scripts/serve_policy.py --port $PORT \
    policy:checkpoint --policy.config=$CFG --policy.dir=$CKPT \
    > $OUT/serve.log 2>&1 &
SPID=$!
echo "=== 策略服务 PID=$SPID，等待就绪"
for i in $(seq 1 90); do
  grep -qE "server listening|Waiting for" $OUT/serve.log 2>/dev/null && { echo "=== 服务就绪（第 $((i*5)) 秒）"; sleep 3; break; }
  kill -0 $SPID 2>/dev/null || { echo "!! 服务进程已死"; tail -25 $OUT/serve.log; exit 1; }
  sleep 5
done

# ---- 1b) 预热：JAX 第一次调用要 JIT 编译整个模型（几十秒），期间服务端不响应
# websocket 的 keepalive ping，评测客户端会直接 1011 掐断连接。先用同步客户端
# 打一次把 JIT 做掉，之后每次推理约 110 ms。
echo "=== 预热策略服务（触发 JIT 编译）"
T_W=$(date +%s)
timeout 600 python $A/env/test_policy_client.py $PORT 2>&1 | tail -3
if [ ${PIPESTATUS[0]:-0} -eq 124 ]; then
  echo "!! 预热 600s 未响应——策略服务没起来。直接退出，不要空转把 --time 烧光"
  kill $WDPID 2>/dev/null
kill $SPID 2>/dev/null; exit 76
fi
echo "=== 预热耗时 $(( $(date +%s) - T_W ))s"
fi

# 策略执行频率：模型训练于 20 Hz，环境是 120 Hz。
# 不传 --policy-target-hz 时 stride=1，动作块被 6 倍加速执行且跳过位姿插值。
HZ=${HZ:-20}
# 动作块执行步数：默认 8（=400ms 重规划）。调小 = 闭环更紧，推理更频繁。
# 指令低通：动作块之间没有融合，拼接处成锯齿，PID 追锯齿会积累误差到推力饱和。
# SMOOTH=0.3 表示 cmd = 0.3*新 + 0.7*上一次。不设则关闭（行为同以前）。
# 块重叠融合（temporal ensembling 的可负担版本）：队列还剩 OVERLAP 步时提前推理
# 下一块，重叠区按线性斜坡从旧块过渡到新块，消掉硬切造成的指令锯齿。
# 夹爪 oracle（诊断）：位置/姿态仍由策略输出，夹爪换成专家规则。
GOARG=""
[ "${GRIPORACLE:-0}" = "1" ] && GOARG="--gripper-oracle"
OVARG=""
[ -n "${OVERLAP:-}" ] && OVARG="--chunk-overlap $OVERLAP"
SMOOTHARG=""
[ -n "${SMOOTH:-}" ] && SMOOTHARG="--action-smoothing $SMOOTH"
NACTARG=""
[ -n "${NACT:-}" ] && NACTARG="--n-action-steps $NACT"
# 视频：默认关（152MB/任务的 tracking 已经够大）；VIDEO=1 时存 ee+scene 两路
EXPARG=""
[ "${EXPERT:-0}" = "1" ] && EXPARG="--expert-policy --expert-hold ${HOLD:-0}"
VIDARG=""
if [ "${VIDEO:-0}" = "1" ]; then
  VIDARG="--save-video --video-camera-names ${CAMS:-ee_camera scene_camera}"
fi
echo "=== policy_target_hz=$HZ  video=${VIDEO:-0}"

export AMBENCH_LOG_INFER=${LOGINFER:-1}   # 每轮推理的输入状态+完整 50 步输出

# 看门狗：Isaac Sim 有多种静默挂死方式（缺相机、渲染死锁、连不上服务），
# 表现都是一样的——进程活着但没有任何输出，一路空转到 --time 耗尽。
# 评测正常时每 100 步就打一行进度，15 分钟无输出即判定挂死。
WDLOG=/public/home/liaodl/ambench/logs/eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
(
  while sleep 300; do
    NOW=$(date +%s)
    MT=$(stat -c %Y "$WDLOG" 2>/dev/null || echo "$NOW")
    AGE=$(( NOW - MT ))
    if [ "$AGE" -gt 900 ]; then
      echo "!! 看门狗：${AGE}s 无输出，判定挂死，终止作业（不要空转烧卡）" >> "$WDLOG"
      scancel "$SLURM_JOB_ID"
      exit 0
    fi
  done
) &
WDPID=$!

# ---- 2) 评测（GPU 1，Isaac 容器）----
cd $A
CUDA_VISIBLE_DEVICES=0 bash $A/env/isaac_py.sh \
  -m ambench_learn.policies.pi.eval \
  --task "$FULL" --num-envs $NENV --num-rollouts $ROLLOUTS \
  --prompt "$P" --host localhost --port $PORT \
  --output-dir "$OUT" --headless --device cuda:0 --kit_args="--/plugins/carb.tasking.plugin/threadCount=$KIT_THREADS" \
  --policy-target-hz $HZ $NACTARG $SMOOTHARG $OVARG $GOARG $VIDARG $EXPARG
ERC=$?

kill $SPID 2>/dev/null
echo "=== EVAL_RC=$ERC"
echo "=== 结果文件:"; find "$OUT" -name "*.json" 2>/dev/null | head -5
for f in $(find "$OUT" -name "*.json" 2>/dev/null | head -2); do
  echo "--- $f"; python3 -c "
import json,sys
d=json.load(open('$f'))
s=d.get('summary', d)
for k in ('num_rollouts','num_successes','success_rate_percent','success_rate'):
    if k in s: print('   %-22s %s' % (k, s[k]))" 2>/dev/null
done

# 评测崩溃时把退出码透传给 Slurm（Isaac 偶发启动崩溃需要靠 afterok/重试拦住）
exit $ERC
