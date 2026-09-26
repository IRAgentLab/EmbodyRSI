#!/bin/bash
#SBATCH --propagate=NONE
#SBATCH --job-name=rec
#SBATCH --partition=gpuA800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/public/home/liaodl/ambench/logs/rec_%x_%j.out
#SBATCH --error=/public/home/liaodl/ambench/logs/rec_%x_%j.err
# ---- 线程预算：按申请的核数算，不按节点 64 核开（公用节点）----
NT=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-${SLURM_JOB_CPUS_PER_NODE:-8}}}
export OMP_NUM_THREADS=$(( NT/2 > 4 ? NT/2 : 4 )) MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export KIT_THREADS=$(( NT*2/3 > 8 ? NT*2/3 : 8 ))      # Isaac carb.tasking 线程池；Isaac Lab 自己按 64核/8卡=8 配，低于 8 会卡死
( while sleep 120; do echo "[threads] $(date +%T) 本用户总线程=$(ps -u $USER -o nlwp= | awk '{s+=$1} END{print s}') 本作业 OMP=$OMP_NUM_THREADS KIT=$KIT_THREADS top3=$(ps -u $USER -o nlwp=,comm= --sort=-nlwp | head -3 | tr -s " " | tr "\n" ";")"; done ) &
THRPID=$!
trap "kill $THRPID 2>/dev/null" EXIT
#
# AM-Bench 演示录制（HPC 版）。一个 job 一张 A800，实测 123 条/小时。
#   sbatch --job-name=<短名> --export=ALL,TASK=<名>,LEN=<秒>,N=<条数> env/record_hpc.sbatch
#
# 注意：
#  - num_envs 固定 16。32 会把 RTX 的 Vulkan descriptor set 池撑爆
#    （Unable to allocate descriptor sets），16 是实测安全值。
#  - 单卡多进程无收益（渲染已饱和），所以 1 job = 1 卡。
#  - 冷启动 2-3 分钟，每个 job 至少录 48 条摊薄。
#  - prompt 必须和 record_all.sh 一字不差 —— 它是训练时的语言条件。


set -uo pipefail
A=/public/home/liaodl/ambench
TASK=${TASK:?必须指定 TASK，如 WipeWindow}
LEN=${LEN:?必须指定 LEN（秒）}
N=${N:-80}
NENV=${NENV:-16}
# FAILED=1 时连失败的 episode 一起存下来 —— 调试脚本专家时必须开，
# 否则只知道"没成功"，看不到球飞到哪、末端停在哪。存下来传回 AutoDL 的回放台看轨迹。
SAVE_FAILED=""
[ "${FAILED:-0}" = "1" ] && SAVE_FAILED="--save_failed_episodes"
# MAXTRY=n：跑满 n 次尝试（无论成败）就干净退出。成功率≈0 的任务必须用它，
# 否则循环永远凑不够 num_demos，只能外部强杀，而强杀会截断正在写的 parquet
# （今天三次都栽在这，最后一次连 meta/episodes 都写坏了）。
[ -n "${MAXTRY:-}" ] && SAVE_FAILED="$SAVE_FAILED --max_attempts $MAXTRY"

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
P=${PROMPT[$TASK]:?未知任务 $TASK}
# EMBODIMENT=EE 用 EE oracle（只有一个浮动末端执行器、无飞行动力学）。
# 论文说它的用途是"隔离任务逻辑与飞行动力学、给出任务可完成性的上界"，
# 拿它跑同一个脚本专家就能分清：失败到底是浮空基座跟不住，还是脚本/判据本身有问题。
EMB=${EMBODIMENT:-FAHexa}
# CTRL=MPC 换成全身 MPC。论文自己的消融里它把按按钮从 73.3% 拉到 100%、
# 推滑块从 63.3% 到 83.3%，而 52.22% 那个 SOTA 是用 IK-PID 测的。
CTRL=${CTRL:-PID}
# BASEJOINT=1：换成 12 维动作的任务变体。录制脚本里的 BaseJointActionAdapter
# 按 action_mode 自动启用，用 IK 把脚本专家的末端航点解算成基座位姿+四个关节角
# 作为动作标签。观测键也要跟着换，否则训练时取不到 base_pos/base_quat/arm_joint_pos。
if [ "${BASEJOINT:-0}" = "1" ]; then
  FULL="${TASK}-Am-${EMB}-BaseJoint-Abs-${CTRL}-Direct-v0"
  SKEYS="base_pos base_quat arm_joint_pos gripper_width"
else
  FULL="${TASK}-Am-${EMB}-Abs-${CTRL}-Direct-v0"
  # STATEKEYS 可覆盖：诊断录制时把 frame_pos / peg_center_pos 一并录下，
  # 好看被操作物体相对末端的位置——训练数据不要这么录，维度会对不上。
  SKEYS="${STATEKEYS:-ee_pos ee_quat gripper_width}"
[ "${WRENCH:-0}" = "1" ] && SKEYS="$SKEYS ee_wrench"   # 腕部六维力（ee_link 固定关节反作用力，机体系）追加进 state
fi
# 输出目录带上本体和控制器：EE oracle 的成功样本和 FAHexa 的失败样本
# 不能挤在同一个任务目录下，否则导出时会混进训练集。
SUF=""
[ "$EMB" != "FAHexa" ] && SUF="${SUF}_${EMB}"
[ "$CTRL" != "PID" ] && SUF="${SUF}_${CTRL}"
OUT=$A/data/lerobot_hpc/${TASK}${SUF}

echo "=== node $(hostname)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# 驱动版本守卫：容器里 bind 的图形用户态库来自 545.23.06 的 RPM。
# 集群在滚动升级驱动（已见 gpu2 = 595.58.03），版本不匹配时 Isaac Sim 会报
# "no suitable CUDA GPU" 然后**挂住不退**，耗满整个 --time（实测浪费 12 GPU 小时）。
# 这里几秒钟就判死，并把坏节点记下来供后续 --exclude。
EXPECT_DRV=${EXPECT_DRV:-545.23.06}
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d " ")
if [ "$DRV" != "$EXPECT_DRV" ]; then
  echo "!! 驱动不匹配：本节点 $DRV，容器图形库要求 $EXPECT_DRV"
  echo "!! 节点 $(hostname -s) 不可用于 Isaac Sim，立即退出（避免挂满 --time）"
  echo "$(date '+%F %T') $(hostname -s) driver=$DRV" >> $A/logs/bad_isaac_nodes.txt
  exit 75
fi
# 相机：默认只录末端（训练只用这一路）。回放台要看三视角时传
# CAMS="ee_camera base_camera scene_camera"——scene_camera 由录制脚本按需创建。
CAMARG=""
[ -n "${CAMS:-}" ] && CAMARG="--camera_names $CAMS"
# 存成 LeRobot 视频而不是逐帧 jpg，三路相机时体积差很多
VIDARG=""
[ "${USEVIDEO:-0}" = "1" ] && VIDARG="--lerobot_use_videos"
NOISEARG=""
[ "${NOISE:-0}" = "1" ] && NOISEARG="--inject_noise"   # DART 式：给脚本专家动作注噪，演示里带扰动+纠正
echo "=== cameras=${CAMS:-ee_camera}  use_videos=${USEVIDEO:-0}"

echo "=== 驱动版本匹配 ($DRV)"
echo "=== task=$FULL  env_length_s=$LEN  num_demos=$N  num_envs=$NENV"
echo "=== state_keys=$SKEYS"
echo "=== prompt=\"$P\""
T0=$(date +%s)
bash $A/env/isaac_py.sh scripts/data/record_demos_scripted.py \
  --task "$FULL" --num_envs "$NENV" --num_demos "$N" \
  --dataset_root "$OUT" --headless --device cuda:0 --kit_args="--/plugins/carb.tasking.plugin/threadCount=$KIT_THREADS" \
  --env_length_s "$LEN" $SAVE_FAILED $CAMARG $VIDARG $NOISEARG \
  --state_keys $SKEYS \
  --task_prompt "$P"
RC=$?
DT=$(( $(date +%s) - T0 ))
OK=$(grep -c "✔️" $A/logs/rec_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out 2>/dev/null || echo 0)
NG=$(grep -c "❌" $A/logs/rec_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out 2>/dev/null || echo 0)
echo "=== RC=$RC  elapsed=${DT}s  成功=$OK 失败=$NG"
[ "$DT" -gt 0 ] && [ "$OK" -gt 0 ] && echo "=== THROUGHPUT: $(python3 -c "print(f'{$OK*3600/$DT:.1f}')") demos/hour"
echo "=== 各子判据（失败样本）"
grep "❌" $A/logs/rec_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out 2>/dev/null \
  | grep -oE "[a-z_0-9]+=[01]" \
  | awk -F= '{t[$1]++; if($2==1) h[$1]++} END{for(k in t) printf "    %-24s %3d/%-3d %5.1f%%\n", k, h[k], t[k], 100*h[k]/t[k]}' | sort
echo "=== dataset:"; du -sh "$OUT" 2>/dev/null
