#!/bin/bash
# =====================================================================================
# env/viewer_publish.sh —— AM-Bench 看板发布的固定入口
#   「（可选）录制/评测 → 转换 → 打包 → rsync 到 AliCPU → curl 核对」一条命令跑完。
#   以后"我想看某任务某条轨迹的三视角"就从这里进。
#
# 用法（在 HPC 登录节点、$A=/public/home/liaodl/ambench 下执行）
#   模型 rollout（评测带三视角视频，再转成看板条目 eval_<jobid>_<Task>）：
#     MODE=model  TASK=PegInHole N=6 CFG=<openpi 配置名> CKPT=<checkpoint 绝对路径> bash env/viewer_publish.sh
#   专家演示（脚本专家录 N 次尝试，失败集也保存并在看板标 ✗，条目 <Task>FAHexaAbsPID<SUFFIX>）：
#     MODE=expert TASK=WipeWindow N=3 bash env/viewer_publish.sh
#       可选：FAILED=1(默认) MAXTRY=<总尝试数,默认 N> NENV=<并行环境,默认 N> SUFFIX=_expert3(默认) LEN=<秒,默认按任务表>
#   只转换已有的评测 / 录制（不再占 GPU）：
#     MODE=convert TASK=PegInHole  JOBID=<eval 作业号>              bash env/viewer_publish.sh
#     MODE=convert TASK=WipeWindow JOBID=<record 作业号> KIND=expert bash env/viewer_publish.sh
#   只打包 + 同步 + 核对（AFTER=作业号:作业号… 让打包等这些转换作业结束）：
#     MODE=sync [AFTER=123:456] bash env/viewer_publish.sh
#
#   通用开关：
#     MAXRETRY=2    Isaac Sim 启动时会随机 Aborted（std::system_error，作业 2 分钟就结束、没有 tracking/数据集）。
#                   转换作业发现产物缺失就用同样参数自动重提 GPU+转换作业（最多 MAXRETRY 次），
#                   并把新转换作业追加进仍在排队的 viewer_pack 依赖里；参数记录在 $A/tmp/viewer_jobs/<gpu jobid>.env
#     DEP=<jobid>   GPU 作业排在该作业之后（afterany），用来把多个 GPU 作业串成链、控制并发 ≤3
#     NOSYNC=1      只提交 GPU + 转换作业就返回（批量发布时最后统一 MODE=sync 一次）
#     EXCLUDE=gpu2  GPU 作业排除的节点（默认 gpu2；logs/bad_isaac_nodes.txt 里的坏节点会自动追加）
#     CAMS          相机列表，默认 "ee_camera scene_camera base_camera"（三视角）
#
# 纪律：登录节点只做 sbatch / rsync / curl；录制评测走 GPU 作业（1 卡）；转换/打包走 CPU 作业。
# 转换作业和打包作业各自用 --dependency=singleton 串行（同名作业同一时刻只跑一个），
# 因为它们都会重写同一份索引文件，并发会互相覆盖。
#
# 每步产物：
#   录制    $A/data/lerobot_hpc/<Task>/<Task>FAHexaAbsPID/demo-<时间戳>/lerobot   日志 $A/logs/rec_vrec_<Task>_<jobid>.out
#   评测    $A/outputs/eval_hpc/<jobid>_<Task>/{tracking,videos,infer_log.jsonl}   日志 $A/logs/eval_veval_<Task>_<jobid>.out
#   转换    $A/webroot/viewer/ambench_data/eval_<jobid>_<Task>/rollout_*.{json,mp4}  → webroot/viewer/ambench_rollout_index.json
#           $A/webroot/viewer/ambench_data/<Task>FAHexaAbsPID<SUFFIX>/demo_*.json + $A/webroot/ambench_rollouts/<同名>/demo_*/frames/*.jpg
#                                                                                   → webroot/viewer/ambench_index.json
#           日志 $A/logs/viewer_conv_<jobid>.out
#   打包    $A/viewer_bundle/（只含 has_video 条目）                                    日志 $A/logs/viewer_pack_<jobid>.out
#   同步    ${VIEWER_SSH}:${VIEWER_DIR}  → ${VIEWER_URL}/ambench.html   日志 /tmp/rsync_viewer.log
#   核对    curl 两份索引，打印每个条目的条数 / 成功数
# =====================================================================================
set -uo pipefail
A=/public/home/liaodl/ambench
cd "$A" || exit 1
# 登录节点 ulimit -u 软/硬限制都只有 1000，sbatch 默认原样传给作业（PropagateResourceLimits=ALL）。
# 这是"每用户每节点的线程总数"上限：节点上已有 700+ 个自己的训练线程时，Isaac Sim + JAX 服务一起启动
# 就撞上 pthread_create: Resource temporarily unavailable → std::system_error Aborted，作业 2 分钟就死
# （2026-09-24 一整批 12 个 GPU 作业全这么死的，你自己的 ev2cam10k_* 也是）。登录节点拉不高硬限制，
# 所以 GPU 作业一律 --propagate=NONE：作业改用计算节点 slurmd 的限制（nproc 1029109 / nofile 51200）。
SBATCH_GPU_OPTS="--propagate=NONE --gres=gpu:1"
MODE=${MODE:-}
TASK=${TASK:-}
CAMS=${CAMS:-"ee_camera scene_camera base_camera"}
EXCLUDE=${EXCLUDE:-gpu2}
ALI=${VIEWER_SSH:?请设置 VIEWER_SSH，例如 user@host}
ALI_DIR=${VIEWER_DIR:-/opt/ambench_viewer/}
ALI_URL=${VIEWER_URL:?请设置 VIEWER_URL，例如 http://host:port/viewer}
PY=$A/env/convert/bin/python

declare -A TASK_LEN=( [PressButton]=20 [PegInHole]=20 [PushSlider]=26 [OpenDoor]=20 [LemonHarvesting]=30
  [PullLever]=20 [RotateValve]=20 [NDT]=30 [CabinetPickPlace]=60 [WipeWindow]=35 [FrameAssembly]=30 [TossBall]=10 )

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "!! $*"; exit 1; }

# 追加 logs/bad_isaac_nodes.txt 里记录的驱动不匹配节点
bad_nodes() {
  local x="$EXCLUDE"
  if [ -f "$A/logs/bad_isaac_nodes.txt" ]; then
    for n in $(grep -oE '\bgpu[0-9]+\b' "$A/logs/bad_isaac_nodes.txt" | sort -u); do
      case ",$x," in *",$n,"*) ;; *) x="$x,$n";; esac
    done
  fi
  echo "$x"
}

# 记录 GPU 作业参数，供转换步骤在产物缺失时原样重提
save_params() {   # save_params <gpu jobid>
  mkdir -p "$A/tmp/viewer_jobs"
  { for v in MODE TASK N CFG CKPT FAILED MAXTRY NENV SUFFIX LEN CAMS HZ NACT EXCLUDE MAXRETRY RETRY; do
      [ -n "${!v:-}" ] && printf '%s=%q\n' "$v" "${!v}"; done; } > "$A/tmp/viewer_jobs/$1.env"
}
# _retry <gpu jobid> <原因>：读回参数重提（RETRY+1），把新转换作业追加进排队中的 viewer_pack 依赖
_retry() {
  local jid=$1 why=$2 f="$A/tmp/viewer_jobs/$1.env"
  [ -f "$f" ] || { log "!! $why；且找不到 $f，无法自动重试"; return 1; }
  local RETRY=0 MAXRETRY=2; source "$f"
  if [ "${RETRY:-0}" -ge "${MAXRETRY:-2}" ]; then log "!! $why；已重试 ${RETRY} 次，放弃"; return 1; fi
  # 重提的作业排在当前最新的 veval_/vrec_ 作业之后，不让重试把 GPU 并发撑过 3
  local dep; dep=$(squeue -h -u "${USER:-liaodl}" -t PENDING,RUNNING -o '%i %j' | grep -E ' (veval|vrec)_' | awk '{print $1}' | sort -n | tail -1)
  log "$why → 第 $((RETRY+1))/${MAXRETRY} 次重试：用同样参数重提 GPU 作业（排在 ${dep:-无} 之后）"
  local out; out=$( set -a; source "$f"; RETRY=$((RETRY+1)); NOSYNC=1; DEP="$dep"; set +a; bash "$A/env/viewer_publish.sh" ) \
    || { echo "$out"; log "!! 重提失败"; return 1; }
  echo "$out"
  local c; c=$(echo "$out" | sed -n 's/.*CONV_JOB=\([0-9]*\).*/\1/p')
  [ -n "$c" ] || return 1
  for pk in $(squeue -h -u "$USER" -n viewer_pack -t PENDING -o %i); do
    local d; d=$(squeue -h -j "$pk" -o %E | sed 's/(unfulfilled)//g; s/(null)//')
    scontrol update JobId="$pk" Dependency="${d:+$d,}afterany:$c" && log "viewer_pack $pk 依赖已追加 afterany:$c"
  done
  return 0
}

# ---------------------------------------------------------------- 内部步骤（在 CPU 作业里执行）
# _convert_model <eval jobid>：rollout_to_viewer(--only) + faststart
_convert_model() {
  local jid=$1
  local ed; ed=$(ls -d "$A"/outputs/eval_hpc/${jid}_* 2>/dev/null | head -1)
  if [ -z "$ed" ] || [ ! -f "$ed/tracking/tracking.jsonl" ]; then
    _retry "$jid" "评测 $jid 没有 tracking.jsonl（Isaac Sim 启动崩溃？看 logs/eval_*_${jid}.err）" && exit 0
    die "评测 $jid 无产物，且无法重试"
  fi
  log "convert model: $ed  视频 $(ls "$ed"/videos/*.mp4 2>/dev/null | wc -l) 个"
  $PY "$A/webroot/viewer/rollout_to_viewer.py" --stride 2 --only "${jid}_" || die "rollout_to_viewer 失败"
  $PY "$A/env/faststart.py" "$A/webroot/viewer/ambench_data/eval_$(basename "$ed")" || die "faststart 失败"
  ls "$A/webroot/viewer/ambench_data/eval_$(basename "$ed")" | head -40
}

# _convert_expert <TASK> <rec jobid> <suffix> <maxeps> <saved_failed 0/1>
_convert_expert() {
  local task=$1 jid=$2 suffix=$3 maxeps=$4 savedf=$5
  local rlog; rlog=$(ls "$A"/logs/rec_*_${jid}.out 2>/dev/null | head -1)
  [ -n "$rlog" ] || die "找不到录制日志 logs/rec_*_${jid}.out"
  local ddir; ddir=$(grep -m1 'Dataset directory:' "$rlog" | sed 's/.*Dataset directory: *//' | tr -d '[:space:]')
  if [ -z "$ddir" ] || [ ! -f "$ddir/lerobot/meta/info.json" ]; then
    _retry "$jid" "录制 $jid 没有完整数据集（目录=$ddir；看 logs/rec_*_${jid}.err）" && exit 0
    die "录制 $jid 无产物，且无法重试"
  fi
  # demo_to_viewer 取任务目录下最新的 demo-*，用符号链接目录把要转的那次会话单独挑出来
  local vroot=$A/tmp/vroot_${jid}
  rm -rf "$vroot"; mkdir -p "$vroot/${task}FAHexaAbsPID"
  ln -s "$ddir" "$vroot/${task}FAHexaAbsPID/$(basename "$ddir")"
  local sf=""; [ "$savedf" = "1" ] && sf="--saved-failed"
  log "convert expert: $ddir  suffix=$suffix max-eps=$maxeps success-log=$rlog saved_failed=$savedf"
  $PY -u "$A/webroot/viewer/demo_to_viewer.py" "$vroot" "$A/webroot" --suffix "$suffix" \
      --max-eps "$maxeps" --success-log "$rlog" $sf || die "demo_to_viewer 失败"
  rm -rf "$vroot"
  ls "$A/webroot/viewer/ambench_data/${task}FAHexaAbsPID${suffix}"
}

case "${1:-}" in
  _convert_model)  shift; _convert_model "$@"; exit $? ;;
  _convert_expert) shift; _convert_expert "$@"; exit $? ;;
esac

# ---------------------------------------------------------------- 提交 GPU 作业
GPU_JID=""
DEPARG=""; [ -n "${DEP:-}" ] && DEPARG="--dependency=afterany:${DEP}"
EXCL=$(bad_nodes)

case "$MODE" in
  model)
    [ -n "$TASK" ] || die "MODE=model 需要 TASK"
    [ -n "${CFG:-}" ] && [ -n "${CKPT:-}" ] || die "MODE=model 需要 CFG 和 CKPT"
    [ -d "$CKPT" ] || die "CKPT 不存在: $CKPT"
    N=${N:-6}
    GPU_JID=$(env TASK="$TASK" ROLLOUTS="$N" HZ=${HZ:-20} NACT=${NACT:-16} VIDEO=1 CAMS="$CAMS" CFG="$CFG" CKPT="$CKPT" \
      sbatch --parsable --job-name="veval_${TASK}" $SBATCH_GPU_OPTS --exclude="$EXCL" $DEPARG --export=ALL "$A/env/eval_hpc.sbatch") \
      || die "提交评测失败"
    GPU_JID=${GPU_JID%%;*}; save_params "$GPU_JID"
    log "GPU 评测作业 $GPU_JID  task=$TASK rollouts=$N cfg=$CFG  → outputs/eval_hpc/${GPU_JID}_${TASK}  日志 logs/eval_veval_${TASK}_${GPU_JID}.out"
    KIND=model; JOBID=$GPU_JID
    ;;
  expert)
    [ -n "$TASK" ] || die "MODE=expert 需要 TASK"
    N=${N:-3}; FAILED=${FAILED:-1}; MAXTRY=${MAXTRY:-$N}; NENV=${NENV:-$N}; SUFFIX=${SUFFIX:-_expert3}
    LEN=${LEN:-${TASK_LEN[$TASK]:-}}; [ -n "$LEN" ] || die "未知任务 $TASK，请给 LEN"
    GPU_JID=$(env TASK="$TASK" N="$N" LEN="$LEN" NENV="$NENV" CAMS="$CAMS" FAILED="$FAILED" MAXTRY="$MAXTRY" \
      sbatch --parsable --job-name="vrec_${TASK}" $SBATCH_GPU_OPTS --exclude="$EXCL" $DEPARG --export=ALL "$A/env/record_hpc.sbatch") \
      || die "提交录制失败"
    GPU_JID=${GPU_JID%%;*}; save_params "$GPU_JID"
    log "GPU 录制作业 $GPU_JID  task=$TASK N=$N LEN=$LEN nenv=$NENV failed=$FAILED maxtry=$MAXTRY cams=[$CAMS]  日志 logs/rec_vrec_${TASK}_${GPU_JID}.out"
    KIND=expert; JOBID=$GPU_JID
    ;;
  convert)
    [ -n "${JOBID:-}" ] || die "MODE=convert 需要 JOBID"
    KIND=${KIND:-model}
    N=${N:-3}; FAILED=${FAILED:-1}; MAXTRY=${MAXTRY:-$N}; SUFFIX=${SUFFIX:-_expert3}
    ;;
  sync) ;;
  *) sed -n '2,45p' "$0"; exit 2 ;;
esac

# ---------------------------------------------------------------- 提交转换作业（CPU，singleton 串行）
CONV_JID=""
if [ "$MODE" != "sync" ]; then
  if [ "$KIND" = "model" ]; then
    CMD="bash $A/env/viewer_publish.sh _convert_model $JOBID"
  else
    [ -n "$TASK" ] || die "expert 转换需要 TASK"
    CMD="bash $A/env/viewer_publish.sh _convert_expert $TASK $JOBID $SUFFIX $MAXTRY $FAILED"
  fi
  CDEP="singleton"; [ -n "$GPU_JID" ] && CDEP="afterany:${GPU_JID},singleton"
  CONV_JID=$(env CMD="$CMD" sbatch --parsable --job-name=viewer_conv --dependency="$CDEP" --export=ALL "$A/env/cpu_run.sbatch") \
    || die "提交转换作业失败"
  CONV_JID=${CONV_JID%%;*}
  log "CPU 转换作业 $CONV_JID（依赖 $CDEP）  日志 logs/viewer_conv_${CONV_JID}.out"
  if [ "${NOSYNC:-0}" = "1" ]; then
    echo "GPU_JOB=$GPU_JID CONV_JOB=$CONV_JID"
    exit 0
  fi
  AFTER="${AFTER:+$AFTER:}$CONV_JID"
fi

# ---------------------------------------------------------------- 打包（CPU，等转换结束）→ rsync（登录节点）→ curl 核对
PDEP="singleton"; [ -n "${AFTER:-}" ] && PDEP="afterany:${AFTER},singleton"
log "提交打包作业（依赖 $PDEP），--wait 等它结束……"
PACK_JID=$(sbatch --parsable --wait --job-name=viewer_pack --dependency="$PDEP" "$A/env/pack.sbatch") ; PRC=$?
PACK_JID=${PACK_JID%%;*}
log "打包作业 $PACK_JID 结束 rc=$PRC  日志 logs/viewer_pack_${PACK_JID}.out"
[ "$PRC" = "0" ] || die "打包失败，见 logs/viewer_pack_${PACK_JID}.out"
tail -3 "$A/logs/viewer_pack_${PACK_JID}.out" 2>/dev/null

# rsync 到 AliCPU：登录节点执行，限速 20 MB/s，串行（flock）；轮询到进程结束
(
  flock 9
  nohup nice rsync -a --bwlimit=20000 -e "ssh -i ${VIEWER_SSH_KEY:-~/.ssh/id_ed25519} -o StrictHostKeyChecking=no" \
      "$A/viewer_bundle/" "$ALI:$ALI_DIR" > /tmp/rsync_viewer.log 2>&1 &
  RPID=$!
  log "rsync PID=$RPID → $ALI:$ALI_DIR（日志 /tmp/rsync_viewer.log）"
  while kill -0 "$RPID" 2>/dev/null && pgrep -f "rsync.*viewer_bundle" >/dev/null; do sleep 20; done
  wait "$RPID"; RRC=$?
  log "rsync 结束 rc=$RRC"; tail -3 /tmp/rsync_viewer.log
  [ "$RRC" = "0" ] || exit 1
) 9>/tmp/viewer_rsync.lock || die "rsync 失败"

# curl 核对线上索引
log "核对线上索引 $ALI_URL/ambench_index.json + ambench_rollout_index.json"
for idx in ambench_index.json ambench_rollout_index.json; do
  curl -s --max-time 30 "$ALI_URL/$idx" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception as e:
    print('   !! 解析失败', e); sys.exit(1)
print('   %s: %d 个条目' % ('$idx', len(d)))
for e in d:
    rs=e.get('rollouts',[]); ok=sum(1 for r in rs if r.get('success'))
    print('     %-44s %2d 条  成功 %2d/%-2d  has_video=%s  cams=%s' % (e.get('dir') or e['task'], len(rs), ok, len(rs), e.get('has_video'), e.get('n_cameras', e.get('n_with_video',''))))
" || log "!! $idx 核对失败"
done
log "完成：$ALI_URL/ambench.html"
