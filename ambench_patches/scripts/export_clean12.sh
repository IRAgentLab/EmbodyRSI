#!/bin/bash
# 扎实版本数据集：12 个任务全部用新环境录制的会话（2026-09-25 09:30 之后、≥32 集的会话），导出 20 Hz 双相机 openpi 格式并合并。
# STEP=export <Task>   STEP=merge
cd /public/home/liaodl/ambench
A=/public/home/liaodl/ambench; PY=$A/env/convert/bin/python; L=$A/lerobot_home/am_bench; D=$A/data/lerobot_hpc
export HF_LEROBOT_HOME=$A/lerobot_home HF_DATASETS_CACHE=$A/hfcache HF_HOME=$A/hfcache TMPDIR=$A/tmp
declare -A PROMPT=( [PressButton]="press the button" [PegInHole]="insert the peg into the hole" [RotateValve]="rotate the valve clockwise"
  [PushSlider]="push the slider to the right" [PullLever]="pull the lever upwards" [OpenDoor]="open the door"
  [NDT]="contact the inspection point and hold" [LemonHarvesting]="pick the yellow lemon and place it in the bin"
  [FrameAssembly]="pick up the frame and install it on the four posts"
  [CabinetPickPlace]="open the cabinet sliding door, pick up the red can, and place it on top of the cabinet"
  [WipeWindow]="clear the red stain on the window" [TossBall]="toss the ball into the bin" )
CUTOFF="2026-09-25 09:00"   # 重录链 09:28 启动，会话目录 mtime 是创建时刻
sessions_for() { # newest-first sessions recorded after CUTOFF with >=32 episodes (+ explicit extras)
  local T=$1; local out=()
  for d in $(ls -dt $D/$T/${T}FAHexaAbsPID/demo-* 2>/dev/null); do
    [ "$(date -r $d +%s)" -ge "$(date -d "$CUTOFF" +%s)" ] || continue
    n=$($PY -c "import json,sys; print(json.load(open('$d/lerobot/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0)
    [ "$n" -ge 32 ] && out+=("$d/lerobot")
  done
  case $T in
    FrameAssembly) out+=("$D/FrameAssembly/FrameAssemblyFAHexaAbsPID/demo-20260925_002954/lerobot") ;;
    WipeWindow)    out=("$D/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260925_002923/lerobot") ;;
    TossBall)      out=("$D/TossBall/TossBallFAHexaAbsPID/demo-20260925_044110/lerobot") ;;
  esac
  printf '%s\n' "${out[@]}"
}
case "${STEP:?}" in
  export)
    T=${2:-${TASK:?}}; mapfile -t ROOTS < <(sessions_for $T)
    echo "=== $T sessions: ${ROOTS[*]}"; [ ${#ROOTS[@]} -ge 1 ] || { echo "!! no sessions for $T"; exit 3; }
    for r in "${ROOTS[@]}"; do $PY env/check_parquet.py $r | grep -E 'bad=[1-9]' && { echo "!! corrupt parquet in $r"; exit 4; }; done
    $PY $A/src/ambench/scripts/data/export_lerobot_to_openpi.py --dataset_roots "${ROOTS[@]}" \
       --output_root $L/clean_$T --repo_id am_bench/clean_$T --target_hz 20 \
       --ee_image_key observation.images.ee_camera --base_image_key observation.images.base_camera \
       --task_prompt "${PROMPT[$T]}" --overwrite && $PY $A/env/st2cam_check.py $L/clean_$T ;;
  merge)
    ARGS=(); for T in PegInHole PressButton PushSlider OpenDoor PullLever LemonHarvesting CabinetPickPlace RotateValve NDT FrameAssembly WipeWindow TossBall; do ARGS+=("$L/clean_$T"); done
    $PY $A/env/merge_openpi_datasets.py $L/mt12_clean_20hz "${ARGS[@]}" ;;
esac
