#!/bin/bash
# 双相机会话 -> 20 Hz openpi 数据集（含 base_image）。在 cpu_run 作业里跑。
# 用法：bash env/st2cam_export.sh SESSION_LEROBOT_DIR OUT_DIR REPO_ID "PROMPT"
set -uo pipefail
A=/public/home/liaodl/ambench
PY=$A/env/convert/bin/python
SESS=$1; OUT=$2; REPO=$3; PROMPT=$4
export HF_LEROBOT_HOME=$A/lerobot_home HF_DATASETS_CACHE=$A/hfcache HF_HOME=$A/hfcache TMPDIR=$A/tmp
mkdir -p $HF_DATASETS_CACHE $TMPDIR
[ -f "$SESS/meta/info.json" ] || { echo "!! 没有 $SESS/meta/info.json"; exit 2; }
echo "=== $(date +%F\ %T) export $SESS -> $OUT (repo=$REPO prompt=\"$PROMPT\")"
T0=$SECONDS
$PY $A/src/ambench/scripts/data/export_lerobot_to_openpi.py \
  --dataset_roots "$SESS" --output_root "$OUT" --repo_id "$REPO" \
  --target_hz 20 --ee_image_key observation.images.ee_camera \
  --base_image_key observation.images.base_camera \
  --task_prompt "$PROMPT" --overwrite
RC=$?
echo "=== export RC=$RC  $((SECONDS-T0))s"
[ $RC -eq 0 ] || exit $RC
$PY $A/env/st2cam_check.py "$OUT"
