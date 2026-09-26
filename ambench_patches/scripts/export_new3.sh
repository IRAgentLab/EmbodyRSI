#!/bin/bash
# 三个新任务的正式录制 -> 20 Hz openpi 数据集（双相机；腕部力默认不导出，8 维 state），再与 9 个老任务合并成 mt12_mixed_20hz。
cd /public/home/liaodl/ambench
A=/public/home/liaodl/ambench; PY=$A/env/convert/bin/python; L=$A/lerobot_home/am_bench; D=$A/data/lerobot_hpc
export HF_LEROBOT_HOME=$A/lerobot_home HF_DATASETS_CACHE=$A/hfcache HF_HOME=$A/hfcache TMPDIR=$A/tmp
exp() { # name roots... (last arg = prompt)
  local NAME=$1; shift; local PROMPT="${@: -1}"; local ROOTS=("${@:1:$#-1}")
  $PY $A/src/ambench/scripts/data/export_lerobot_to_openpi.py --dataset_roots "${ROOTS[@]}" \
     --output_root $L/st2cam_$NAME --repo_id am_bench/st2cam_$NAME --target_hz 20 \
     --ee_image_key observation.images.ee_camera --base_image_key observation.images.base_camera \
     --task_prompt "$PROMPT" --overwrite && $PY $A/env/st2cam_check.py $L/st2cam_$NAME
}
case "${STEP:?}" in
  frame) exp FrameAssembly $D/FrameAssembly/FrameAssemblyFAHexaAbsPID/demo-20260925_002954/lerobot "pick up the frame and install it on the four posts" ;;
  wipe)  exp WipeWindow $D/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260925_002923/lerobot "clear the red stain on the window" ;;
  toss)  exp TossBall $D/TossBall/TossBallFAHexaAbsPID/demo-20260925_044110/lerobot "toss the ball into the bin" ;;
  merge) $PY $A/env/merge_openpi_datasets.py $L/mt12_mixed_20hz $L/st2cam_PegInHole $L/st2cam_PressButton $L/st2cam_PushSlider $L/st2camN_PushSlider $L/st2cam_OpenDoor $L/st2cam_PullLever $L/st2cam_LemonHarvesting $L/st2cam_CabinetPickPlace $L/st2cam_RotateValve $L/st2cam_NDT $L/clean_FrameAssembly $L/st2cam_WipeWindow $L/st2cam_TossBall ;;
esac
