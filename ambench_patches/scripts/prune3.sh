#!/bin/bash
# 2026-09-25 第三轮清理清单（dry-run 默认；DO=1 bash env/prune3.sh 才真正删除）。
# 依据：成绩页“固定配方（5k/10k 取优）”每任务只保留胜出的那个 step；诊断/失败专家的录制会话全部可删。
set -u
A=/public/home/liaodl/ambench
if squeue -u liaodl -h -t R -o '%j' | grep -qE '^(tr2cam|stes|train|stown|st2cam)'; then echo "!! 有 openpi 训练在跑，先等它结束"; exit 1; fi
C=$A/src/ambench/ext/openpi/checkpoints/pi05_am_bench
LIST=(
  # --- 双相机 ST：每任务删掉 5k/10k 里较差的那个（各 42 GB）---
  $C"_st2cam_peginhole/st2cam_PegInHole_m7/5000"            # 63.3 < 96.7   (已不存在则跳过)
  $C"_st2camn_pushslider/st2camN_PushSlider_m7/9999"        # 96.7 < 100
  $C"_st2cam_pressbutton/st2cam_PressButton_m7/5000"        # 100 = 100，留 9999
  $C"_st2cam_opendoor/st2cam_OpenDoor_m7/9999"              # 96.7 < 100
  $C"_st2cam_pulllever/st2cam_PullLever_m7/5000"            # — < 93.3
  $C"_st2cam_lemonharvesting/st2cam_LemonHarvesting_m7/5000" # 66.7 < 86.7
  $C"_st2cam_cabinetpickplace/st2cam_CabinetPickPlace_m7/9999" # 5000 是 13.3，9999 未评/更差
  $C"_st2cam_rotatevalve/st2cam_RotateValve_m7/9999"        # 10.0 < 26.7
  $C"_st2cam_ndt/st2cam_NDT_a/5000"                         # 46.7 < 63.3
  $C"_st2cam_ndt/st2cam_NDT_m7/5000"                        # m7 起步两档都不如 NDT_a
  $C"_st2cam_ndt/st2cam_NDT_m7/9999"
  # --- 失败专家 / 诊断录制（FrameAssembly 保留 0924_223452(32) + 0925_002954(48)）---
  $A/data/lerobot_hpc/FrameAssembly/FrameAssemblyEEAbsPID
  $A/data/lerobot_hpc/FrameAssembly/FrameAssemblyFAHexaAbsMPC
  $A/data/lerobot_hpc/FrameAssembly/FrameAssemblyFAHexaAbsPID/demo-20260919_211134
  $A/data/lerobot_hpc/FrameAssembly/FrameAssemblyFAHexaAbsPID/demo-20260923_005128
  $A/data/lerobot_hpc/FrameAssembly/FrameAssemblyFAHexaAbsPID/demo-20260924_104118
  $A/data/lerobot_hpc/FrameAssembly/FrameAssemblyFAHexaAbsPID/demo-20260924_123757
  $A/data/lerobot_hpc/FrameAssembly/FrameAssemblyFAHexaAbsPID/demo-20260924_140852
  $A/data/lerobot_hpc/FrameAssembly_EE
  # WipeWindow：保留 0925_002923(80 集正式) 与 0925_020444(servo4 看板 6 集)
  $A/data/lerobot_hpc/WipeWindow/WipeWindowEEAbsPID
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsMPC
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260919_174809
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_101454
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_104115
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_112351
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_113751
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_133049
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_152230
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_152932
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_154800
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_160457
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_160602
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_180707
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_182654
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_184606
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_185158
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_222858
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_225721
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_230955
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_230956
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_232025
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_233128
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260924_233516
  $A/data/lerobot_hpc/WipeWindow/WipeWindowFAHexaAbsPID/demo-20260925_001417
  # TossBall：保留 0925_044110(80) 与 0925_024300(弹道看板 6 集)
  $A/data/lerobot_hpc/TossBall/TossBallEEAbsPID
  $A/data/lerobot_hpc/TossBall/TossBallFAHexaAbsMPC
  $A/data/lerobot_hpc/TossBall/TossBallFAHexaAbsPID/demo-20260919_211134
  $A/data/lerobot_hpc/TossBall/TossBallFAHexaAbsPID/demo-20260924_134047
  $A/data/lerobot_hpc/TossBall/TossBallFAHexaAbsPID/demo-20260924_204300
  $A/data/lerobot_hpc/TossBall/TossBallFAHexaAbsPID/demo-20260924_210858
  $A/data/lerobot_hpc/TossBall_EE
  # PushSlider 的早期残片（正式集：0924_011046 80 + 0923_233758 40 注噪）
  $A/data/lerobot_hpc/PushSlider/PushSliderFAHexaAbsPID/demo-20260921_130936
  $A/data/lerobot_hpc/PushSlider/PushSliderFAHexaAbsPID/demo-20260921_173058
  $A/data/lerobot_hpc/PushSlider/PushSliderFAHexaAbsPID/demo-20260921_213703
  $A/data/lerobot_hpc/PushSlider/PushSliderFAHexaAbsPID/demo-20260922_205933
  $A/data/lerobot_hpc/PushSlider/PushSliderFAHexaBaseJointAbsPID
  $A/data/lerobot_hpc/PegInHole/PegInHoleFAHexaAbsPID/demo-20260922_204820
  $A/data/lerobot_hpc/PegInHole/PegInHoleFAHexaBaseJointAbsPID
)
n=0; tot=0
for d in "${LIST[@]}"; do
  [ -e "$d" ] || continue
  sz=$(du -sk "$d" 2>/dev/null | cut -f1); tot=$((tot+sz)); n=$((n+1))
  if [ "${DO:-0}" = "1" ]; then echo "rm -rf $d ($((sz/1048576)) GB)"; rm -rf "$d"; else echo "[dry-run] $((sz/1048576))G  $d"; fi
done
echo "=== $n 个目录，合计 $((tot/1048576)) GB $([ "${DO:-0}" = 1 ] && echo 已删除 || echo 待删除（DO=1 执行）)"
