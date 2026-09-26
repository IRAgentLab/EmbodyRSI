#!/bin/bash
# 扎实版本：9 个老任务用新环境统一重录（双相机 + 腕部力，80 集；PushSlider 另加 40 集注噪）。
# 3 条并行链（每链串行），同时最多占 3 张卡。
cd /public/home/liaodl/ambench
export CAMS="ee_camera base_camera" WRENCH=1
declare -A LEN
for T in PegInHole PushSlider PressButton OpenDoor PullLever LemonHarvesting CabinetPickPlace RotateValve NDT; do
  f=$(ls -t logs/rec_rec2cam_${T}_*.out 2>/dev/null | head -1)
  L=$(grep -oE 'env_length_s=[0-9]+' $f 2>/dev/null | head -1 | cut -d= -f2)
  LEN[$T]=${L:-20}
done
lane() { local DEP=""; for T in "$@"; do
  local NE=16; [ "$T" = "CabinetPickPlace" ] && NE=8
  J=$(sbatch --parsable --job-name=rr_${T} --exclude=gpu2 $DEP --export=ALL,TASK=$T,LEN=${LEN[$T]},N=80,NENV=$NE env/record_hpc.sbatch)
  echo "rr_$T LEN=${LEN[$T]} NENV=$NE job $J ${DEP:+(after $DEP)}"; DEP="--dependency=afterany:$J"
  if [ "$T" = "PushSlider" ]; then
    J2=$(NOISE=1 sbatch --parsable --job-name=rrN_PushSlider --exclude=gpu2 $DEP --export=ALL,TASK=PushSlider,LEN=${LEN[$T]},N=40,NENV=16 env/record_hpc.sbatch)
    echo "rrN_PushSlider (noise, 40) job $J2"; DEP="--dependency=afterany:$J2"
  fi
done; }
lane PegInHole PressButton NDT
lane PushSlider PullLever RotateValve
lane LemonHarvesting OpenDoor CabinetPickPlace
squeue -u liaodl -h -o '%.9i %.20j %.2t %.8M %R' | grep -vE 'ehm_'
