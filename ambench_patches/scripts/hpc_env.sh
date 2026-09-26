#!/bin/bash
# set -u 安全：调用方可能开 nounset，这里先给可能未设置的变量兜底
: "${http_proxy:=}" "${https_proxy:=}" "${no_proxy:=}" "${LD_LIBRARY_PATH:=}"
# SCUT HPC cluster2 - openpi environment (source this)
source /public/software/apps/anaconda3/2024.10/etc/profile.d/conda.sh
export PATH=/public/home/liaodl/ambench/env/openpi/bin:$PATH
A=/public/home/liaodl/ambench
export HF_LEROBOT_HOME=$A/lerobot_home          # openpi reads $HF_LEROBOT_HOME/<repo_id>
export HF_DATASETS_CACHE=$A/hfcache HF_HOME=$A/hfcache
export TMPDIR=$A/tmp
export OPENPI_DATA_HOME=$A/openpi_assets        # gs://openpi-assets/X -> $OPENPI_DATA_HOME/openpi-assets/X
export OPENPI_SRC=$A/src/ambench/ext/openpi
mkdir -p $HF_LEROBOT_HOME $HF_DATASETS_CACHE $TMPDIR $OPENPI_DATA_HOME/openpi-assets/checkpoints
# weights live directly under $OPENPI_DATA_HOME (symlinks break maybe_download resolve())
# Compute nodes have no direct internet: use the campus proxy unless we are on a login node.
case "$(hostname -s)" in
  admin|login*) : ;;                    # login nodes: direct internet
  *) export http_proxy=http://login5:3128 https_proxy=http://login5:3128
     export HTTP_PROXY=${http_proxy:-} HTTPS_PROXY=${https_proxy:-}
     export no_proxy=localhost,127.0.0.1 NO_PROXY=${no_proxy:-localhost,127.0.0.1} ;;
esac
