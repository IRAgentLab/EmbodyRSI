#!/bin/bash
# Run a command inside the Isaac Sim container.
#   bash env/isaac_run.sh [--nv] <cmd...>
#
# NOTE: do NOT add `--bind $A:$A`. $A lives under $HOME which apptainer binds
# automatically; the redundant nested bind races with fuse-overlayfs and the
# process intermittently loses its cwd (pip dies in os.getcwd()).
# --nv adds GPU support: apptainer's nvliblist bind PLUS /opt/nvgl, the graphics
# userspace libs the compute-only driver on the gpuA800 nodes is missing
# (from nvidia-driver-libs-545.23.06 el8 rpm -- see env/mk_nvgl.sh).
source /etc/profile.d/modules.sh 2>/dev/null
module load apps/apptainer/1.4.4 2>/dev/null
A=/public/home/liaodl/ambench
export APPTAINER_CACHEDIR=$A/apptainer_cache APPTAINER_TMPDIR=$A/isaac/aptmp
IMG=$A/isaac/isaac311.sif
[ -f "$IMG" ] || IMG=$A/isaac/isaac_sbx

NVFLAG=""
BINDS="--bind $A/isaac/nvgl:/opt/nvgl"
ENVS=""
if [ "$1" = "--nv" ]; then
    NVFLAG="--nv"
    ENVS="--env VK_ICD_FILENAMES=/opt/nvgl/icd.d/nvidia_icd.json \
          --env VK_DRIVER_FILES=/opt/nvgl/icd.d/nvidia_icd.json \
          --env __EGL_VENDOR_LIBRARY_FILENAMES=/opt/nvgl/egl_vendor.d/10_nvidia.json \
          --env LD_LIBRARY_PATH=/.singularity.d/libs:/opt/nvgl/lib"
    shift
fi
case "$(hostname -s)" in
  admin|login*) PROXY="" ;;
  *) PROXY="--env http_proxy=http://192.168.5.249:3128 --env https_proxy=http://192.168.5.249:3128 --env no_proxy=localhost,127.0.0.1" ;;
esac
exec apptainer exec $NVFLAG $BINDS $PROXY $ENVS \
  --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y \
  --env TMPDIR=$A/isaac/tmp --env PIP_CACHE_DIR=$A/tmp/pipcache \
  --pwd "$A" "$IMG" "$@"
