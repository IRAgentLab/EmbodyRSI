#!/bin/bash
# Run a python script from the Isaac Sim venv inside the container, on a GPU node.
#   bash env/isaac_py.sh <script.py> [args...]
# Sets the AM-Bench runtime env (JAX cpu, acados, LD_LIBRARY_PATH incl. /opt/nvgl).
A=/public/home/liaodl/ambench
B=$A/src/ambench
V=$A/isaac/venv511
ARGS=$(printf ' %q' "$@")
exec bash $A/env/isaac_run.sh --nv bash -c "
  export JAX_PLATFORMS=cpu
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
  export ACADOS_SOURCE_DIR=$B/ext/acados
  export LD_LIBRARY_PATH=/.singularity.d/libs:/opt/nvgl/lib:$B/ext/acados/lib:$B/ext/acados/build
  export OMNI_KIT_ACCEPT_EULA=YES
  export HOME=/public/home/liaodl
  cd $B
  exec $V/bin/python$ARGS
"
