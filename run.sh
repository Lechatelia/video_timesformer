#!/bin/bash

CONFIG=$1
JOB_NAME=${2:-"baseline"}
GPUS=${3:-8}
SRUN=${4:-'srun'} # srun spring local

GPUS_PER_NODE=${GPUS:-8}
if [ $GPUS_PER_NODE -ge 8 ]; then
  GPUS_PER_NODE=8
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}

PY_ARGS=${@:5}

# SCRIPTPATH=$(dirname "$0")
WORK_DIR=${CONFIG//configs/work_dirs}
WORK_DIR=${WORK_DIR//.yaml//$JOB_NAME}
echo $WORK_DIR
mkdir  -p $WORK_DIR


now=$(date +"%Y%m%d_%H%M%S")


a=$(echo $HOSTNAME | cut  -c12-16)


  srun --partition=vc_research_2 --mpi=pmi2 \
  --job-name=${JOB_NAME} -n1 \
  --gres=gpu:${GPUS_PER_NODE}  --ntasks-per-node=1 \
  --kill-on-bad-exit=1  --cpus-per-task 32 \
  python -u run_net.py \
  --cfg ${CONFIG}  \
  ${PY_ARGS} OUTPUT_DIR $WORK_DIR \
  2>&1 | tee -a $WORK_DIR/exp_$now.txt

