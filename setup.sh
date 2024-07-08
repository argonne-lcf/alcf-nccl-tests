#!/bin/bash
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.4.1/
export NCCL_HOME=${HOME}/alcf-nccl-tests//v2.18.3-1/build/
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH

