#!/bin/bash
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.4.1/
export NCCL_HOME=$CRAY_NVIDIA_PREFIX/comm_libs/nccl
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/PolarisAT/soft/hwloc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH

