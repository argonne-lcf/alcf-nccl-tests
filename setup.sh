#!/bin/bash
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.4.1/
export NCCL_HOME=/home/hzheng/PolarisAT/NCCL/2.21/nccl_2.21.5-1+cuda12.4_x86_64
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH

