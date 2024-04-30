#!/bin/bash
source ./setup.sh
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests/
CC=cc CXX=CC make MPI=1 CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME MPI_HOME=$(dirname $(dirname `which cc`))
cd -rf build ../
