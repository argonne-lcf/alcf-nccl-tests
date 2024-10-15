#!/bin/bash
PLATFORM=${PLATFORM:-"polaris"}
source ./setup/$PLATFORM.sh
rm -rf aws-ofi-nccl
[ -e aws-ofi-nccl ] || git clone -b v1.6.0 https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl
./autogen.sh
CC=gcc-12 CXX=g++-12 ./configure --prefix=/soft/libraries/aws-ofi-nccl/v1.6.0 --with-libfabric=/opt/cray/libfabric/1.15.2.0/ --with-cuda=${CUDA_HOME} --disable-tests
cd -
