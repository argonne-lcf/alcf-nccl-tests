#!/bin/bash
source ./setup.sh
rm -rf aws-ofi-nccl
[ -e aws-ofi-nccl ] || git clone -b v1.6.0-hcopy https://github.com/aws/aws-ofi-nccl.git v1.6.0-hcopy
cd v1.6.0-hcopy
./autogen.sh
CC=gcc-12 CXX=g++-12 ./configure --prefix=/soft/libraries/aws-ofi-nccl/v1.6.0-hcopy --with-libfabric=/opt/cray/libfabric/1.15.2.0/ --with-cuda=/soft/compilers/cudatoolkit/cuda-12.4.1 --with-gdrcopy=$HOME/PolarisAT/soft/gdrcopy --disable-tests
cd -
