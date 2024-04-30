#!/bin/bash
source ./setup.sh
[ -e aws-ofi-nccl ] || git clone -b v1.9.1-aws https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl
./autogen.sh
./configure --prefix=/home/hzheng/PolarisAT/soft/aws-ofi-nccl --with-libfabric=/opt/cray/libfabric/1.15.2.0/ --with-cuda=/soft/compilers/cudatoolkit/cuda-12.4.1 --with-hwloc=/home/hzheng/PolarisAT/soft/hwloc/
cd -
