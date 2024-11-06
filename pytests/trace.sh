#!/bin/bash
#PBS -l walltime=0:10:00 -q workq -l filesystems=home:tegu -A datascience -l select=2

cd $PBS_O_WORKDIR
module use /soft/modulefiles
module load conda/2024-04-29
source venvs/torch_2.3.0.nccl_2.18/bin/activate
echo "which python: " `which python`

export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.6.0-gnu/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export NCCL_CROSS_NIC=1
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_REQ_BUF_SIZE=8388608
export FI_CXI_DEFAULT_TX_SIZE=1028576
export FI_CXI_RDZV_THRESHOLD=16384
export NCCL_DEBUG=TRACE
export NCCL_NET_GDR_LEVEL=PHB
export OUTPUT=trace/$(date +"%Y-%m-%d-%H-%M-%S")

mkdir -p $OUTPUT

env >& $OUTPUT/env.dat
echo "ldd /home/hzheng/PolarisAT/alcf-nccl-tests/pytests/venvs/torch_2.3.0.nccl_2.18/lib/python3.11/site-packages/torch/lib/libtorch_python.so >"  >& $OUTPUT/ldd.dat
ldd /home/hzheng/PolarisAT/alcf-nccl-tests/pytests/venvs/torch_2.3.0.nccl_2.18/lib/python3.11/site-packages/torch/lib/libtorch_python.so >> $OUTPUT/ldd.dat
echo "\n ldd /home/hzheng/PolarisAT/alcf-nccl-tests//v2.18.3-1/build/lib/libnccl.so > " >> $OUTPUT/ldd.dat
ldd /home/hzheng/PolarisAT/alcf-nccl-tests//v2.18.3-1/build/lib/libnccl.so >> $OUTPUT/ldd.dat

echo "Running on single node" 
piexec -np 4 --ppn 4 --cpu-bind depth -d 16 ./launcher.sh python torch_nccl.py >& $OUTOUT/np4.ppn4.dat
echo "-------------------------------\n"
echo "Running on two nodes with 1 rank"
mpiexec -np 2 --ppn 1 --cpu-bind depth -d 16 ./launcher.sh python torch_nccl.py >& $OUTPUT/np2.ppn1.dat
echo "-------------------------------\n"
echo "Running on two nodes with 4 ranks"
mpiexec -np 4 --ppn 2 --cpu-bind depth -d 16 ./launcher.sh python torch_nccl.py >& $OUTPUT/np4.ppn2.dat
echo "-------------------------------\n"
echo "Running on two nodes with 8 ranks"
mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 --gpu-bind none ./launcher.sh python torch_nccl.py >& $OUTPUT/np8.ppn4.dat
