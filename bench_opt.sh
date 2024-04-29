#!/bin/bash
#PBS -l walltime=0:10:00
#PBS -A datascience
#PBS -l nodes=2:ppn=4
#PBS -l filesystems=home:tegu
module restore
cd $PBS_O_WORKDIR
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | sed -n $=)
export DATE_TAG=$(date +"%Y-%m-%d-%H-%M-%S")
OUTPUT=results_ss11/n$PBS_JOBSIZE.g4/$DATE_TAG/
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.4.1/
export NCCL_HOME=/home/hzheng/PolarisAT/NCCL/2.21/nccl_2.21.5-1+cuda12.4_x86_64
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
mkdir -p $OUTPUT/
env >& $OUTPUT/env.dat
module list >& $OUTPUT/modules.dat
ldd ./build/all_reduce_perf >& $OUTPUT/ldd.dat
cat $PBS_NODEFILE | uniq >& $OUTPUT/nodes.dat

aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_reduce_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_reduce_output.dat
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/alltoall_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/alltoall_output.dat
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_gather_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_gather_output.dat


export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1

aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_reduce_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_reduce_output.dat.2
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/alltoall_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/alltoall_output.dat.2
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_gather_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_gather_output.dat.2

export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=${HOME}/PolarisAT/soft/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}

aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_reduce_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_reduce_output.dat.3
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/alltoall_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/alltoall_output.dat.3
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_gather_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_gather_output.dat.3


export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_reduce_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_reduce_output.dat.4
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/alltoall_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/alltoall_output.dat.4
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_gather_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_gather_output.dat.4

export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_REQ_BUF_SIZE=8388608
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_reduce_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_reduce_output.dat.5
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/alltoall_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/alltoall_output.dat.5
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_gather_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_gather_output.dat.5
