#!/bin/bash
#PBS -l walltime=0:30:00
#PBS -A datascience
#PBS -l nodes=2:ppn=4
#PBS -q datascience
#PBS -l filesystems=home:eagle
module restore
cd $PBS_O_WORKDIR
source ./setup.sh
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | sed -n $=)
export DATE_TAG=$(date +"%Y-%m-%d-%H-%M-%S")
read system b <<<$(echo $PBS_O_HOST | tr - "\n")
OUTPUT=results_${system}/n$PBS_JOBSIZE.g4/$DATE_TAG/
mkdir -p $OUTPUT/

env >& $OUTPUT/env.dat.1
module list >& $OUTPUT/modules.dat
ldd ./build/all_reduce_perf >& $OUTPUT/ldd.dat
cat $PBS_NODEFILE | uniq >& $OUTPUT/nodes.dat
sleep 10
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_reduce_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_reduce_output.dat.1
sleep 10
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/alltoall_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/alltoall_output.dat.1
sleep 10
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_gather_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_gather_output.dat.1
sleep 10


export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
env >& $OUTPUT/env.dat.2
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_reduce_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_reduce_output.dat.2
sleep 10
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/alltoall_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/alltoall_output.dat.2
sleep 10
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_gather_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_gather_output.dat.2
sleep 10

export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:${LD_LIBRARY_PATH}
env >& $OUTPUT/env.dat.3
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_reduce_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_reduce_output.dat.3
sleep 10
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/alltoall_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/alltoall_output.dat.3
sleep 10
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_gather_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_gather_output.dat.3
sleep 10


export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
env >& $OUTPUT/env.dat.4
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_reduce_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_reduce_output.dat.4
sleep 10
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/alltoall_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/alltoall_output.dat.4
sleep 10
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_gather_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_gather_output.dat.4
sleep 10

export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_REQ_BUF_SIZE=8388608
env >& $OUTPUT/env.dat.5
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_reduce_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_reduce_output.dat.5
sleep 10
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/alltoall_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/alltoall_output.dat.5
sleep 10
aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -d 16 ./build/all_gather_perf -b 4 -e 1073741824 -f 2 -g 1 >& $OUTPUT/all_gather_output.dat.5
sleep 10
