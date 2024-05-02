#!/bin/bash
#PBS -l walltime=0:10:00 -q workq -l filesystems=home:tegu -A datascience -l select=2
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO
cd $PBS_O_WORKDIR
export OUTPUT=logs3/$(date +"%Y-%m-%d-%H-%M-%S")
mkdir -p $OUTPUT
env >& $OUTPUT/env.dat
module use /soft/modulefiles
module load conda
conda activate
echo "Running on single node"
MASTER_ADDR=`hostname` mpiexec -np 4 --ppn 4 --cpu-bind depth -d 16 ./launcher.sh python torch_nccl.py >& $OUTPUT/n4.p4.dat
echo "Running on two nodes with 1 rank"
MASTER_ADDR=`hostname` mpiexec -np 2 --ppn 1 --cpu-bind depth -d 16 ./launcher.sh python torch_nccl.py >& $OUTPUT/n2.p1.dat
echo "Running on two nodes with 4 ranks"
MASTER_ADDR=`hostname` mpiexec -np 4 --ppn 2 --cpu-bind depth -d 16 ./launcher.sh python torch_nccl.py >& $OUTPUT/n4.p2.dat
echo "Running on two nodes with 8 ranks"
MASTER_ADDR=`hostname` mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 ./launcher.sh python torch_nccl.py >& $OUTPUT/n8.p4.dat

