#!/bin/bash
#PBS -l walltime=0:10:00 -q debug-scaling -l filesystems=home:eagle -A datascience -l select=2

module load conda
conda activate

unset NCCL_NET
unset NCCL_NET_GDR_LEVEL
echo "Run without AWS"
cd $PBS_O_WORKDIR

function run_jobs() {
    echo "Running on single node"
    mpiexec -np 4 --ppn 4 --cpu-bind depth -d 16 ./launcher.sh python torch_nccl.py
    echo "-------------------------------"
    echo "Running on two nodes with 1 rank per node"
    mpiexec -np 2 --ppn 1 --cpu-bind depth -d 16 ./launcher.sh python torch_nccl.py
    echo "-------------------------------"
    echo "Running on two nodes with 2 ranks per node"
    mpiexec -np 4 --ppn 2 --cpu-bind depth -d 16 ./launcher.sh python torch_nccl.py
    echo "-------------------------------"
    echo "Running on two nodes with 4 ranks per node"
    mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 ./launcher.sh python torch_nccl.py
}
export run_jobs

OUTPUT=results/$(date +'%Y-%m-%d-%H-%M-%S')
mkdir -p $OUTPUT
run_jobs >& $OUTPUT/without_aws.log

echo "Run with AWS"
cd $PBS_O_WORKDIR
source ../aws_2025.01.15.sh
env $OUTPUT/env_aws.dat
run_jobs >& $OUTPUT/with_aws.log

