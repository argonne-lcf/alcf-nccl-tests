#!/bin/bash
#PBS -S /bin/bash
#PBS -l select=2
#PBS -l walltime=00:10:00
#PBS -N VAE
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:eagle

module load conda
conda activate

unset NCCL_NET
unset NCCL_NET_GDR_LEVEL

cd $PBS_O_WORKDIR

[ -e local/lib/python3.11/site-packages/ ] || sh ./install.sh
export PYTHONPATH=$PBS_O_WORKDIR/local/lib/python3.11/site-packages/:$PYTHONPATH
  
OUTPUT=results/$(date +'%Y-%m-%d-%H-%M-%S')
mkdir -p $OUTPUT

echo "Run without AWS"

mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 python ./VAE_with_LSTM.py >& $OUTPUT/without_aws.log

echo "Run with AWS"
source ../aws_2025.01.15.sh
export NCCL_DEBUG=TRACE
env >& $OUTPUT/env_aws.dat
mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 python ./VAE_with_LSTM.py >& $OUTPUT/with_aws.log


