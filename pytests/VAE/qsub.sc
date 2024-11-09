#!/bin/bash
#PBS -S /bin/bash
#PBS -l select=2
#PBS -l walltime=00:10:00
#PBS -N VAE
#PBS -A datascience
#PBS -l filesystems=home:tegu

module load conda
conda activate
unset NCCL_NET
unset NCCL_NET_GDR_LEVEL


cd $PBS_O_WORKDIR
OUTPUT=results/$(date +'%Y-%m-%d-%H-%M-%S')
mkdir -p $OUTPUT

echo "Run without AWS"

mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 python ./VAE_with_LSTM.py >& $OUTPUT/without_aws.log

echo "Run with AWS"
source ../aws.sh
export NCCL_DEBUG=TRACE
mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 python ./VAE_with_LSTM.py >& $OUTPUT/with_aws.log


