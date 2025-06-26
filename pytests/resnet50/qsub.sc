#!/bin/bash
#-------------------------------------------------------------------------------
#PBS -S /bin/bash
#PBS -W block=false
#PBS -l select=2
#PBS -q debug
#PBS -l walltime=00:10:00
#PBS -N resnet
#PBS -A datascience
#PBS -l filesystems=home:eagle
#-------------------------------------------------------------------------------
module load conda
conda activate

cd ${PBS_O_WORKDIR}

export HOROVOD_LOG_LEVEL=ERROR
WKDIR=${PBS_O_WORKDIR}/
EXE=${WKDIR}/pytorch_imagenet_ddp.py
EPOCHS=10


TAG=$(date +%F_%H%M)
TAG_DATE_ONLY=$(date +%F)
export LOG_ROOT=./logs/
JOBID=$(echo ${PBS_JOBID} | cut -d '.' -f 1)
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
export PPN=4
WKLD_OPTS="--dummy --epochs ${EPOCHS}"
echo "Running on host:"
hostname
date

export HYDRA_TOPO_DEBUG=1

# Launch using MPICH
#mpirun -np $TOTAL_NUMBER_OF_RANKS -ppn $RANKS_PER_NODE <additional MPICH affinity settings> ./wrapper.sh <your application command>
###########################################################################
# End of section for wrapper.sh script
###########################################################################
echo $LD_LIBRARY_PATH

export TOTAL_NUMBER_OF_RANKS=12
export RANKS_PER_NODE=4

#echo "nodelist:"
if [ -z "$PBS_NODEFILE" ]; then
#    echo "Empty nodefile"
    export HOST_FILE=
    export NUM_NODES=1
else
    cat $PBS_NODEFILE
    export HOST_FILE="-hostfile $PBS_NODEFILE"
    export NUM_NODES=$(cat $PBS_NODEFILE | uniq | wc -l)
fi

export NJOBS=$((NUM_NODES * RANKS_PER_NODE))
#echo -e "#nodes: $NUM_NODES\n#ntasks: $RANKS_PER_NODE\n#jobs: $NJOBS"
export TOTAL_NUMBER_OF_RANKS=$NJOBS

OUTPUTFILE=output.log
#CMD=
export HOROVOD_THREAD_AFFINITY=1,9,17,25,33,41,52,60,68,76,84,92
export PYTORCH_WORKER_AFFINITY=5,13,21,29,37,45,56,64,72,80,88,96

echo "Run without AWS plugin"


OUTPUT=results/$(date +'%Y-%m-%d-%H-%M-%S')
mkdir -p $OUTPUT

unset NCCL_NET NCCL_COLLNET_ENABLE NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC
export NCCL_DEBUG=TRACE
echo "start time: `date`"
mpiexec -np $TOTAL_NUMBER_OF_RANKS -ppn $RANKS_PER_NODE --cpu-bind depth -d 16 python ${EXE} ${WKLD_OPTS} >& $OUTPUT/without_aws.log
echo "end time: `date`"

source ../aws_2025.01.15.sh
echo "Rerun with AWS"
echo "start time: `date`"
env >& $OUTPUT/env_aws.dat
mpiexec -np $TOTAL_NUMBER_OF_RANKS -ppn $RANKS_PER_NODE --cpu-bind depth -d 16 python ${EXE} ${WKLD_OPTS} >& $OUTPUT/with_aws.log
echo "end time: `date`"
