git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests/
CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.4.1/
NCCL_HOME=/home/hzheng/PolarisAT/NCCL/2.21/nccl_2.21.5-1+cuda12.4_x86_64
CC=cc CXX=CC make MPI=1 CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME MPI_HOME=/opt/cray/pe/craype/2.7.20/
cd -rvf build ../
