# Testing NCCL performance on ALCF machines

This is for performing NCCL tests with different environment variables. 

(1) Default settting, no NCCL environement setup. This will use TCP, resulting very bad performance. 

(2) with following setup
```bash 
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
```

(3) the following setup is to see how aws libfabric plugin helps
```bash
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=${HOME}/PolarisAT/soft/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}
```

Note that this will requires the AWS plugin, which can be built on Polaris
```bash 
git clone v1.9.1-aws https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl
./configure --prefix=/home/hzheng/PolarisAT/soft/aws-ofi-nccl --with-libfabric=/opt/cray/libfabric/1.15.2.0/ --with-cuda=/soft/compilers/cudatoolkit/cuda\
-12.4.1 --with-hwloc=/home/hzheng/PolarisAT/pyenvs/hwloc/
```

(4) the following setup is to see how FI environment variables help
```bash
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=${HOME}/PolarisAT/soft/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
```

(5) The following setup is for testing alltoall 
```bash
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=${HOME}/PolarisAT/soft/aws-ofi-nccl/lib:${LD_LIBRARY_PATH}
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_REQ_BUF_SIZE=8388608
```
