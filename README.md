# Testing NCCL performance on ALCF systems

This is for performing NCCL tests with different environment setups (1)-(5) to identify the best setup for NCCL on ALCF systems. We find out that the optimal setup is

```bash
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.6.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

# Additional variables that might be critical to address any potential hang issue in Python applications. 
export FI_CXI_DEFAULT_TX_SIZE=131072
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_REQ_BUF_SIZE=16MB
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16000
export FI_CXI_RDZV_THRESHOLD=2000
```
This achieves 5-10x improvement over the default setup. 

## Different environment setups

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
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
```

Note that this will requires the AWS plugin, which can be built on Polaris
```bash 
git clone v1.9.1-aws https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl
./configure --prefix=/home/hzheng/PolarisAT/soft/aws-ofi-nccl --with-libfabric=/opt/cray/libfabric/1.15.2.0/ --with-cuda=/soft/compilers/cudatoolkit/cuda\
-12.4.1 --with-hwloc=/soft/libraries/hwloc/
```

(4) the following setup is to see how FI environment variables help
```bash
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
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
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_REQ_BUF_SIZE=8388608
```
## Results & Conclusion

The results are shown in [./results_polaris](./results_polaris). One can find analysis here: [./nccl-performance-evaluation.ipynb](./nccl-performance-evaluation.ipynb)

The main conclusions are: 

* For allreduce and all_gather, the best setup is following. We achieve 5-10x improvement over the default setup. 

    ```bash
    export NCCL_NET_GDR_LEVEL=PHB
    export NCCL_CROSS_NIC=1
    export NCCL_COLLNET_ENABLE=1
    export NCCL_NET="AWS Libfabric"
    export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_MR_CACHE_MONITOR=userfaultfd
    export FI_CXI_DEFAULT_CQ_SIZE=131072
    ```

* For alltoall with message size larger than 8MB, additional setup is needed. But this will influence the latency for allreduce and all_gather at smaller message size (<1MB)
    ```bash
    export FI_CXI_RX_MATCH_MODE=software
    export FI_CXI_RDZV_PROTO=alt_read
    export FI_CXI_REQ_BUF_SIZE=8388608
    ```
    
* For allreduce, Slingshot 11 with (4) setup is 3x speed up over slingshot 10 results (for message size over 10 MB)
* We were able to run up to 540 nodes with (4) and (5).
* With (1), we were not able to go beyond 128 nodes. all_gather will cause node failure. 
