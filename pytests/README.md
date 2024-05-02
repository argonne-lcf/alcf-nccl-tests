# Torch NCCL issue with AWS plugin


This study is for investigating the issues

Different columns correspond different environment setups. Different rows correspond to different process grids (nodes x ppn). 1 means sucess, and 0 means failure. 

| Setup  |    (0)   |   (1)   |   (2)   |   (3)  |  
| ------ |  ------- | ------- | ------- | ------ |
| 1x4    |    1     |    1    |    1    |    1   |
| 2x1    |    1     |    1    |    1    |    1   |
| 2x2    |    0     |    0    |    1    |    1   |
| 2x4    |    0     |    0    |    1    |    1   |

We find that anytime when we have a set of NCCL environment, it will fail going beyond single nodes. 
```bash
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
```

Now we would like see what are the problem for each of the three NCC environment variables [let us designated as (a-c)]

| Setup  |   (0-a)  |  (0-b)  |  (0-c)  | (0-ab) | (0-ac) | (0-bc) |
| ------ |  ------- | ------- | ------- | ------ |  ----- | ------ |
| 1x4    |    1     |    1    |    1    |    1   |    1   |    1   |
| 2x1    |    1     |    1    |    1    |    1   |    1   |    1   |
| 2x2    |    0     |    1    |    1    |    0   |    0   |    1   |
| 2x4    |    0     |    1    |    1    |    0   |    0   |    1   |

We found that the true problem is from 
```bash
export NCCL_NET_GDR_LEVEL=PHB
```
Unfortunately, this environment is very key to NCCL performance. 

Below is the busbw for 2 nodes (4 processes per node), message size 1073741824. 

| bsbw       |   (0)    |  (0-b)  |  (0-c)  | (0-bc) |   (2)  |
| ---------- |  ------- | ------- | ------- | ------ | ------ |
|  allreduce |  36.07   |   9.61  |   9.89  | 15.67  | 12.79  |
|  alltoall  |  8.56    |   7.88  |   7.67  |  7.12  |  7.66  |
| allgather  |  35.76   |   9.52  |   14.14 |  14.46 | 15.62  |

Therefore, with pytorch, the best env setup is. However, it would be best to figure our why ```NCCL_NET_GDR_LEVEL=PHB``` causing the issue. 

```bash
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
```


(0) Full environment setup identified through NCCL-tests
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

(1) Remove FI* envs
```bash
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
```

(2) Remove other NCCL envs and keep FI
```bash
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
```

(3) Only aws-ofi-nccl
```bash
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
```
