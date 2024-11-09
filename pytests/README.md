# Torch NCCL issue with AWS plugin


This study is for investigating the issue that we run into when going beyond one nodes. This will cause hang or have the following error. 

```
x3200c0s19b1n0:191643:191866 [2] NCCL INFO comm 0x5590ec4762b0 rank 2 nranks 8 cudaDev 2 nvmlDev 2 busId 85000 commId 0xfa38415269cf0cf7 - Init COMPLETE
terminate called after throwing an instance of 'c10::DistBackendError'
  what():  [PG 0 Rank 1] Process group watchdog thread terminated with exception: [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1048576, NumelOut=1048576, Timeout(ms)=120000) ran for 120092 milliseconds before timing out.
Exception raised from checkTimeout at /soft/applications/conda/2024-04-29/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:565 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xa9 (0x15325c0b76f9 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x10584a1 (0x15325d20c4a1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x1d1 (0x15325d1e5b61 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::watchdogHandler() + 0x1e5 (0x15325d1e6035 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0xfd (0x15325d1e6e6d in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #5: <unknown function> + 0xd3e95 (0x153276642e95 in /soft/applications/conda/2024-04-29/mconda3/bin/../lib/libstdc++.so.6)
frame #6: <unknown function> + 0xa6ea (0x1532814896ea in /lib64/libpthread.so.0)
frame #7: clone + 0x41 (0x15328124950f in /lib64/libc.so.6)

Exception raised from ncclCommWatchdog at /soft/applications/conda/2024-04-29/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1418 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xa9 (0x15325c0b76f9 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x10584a1 (0x15325d20c4a1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #2: <unknown function> + 0xcf2d40 (0x15325cea6d40 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #3: <unknown function> + 0xd3e95 (0x153276642e95 in /soft/applications/conda/2024-04-29/mconda3/bin/../lib/libstdc++.so.6)
frame #4: <unknown function> + 0xa6ea (0x1532814896ea in /lib64/libpthread.so.0)
frame #5: clone + 0x41 (0x15328124950f in /lib64/libc.so.6)

[rank2]:[E ProcessGroupNCCL.cpp:1537] [PG 0 Rank 2] Timeout at NCCL work: 1, last enqueued NCCL work: 1, last completed NCCL work: 76280750235782912.
[rank2]:[E ProcessGroupNCCL.cpp:577] [Rank 2] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank2]:[E ProcessGroupNCCL.cpp:583] [Rank 2] To avoid data inconsistency, we are taking the entire process down.
[rank2]:[E ProcessGroupNCCL.cpp:1414] [PG 0 Rank 2] Process group watchdog thread terminated with exception: [Rank 2] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1048576, NumelOut=1048576, Timeout(ms)=120000) ran for 120094 milliseconds before timing out.
Exception raised from checkTimeout at /soft/applications/conda/2024-04-29/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:565 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xa9 (0x1547d6f756f9 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x10584a1 (0x1547d80ca4a1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x1d1 (0x1547d80a3b61 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::watchdogHandler() + 0x1e5 (0x1547d80a4035 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0xfd (0x1547d80a4e6d in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #5: <unknown function> + 0xd3e95 (0x1547f1442e95 in /soft/applications/conda/2024-04-29/mconda3/bin/../lib/libstdc++.so.6)
frame #6: <unknown function> + 0xa6ea (0x1547fc2c96ea in /lib64/libpthread.so.0)
frame #7: clone + 0x41 (0x1547fc08950f in /lib64/libc.so.6)

terminate called after throwing an instance of 'c10::DistBackendError'
  what():  [PG 0 Rank 2] Process group watchdog thread terminated with exception: [Rank 2] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1048576, NumelOut=1048576, Timeout(ms)=120000) ran for 120094 milliseconds before timing out.
Exception raised from checkTimeout at /soft/applications/conda/2024-04-29/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:565 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xa9 (0x1547d6f756f9 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x10584a1 (0x1547d80ca4a1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x1d1 (0x1547d80a3b61 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::watchdogHandler() + 0x1e5 (0x1547d80a4035 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #4: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0xfd (0x1547d80a4e6d in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #5: <unknown function> + 0xd3e95 (0x1547f1442e95 in /soft/applications/conda/2024-04-29/mconda3/bin/../lib/libstdc++.so.6)
frame #6: <unknown function> + 0xa6ea (0x1547fc2c96ea in /lib64/libpthread.so.0)
frame #7: clone + 0x41 (0x1547fc08950f in /lib64/libc.so.6)

Exception raised from ncclCommWatchdog at /soft/applications/conda/2024-04-29/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1418 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xa9 (0x1547d6f756f9 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x10584a1 (0x1547d80ca4a1 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #2: <unknown function> + 0xcf2d40 (0x1547d7d64d40 in /soft/applications/conda/2024-04-29/mconda3/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
frame #3: <unknown function> + 0xd3e95 (0x1547f1442e95 in /soft/applications/conda/2024-04-29/mconda3/bin/../lib/libstdc++.so.6)
frame #4: <unknown function> + 0xa6ea (0x1547fc2c96ea in /lib64/libpthread.so.0)
frame #5: clone + 0x41 (0x1547fc08950f in /lib64/libc.so.6)

./launcher.sh: line 14: 191644 Aborted                 (core dumped) $@
x3200c0s19b1n0.hsn.cm.sirius.alcf.anl.gov: rank 0 exited with code 134
x3200c0s19b1n0.hsn.cm.sirius.alcf.anl.gov: rank 1 died from signal 15
```

In order to identify the issue, we check different environement setups and see which one is causing the issue. 

## Results and Conclusion

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
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
```

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


## Appendix: Different Environment Settings
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
