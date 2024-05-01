from mpi4py import MPI
import torch.nn.parallel
import os
import socket
import datetime
t1 = datetime.datetime.now()
import torch.distributed as dist
t2 = datetime.datetime.now()
elapsed = (t2 - t1).total_seconds()
print(f"F-profiling complted at t2 {t2} (import torch dist - time : {elapsed:.5f})")


MPI.COMM_WORLD.Barrier()

os.environ['RANK']          = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE']    = str(os.environ.get('PMI_SIZE', 1))
mpi_world_size              = MPI.COMM_WORLD.Get_size()
mpi_my_rank                 = MPI.COMM_WORLD.Get_rank()

if mpi_my_rank == 0:
   master_addr              = socket.gethostname()
   sock                     = socket.socket()
   sock.bind(('',0))
   # master_port  = sock.getsockname()[1] 
   master_port              = 2345
else:
   master_addr              = None
   master_port              = None

master_addr                 = MPI.COMM_WORLD.bcast(master_addr, root=0)
master_port                 = MPI.COMM_WORLD.bcast(master_port, root=0)
os.environ["MASTER_ADDR"]   = master_addr
os.environ["MASTER_PORT"]   = str(master_port)

t3 = datetime.datetime.now()
dist.init_process_group(backend = "nccl", init_method = 'env://', world_size = mpi_world_size, rank = mpi_my_rank, timeout = datetime.timedelta(seconds=120))
t4 = datetime.datetime.now()
elapsed = (t4 - t3).total_seconds()
print(f"F-profiling complted at t4 {t4} (import hvd init - time : {elapsed:.5f})")


dist_my_rank        = dist.get_rank()
dist_world_size     = dist.get_world_size()
print("dist_my_rank = %d  dist_world_size = %d" % (dist_my_rank, dist_world_size))

def get_default_device():
   return torch.device(f"cuda:{dist_my_rank%12}")

device  = get_default_device()
print(device)
# torch.xpu.set_device(hvd_local_rank if torch.xpu.device_count() > 1 else 0)

for _ in range(50):
    x = torch.ones([1024, 1024]).to(device, non_blocking=True)
    # print(x)
    t5 = datetime.datetime.now() 
    dist.all_reduce(x,op=dist.ReduceOp.SUM)  # Added Extra op
    t6 = datetime.datetime.now()
    elapsed = (t6 - t5).total_seconds() 
    print(f"Python: Elapsed time in each iter for all_reduce : {elapsed:.5f})")
    
