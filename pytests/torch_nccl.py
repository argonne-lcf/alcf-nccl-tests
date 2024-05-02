print("imported all the libraries")
import torch.nn.parallel
import os
import socket
import datetime
t1 = datetime.datetime.now()
import torch.distributed as dist
t2 = datetime.datetime.now()
elapsed = (t2 - t1).total_seconds()

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
master_port              = 2345
if (rank==0):
   master_addr=socket.gethostname()
   print(master_addr)
   assert(master_addr.find(os.environ["MASTER_ADDR"])!=-1)

os.environ["MASTER_PORT"]   = str(master_port)

t3 = datetime.datetime.now()
dist.init_process_group(backend = "nccl", init_method = 'env://', world_size = world_size, rank = rank, timeout = datetime.timedelta(seconds=120))
t4 = datetime.datetime.now()
elapsed = (t4 - t3).total_seconds()
print(f"F-profiling complted at t4 {t4} (import hvd init - time : {elapsed:.5f})")
dist_my_rank        = dist.get_rank()
dist_world_size     = dist.get_world_size()
print("dist_my_rank = %d  dist_world_size = %d" % (dist_my_rank, dist_world_size))

def get_default_device():
   return torch.device(f"cuda:{local_rank}")

device  = get_default_device()
print(f"[{dist_my_rank}-{local_rank}] {device}")

for _ in range(2):
    x = torch.ones([1024, 1024]).to(device, non_blocking=True)
    # print(x)
    t5 = datetime.datetime.now() 
    dist.all_reduce(x,op=dist.ReduceOp.SUM)  # Added Extra op
    t6 = datetime.datetime.now()
    elapsed = (t6 - t5).total_seconds() 
    print(f"[{dist_my_rank}] Python: Elapsed time in each iter for all_reduce : {elapsed:.5f})")
    
