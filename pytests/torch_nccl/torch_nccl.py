import datetime
t1 = datetime.datetime.now()

from mpi4py import MPI
comm = MPI.COMM_WORLD

import torch.nn.parallel
import os
import socket
import torch.distributed as dist
import numpy as np

import torch
t2 = datetime.datetime.now()
import_time = (t2 - t1).total_seconds()

def init_process_group():
   try:
      rank = int(os.environ['RANK'])
      world_size = int(os.environ['WORLD_SIZE'])
   except:
      rank = comm.rank
      world_size = comm.size
   try:
      local_rank = int(os.environ['LOCAL_RANK'])
   except:
      local_rank = rank%torch.cuda.device_count()
      os.environ['LOCAL_RANK'] = str(local_rank)
   master_port              = 2345
   os.environ["MASTER_PORT"]   = str(master_port)
   if "MASTER_ADDR" not in os.environ.keys():
      master_addr = socket.gethostname()
      comm.bcast(master_addr, root=0)
      os.environ["MASTER_ADDR"] = master_addr

   t3 = datetime.datetime.now()
   dist.init_process_group(backend = "nccl", init_method = 'env://', world_size = world_size, rank = rank, timeout = datetime.timedelta(seconds=30))
   t4 = datetime.datetime.now()
   elapsed = (t4 - t3).total_seconds()
   if rank==0:
      print(f"import time: {import_time:.8f}")
      print(f"torch init time : {elapsed:.8f}")
   return rank, local_rank, world_size

rank, local_rank, world_size = init_process_group()

dist_my_rank        = dist.get_rank()
dist_world_size     = dist.get_world_size()

if rank == 0:    
   print(f"{torch.__version__}")
   print(f"{torch.__file__}")
   print(f"{torch.cuda.nccl.version()}")

print("dist_my_rank = %d  dist_world_size = %d" % (dist_my_rank, dist_world_size))

def get_default_device():
   return torch.device(f"cuda:{local_rank}")

device  = get_default_device()
print(f"[{dist_my_rank}-{local_rank}] {device}")

#exit()

def print_rank_0(msg):
   if rank == 0:
      print(msg)

comm.barrier()
niters = 10

time_iters = np.zeros(niters)
for i in range(niters):
   x = torch.ones([1024, 1024]).to(device, non_blocking=True)
   # print_rank_0(x)
   t5 = datetime.datetime.now() 
   dist.broadcast(x, src=0, async_op=True)  # Added Extra op
   t6 = datetime.datetime.now()
   elapsed = (t6 - t5).total_seconds()
   time_iters[i] = elapsed
   print_rank_0(f"[{dist_my_rank}] Python: Elapsed time in each iter for broadcast: {elapsed:.8f})")
print_rank_0(f"Average time for broadcast (exclude first iter): {np.mean(time_iters[1:]):.8f} ")

comm.barrier()   
for i in range(niters):
   x = torch.ones([1024, 1024]).to(device, non_blocking=True)
   # print_rank_0(x)
   t5 = datetime.datetime.now() 
   dist.all_reduce(x,op=dist.ReduceOp.SUM)  # Added Extra op
   t6 = datetime.datetime.now()
   elapsed = (t6 - t5).total_seconds()
   time_iters[i] = elapsed   
   print_rank_0(f"[{dist_my_rank}] Python: Elapsed time in each iter for all_reduce : {elapsed:.8f})")
print_rank_0(f"Average time for all_reduce (exclude first iter): {np.mean(time_iters[1:]):.8f} ")

comm.barrier()



for i in range(niters):
   x = torch.ones([1024, 1024]).to(device, non_blocking=True)
   # print_rank_0(x)
   t5 = datetime.datetime.now() 
   dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)  # Added Extra op
   t6 = datetime.datetime.now()
   elapsed = (t6 - t5).total_seconds()
   time_iters[i] = elapsed   
   print_rank_0(f"[{dist_my_rank}] Python: Elapsed time in each iter for reduce : {elapsed:.8f})")
print_rank_0(f"Average time for reduce (exclude first iter): {np.mean(time_iters[1:]):.8f} ")      
comm.barrier()

for i in range(niters):
   x = torch.ones(4).to(device, non_blocking=True)
   y = [torch.zeros(4).to(device, non_blocking=True) for _ in range(world_size)]
   # print_rank_0(x)
   t5 = datetime.datetime.now() 
   dist.all_gather(y, x)
   t6 = datetime.datetime.now()
   elapsed = (t6 - t5).total_seconds()
   print_rank_0(f"[{dist_my_rank}] Python: Elapsed time in each iter for all_gather : {elapsed:.8f})")
print_rank_0(f"Average time for all_gather (exclude first iter): {np.mean(time_iters[1:]):.8f} ")      

comm.barrier()
for i in range(niters):
   x = torch.ones(world_size).to(device, non_blocking=True)
   y = torch.zeros(1).to(device, non_blocking=True)
   # print_rank_0(x)
   t5 = datetime.datetime.now() 
   dist.reduce_scatter_tensor(y, x, op=dist.ReduceOp.SUM)
   t6 = datetime.datetime.now()
   elapsed = (t6 - t5).total_seconds()
   time_iters[i] = elapsed
   print_rank_0(f"[{dist_my_rank}] Python: Elapsed time in each iter for reduce_scatter : {elapsed:.8f})")
print_rank_0(f"Average time for reduce_scatter (exclude first iter): {np.mean(time_iters[1:]):.8f} ")      
   
comm.barrier()
for i in range(niters):
   x_in = torch.ones(1024).to(device, non_blocking=True)
   x_out = torch.ones(1024).to(device, non_blocking=True)
   t5 = datetime.datetime.now()       
   dist.all_to_all_single(x_out, x_in)
   t6 = datetime.datetime.now()
   elapsed = (t6 - t5).total_seconds()   
   time_iters[i] = elapsed   
   print_rank_0(f"[{dist_my_rank}] Python: Elapsed time in each iter for all_to_all : {elapsed:.8f})")
print_rank_0(f"Average time for all_to_all (exclude first iter): {np.mean(time_iters[1:]):.8f} ")      
