from multiprocessing import Process
import argparse
import torch
import logging
import time
import csv

from torch import distributed as dist

""" Implementation of a ring-reduce with addition. """
def allreduce(send, recv):
       rank = dist.get_rank()
       size = dist.get_world_size()
       send_buff = send.clone()
       recv_buff = send.clone()
       accum = send.clone()
       left = ((rank - 1) + size) % size
       right = (rank + 1) % size
       for i in range(size - 1):
           if i % 2 == 0:
               # Send send_buff
               send_req = dist.isend(send, right)
               dist.recv(recv, left)
               accum[:] += recv[:]
           else:
               # Send recv_buff
               send_req = dist.isend(recv, right)
               dist.recv(send, left)
               accum[:] += send[:]
           send_req.wait()
       recv[:] = accum[:]


def run_allreduce(rank, size):
   data = torch.rand(size // 4, dtype=torch.float32)    
   recv = torch.zeros_like(data)
   print(data)
   print("starting all reduce")
   st = time.time()
   allreduce(send=data, recv=recv)
   en = time.time()
   f = open('meas.csv', 'a+')
   wrt = csv.writer(f)
   rw = []
   rw.append(en - st)
   rw.append(size)
   wrt.writerow(rw)
   print(recv)

def init_process(rank, size, fn, world_size, backend='gloo'):
   """ Initialize the distributed environment. """
   print("Hello in init_process")
   dist.init_process_group(backend="gloo", init_method="tcp://10.10.1.1:6588", rank=rank, world_size=world_size)
   print("Init done")
   fn(rank, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip", "-m", required=True, type=str)
    parser.add_argument("--num-nodes", "-n", required=True, type=int)
    parser.add_argument("--rank", "-r", required=True, type=int)
    parser.add_argument("--size", "-s", required=True, type=int)
    args = parser.parse_args()
    sz = args.size
    init_process(
                 rank=args.rank,
                 size = args.size,
                 fn = run_allreduce,
                 world_size=args.num_nodes) 


