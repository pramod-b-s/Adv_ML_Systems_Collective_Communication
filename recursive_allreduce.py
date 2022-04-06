from multiprocessing import Process
import argparse
import torch
import logging
import time

from torch import distributed as dist
""" Implementation of a ring-reduce with addition. """


def reduceScatter(x, left, right):
        if (left == right): return
        size = right - left + 1
        mid = (left + right)//2
        if dist.get_rank() <= mid:
            partner = dist.get_rank() + size//2
        else:
            partner = dist.get_rank() - size//2

        if dist.get_rank() <= mid:
            send_req = dist.isend(x[mid:right + 1], partner)
            tmp = torch.zeros_like(x[left:mid + 1])
            dist.recv(tmp, partner)
            x[left:mid + 1] = x[left:mid + 1] + tmp
        else:
            send_req = dist.isend(x[left:mid + 1], partner)
            tmp = torch.zeros_like(x[mid:right + 1])
            dist.recv(tmp, partner)
            x[mid:right + 1] = x[mid:right + 1] + tmp

        send_req.wait()

        if dist.get_rank() <= mid:
            reduceScatter(x, left, mid)
        else:
            reduceScatter(x, mid+1, right)

def allGather(x, left, right):
        if (left == right): return
        size = right - left + 1
        mid = (left + right)//2
        if dist.get_rank() <= mid:
            partner = dist.get_rank() + size//2
        else:
            partner = dist.get_rank() - size//2

        if dist.get_rank() <= mid:
            allGather(x, left, mid)
        else:
            allGather(x, mid+1, right)

        if dist.get_rank() <= mid:
            send_req = dist.isend(x[left:mid + 1], partner)
            dist.recv(x[mid: right + 1], partner)
        else:
            send_req = dist.isend(x[mid:right + 1], partner)
            dist.recv(x[left: mid + 1], partner)
        send_req.wait()



def run_allreduce(rank, size):

    data = torch.rand(1, 1024 * 1024 * (10  // 4), dtype=torch.float32)
    recv = torch.zeros_like(data)
    tot_time = 0
    print("Original data", data)
    st = time.time()
    reduceScatter(data, 0, size-1)
    allGather(data, 0, size-1)
    en = time.time()
    tot_time += (en - st)
    print("All Reduce output", data)

    avg_time = tot_time/1
    print("Average time ", avg_time)

def init_process(rank, size, fn, world_size, backend='gloo'):
   """ Initialize the distributed environment. """

   dist.init_process_group(backend="gloo", init_method="tcp://10.10.1.1:16913", rank=rank, world_size=world_size)
   fn(rank, world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip", "-m", required=True, type=str)
    parser.add_argument("--num-nodes", "-n", required=True, type=int)
    parser.add_argument("--rank", "-r", required=True, type=int)
    parser.add_argument("--size", "-s", required=True, type=int)
    args = parser.parse_args()
    init_process(
                 rank=args.rank,
                 size = args.size,
                 fn = run_allreduce,
                 world_size=args.num_nodes)
