import argparse
import os
import sys
import tempfile
from time import sleep
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import datetime


from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def train():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
    ranks_list = [[0,1], [0,2], [2,3], [1,3]]
    for ranks in ranks_list:
        if rank in ranks:
            group = torch.distributed.new_group(ranks, backend="nccl")
            print("finish init group for rank" + str(rank) + ' in ranks ' + str(ranks))

def run():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    print(init_method)
    dist.init_process_group(backend="nccl", 
                init_method=init_method,
                world_size=int(env_dict["WORLD_SIZE"]), rank=int(env_dict["RANK"]),
                timeout=datetime.timedelta(seconds=30))
    print("finish init_process_group")
    torch.distributed.barrier()
    print("finish barrier")
    train()
    print("finish training")
    dist.destroy_process_group()



if __name__ == "__main__":
    run()