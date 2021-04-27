import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as ddp



def setup(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # dist.init_process_group(
    #     "nccl",
    #     rank=rank,
    #     world_size=world_size,
    #     init_method="env://"
    # )


    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()


class Toy(nn.Module):

    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic ddp example on rank {rank}.")

    setup(rank, world_size)
    model = Toy().to(rank)
    ddp_model = ddp(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    print(f"rank {rank}: loss = {loss_fn(outputs, labels).item()}")

    cleanup()


def run(fnc, world_size):
    mp.spawn(
        fnc,
        nprocs=world_size,
        args=(world_size,)
    )


if __name__ == "__main__":
    run(demo_basic, world_size=8)


