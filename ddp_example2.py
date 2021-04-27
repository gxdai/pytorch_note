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


def demo_basic(rank, world_size, input, target):
    print(f"Running basic ddp example on rank {rank}.")

    setup(rank, world_size)
    model = Toy().to(rank)
    ddp_model = ddp(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    for idx in range(50000):
        optimizer.zero_grad()
        outputs = ddp_model(input.to(rank))
        target = target.to(rank)
        loss_fn(outputs, target).backward()
        optimizer.step()

        if idx % 100 == 0 and idx > 0:
            print(f"rank {rank}: loss = {loss_fn(outputs, target).item()}")

    if rank == 0:
        torch.save(ddp_model.state_dict(), "ddp_model.pt")

    # block other process to load the checkpoint before process-0 to save the
    # checkpoint.

    dist.barrier()

    map_location = {"cuda:0": f"cuda:{rank}"}

    ddp_model.load_state_dict(
        torch.load("ddp_model.pt",
        map_location=map_location
        )
    )


    cleanup()

    if rank == 0:
        os.remove("ddp_model.pt")


def run(fnc, world_size, input, target):
    mp.spawn(
        fnc,
        nprocs=world_size,
        args=(world_size,input, target)
    )


if __name__ == "__main__":
    input = torch.randn(20, 10)
    target = torch.randn(20, 5)
    run(demo_basic, world_size=8, input=input, target=target)


