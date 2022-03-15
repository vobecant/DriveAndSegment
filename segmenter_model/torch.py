import os
import torch

"""
GPU wrappers
"""

use_gpu = False
gpu_id = 0
device = None

distributed = False
dist_rank = 0
world_size = 1


def set_gpu_mode(mode, pbs=False):
    global use_gpu
    global device
    global gpu_id
    global distributed
    global dist_rank
    global world_size
    if pbs:
        gpu_id = int(os.environ.get("MPI_LOCALRANKID", 0))
        dist_rank = int(os.environ.get("PMI_RANK", 0))
        world_size = int(os.environ.get("PMI_SIZE", 1))
    else:
        gpu_id = int(os.environ.get("SLURM_LOCALID", 0))
        dist_rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))

    distributed = world_size > 1
    use_gpu = mode
    print('gpu_id: {}, dist_rank: {}, world_size: {}, distributed: {}'.format(gpu_id, dist_rank, world_size,
                                                                              distributed))
    device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")
    torch.backends.cudnn.benchmark = True
