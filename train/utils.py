"""
训练工具函数集合
"""
import os
import sys
__package__ = "train"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr, warmup_steps=0):
    """
    学习率调度器：Warmup + Cosine Decay
    - warmup_steps: 线性warmup的步数
    - 之后使用cosine decay衰减
    """
    if current_step < warmup_steps:
        # 线性warmup: 从 0 增长到 lr
        return lr * (current_step / warmup_steps)
    else:
        # Cosine decay: 从 lr 衰减到 0.1*lr
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def init_distributed_mode():
    """
    初始化分布式训练环境（多 GPU DDP）。
    若未设置 RANK（如直接 python 跑单卡），则按单机模式处理并返回 0。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP：单进程，使用默认设备（如 cuda:0）

    dist.init_process_group(backend="nccl")  # NCCL 用于多 GPU 通信
    local_rank = int(os.environ["LOCAL_RANK"])  # 当前进程在本机上的 GPU 编号
    torch.cuda.set_device(local_rank)  # 绑定当前进程到对应 GPU
    return local_rank


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
