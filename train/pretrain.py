"""
SpongeBob 预训练脚本（支持多卡 DDP）
与 pretrain_without_ddp.py 的差异已用 [DDP] 标出：主要为分布式初始化、Sampler、DDP 包模型、
主进程判断（保存/评测）、总步数按 world_size 分片、训练循环用 train_sampler、结束时 destroy_process_group。
"""
import os
import sys

# 禁用 tokenizers 并行警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

__package__ = "train"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist  # [DDP] 多进程/多 GPU 通信；without_ddp 无此 import
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel  # [DDP] DDP 包模型；without_ddp 无
from torch.utils.data import DataLoader, DistributedSampler  # [DDP] 多卡用 DistributedSampler；without_ddp 仅 DataLoader
from model.config import SpongeBobConfig
from model.model_spongebob_pro import SpongeBobForCausalLM
from dataset.pretrain_dataset import PretrainDataset
from utils import get_lr, Logger, is_main_process, init_distributed_mode, SkipBatchSampler  # [DDP] is_main_process/init_distributed_mode 仅 DDP 用；without_ddp 无
from benchmark.pretrain.evaluator import run_benchmark

_BENCH_PRETRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark", "pretrain")

warnings.filterwarnings('ignore')

###auto_cast用法
# from torch.cuda.amp import GradScaler, autocast

# scaler = GradScaler() # 初始化

# for data, target in data_loader:
#     optimizer.zero_grad()

#     with autocast(): # 开启半精度前向传播
#         output = model(data)
#         loss = loss_fn(output, target)

#     # 1. 放大 Loss 并计算梯度
#     scaler.scale(loss).backward()

#     # 2. scaler.step() 会先取消缩放梯度 (unscale)
#     # 如果梯度没溢出，则调用 optimizer.step() 更新参数
#     # 如果溢出了，则跳过这一步
#     scaler.step(optimizer)

#     # 3. 更新缩放因子（根据是否有溢出来决定下次放大多少）
#     scaler.update()


def train_epoch(epoch, loader, iters, start_step=0, swanlab=None, total_steps=None, warmup_steps=None, full_save_dir=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        current_step = epoch * iters + step
        lr = get_lr(current_step, total_steps, args.learning_rate, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        global_step = epoch * iters + step
        
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if swanlab: swanlab.log({"loss": current_loss, "learning_rate": current_lr, "eta_time": eta_min}, step=global_step)
        
        # 保存 checkpoint [DDP] 仅主进程写盘；without_ddp 无 is_main_process() 判断
        if (global_step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            ckp_dir = f'{full_save_dir}/global_step_{global_step}'
            os.makedirs(ckp_dir, exist_ok=True)
            # [DDP] DDP 下取 .module 才是真实模型；without_ddp 仅 getattr(model, '_orig_mod', model)
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
            
            torch.save(state_dict, f'{ckp_dir}/{args.save_weight}_{lm_config.hidden_size}.pth')
            torch.save({
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'step': step,
                'global_step': global_step,
                'swanlab_id': getattr(swanlab, 'id', None) if swanlab else None
            }, f'{ckp_dir}/resume.pth')
            
            Logger(f'Saved checkpoint: {ckp_dir}')
            model.train()
        
        # Benchmark 评测 [DDP] 仅主进程跑；without_ddp 无 is_main_process() 判断
        if args.eval_bench == 1 and tokenizer is not None and global_step % args.eval_interval == 0 and is_main_process():
            model.eval()
            c3_path = os.path.join(_BENCH_PRETRAIN_DIR, "clue_c3_eval_500.jsonl")
            xcopa_path = os.path.join(_BENCH_PRETRAIN_DIR, "xcopa_zh_merged.jsonl")
            eval_results = run_benchmark(model, tokenizer, c3_path, xcopa_path)
            if swanlab_run:
                swanlab_run.log(eval_results, step=global_step)
            Logger(f'Benchmark results: {eval_results}')
            model.train()

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpongeBob Pretraining")
    parser.add_argument("--save_dir", type=str, default="../pretrain_out/exp_mini", help="模型保存根目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=3000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="序列长度")
    parser.add_argument("--data_path", type=str, default="", help="预处理后的.bin文件路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_swanlab", type=int, default=1, choices=[0, 1], help="是否使用swanlab（0=否，1=是）")
    parser.add_argument("--swanlab_project", type=str, default="SpongeBob-Pretrain", help="swanlab项目名")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--eval_bench", default=1, type=int, choices=[0, 1], help="是否评测benchmark（0=否，1=是）")
    parser.add_argument("--eval_interval", type=int, default=1000, help="评测间隔步数")
    args = parser.parse_args()

    # ========== 1. [DDP] 初始化分布式环境 ==========
    # without_ddp 无此步骤，直接进入配置目录
    local_rank = init_distributed_mode()  # 多卡时初始化进程组并返回本卡 GPU 号
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"  # DDP 时每进程用不同 GPU

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    lm_config = SpongeBobConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    
    # 生成 run_name（用于后续创建子目录）
    run_name = f"h{args.hidden_size}_l{args.num_hidden_layers}_bs{args.batch_size}_lr{args.learning_rate}"
    full_save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(full_save_dir, exist_ok=True)
    
    # 从最新的 checkpoint 恢复
    ckp_data = None
    if args.from_resume == 1:
        ckp_dirs = [d for d in os.listdir(full_save_dir) if d.startswith('global_step_')]
        if ckp_dirs:
            latest_ckp = max(ckp_dirs, key=lambda x: int(x.split('_')[-1]))
            resume_path = f'{full_save_dir}/{latest_ckp}/resume.pth'
            if os.path.exists(resume_path):
                ckp_data = torch.load(resume_path, map_location='cpu')
                Logger(f'Found checkpoint: {full_save_dir}/{latest_ckp}')
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    # ========== 4. 配置swanlab ==========
    swanlab_run = None
    if args.use_swanlab and is_main_process():  # [DDP] 仅主进程上报；without_ddp 无 is_main_process()
        import swanlab
        swanlab.login(api_key="4jqfbuJs9zDRcLAMPoDQv")
        
        swanlab_id = ckp_data.get('swanlab_id') if ckp_data else None
        
        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=run_name,
            id=swanlab_id,
            resume=True,
            config=vars(args)
        )
        Logger(f'SwanLab initialized: {run_name}')
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 创建模型
    if args.from_weight != 'none' and os.path.exists(args.from_weight):
        Logger(f'Loading model from {args.from_weight}')
        model = SpongeBobForCausalLM.from_pretrained(args.from_weight)
    else:
        Logger(f'Creating new model: hidden_size={args.hidden_size}, num_layers={args.num_hidden_layers}')
        model = SpongeBobForCausalLM(lm_config)
    
    model = model.to(args.device)
    Logger(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    # 加载 tokenizer（用于 benchmark 评测）
    if args.eval_bench == 1:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('')
        Logger('Tokenizer loaded for benchmark evaluation')
    else:
        tokenizer = None
    
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    # 数据集（加载预处理好的.bin文件）
    Logger('Loading dataset...')
    train_ds = PretrainDataset(args.data_path, seq_len=args.max_seq_len)
    # [DDP] 多卡用 DistributedSampler 分片；without_ddp 无 train_sampler，后面用 indices
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    Logger('Dataset ready')
    
    # 优化器
    Logger('Initializing optimizer...')
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    Logger('Optimizer ready')
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        Logger('Loading checkpoint...')
        if args.use_compile == 1:
            raw_model = getattr(model, '_orig_mod', model)
            raw_model.load_state_dict(ckp_data['model'])
        else:
            model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'Checkpoint loaded: epoch={start_epoch}, step={start_step}')
    
    # ========== 7. [DDP] DDP 包模型 ==========
    # without_ddp 无此整段，不包 DDP
    if dist.is_initialized():
        Logger('Wrapping model with DDP...')
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
        Logger('DDP ready')

    # ========== 8. 计算总步数 ==========
    # [DDP] 多卡时除以 world_size；without_ddp 为 len(train_ds) // args.batch_size
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    steps_per_epoch = len(train_ds) // (args.batch_size * world_size)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * 0.03)  # 3% warmup
    Logger(f'World size: {world_size}, Steps per epoch: {steps_per_epoch}')
    Logger(f'Total training steps: {total_steps}, Warmup steps: {warmup_steps} (3%)')
    
    # ========== 8.5. 初始评测 (step 0) ==========
    # [DDP] 仅主进程评测；without_ddp 无 is_main_process()
    if args.eval_bench == 1 and tokenizer is not None and is_main_process() and start_epoch == 0 and start_step == 0:
        Logger('Running initial benchmark evaluation (step 0)...')
        model.eval()
        c3_path = os.path.join(_BENCH_PRETRAIN_DIR, "clue_c3_eval_500.jsonl")
        xcopa_path = os.path.join(_BENCH_PRETRAIN_DIR, "xcopa_zh_merged.jsonl")
        eval_results = run_benchmark(model, tokenizer, c3_path, xcopa_path)
        if swanlab_run:
            swanlab_run.log(eval_results, step=0)
        Logger(f'Initial benchmark results (step 0): {eval_results}')
        model.train()
    
    # ========== 9. 开始训练 ==========
    Logger(f'Starting training: {args.epochs} epochs, batch_size={args.batch_size}')
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)  # [DDP] 多卡时打乱各卡分片；without_ddp 无此行
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # [DDP] 多卡用 train_sampler，单卡用 indices；without_ddp 仅 SkipBatchSampler(indices, ...)
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        Logger(f'Creating DataLoader for epoch {epoch+1}...')
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        Logger(f'DataLoader ready, starting epoch {epoch+1}...')
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, swanlab_run, total_steps, warmup_steps, full_save_dir)
        else:
            train_epoch(epoch, loader, len(loader), 0, swanlab_run, total_steps, warmup_steps, full_save_dir)
    
    # ========== 10. [DDP] 清理分布式进程组 ==========
    # without_ddp 无此步骤，仅 Logger('Training done.')
    if dist.is_initialized(): dist.destroy_process_group()