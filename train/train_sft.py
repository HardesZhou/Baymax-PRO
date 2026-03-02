"""
SFT 训练脚本：由 pretrain.py 复制后做少量修改得到，便于与预训练对比讲解。
改动处均用 # [SFT] 标出。

主要差异：
1. 数据集：PretrainDataset(.bin) → SFTDataset(jsonl 对话数据，只算 assistant loss)
2. Tokenizer：SFT 需提前加载（Dataset 需要），pretrain 仅评测时加载
3. 模型加载：pretrain 用 from_pretrained，SFT 用 load_state_dict 加载 .pth
4. 评测方式：pretrain 用 C3/XCOPA benchmark，SFT 用 mini_bench + DeepSeek Judge 生成式评测
5. 新增参数：tokenizer_path, judge_api_key, judge_model
"""
import os
import sys

# 禁用 tokenizers 并行警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

__package__ = "train"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import time
import warnings
import torch
import torch.distributed as dist  # 多进程/多 GPU 通信
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel  # DDP：多卡同步梯度
from torch.utils.data import DataLoader, DistributedSampler  # 每卡分片数据，不重复
from transformers import AutoTokenizer  # [SFT] SFT 需一开始就加载 tokenizer（给 SFTDataset 用），pretrain 仅在 eval_bench=1 时加载
from model.config import SpongeBobConfig
from model.model_spongebob_pro import SpongeBobForCausalLM
from dataset.sft_dataset import SFTDataset  # [SFT] pretrain 用 PretrainDataset(.bin)，SFT 用 SFTDataset(jsonl + 只算 assistant loss)
from utils import get_lr, Logger, is_main_process, init_distributed_mode, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, swanlab=None, total_steps=None, warmup_steps=None, full_save_dir=None):
    """与 pretrain 相同（含 checkpoint 保存）。"""
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

        # 保存 checkpoint（仅主进程写盘，逻辑与 pretrain.py 完全相同）
        if (global_step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            ckp_dir = f'{full_save_dir}/global_step_{global_step}'
            os.makedirs(ckp_dir, exist_ok=True)
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

        # [SFT] mini_bench 评测：每到 eval_interval 用当前模型推理，异步 DeepSeek Judge
        # pretrain 用 run_benchmark (C3/XCOPA)，SFT 用 run_inference + run_judge_async (生成式评测)
        if args.enable_eval and getattr(args, "eval_interval", 0) > 0 and global_step % args.eval_interval == 0 and is_main_process():
            from benchmark.mini_bench.eval import run_inference, run_judge_async
            raw = model.module if isinstance(model, DistributedDataParallel) else model
            raw = getattr(raw, "_orig_mod", raw)
            model.eval()
            pairs = run_inference(raw, tokenizer, device=args.device, num_samples=3)
            model.train()
            
            valid_dir = os.path.join(full_save_dir, "valid_samples")
            valid_file = os.path.join(valid_dir, f"global_step_{global_step}.jsonl")
            run_judge_async(pairs, args.judge_api_key, args.judge_model,
                           output_file=valid_file,
                           swanlab_log_fn=swanlab_run.log if swanlab_run else None,
                           global_step=global_step)
            Logger(f'[eval] step={global_step} 推理完成，Judge 后台运行中...')

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpongeBob SFT Training")
    # [SFT] 以下参数默认值与 pretrain.py 不同：save_dir, save_weight, epochs, data_path, from_weight, swanlab_project
    # [SFT] 新增参数：tokenizer_path, ollama_url, ollama_model
    parser.add_argument("--save_dir", type=str, default="../out_sft/exp_1", help="模型保存目录")
    parser.add_argument('--save_weight', default='sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数（SFT 推荐 2-3 epoch，过多会过拟合）")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="初始学习率（SFT 推荐 1e-5 ~ 1e-4，从预训练继续可用 5e-5）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=12, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="序列长度")
    parser.add_argument("--data_path", type=str, default="", help="SFT 数据 jsonl 路径")
    parser.add_argument("--tokenizer_path", type=str, default="../tokenizer_15k", help="tokenizer 路径")  # [SFT] 新增：SFTDataset 需要 tokenizer
    parser.add_argument('--from_weight', default='', type=str, help="基于哪个权重训练，为 none 则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_swanlab", type=int, default=1, choices=[0, 1], help="是否使用 swanlab（0=否，1=是）")
    parser.add_argument("--swanlab_project", type=str, default="SpongeBob-SFT", help="swanlab 项目名")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="是否使用 torch.compile 加速（0=否，1=是）")
    # [SFT] 新增：mini_bench 评测参数（使用 DeepSeek API）
    parser.add_argument("--enable_eval", type=int, default=0, choices=[0, 1], help="是否启用评估（0=关闭，1=开启）")
    parser.add_argument("--eval_interval", type=int, default=1000, help="每隔多少 step 跑 mini_bench（0=关闭），用当前模型推理+DeepSeek Judge 打分")
    parser.add_argument("--judge_api_key", type=str, default='', help="Judge API Key（可直接传入或从环境变量 DEEPSEEK_API_KEY 读取）")
    parser.add_argument("--judge_model", type=str, default="deepseek-chat", help="Judge 模型名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"

    # ========== 2. 配置目录、模型参数、检查 ckp ==========
    lm_config = SpongeBobConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)

    # 与 pretrain 一致：用 run_name 做子目录
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

    # ========== 4. 配置 swanlab ==========
    swanlab_run = None
    if args.use_swanlab and is_main_process():
        import swanlab
        swanlab.login(api_key="4jqfbuJs9zDRcLAMPoDQv")

        swanlab_id = ckp_data.get('swanlab_id') if ckp_data else None
        swanlab_run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=run_name,
            id=swanlab_id,
            config=vars(args)
        )
        Logger(f'SwanLab initialized: {run_name}')

    # ========== 5. 定义模型、数据、优化器 ==========
    # [SFT] 先加载 tokenizer（SFT 数据集需要，pretrain 仅在 eval_bench=1 时加载）
    Logger(f'Loading tokenizer from {args.tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 创建/加载模型
    # [SFT] 用 load_state_dict 加载 .pth 权重文件，pretrain 用 from_pretrained 加载模型目录
    if args.from_weight != 'none' and os.path.exists(args.from_weight):
        Logger(f'Loading model from {args.from_weight}')
        model = SpongeBobForCausalLM(lm_config)
        model.load_state_dict(torch.load(args.from_weight, map_location='cpu'), strict=False)
    else:
        Logger(f'Creating new model: hidden_size={args.hidden_size}, num_layers={args.num_hidden_layers}')
        model = SpongeBobForCausalLM(lm_config)

    model = model.to(args.device)
    Logger(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # [SFT] 数据集：SFTDataset 读取 jsonl 对话数据，只计算 assistant 部分的 loss
    # pretrain 用 PretrainDataset 读取预处理好的 .bin 文件，计算全部 token 的 loss
    Logger('Loading dataset...')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    Logger('Dataset ready')

    # 优化器
    Logger('Initializing optimizer...')
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    Logger('Optimizer ready')

    # ========== 6. 从 ckp 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        Logger('Loading checkpoint...')
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'Checkpoint loaded: epoch={start_epoch}, step={start_step}')

    # ========== 7. DDP 包模型 ==========
    if dist.is_initialized():
        Logger('Wrapping model with DDP...')
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
        Logger('DDP ready')

    # ========== 8. 计算总步数（考虑 DDP 分片）==========
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    steps_per_epoch = len(train_ds) // (args.batch_size * world_size)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * 0.1)  # 10% warmup（SFT 推荐更长 warmup）
    Logger(f'World size: {world_size}, Steps per epoch: {steps_per_epoch}')
    Logger(f'Total training steps: {total_steps}, Warmup steps: {warmup_steps} (3%)')

    # ========== 8.5. [SFT] 初始评测 (step 0) ==========
    # 仅主进程评测，且仅在从头训练时（start_epoch=0, start_step=0）执行
    if args.enable_eval and getattr(args, "eval_interval", 0) > 0 and is_main_process() and start_epoch == 0 and start_step == 0:
        Logger('Running initial mini_bench evaluation (step 0)...')
        from benchmark.mini_bench.eval import run_inference, run_judge_async
        raw = model.module if isinstance(model, DistributedDataParallel) else model
        raw = getattr(raw, "_orig_mod", raw)
        model.eval()
        pairs = run_inference(raw, tokenizer, device=args.device, num_samples=3)
        model.train()
        
        valid_dir = os.path.join(full_save_dir, "valid_samples")
        valid_file = os.path.join(valid_dir, f"global_step_0.jsonl")
        run_judge_async(pairs, args.judge_api_key, args.judge_model,
                       output_file=valid_file,
                       swanlab_log_fn=swanlab_run.log if swanlab_run else None,
                       global_step=0)
        Logger('[eval] step=0 初始评测完成，Judge 后台运行中...')

    # ========== 9. 开始训练 ==========
    Logger(f'Starting training: {args.epochs} epochs, batch_size={args.batch_size}')
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        Logger(f'Creating DataLoader for epoch {epoch+1}...')
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        Logger(f'DataLoader ready, starting epoch {epoch+1}...')
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, swanlab_run, total_steps, warmup_steps, full_save_dir)
        else:
            train_epoch(epoch, loader, len(loader), 0, swanlab_run, total_steps, warmup_steps, full_save_dir)

    # ========== 10. 清理 ==========
    if dist.is_initialized(): dist.destroy_process_group()
