"""
单卡预训练脚本（无 DDP）
用法: python pretrain_without_ddp.py [args]
"""
import os
import sys

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

__package__ = "train"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader
from model.config import SpongeBobConfig
from model.model_spongebob_pro import SpongeBobForCausalLM
from dataset.pretrain_dataset import PretrainDataset
from utils import get_lr, Logger, SkipBatchSampler
from benchmark.evaluator import run_benchmark

warnings.filterwarnings('ignore')


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
            if swanlab:
                swanlab.log({"loss": current_loss, "learning_rate": current_lr, "eta_time": eta_min}, step=global_step)

        # 保存 checkpoint
        if global_step % args.save_interval == 0 or step == iters - 1:
            model.eval()
            ckp_dir = f'{full_save_dir}/global_step_{global_step}'
            os.makedirs(ckp_dir, exist_ok=True)
            raw_model = getattr(model, '_orig_mod', model)
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

        # Benchmark 评测
        if args.eval_bench == 1 and tokenizer is not None and global_step % args.eval_interval == 0:
            model.eval()
            c3_path = ''
            xcopa_path = ''
            eval_results = run_benchmark(model, tokenizer, c3_path, xcopa_path)
            if swanlab_run:
                swanlab_run.log(eval_results, step=global_step)
            Logger(f'Benchmark results: {eval_results}')
            model.train()

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpongeBob Pretraining (Single GPU)")
    parser.add_argument("--save_dir", type=str, default="../pretrain_out", help="模型保存根目录")
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
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
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
    parser.add_argument("--eval_interval", type=int, default=100, help="评测间隔步数")
    args = parser.parse_args()

    # ========== 1. 配置目录、模型参数、检查 ckp ==========
    lm_config = SpongeBobConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    run_name = f"h{args.hidden_size}_l{args.num_hidden_layers}_bs{args.batch_size}_lr{args.learning_rate}"
    full_save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(full_save_dir, exist_ok=True)

    ckp_data = None
    if args.from_resume == 1:
        ckp_dirs = [d for d in os.listdir(full_save_dir) if d.startswith('global_step_')]
        if ckp_dirs:
            latest_ckp = max(ckp_dirs, key=lambda x: int(x.split('_')[-1]))
            resume_path = f'{full_save_dir}/{latest_ckp}/resume.pth'
            if os.path.exists(resume_path):
                ckp_data = torch.load(resume_path, map_location='cpu')
                Logger(f'Found checkpoint: {full_save_dir}/{latest_ckp}')

    # ========== 3. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)

    # ========== 4. SwanLab ==========
    swanlab_run = None
    if args.use_swanlab:
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

    # ========== 5. 模型、数据、优化器 ==========
    if args.from_weight != 'none' and os.path.exists(args.from_weight):
        Logger(f'Loading model from {args.from_weight}')
        model = SpongeBobForCausalLM.from_pretrained(args.from_weight)
    else:
        Logger(f'Creating new model: hidden_size={args.hidden_size}, num_layers={args.num_hidden_layers}')
        model = SpongeBobForCausalLM(lm_config)
    model = model.to(args.device)
    Logger(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    if args.eval_bench == 1:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('')
        Logger('Tokenizer loaded for benchmark evaluation')
    else:
        tokenizer = None

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    Logger('Loading dataset...')
    train_ds = PretrainDataset(args.data_path, seq_len=args.max_seq_len)
    Logger('Dataset ready')

    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    Logger('Optimizer ready')

    # ========== 6. 从 ckp 恢复 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        Logger('Loading checkpoint...')
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f'Checkpoint loaded: epoch={start_epoch}, step={start_step}')

    # ========== 7. 总步数（单卡）==========
    steps_per_epoch = len(train_ds) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * 0.03)
    Logger(f'Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}, Warmup: {warmup_steps}')

    # ========== 8. 初始评测 (step 0) ==========
    if args.eval_bench == 1 and tokenizer is not None and start_epoch == 0 and start_step == 0:
        Logger('Running initial benchmark evaluation (step 0)...')
        model.eval()
        c3_path = ''
        xcopa_path = ''
        eval_results = run_benchmark(model, tokenizer, c3_path, xcopa_path)
        if swanlab_run:
            swanlab_run.log(eval_results, step=0)
        Logger(f'Initial benchmark results (step 0): {eval_results}')
        model.train()

    # ========== 9. 训练循环 ==========
    Logger(f'Starting training: {args.epochs} epochs, batch_size={args.batch_size} (single GPU)')
    for epoch in range(start_epoch, args.epochs):
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, swanlab_run, total_steps, warmup_steps, full_save_dir)
        else:
            train_epoch(epoch, loader, len(loader), 0, swanlab_run, total_steps, warmup_steps, full_save_dir)

    Logger('Training done.')
