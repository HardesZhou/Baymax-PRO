"""
GRPO (Group Relative Policy Optimization) 训练脚本

核心逻辑：
1. 格式检查：<think>\n...\n</think>\n... (严格匹配)
2. 格式错误 → reward=0
3. 格式正确 → DeepSeek Judge 评分 → reward=三指标均值
"""
import os
import sys
import json
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.config import SpongeBobConfig
from model.model_spongebob_pro import SpongeBobForCausalLM
from dataset.grpo_dataset import GRPODataset
from train.utils import Logger, is_main_process, init_distributed_mode, SkipBatchSampler

warnings.filterwarnings('ignore')


# ==================== 格式检查和奖励计算 ====================

def clean_special_tokens(text):
    """清理不需要的特殊 token（保留 <think> 和 </think>）"""
    tokens_to_remove = ['<|im_end|>', '<|im_start|>', '<|endoftext|>', '<|user|>', '<|assistant|>']
    for token in tokens_to_remove:
        text = text.replace(token, '')
    return text.strip()


def check_format(response):
    """检查格式：<think>\n...\n</think>\n..."""
    if response.count("<think>") != 1 or response.count("</think>") != 1:
        return False
    return re.match(r"^<think>\n.*?\n</think>\n.+$", response, re.DOTALL) is not None


def parse_answer(response):
    """提取答案：最后一个 </think>\n 后的内容"""
    if "</think>\n" in response:
        answer = response.split("</think>\n")[-1]
        return clean_special_tokens(answer)
    return ""


def parse_judge_json(text):
    """从 Judge 输出解析 JSON"""
    for pattern in [r"```(?:json)?\s*(\{[^`]*\})\s*```", r"(\{[^{}]*\})"]:
        for raw in re.findall(pattern, text, re.DOTALL | re.IGNORECASE):
            try:
                d = json.loads(raw.strip())
                result = {k: 1 if d.get(k, 0) >= 1 else 0 
                         for k in ["fluency", "factuality", "instruction_following"]}
                if len(result) == 3:
                    return result
            except:
                continue
    return None


def call_judge(prompt, answer, api_key, model="deepseek-chat"):
    """调用 DeepSeek Judge API"""
    PROMPT = """请根据问题对以下回答进行评分（0-1 二值）：

【问题】{question}
【回答】{response}

请从三个维度评分（0=不通过，1=通过）：
1. fluency: 回答是否流畅、语言自然
2. factuality: 回答是否准确、符合事实
3. instruction_following: 是否正确理解并遵循了指令并回答用户问题

请务必严格，如果无法判断，则视为不通过。

以 JSON 格式输出：
```json
{{
  "fluency": 0 或 1,
  "factuality": 0 或 1,
  "instruction_following": 0 或 1
}}
```"""
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT.format(question=prompt, response=answer)}],
            stream=False
        )
        return parse_judge_json(response.choices[0].message.content)
    except Exception as e:
        print(f"  [judge] API 失败: {str(e)[:80]}")
        return None


def calculate_rewards(raw_prompts, responses, args):
    """
    计算奖励：格式检查 + Judge 评分
    
    Returns:
        rewards: tensor [N], 奖励值
        stats: dict, 统计信息
        detailed_results: list, 每个样本的详细结果
    """
    num_resp = len(responses)
    rewards = torch.zeros(num_resp, device=args.device)
    
    # 1. 格式检查
    format_ok = [check_format(r) for r in responses]
    format_pass = sum(format_ok)
    
    # 2. 并发调用 Judge（仅格式正确的）
    tasks = []
    for i, (ok, resp) in enumerate(zip(format_ok, responses)):
        if ok:
            answer = parse_answer(resp)
            prompt_idx = i // args.num_generations
            tasks.append((i, raw_prompts[prompt_idx], answer))
    
    judge_results = {}
    if tasks:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(call_judge, p, a, args.judge_api_key, args.judge_model): idx
                      for idx, p, a in tasks}
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                if result:
                    judge_results[idx] = result
    
    # 3. 计算 rewards 并构建详细结果
    detailed_results = []
    for i in range(num_resp):
        prompt_idx = i // args.num_generations
        gen_idx = i % args.num_generations
        
        judge_scores = None
        if format_ok[i] and i in judge_results:
            scores = judge_results[i]
            rewards[i] = sum(scores.values()) / 3.0
            judge_scores = scores
        # else: rewards[i] = 0.0 (已初始化)
        
        # 保存详细结果
        detailed_results.append({
            'prompt': raw_prompts[prompt_idx],
            'response': responses[i],
            'reward': rewards[i].item(),
            'format_ok': format_ok[i],
            'judge_scores': judge_scores,
            'generation_idx': gen_idx
        })
    
    # 4. 计算 solve_all 和 solve_none（基于 judge 综合分数）
    # solve_all: 所有 generation 的 judge 三指标之和都是 3（即 reward=1.0）
    # solve_none: 所有 generation 的 judge 综合分数都是 0（格式错误或 judge=0）
    num_prompts = len(raw_prompts)
    solve_all_count = 0
    solve_none_count = 0
    
    for prompt_idx in range(num_prompts):
        start_idx = prompt_idx * args.num_generations
        end_idx = start_idx + args.num_generations
        group_rewards = rewards[start_idx:end_idx]
        
        # 检查是否所有 reward 都是 1.0（满分）
        if torch.all(group_rewards == 1.0).item():
            solve_all_count += 1
        # 检查是否所有 reward 都是 0.0（零分）
        elif torch.all(group_rewards == 0.0).item():
            solve_none_count += 1
    
    # 5. 统计（仅格式正确的）
    if judge_results:
        all_scores = list(judge_results.values())
        stats = {
            'format_pass_rate': format_pass / num_resp,
            'format_pass_count': format_pass,
            'solve_all_rate': solve_all_count / num_prompts,
            'solve_none_rate': solve_none_count / num_prompts,
            'judge_fluency': sum(s['fluency'] for s in all_scores) / len(all_scores),
            'judge_factuality': sum(s['factuality'] for s in all_scores) / len(all_scores),
            'judge_instruction_following': sum(s['instruction_following'] for s in all_scores) / len(all_scores),
            'judge_mean': sum(sum(s.values()) for s in all_scores) / (3 * len(all_scores))
        }
    else:
        stats = {
            'format_pass_rate': format_pass / num_resp, 
            'format_pass_count': format_pass,
            'solve_all_rate': solve_all_count / num_prompts,
            'solve_none_rate': solve_none_count / num_prompts,
            'judge_fluency': 0, 'judge_factuality': 0, 
            'judge_instruction_following': 0, 'judge_mean': 0
        }
    
    return rewards, stats, detailed_results


# ==================== 训练循环 ====================

def compute_logprobs(model, outputs, completion_len):
    """
    计算 completion 部分每个 token 的 log probability
    
    Args:
        model: 语言模型（policy 或 reference）
        outputs: 完整序列 [B*num_gen, prompt_len + completion_len]
        completion_len: 生成部分的长度
    
    Returns:
        log_probs: [B*num_gen, completion_len]，每个生成 token 的 log probability
    """
    # 1. 前向传播获取 logits: [B*num_gen, seq_len-1, vocab_size]
    #    去掉最后一个位置（因为它没有 next token）
    logits = model(outputs).logits[:, :-1, :]
    
    # 2. 只保留 completion 部分的 logits: [B*num_gen, completion_len, vocab_size]
    logits = logits[:, -completion_len:, :]
    
    # 3. 提取 completion 部分的 target tokens: [B*num_gen, completion_len]
    target_ids = outputs[:, -completion_len:]
    
    # 4. 计算 log softmax 并用 gather 提取对应 token 的概率
    #    log_softmax: [B*num_gen, completion_len, vocab_size]
    #    gather: 从 vocab 维度提取 target_ids 对应的概率
    #    结果: [B*num_gen, completion_len]
    return torch.gather(logits.log_softmax(dim=-1), 2, target_ids.unsqueeze(2)).squeeze(2)


def create_eos_mask(completion_ids, eos_token_id):
    """创建 EOS mask（只计算到第一个 EOS）"""
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    return (torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1) 
            <= eos_idx.unsqueeze(1)).int()


def save_checkpoint(model, optimizer, epoch, step, global_step, swanlab_id, save_dir, weight_name, hidden_size):
    """保存 checkpoint"""
    ckp_dir = f'{save_dir}/global_step_{global_step}'
    os.makedirs(ckp_dir, exist_ok=True)
    
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
    
    torch.save(state_dict, f'{ckp_dir}/{weight_name}_{hidden_size}.pth')
    torch.save({'model': state_dict, 'optimizer': optimizer.state_dict(), 
                'epoch': epoch, 'step': step, 'global_step': global_step, 'swanlab_id': swanlab_id},
               f'{ckp_dir}/resume.pth')
    Logger(f'Checkpoint saved: {ckp_dir}')


def train_epoch(epoch, loader, iters, model, ref_model, optimizer, tokenizer, autocast_ctx, args,
                start_step=0, swanlab=None, save_dir=None):
    """训练一个 epoch"""
    start_time = time.time()
    
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']
        raw_prompts = [p.split("<|user|>")[1].split("<|im_end|>")[0] for p in prompts]
        
        # 1. 生成（左 padding）
        orig_pad_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, 
                          return_token_type_ids=False, add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            inputs["input_ids"] = inputs["input_ids"][:, -args.max_seq_len:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -args.max_seq_len:]
        
        input_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            gen_model = model.module if isinstance(model, DDP) else model
            outputs = gen_model.generate(**inputs, max_new_tokens=args.max_gen_len, do_sample=True,
                                        temperature=0.3, num_return_sequences=args.num_generations,
                                        pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.2)
        
        tokenizer.padding_side = orig_pad_side
        completion_ids = outputs[:, input_len:]
        
        # 2. 计算 log probs
        with autocast_ctx:
            policy_logps = compute_logprobs(model, outputs, completion_ids.size(1))
        with torch.no_grad():
            ref_logps = compute_logprobs(ref_model, outputs, completion_ids.size(1))
        
        # 3. 计算奖励
        # 重要：不能 skip_special_tokens，否则 <think> 标签会被过滤掉！
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=False)
        # 清理不需要的特殊 token（保留 <think> 和 </think>）
        completions = [clean_special_tokens(c) for c in completions]
        rewards, stats, detailed_results = calculate_rewards(raw_prompts, completions, args)
        
        # 4. 计算优势函数
        grouped_r = rewards.view(-1, args.num_generations)
        mean_r = grouped_r.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_r.std(dim=1).repeat_interleave(args.num_generations)
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 5. GRPO loss
        mask = create_eos_mask(completion_ids, tokenizer.eos_token_id)
        kl = torch.exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1
        per_token_loss = -(torch.exp(policy_logps - policy_logps.detach()) * advantages.unsqueeze(1) - args.beta * kl)
        loss = ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)).mean() / args.accumulation_steps
        
        # 计算平均 KL 散度（用于监控）
        kl_mean = ((kl * mask).sum(dim=1) / mask.sum(dim=1)).mean().item()
        
        # 6. 更新
        loss.backward()
        grad_norm = 0.0
        if (step + 1) % args.accumulation_steps == 0:
            # 计算梯度范数（在 clip 之前）
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # 7. 日志和保存
        global_step = epoch * iters + step
        if step % args.log_interval == 0 or step == iters - 1:
            eta = (time.time() - start_time) / (step + 1) * iters // 60 - (time.time() - start_time) // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), '
                   f'Loss:{loss.item()*args.accumulation_steps:.4f}, Reward:{rewards.mean():.3f}, '
                   f'Format:{stats["format_pass_rate"]:.2%} (All:{stats["solve_all_rate"]:.2%}, None:{stats["solve_none_rate"]:.2%}), '
                   f'Judge:{stats["judge_mean"]:.3f}, KL:{kl_mean:.4f}, GradNorm:{grad_norm:.3f}, '
                   f'Len:{mask.sum(dim=1).float().mean():.0f}, ETA:{eta:.0f}min')
            
            # 打印第一个生成样本（用于调试）
            if step == 1:
                Logger(f'\n[Sample] Prompt: {raw_prompts[0][:50]}...')
                Logger(f'[Sample] Response: {completions[0][:200]}...\n')
            
            if swanlab and is_main_process():
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps, 
                    "reward_mean": rewards.mean().item(),
                    "kl_divergence": kl_mean,
                    "grad_norm": grad_norm,
                    **{k: v for k, v in stats.items()}, 
                    "avg_len": mask.sum(dim=1).float().mean().item(),
                    "lr": optimizer.param_groups[0]['lr']
                }, step=global_step)
        
        # 8. 保存 rollout 数据
        if is_main_process():
            data_log_dir = os.path.join(save_dir, 'data_log')
            os.makedirs(data_log_dir, exist_ok=True)
            rollout_file = os.path.join(data_log_dir, f'global_step_{global_step}.jsonl')
            
            with open(rollout_file, 'w', encoding='utf-8') as f:
                for result in detailed_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # 9. 保存 checkpoint
        if (global_step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            save_checkpoint(model, optimizer, epoch, step, global_step, 
                          getattr(swanlab, 'id', None) if swanlab else None,
                          save_dir, args.save_weight, lm_config.hidden_size)
            model.train()


# ==================== 主程序 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpongeBob GRPO Training")
    
    # 训练参数
    parser.add_argument("--save_dir", type=str, default="../out_grpo/exp_1")
    parser.add_argument('--save_weight', default='grpo', type=str)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=0.2)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=20)
    
    # 模型配置
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=12, type=int)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument("--max_gen_len", type=int, default=512)
    
    # 路径
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--tokenizer_path", type=str, default="../tokenizer_15k")
    parser.add_argument("--sft_model_path", type=str, 
                       default="../out_think_distill/exp_1/h768_l12_bs128_lr2e-05/global_step_2971/sft_768.pth")
    
    # GRPO 参数
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--judge_api_key", type=str, default='')
    parser.add_argument("--judge_model", type=str, default="deepseek-chat")
    
    # 控制
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_swanlab", type=int, default=1, choices=[0, 1])
    parser.add_argument("--swanlab_project", type=str, default="SpongeBob-GRPO")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1])
    
    args = parser.parse_args()
    
    # 初始化
    local_rank = init_distributed_mode()
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    
    # 配置
    lm_config = SpongeBobConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                max_position_embeddings=args.max_seq_len + args.max_gen_len)
    run_name = f"h{args.hidden_size}_l{args.num_hidden_layers}_bs{args.batch_size}_lr{args.learning_rate}"
    full_save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(full_save_dir, exist_ok=True)
    
    # 恢复 checkpoint
    ckp_data = None
    if args.from_resume:
        ckp_dirs = [d for d in os.listdir(full_save_dir) if d.startswith('global_step_')] if os.path.exists(full_save_dir) else []
        if ckp_dirs:
            latest = max(ckp_dirs, key=lambda x: int(x.split('_')[-1]))
            resume_path = f'{full_save_dir}/{latest}/resume.pth'
            if os.path.exists(resume_path):
                ckp_data = torch.load(resume_path, map_location='cpu')
                Logger(f'Resume from: {latest}')
    
    # 混合精度
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if "cpu" in args.device else torch.amp.autocast(device_type="cuda", dtype=dtype)
    
    # SwanLab
    swanlab_run = None
    if args.use_swanlab and is_main_process():
        import swanlab
        swanlab.login(api_key="4jqfbuJs9zDRcLAMPoDQv")
        swanlab_run = swanlab.init(project=args.swanlab_project, experiment_name=run_name,
                                   id=ckp_data.get('swanlab_id') if ckp_data else None, config=vars(args))
        Logger(f'SwanLab: {run_name}')
    
    # 加载模型
    Logger(f'Loading models from {args.sft_model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else '[PAD]'
    
    def load_model(trainable=True):
        m = SpongeBobForCausalLM(lm_config)
        if os.path.exists(args.sft_model_path):
            m.load_state_dict(torch.load(args.sft_model_path, map_location='cpu'), strict=False)
        m = m.to(args.device)
        return m if trainable else m.eval().requires_grad_(False)
    
    model = load_model(trainable=True)
    ref_model = load_model(trainable=False)
    Logger(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')
    
    if args.use_compile:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    # 数据和优化器
    train_ds = GRPODataset(args.data_path, tokenizer, args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    Logger(f'Dataset: {len(train_ds)} samples')
    
    # 恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        start_epoch, start_step = ckp_data['epoch'], ckp_data.get('step', 0)
    
    # DDP
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DDP(model, device_ids=[local_rank])
    
    # 训练
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    steps_per_epoch = len(train_ds) // (args.batch_size * world_size)
    Logger(f'Training: {args.epochs} epochs, {steps_per_epoch} steps/epoch')
    
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        skip = start_step if epoch == start_epoch and start_step > 0 else 0
        batch_sampler = SkipBatchSampler(train_sampler or torch.randperm(len(train_ds)).tolist(), 
                                        args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        train_epoch(epoch, loader, len(loader) + skip, model, ref_model, optimizer, tokenizer, 
                   autocast_ctx, args, skip, swanlab_run, full_save_dir)
    
    if dist.is_initialized(): 
        dist.destroy_process_group()
    Logger('Training completed!')
