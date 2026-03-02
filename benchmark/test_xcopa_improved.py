#!/usr/bin/env python3
"""
测试改进后的 XCOPA 评测
"""
import sys
sys.path.append('/apdcephfs_qy4/share_302593112/huaibingxie/SpongeBob')

import torch
from transformers import AutoTokenizer
from model.config import SpongeBobConfig
from model.model_spongebob_pro import SpongeBobForCausalLM
from benchmark.evaluator import eval_xcopa

print("="*60)
print("测试 XCOPA 评测（改进后）")
print("="*60)

# 1. 加载 tokenizer
print("\n1. 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('/apdcephfs_qy4/share_302593112/huaibingxie/SpongeBob/tokenizer_15k')
print(f"   词表大小: {len(tokenizer)}")

# 2. 创建小模型测试
print("\n2. 创建随机初始化模型...")
config = SpongeBobConfig(
    hidden_size=256,
    num_layers=4,
    num_attention_heads=8,
    max_position_embeddings=512,
    vocab_size=len(tokenizer)
)
model = SpongeBobForCausalLM(config)
model.eval()
print(f"   模型参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# 3. 测试 XCOPA 评测
print("\n3. 运行 XCOPA 评测...")
xcopa_path = '/apdcephfs_qy4/share_302593112/huaibingxie/SpongeBob/benchmark/xcopa_zh_merged.jsonl'
accuracy = eval_xcopa(model, tokenizer, xcopa_path)

print(f"\n{'='*60}")
print(f"XCOPA 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*60}")

# 随机基线应该是 50%
print(f"\n随机基线: 50.00%")
print(f"当前模型: {accuracy*100:.2f}%")
if abs(accuracy - 0.5) < 0.05:
    print("✓ 结果正常（接近随机猜测，符合未训练模型预期）")
else:
    print("⚠️ 结果异常（偏离随机基线较远）")

print("\n" + "="*60)
