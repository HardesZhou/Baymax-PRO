#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 mini_bench"""
import argparse, os, sys, torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from model.config import SpongeBobConfig
from model.model_spongebob_pro import SpongeBobForCausalLM
from eval import run_inference, run_judge

WEIGHT = "/apdcephfs_qy4/share_302593112/huaibingxie/SpongeBob/pretrain_out/h768_l12_bs128_lr0.001/global_step_5779/pretrain_768.pth"
TOKENIZER = os.path.join(_REPO, "tokenizer_15k")

parser = argparse.ArgumentParser()
parser.add_argument("--max_prompts", type=int, default=5)
args = parser.parse_args()

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
model = SpongeBobForCausalLM(SpongeBobConfig(hidden_size=768, num_hidden_layers=12))
model.load_state_dict(torch.load(WEIGHT, map_location="cpu"), strict=False)
model = model.to("cuda:0" if torch.cuda.is_available() else "cpu").eval()

print("Inference...")
pairs = run_inference(model, tokenizer, device=model.device, max_prompts=args.max_prompts)

print("Judge...")
metrics = run_judge(pairs)

print("\n=== 结果 ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
print("============")
