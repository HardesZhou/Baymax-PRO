"""
GRPO 数据集（用于 GRPO 训练）
只需要 prompt，不需要 labels（因为 GRPO 会动态生成 completions）

数据格式：{"id": 1, "category": "开放问答", "prompt": "问题文本"}
"""
import json
from torch.utils.data import Dataset
from datasets import load_dataset


class GRPODataset(Dataset):
    """
    GRPO 数据集（用于强化学习训练）
    
    数据格式（mini_bench 格式）：
        {"id": 1, "category": "开放问答", "prompt": "问题文本"}
    
    返回格式：
        {"prompt": str}  # 格式化后的 prompt：<|im_start|><|user|>问题<|im_end|><|assistant|>
    """
    
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        """
        Args:
            jsonl_path: JSONL 文件路径（mini_bench 格式）
            tokenizer: HuggingFace tokenizer（未使用，保持接口一致）
            max_length: 最大序列长度（未使用，保持接口一致）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 使用 datasets 加载（自动内存映射）
        print(f"Loading GRPO data from {jsonl_path}...")
        self.data = load_dataset('json', data_files=jsonl_path, split='train')
        print(f"Loaded {len(self.data)} prompts")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        返回一个样本的 prompt
        
        返回:
            {"prompt": str}  # 格式化后的 prompt
        """
        item = self.data[idx]
        
        # 从 mini_bench 格式提取 prompt
        raw_prompt = item['prompt']
        
        # 格式化为训练格式（与 eval.py 保持一致）
        formatted_prompt = f"<|im_start|><|user|>{raw_prompt}<|im_end|><|assistant|>"
        
        return {"prompt": formatted_prompt}


if __name__ == "__main__":
    # 简单测试
    from transformers import AutoTokenizer
    
    # 使用实际的 mini_bench 数据测试
    tokenizer = AutoTokenizer.from_pretrained("../tokenizer_15k")
    dataset = GRPODataset("../benchmark/mini_bench/100miniSponge.jsonl", tokenizer, max_length=512)
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 查看前3个样本
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n样本 {i}:")
        print(f"Prompt: {sample['prompt']}")
