"""
SpongeBob SFT 数据集
支持多轮对话，只计算 assistant 部分的 loss
"""
import json
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class SFTDataset(Dataset):
    """
    SFT 数据集（使用 HuggingFace datasets 自动内存映射）
    
    数据格式：{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    对话格式：<|im_start|><|user|>content<|im_end|><|assistant|>content<|im_end|>...
    
    Labels：只计算 assistant 部分的 loss（包括 <|assistant|> token、content 和 <|im_end|>）
    """
    
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        """
        Args:
            jsonl_path: JSONL 文件路径
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 使用 datasets 加载（自动内存映射，不会占用大量内存）
        print(f"Loading SFT data from {jsonl_path}...")
        self.data = load_dataset('json', data_files=jsonl_path, split='train')
        print(f"Loaded {len(self.data)} conversations")
        
        # 特殊 token IDs
        self.im_start_id = 1  # <|im_start|>
        self.im_end_id = 2    # <|im_end|>
        self.user_id = 8      # <|user|>
        self.assistant_id = 9 # <|assistant|>
        self.pad_id = tokenizer.pad_token_id or 0  # <|endoftext|>
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        返回一个样本的 input_ids 和 labels
        
        格式示例：
            input_ids: [1, 8, tok1, tok2, 2, 9, tok3, tok4, 2, 8, tok5, 2, 9, tok6, 2]
            labels:    [-100, -100, -100, -100, -100, 9, tok3, tok4, 2, -100, -100, -100, 9, tok6, 2]
                        ↑开始  ↑user部分全部mask           ↑assistant部分计算loss  ↑user mask    ↑assistant计算loss
        """
        conversations = self.data[idx]['conversations']
        
        input_ids = [self.im_start_id]  # 开头 <|im_start|>
        labels = [-100]  # <|im_start|> 不计算 loss
        
        for conv in conversations:
            role = conv['role']
            content = conv['content']
            
            if role == 'user':
                # User 部分：<|user|>content<|im_end|>
                # 全部 mask 掉（不计算 loss）
                user_tokens = [self.user_id] + \
                              self.tokenizer.encode(content, add_special_tokens=False) + \
                              [self.im_end_id]
                input_ids.extend(user_tokens)
                labels.extend([-100] * len(user_tokens))
                
            elif role == 'assistant':
                # Assistant 部分：<|assistant|>content<|im_end|>
                # 全部计算 loss（包括 <|assistant|> token）
                assistant_tokens = [self.assistant_id] + \
                                   self.tokenizer.encode(content, add_special_tokens=False) + \
                                   [self.im_end_id]
                input_ids.extend(assistant_tokens)
                labels.extend(assistant_tokens)  # 全部计算 loss
        
        # 截断或填充到 max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        else:
            padding_length = self.max_length - len(input_ids)
            input_ids.extend([self.pad_id] * padding_length)
            labels.extend([-100] * padding_length)  # padding 不计算 loss
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


if __name__ == "__main__":
    # 简单测试
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("../tokenizer_15k")
    dataset = SFTDataset("../data_raw/mini/sft_512_2M.jsonl", tokenizer, max_length=512)
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 查看第一个样本
    input_ids, labels = dataset[0]
    print(f"\n第一个样本:")
    print(f"input_ids shape: {input_ids.shape}")
    print(f"labels shape: {labels.shape}")
    
    # 解码前 100 个 tokens
    print(f"\n前 100 个 tokens:")
    for i in range(min(100, len(input_ids))):
        token = input_ids[i].item()
        label = labels[i].item()
        decoded = tokenizer.decode([token])
        loss_marker = "✓" if label != -100 else "✗"
        print(f"{i:3d}: [{token:5d}] {decoded:20s} | label={label:5d} {loss_marker}")
