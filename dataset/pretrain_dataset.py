"""
SpongeBob 预训练数据集
加载预处理好的二进制数据
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    预训练数据集：加载.bin文件
    数据格式：(num_chunks, seq_len) 的 uint16 数组
    """
    def __init__(self, data_path, seq_len=512):
        """
        Args:
            data_path: .bin文件路径
            seq_len: 序列长度（用于验证，实际从.meta读取）
        """
        if not data_path.endswith('.bin'):
            data_path = data_path + '.bin'
        
        # 加载元信息
        meta_path = data_path.replace('.bin', '.meta')
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        # 验证序列长度
        assert self.meta['seq_len'] == seq_len, f"seq_len mismatch: {self.meta['seq_len']} vs {seq_len}"
        
        # 使用内存映射加载数据（不占用内存）
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r', shape=tuple(self.meta['shape']))
        
        print(f"Dataset loaded: {len(self)} chunks from {data_path}")
    
    def __len__(self):
        return self.meta['num_chunks']
    
    def __getitem__(self, idx):
        """
        返回 (input_ids, labels)
        注意：不做 shift，让模型自己处理
        input_ids 和 labels 是同一个序列，模型会在内部做 shift
        """
        chunk = torch.from_numpy(self.data[idx].astype(np.int64))
        # 直接返回整个 chunk 作为 input_ids 和 labels
        return chunk.clone(), chunk.clone()




