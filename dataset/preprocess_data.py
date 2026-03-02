#!/usr/bin/env python3
"""
预训练数据预处理脚本
离线将jsonl处理成二进制文件(.bin)，用于高效训练
且由于tokenizer是cpu密集型操作,当进行多卡训练时,每个进程都要加载tokenizer,会占用大量内存,且速度较慢,可能成为训练速度的瓶颈
"""
import os
import sys
import json
import argparse
import tempfile
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count


# 全局变量，供子进程使用
_tokenizer = None
_eos_id = None

def _init_worker(tokenizer_path):
    """初始化子进程的tokenizer"""
    global _tokenizer, _eos_id
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _eos_id = _tokenizer.eos_token_id

def _tokenize_line(line):
    """处理单行数据"""
    try:
        line = line.strip()
        if not line:
            return []
        data = json.loads(line)
        text = data.get('text', '')
        if not text:
            return []
        tokens = _tokenizer.encode(text, add_special_tokens=False)
        tokens.append(_eos_id)
        return tokens
    except Exception as e:
        # 静默失败，避免中断整个流程
        return []

def preprocess(input_path, output_path, tokenizer_path, seq_len=512, num_workers=None):
    """
    预处理：tokenize + 拼接 + 切分 + 保存为.bin
    
    输出文件：
    - output_path.bin: 所有token数据 (int16格式)
    - output_path.meta: 元信息 (json格式)
    """
    if num_workers is None:
        num_workers = cpu_count()  # 最多32个进程
    
    print(f"{'='*60}")
    print(f"预训练数据预处理")
    print(f"{'='*60}")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}.bin")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"序列长度: {seq_len}")
    print(f"进程数: {num_workers}")
    print(f"{'='*60}\n")
    
    # 获取词表信息
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)
    eos_id = tokenizer.eos_token_id
    print(f"词表大小: {vocab_size}")
    print(f"EOS token: {eos_id} ('{tokenizer.decode([eos_id])}')\n")
    
    # 统计行数
    print("步骤1: 统计样本数...")
    with open(input_path, 'r', encoding='utf-8') as f:
        num_samples = sum(1 for _ in f)
    print(f"样本数: {num_samples:,}\n")
    
    # 流式处理：多进程tokenize + 分批写入
    print(f"步骤2: Tokenizing (使用 {num_workers} 个进程)...")
    
    def line_generator():
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line
    
    # 临时文件：边处理边写入，避免内存爆炸
    temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.tmp')
    temp_path = temp_file.name
    
    total_tokens = 0
    buffer = []
    BUFFER_SIZE = 10_000_000  # 每1000万个token写一次磁盘（~20MB）
    
    try:
        with Pool(num_workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
            for tokens in tqdm(pool.imap(_tokenize_line, line_generator(), chunksize=100), total=num_samples):
                if tokens:
                    buffer.extend(tokens)
                    total_tokens += len(tokens)
                    
                    # 缓冲区满了，写入磁盘
                    if len(buffer) >= BUFFER_SIZE:
                        np.array(buffer, dtype=np.uint16).tofile(temp_file)
                        buffer = []
        
        # 写入剩余数据
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(temp_file)
            buffer = []
        
        temp_file.close()
        
        print(f"\n总tokens: {total_tokens:,}")
        print(f"平均长度: {total_tokens/num_samples:.1f} tokens/sample\n")
        
        # 切分成固定长度chunks
        print(f"步骤3: 切分成 {seq_len} 长度的chunks...")
        num_chunks = total_tokens // seq_len
        dropped = total_tokens % seq_len
        
        # 从临时文件读取并切分（内存映射）
        all_tokens = np.fromfile(temp_path, dtype=np.uint16)
        all_tokens = all_tokens[:num_chunks * seq_len]  # 只保留完整chunks
        arr = all_tokens.reshape(-1, seq_len)
        
        
        print(f"Chunks数: {num_chunks:,}")
        print(f"丢弃tokens: {dropped} ({dropped/total_tokens*100:.3f}%)")
        print(f"数组形状: {arr.shape}")
        print(f"内存占用: {arr.nbytes / (1024**3):.2f} GB\n")
        
        # 保存为.bin文件
        print(f"步骤4: 保存为二进制文件...")
        bin_path = f"{output_path}.bin"
        arr.tofile(bin_path)
        print(f"✓ 已保存: {bin_path} ({os.path.getsize(bin_path)/(1024**3):.2f} GB)")
        
        # 保存元信息
        meta = {
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "num_chunks": num_chunks,
            "total_tokens": total_tokens,
            "num_samples": num_samples,
            "dropped_tokens": dropped,
            "dtype": "uint16",
            "shape": list(arr.shape),
        }
        
        meta_path = f"{output_path}.meta"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"✓ 已保存: {meta_path}\n")
        
        print(f"{'='*60}")
        print(f"预处理完成！")
        print(f"{'='*60}")
        print(f"训练时使用: --data_path {bin_path}")
        print(f"{'='*60}\n")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预训练数据预处理")
    parser.add_argument("--input", type=str, required=True, help="输入jsonl文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件前缀（不含扩展名）")
    parser.add_argument("--tokenizer", type=str, default="../tokenizer_15k", help="tokenizer路径")
    parser.add_argument("--seq_len", type=int, default=512, help="序列长度")
    parser.add_argument("--num_workers", type=int, default=None, help="进程数（默认=CPU核心数，最多32）")
    args = parser.parse_args()
    
    preprocess(args.input, args.output, args.tokenizer, args.seq_len, args.num_workers)
