"""
训练15k BPE tokenizer (中英文双语)
"""
import os
import json
import time
from datetime import datetime
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

# 设置线程数
NUM_THREADS = 300
os.environ['RAYON_NUM_THREADS'] = str(NUM_THREADS)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# 配置
DATA_PATH = '/apdcephfs_qy4/share_302593112/huaibingxie/SpongeBob/data/pretrain_data/merged_pretrain_data_zh_en_only_v2.jsonl'
TOKENIZER_DIR = '/apdcephfs_qy4/share_302593112/huaibingxie/SpongeBob/tokenizer_15k/'
VOCAB_SIZE = 15000

# 特殊tokens（15个）
SPECIAL_TOKENS = [
    "<|endoftext|>",   # 0
    "<|im_start|>",    # 1
    "<|im_end|>",      # 2
    "<think>",         # 3
    "</think>",        # 4
    "<pad>",           # 5
    "<unk>",           # 6
    "<|system|>",      # 7
    "<|user|>",        # 8
    "<|assistant|>",   # 9
    "<tool_call>",     # 10
    "</tool_call>",    # 11
    "<function>",      # 12
    "</function>",     # 13
    "<unused_0>",      # 14
]

def get_texts(data_path, max_lines=None):
    """读取训练数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            try:
                data = json.loads(line)
                text = data.get('text', '')
                if text:
                    yield text
            except:
                continue

def train_tokenizer(data_path, tokenizer_dir, vocab_size, special_tokens, max_lines=None):
    """训练tokenizer"""
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n开始时间: {start_datetime}")
    print(f"训练配置:")
    print(f"  数据: {data_path}")
    print(f"  词表: {vocab_size} (BPE: {vocab_size - len(special_tokens)}, 特殊: {len(special_tokens)})")
    print(f"  模式: {'测试' if max_lines else '全量'}")
    print(f"  线程: {NUM_THREADS} (总核心: {os.cpu_count()})\n")
    
    # 初始化
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 训练
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,  # 终端会显示，重定向时无效
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        limit_alphabet=6000,
        continuing_subword_prefix="",
    )
    
    print("开始训练...")
    print("(注意: 训练过程较长，日志输出可能延迟，请耐心等待...)")
    texts = get_texts(data_path, max_lines=max_lines)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    print("训练阶段完成，开始保存...")
    
    # 验证特殊tokens
    for i, token in enumerate(special_tokens[:5]):
        assert tokenizer.token_to_id(token) == i, f"{token} ID错误"
    
    # 保存
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)
    
    # 配置文件
    added_tokens_decoder = {
        str(i): {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        } for i, token in enumerate(special_tokens)
    }
    
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": special_tokens[5:],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "model_max_length": 8192,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": "{{- '<|im_start|>' -}}{%- for message in messages -%}{%- if message.role == 'user' -%}{{- '<|user|>' + message.content + '<|im_end|>' -}}{%- elif message.role == 'assistant' -%}{{- '<|assistant|>' + message.content + '<|im_end|>' -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|assistant|>' -}}{%- endif -%}"
    }
    
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # 计算训练时间
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n训练完成! 保存到: {tokenizer_dir}")
    print(f"实际词表大小: {len(tokenizer.get_vocab())}")
    print(f"结束时间: {end_datetime}")
    print(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒 ({elapsed_time:.1f}秒)\n")

def eval_tokenizer(tokenizer_dir):
    """测试tokenizer"""
    from transformers import AutoTokenizer
    
    print("测试tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    
    # 基础测试
    test_text = "Hello World! 你好世界！This is a test. 这是测试。"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"  词表大小: {len(tokenizer)}")
    print(f"  测试文本: {test_text}")
    print(f"  Token数量: {len(tokens)}")
    print(f"  解码一致: {'✓' if decoded == test_text else '✗'}")
    
    # 对话测试
    messages = [
        {"role": "system", "content": "You are helpful. 你很有帮助。"},
        {"role": "user", "content": "Hi! 你好！"},
        {"role": "assistant", "content": "Hello! 你好！"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"\n对话模板:\n{prompt}")

if __name__ == '__main__':
    import sys
    
    total_start = time.time()
    
    # --test 参数使用测试模式（前10000行）
    test_mode = '--test' in sys.argv
    max_lines = 10000 if test_mode else None
    
    # 训练
    # train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE, SPECIAL_TOKENS, max_lines)
    
    # 测试
    eval_tokenizer(TOKENIZER_DIR)
    
    # 总时间统计
    total_elapsed = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"总运行时间: {total_elapsed/60:.1f} 分钟 ({total_elapsed:.1f} 秒)")
    print(f"{'='*50}\n")