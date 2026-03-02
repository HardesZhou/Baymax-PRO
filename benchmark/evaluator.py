"""
简单的 Benchmark 评测模块
支持 C3 和 XCOPA 数据集的评测
"""
import json
import torch
import torch.nn.functional as F


def eval_multiple_choice(model, tokenizer, context, choices, label_idx, max_length=512):
    """
    多选题评测：计算每个选项的困惑度，选择困惑度最低的
    
    Args:
        model: 语言模型
        tokenizer: tokenizer
        context: 问题上下文（字符串）
        choices: 选项列表（字符串列表）
        label_idx: 正确答案的索引
        max_length: 最大序列长度
    
    Returns:
        1 表示预测正确，0 表示预测错误
    """
    losses = []
    
    for choice in choices:
        # 拼接 context + choice
        full_text = context + choice
        
        # 编码（添加最大长度限制）
        inputs = tokenizer(
            full_text, 
            return_tensors="pt", 
            max_length=max_length,
            truncation=True
        ).to(model.device)
        input_ids = inputs.input_ids
        
        # 计算 context 的长度（用于定位 choice 的起始位置）
        # 注意：使用与完整文本编码相同的 add_special_tokens 设置
        context_tokens = tokenizer(context, add_special_tokens=True).input_ids
        context_len = len(context_tokens)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
        
        # 计算 loss（只计算 choice 部分）
        # shift_logits[i] 用于预测 input_ids[i+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Choice 在 shift 后的起始位置应该是 context_len - 1
        # 因为 loss[i] 预测的是原始序列中位置 i+1 的 token
        # 如果 context 占据位置 0 到 context_len-1，
        # 那么 choice 从位置 context_len 开始
        # 在 loss 中，预测位置 context_len 的 token 对应 loss[context_len-1]
        choice_start = max(0, context_len - 1)
        
        # 只取 choice 部分的平均 loss
        if choice_start < len(loss):
            choice_loss = loss[choice_start:].mean().item()
        else:
            # 如果 context 太长，导致 choice 被截断，则使用整个序列的 loss
            choice_loss = loss.mean().item()
        
        losses.append(choice_loss)
    
    # 选择 loss 最小的作为预测
    pred_idx = losses.index(min(losses))
    return 1 if pred_idx == label_idx else 0


def eval_c3(model, tokenizer, data_path):
    """
    评测 C3 数据集
    
    Args:
        model: 模型
        tokenizer: tokenizer
        data_path: C3 数据集路径（jsonl 格式）
    
    Returns:
        准确率（0-1之间的浮点数）
    """
    correct = 0
    total = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # C3 数据格式：context (list), question (str), choice (list), answer (str)
            context_text = ''.join(data['context'])  # 合并上下文
            question = data['question']
            choices = data['choice']
            answer = data['answer']
            
            # 将答案文本转换为索引
            if answer not in choices:
                continue
            label_idx = choices.index(answer)
            
            # 构建完整的问题上下文
            full_context = context_text + question
            
            # 评测
            result = eval_multiple_choice(model, tokenizer, full_context, choices, label_idx)
            correct += result
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def eval_xcopa(model, tokenizer, data_path):
    """
    评测 XCOPA 数据集
    
    Args:
        model: 模型
        tokenizer: tokenizer
        data_path: XCOPA 数据集路径（jsonl 格式）
    
    Returns:
        准确率（0-1之间的浮点数）
    """
    correct = 0
    total = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # XCOPA 数据格式：premise (str), choice1 (str), choice2 (str), question (str), label (int)
            premise = data['premise']
            choices = [data['choice1'], data['choice2']]
            label_idx = data['label']
            question_type = data['question']  # 'cause' 或 'effect'
            
            # 构建上下文（根据问题类型调整提示，使用更明确的格式）
            if question_type == 'cause':
                context = f"{premise}这是因为："
            else:  # effect
                context = f"{premise}所以："
            
            # 评测
            result = eval_multiple_choice(model, tokenizer, context, choices, label_idx)
            correct += result
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def run_benchmark(model, tokenizer, c3_path, xcopa_path):
    """
    运行所有 benchmark 评测
    
    Args:
        model: 模型（会自动解包 DDP）
        tokenizer: tokenizer
        c3_path: C3 数据集路径
        xcopa_path: XCOPA 数据集路径
    
    Returns:
        包含所有评测结果的字典
    """
    results = {}
    
    # 解包模型（如果是 DDP 或 compile 包装的）
    from torch.nn.parallel import DistributedDataParallel
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    
    raw_model.eval()  # 设置为评测模式
    
    print("\n" + "="*60)
    print("开始 Benchmark 评测")
    print("="*60)
    
    # 评测 C3
    try:
        print(f"评测 C3 数据集: {c3_path}")
        c3_acc = eval_c3(raw_model, tokenizer, c3_path)
        results['c3_accuracy'] = c3_acc
        print(f"✓ C3 Accuracy: {c3_acc:.4f} ({c3_acc*100:.2f}%)")
    except Exception as e:
        print(f"✗ C3 evaluation failed: {e}")
        results['c3_accuracy'] = 0.0
    
    # 评测 XCOPA
    try:
        print(f"评测 XCOPA 数据集: {xcopa_path}")
        xcopa_acc = eval_xcopa(raw_model, tokenizer, xcopa_path)
        results['xcopa_accuracy'] = xcopa_acc
        print(f"✓ XCOPA Accuracy: {xcopa_acc:.4f} ({xcopa_acc*100:.2f}%)")
    except Exception as e:
        print(f"✗ XCOPA evaluation failed: {e}")
        results['xcopa_accuracy'] = 0.0
    
    print("="*60 + "\n")
    
    raw_model.train()  # 恢复训练模式
    
    return results
