#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mini_bench 评测：推理 + Judge（使用 DeepSeek API）
依赖：pip install openai
环境变量：export DEEPSEEK_API_KEY="your-key-here"
"""
import os, sys, json, re, torch
from pathlib import Path

_BENCH_JSONL = Path(__file__).parent / "100miniSponge.jsonl"
DIMENSIONS = ["fluency", "factuality", "instruction_following"]

JUDGE_PROMPT = """请根据问题对以下回答进行评分（0-1 二值）：

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


def run_inference(model, tokenizer, device=None, num_samples=3, max_prompts=None):
    """批量推理：每条 prompt 生成 num_samples 个 response"""
    device = device or next(model.parameters()).device
    
    with open(_BENCH_JSONL, "r", encoding="utf-8") as f:
        prompts = [json.loads(line.strip())["prompt"] for line in f if line.strip()]
    
    if max_prompts:
        prompts = prompts[:max_prompts]
    
    batch_size = 20
    all_pairs = []
    
    # 保存原始 padding 方向，生成时使用左 padding
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    # 确保 tokenizer 有 pad_token（左 padding 必需）
    # 重要：pad_token 不能等于 eos_token，否则模型会立即停止生成
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # 如果连 unk_token 也没有，添加一个新的 pad token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start+batch_size]
        
        # 关键修复：使用和训练时一致的对话格式
        # 训练格式：<|im_start|><|user|>content<|im_end|><|assistant|>
        formatted_prompts = [f"<|im_start|><|user|>{p}<|im_end|><|assistant|>" for p in batch_prompts]
        
        inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # 移除模型不支持的 token_type_ids
        inputs.pop("token_type_ids", None)
        
        # 左 padding 后，所有 prompt 在 batch 中对齐到右侧，输入长度统一为 max_len
        input_len = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,  # 明确指定 eos_token_id
                repetition_penalty=1.2
            )
        
        for i, p in enumerate(batch_prompts):
            responses = []
            responses_with_special = []  # 包含特殊token的版本
            for j in range(num_samples):
                idx = i * num_samples + j
                if idx < len(outputs):
                    # 左 padding 后，从统一的 input_len 位置截取生成内容
                    resp = tokenizer.decode(outputs[idx][input_len:], skip_special_tokens=True).strip()
                    resp_special = tokenizer.decode(outputs[idx][input_len:], skip_special_tokens=False)
                    responses.append(resp)
                    responses_with_special.append(resp_special)
            all_pairs.append((p, responses, responses_with_special))
    
    # 恢复原始 padding 方向
    tokenizer.padding_side = original_padding_side
    
    return all_pairs


def _parse_judge_json(text):
    """从 Judge 输出提取 JSON 并解析三个维度（0/1 二值）"""
    for pattern in [r"```(?:json)?\s*(\{[^`]*\})\s*```", r"(\{[^{}]*\})"]:
        for raw in re.findall(pattern, text, re.DOTALL | re.IGNORECASE):
            try:
                d = json.loads(raw.strip())
                # 二值化：>= 1 认为通过
                result = {k: 1 if d.get(k, d.get(k.replace("_", " "), 0)) >= 1 else 0 for k in DIMENSIONS}
                if len(result) == 3:
                    return result
            except:
                continue
    return None


def _judge_one(prompt, response, api_key, model="deepseek-chat"):
    """单次 Judge 请求（DeepSeek API）"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=prompt, response=response)}],
            stream=False
        )
        out = completion.choices[0].message.content
        return _parse_judge_json(out), None
    except Exception as e:
        return None, str(e)[:200]


def run_judge(pairs, api_key=None, model="deepseek-chat", return_details=False, max_workers=10):
    """Judge 评测（DeepSeek API）"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if not pairs:
        return ({}, []) if return_details else {}
    
    # 从环境变量读取 API key
    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    
    # 自动检测 num_samples（从第一个 pair 的 responses 数量）
    # pairs 现在是 (prompt, responses, responses_with_special) 的格式
    n = len(pairs[0][1]) if pairs[0][1] else 1
    
    tasks = [(i, j, p, r) for i, (p, rs, _) in enumerate(pairs) for j, r in enumerate(rs)]
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_judge_one, p, r, api_key, model): (i, j) 
                         for i, j, p, r in tasks}
        for future in as_completed(future_to_idx):
            i, j = future_to_idx[future]
            results[(i, j)] = future.result()[0]
    
    dim_data = {d: {"scores": [], "pass_per_prompt": []} for d in DIMENSIONS}
    details = []
    
    for i, (prompt, responses, responses_with_special) in enumerate(pairs):
        passed_any = {d: False for d in DIMENSIONS}
        judge_results = []
        for j in range(len(responses)):
            parsed = results.get((i, j))
            judge_results.append(parsed)
            # 关键修复：即使解析失败，也要添加 0 分，保持 avg 和 pass 分母一致
            for d in DIMENSIONS:
                v = parsed.get(d, 0) if parsed else 0
                dim_data[d]["scores"].append(float(v))
                if v == 1:
                    passed_any[d] = True
        for d in DIMENSIONS:
            dim_data[d]["pass_per_prompt"].append(passed_any[d])
        if return_details:
            details.append({
                "prompt": prompt, 
                "responses": responses,
                "responses_with_special_tokens": responses_with_special,  # 添加包含特殊token的版本
                "judge_results": judge_results, 
                "pass_any": passed_any
            })
    
    metrics = {}
    for d in DIMENSIONS:
        scores = dim_data[d]["scores"]
        pass_pp = dim_data[d]["pass_per_prompt"]
        metrics[f"{d}_avg{n}"] = sum(scores) / len(scores) if scores else 0.0
        metrics[f"{d}_pass{n}"] = sum(pass_pp) / len(pairs) if pairs else 0.0
    metrics[f"mean_avg{n}"] = sum(metrics[f"{d}_avg{n}"] for d in DIMENSIONS) / 3
    metrics[f"mean_pass{n}"] = sum(metrics[f"{d}_pass{n}"] for d in DIMENSIONS) / 3
    
    return (metrics, details) if return_details else metrics


def run_judge_async(pairs, api_key=None, model="deepseek-chat",
                     output_file=None, swanlab_log_fn=None, global_step=None, max_workers=10):
    """异步 Judge：后台执行，不阻塞训练（DeepSeek API）"""
    import threading
    
    def _background():
        try:
            metrics, details = run_judge(pairs, api_key, model, return_details=True, max_workers=max_workers)
            
            # 从 metrics 推断 num_samples（避免重复计算）
            n = 1
            for key in metrics:
                if key.startswith("fluency_avg"):
                    n = int(key.replace("fluency_avg", ""))
                    break
            
            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    for item in details:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if swanlab_log_fn:
                swanlab_log_fn({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)
            success = sum(1 for d in details for j in d["judge_results"] if j)
            total = sum(len(d["responses"]) for d in details)
            if success == 0:
                print(f"  [judge] WARNING: 所有指标为 0！成功 {success}/{total} 次，检查 Judge 是否正常")
            print(f"  [judge] step={global_step} fluency={metrics[f'fluency_avg{n}']:.3f}/{metrics[f'fluency_pass{n}']:.3f} "
                  f"factuality={metrics[f'factuality_avg{n}']:.3f}/{metrics[f'factuality_pass{n}']:.3f} "
                  f"if={metrics[f'instruction_following_avg{n}']:.3f}/{metrics[f'instruction_following_pass{n}']:.3f} "
                  f"mean={metrics[f'mean_avg{n}']:.3f}/{metrics[f'mean_pass{n}']:.3f}")
        except Exception as e:
            print(f"  [judge] 后台评测异常: {e}")
    
    threading.Thread(target=_background, daemon=True).start()
