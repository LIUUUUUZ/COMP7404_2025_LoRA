import os
import time
import torch
import numpy as np
import random
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from parser import parse_args
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    AdapterConfig,
    PeftType,
    PeftModel
)

def set_seed(seed):
    """为可重复性设置随机种子."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_preprocess_data(args):
    """加载并预处理数据集."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    
    if args.dataset == "snli":
        dataset = load_dataset("snli")
        # SNLI也类似于MNLI，具有3个标签
        num_labels = 3
        metric_name = "accuracy"
        
        # 过滤掉标签无效的样本（例如标签为 -1）
        dataset["train"] = dataset["train"].filter(lambda x: x["label"] in [0, 1, 2])
        dataset["validation"] = dataset["validation"].filter(lambda x: x["label"] in [0, 1, 2])
        def preprocess_function(examples):
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            )
    
    elif args.dataset == "stsb":
        dataset = load_dataset("glue", "stsb")
        # STSB是一个回归任务
        num_labels = 1
        metric_name = "pearson"
        
        def preprocess_function(examples):
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            )
    
    else:
        raise ValueError(f"不支持的数据集 {args.dataset}.")
    
    # 应用预处理
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # 将标签列添加回去
    if args.dataset == "snli":
        tokenized_datasets = tokenized_datasets.map(
            lambda examples, indices: {"labels": dataset["train"][indices]["label"]},
            with_indices=True,
            batched=True,
            input_columns=["idx"]
            # input_columns=["idx"]
        )
    elif args.dataset == "stsb":
        tokenized_datasets = tokenized_datasets.map(
            lambda examples, indices: {"labels": dataset["train"][indices]["label"]},
            with_indices=True,
            batched=True,
            input_columns=["idx"]
            # input_columns=["idx"]
        )
    
    # 设置PyTorch格式
    tokenized_datasets.set_format("torch")
    
    # 创建DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    
    if args.dataset == "mnli":
        eval_dataloader = DataLoader(
            tokenized_datasets["validation_matched"],
            batch_size=args.eval_batch_size,
            collate_fn=data_collator,
        )
    else:
        eval_dataloader = DataLoader(
            tokenized_datasets["validation"],
            batch_size=args.eval_batch_size,
            collate_fn=data_collator,
        )
    
    return train_dataloader, eval_dataloader, num_labels, metric_name

def load_model(args, num_labels):
    """根据方法加载模型, 并根据需要应用LoRA或Adapter."""
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_labels
    )
    
    if args.method == "lora":
        # 配置LoRA
        target_modules = []
        for target in args.lora_target:
            if target == "q":
                target_modules.append("query")
            elif target == "k":
                target_modules.append("key")
            elif target == "v":
                target_modules.append("value")
            elif target == "o":
                target_modules.append("output")
        
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        
        # 应用LoRA配置
        model = get_peft_model(model, lora_config)
        
    elif args.method == "adapter":
        # 配置Adapter
        adapter_config = AdapterConfig(
            peft_type=PeftType.ADAPTER,
            task_type=TaskType.SEQ_CLS,
            adapter_size=args.lora_rank * 4,  # 使用rank*4作为adapter大小，以进行公平的比较
            r=args.lora_rank,
            bias="none"
        )
        
        # 应用Adapter配置
        model = get_peft_model(model, adapter_config)
        
    elif args.method == "full_ft":
        # 对于全微调，所有参数已经是可训练的
        pass
    
    else:
        raise ValueError(f"不支持的方法 {args.method}.")
    
    return model

def compute_metrics(preds, labels, metric_name):
    """计算评估指标."""
    if metric_name == "accuracy":
        return {"accuracy": (preds == labels).mean()}
    elif metric_name == "pearson":
        from scipy.stats import pearsonr
        return {"pearson": pearsonr(preds, labels)[0]}
    else:
        raise ValueError(f"不支持的指标 {metric_name}.")

def train(args, model, train_dataloader, eval_dataloader, metric_name):
    """训练模型."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 计算可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params} ({trainable_params/all_params:.2%} of all parameters)")
    
    # 准备优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练循环
    model.train()
    total_train_time = 0
    best_metric = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        epoch_start_time = time.time()
        
        progress_bar = tqdm(train_dataloader, desc="训练")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item()})
        
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
        print(f"Epoch {epoch+1} 完成，耗时 {epoch_time:.2f} 秒。")
        
        # 评估
        eval_result, eval_time = evaluate(args, model, eval_dataloader, metric_name, predict=False)
        print(f"评估完成，耗时 {eval_time:.2f} 秒。")
        print(f"评估结果: {eval_result}")
        
        # 保存最佳模型
        if args.save_model:
            current_metric = list(eval_result.values())[0]
            if current_metric > best_metric:
                best_metric = current_metric
                output_dir = os.path.join(
                    args.output_dir, 
                    f"{args.dataset}_{args.method}_rank{args.lora_rank if args.method == 'lora' else ''}"
                )
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                print(f"模型已保存到 {output_dir}")
    
    # 如果需要输出矩阵
    if args.output_matrices and args.method == "lora":
        output_lora_matrices(model, args)
    
    return total_train_time

def output_lora_matrices(model, args):
    """输出预训练矩阵和LoRA微调后的矩阵."""
    print("正在保存预训练矩阵和LoRA微调后的矩阵...")
    matrices_dir = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.method}_rank{args.lora_rank}_matrices"
    )
    os.makedirs(matrices_dir, exist_ok=True)
    
    # 用于存储所有矩阵信息的列表
    all_matrices_info = []
    
    # 遍历所有LoRA层
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # 获取原始权重
            original_weight = module.weight.detach().cpu().numpy()
            
            # 获取LoRA权重
            lora_A = module.lora_A.detach().cpu().numpy()
            lora_B = module.lora_B.detach().cpu().numpy()
            
            # 计算LoRA增量
            lora_weight = (lora_B @ lora_A) * (module.scaling if hasattr(module, 'scaling') else module.lora_alpha / module.r)
            
            # 保存矩阵
            original_path = os.path.join(matrices_dir, f"{name.replace('.', '_')}_original.npy")
            lora_A_path = os.path.join(matrices_dir, f"{name.replace('.', '_')}_lora_A.npy")
            lora_B_path = os.path.join(matrices_dir, f"{name.replace('.', '_')}_lora_B.npy")
            lora_weight_path = os.path.join(matrices_dir, f"{name.replace('.', '_')}_lora_weight.npy")
            
            np.save(original_path, original_weight)
            np.save(lora_A_path, lora_A)
            np.save(lora_B_path, lora_B)
            np.save(lora_weight_path, lora_weight)
            
            # 收集矩阵信息
            matrix_info = {
                "name": name,
                "shape_original": original_weight.shape,
                "shape_lora_A": lora_A.shape,
                "shape_lora_B": lora_B.shape,
                "shape_lora_weight": lora_weight.shape,
                "path_original": original_path,
                "path_lora_A": lora_A_path,
                "path_lora_B": lora_B_path,
                "path_lora_weight": lora_weight_path
            }
            all_matrices_info.append(matrix_info)
    
    # 保存矩阵信息索引到JSON文件
    import json
    with open(os.path.join(matrices_dir, "matrices_index.json"), "w") as f:
        json.dump(all_matrices_info, f, indent=2)
            
    print(f"矩阵已保存到 {matrices_dir}")
    print(f"矩阵索引文件已保存到 {os.path.join(matrices_dir, 'matrices_index.json')}")

def evaluate(args, model, eval_dataloader, metric_name, predict=False):
    """评估模型."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.eval()
    start_time = time.time()
    
    all_preds = []
    all_labels = []
    
    # 如果在预测模式下，限制批次数量
    if predict:
        eval_dataloader = list(eval_dataloader)[:args.test_iters]
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="评估"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            
            if metric_name == "accuracy":
                predictions = outputs.logits.argmax(dim=-1)
            else:  # 回归
                predictions = outputs.logits.squeeze()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = compute_metrics(all_preds, all_labels, metric_name)
    eval_time = time.time() - start_time
    
    return metrics, eval_time

def run_experiments(args):
    """根据参数运行实验."""
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    train_dataloader, eval_dataloader, num_labels, metric_name = load_and_preprocess_data(args)
    
    if args.mode == "train":
        # 加载模型
        model = load_model(args, num_labels)
        
        # 训练模型
        train_time = train(args, model, train_dataloader, eval_dataloader, metric_name)
        print(f"总训练时间: {train_time:.2f} 秒")
        
        # 评估模型（用于预测时间）
        _, predict_time = evaluate(args, model, eval_dataloader, metric_name, predict=True)
        print(f"预测 {args.test_iters} 批次的时间: {predict_time:.2f} 秒")
        
    elif args.mode == "eval":
        # 加载保存的模型
        model_path = os.path.join(
            args.output_dir, 
            f"{args.dataset}_{args.method}_rank{args.lora_rank if args.method == 'lora' else ''}"
        )
        
        if args.method in ["lora", "adapter"]:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
            model = PeftModel.from_pretrained(model, model_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # 评估模型
        metrics, eval_time = evaluate(args, model, eval_dataloader, metric_name)
        print(f"评估结果: {metrics}")
        print(f"评估时间: {eval_time:.2f} 秒")
        
        # 测量预测时间
        _, predict_time = evaluate(args, model, eval_dataloader, metric_name, predict=True)
        print(f"预测 {args.test_iters} 批次的时间: {predict_time:.2f} 秒")
        
        # 如果需要输出矩阵
        if args.output_matrices and args.method == "lora":
            output_lora_matrices(model, args)

def main():
    args = parse_args()
    
    if args.time_count:
        start_time = time.time()
    
    run_experiments(args)
    
    if args.time_count:
        total_time = time.time() - start_time
        print(f"总执行时间: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()