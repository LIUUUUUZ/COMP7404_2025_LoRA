import os
import time
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
)
from peft import (
    PeftModel
)
from model import set_seed, load_model, train, evaluate
from dataloader import load_and_preprocess_data


def run_experiments(args):
    """根据参数运行实验."""
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    train_dataloader, eval_dataloader, num_labels, metric_name = load_and_preprocess_data(args)

    if args.mode == "train":
        if args.method == "lora" and args.compare_ranks:
            run_lora_experiments(args, train_dataloader, eval_dataloader, num_labels, metric_name)
        else:
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


def run_lora_experiments(args, train_dataloader, eval_dataloader, num_labels, metric_name):
    """测试不同 LoRA Rank 对训练时间和评估性能的影响"""
    lora_ranks = [2, 3, 4, 8, 16, 32]  # 选择不同的 LoRA 秩
    results = []

    for r in lora_ranks:
        args.lora_rank = r
        print(f"运行 LoRA rank={r} 的实验...")

        set_seed(args.seed)
        model = load_model(args, num_labels)

        # 记录训练时间
        start_time = time.time()
        train_time = train(args, model, train_dataloader, eval_dataloader, metric_name)
        eval_result, eval_time = evaluate(args, model, eval_dataloader, metric_name)
        end_time = time.time()

        print(f"LoRA rank={r} 训练时间: {train_time:.2f} 秒")
        print(f"LoRA rank={r} 评估结果: {eval_result}")

        results.append({
            "LoRA Rank": r,
            "Train Time (s)": train_time,
            "Eval Time (s)": eval_time,
            "Metric": metric_name,
            "Score": list(eval_result.values())[0]
        })

    # 保存实验结果
    df = pd.DataFrame(results)
    df.to_csv("lora_rank_comparison.csv", index=False)
    print("LoRA Rank 影响实验结果已保存到 lora_rank_comparison.csv")