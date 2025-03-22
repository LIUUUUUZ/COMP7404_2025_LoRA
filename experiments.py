import os
import time
import pandas as pd
from transformers import (
	AutoModelForSequenceClassification,
)
from adapters import AutoAdapterModel
from peft import (
	PeftModel
)
from model import set_seed, load_model, train, evaluate
from dataloader import load_and_preprocess_data
import json
import numpy as np
import torch
import gc

def run_experiments(args):
	"""根据参数运行实验."""
	set_seed(args.seed)

	# 创建输出目录
	os.makedirs(args.output_dir, exist_ok=True)


	# 加载数据
	if not args.test_weight_matrix_comb_with_rank:
		train_dataloader, eval_dataloader, num_labels, metric_name = load_and_preprocess_data(args)

	if args.mode == "train":
		if args.method == "lora" and args.compare_ranks:
			run_lora_experiments(args, train_dataloader, eval_dataloader, num_labels, metric_name)

		elif args.test_weight_matrix_comb_with_rank:
			run_weight_matrix_comb_with_rank_experiment(args)

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

		if args.method in ["lora"]:
			model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
			model = PeftModel.from_pretrained(model, model_path)
		elif args.method == "adapter":
			model = AutoAdapterModel.from_pretrained(model_path)
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

def numpy_to_python(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    return obj

def run_weight_matrix_comb_with_rank_experiment(args):
	dataset_list = ["stsb", "ag_news"]
	lora_rank = [1, 2, 4, 8, 32]
	lora_target = ["k","v","q","o","kv","qv","kvq","kvqo"]

	results = []
	'''
	每个实验记录：{
	dataset:,
	lora_rank:,
	lora_target:,
	parameters:,
	total_training_time:,
	eval_time:,
	metric_name:,
	result:,
	}
	'''

	for dataset in dataset_list:
		args.dataset = dataset
		print(f"\n开始数据集 {args.dataset} 的实验...")
		train_dataloader, eval_dataloader, num_labels, metric_name = load_and_preprocess_data(args)

		for rank in lora_rank:
			args.lora_rank = rank
			for target in lora_target:
				print(f"\n运行实验: 数据集={args.dataset}, LoRA rank={args.lora_rank}, 目标矩阵={args.lora_target}")
				# 实验开始时清除上一个模型的缓存
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
				gc.collect()  # 清理内存中的未使用对象
				
				# 将字符串转换为列表形式的目标矩阵
				if isinstance(target, str):
					args.lora_target = list(target)
				else:
					args.lora_target = target
					
				# 加载模型
				model = load_model(args, num_labels)

				
				# 训练模型并记录训练时间
				train_time = train(args, model, train_dataloader, eval_dataloader, metric_name)

				
				# 计算可训练参数数量
				trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
				
				# 评估模型
				metrics, eval_time = evaluate(args, model, eval_dataloader, metric_name)
				
				# 记录实验结果
				result = {
					"dataset": dataset,
					"lora_rank": rank,
					"lora_target": "".join(args.lora_target),
					"parameters": trainable_params,
					"total_train_time": train_time ,
					"eval_time": eval_time,
					"metric_name": metric_name,
					"result": list(metrics.values())[0],
					"num_epochs": args.num_epochs
				}
				results.append(result)
				
				print(f"实验结果: {metrics}")
				print(f"评估时间: {eval_time:.2f} 秒")
				print(f"可训练参数数量: {trainable_params}")

				with open("lora_experiments_results.json", "w") as f:
					json.dump(results, f, ensure_ascii=False, indent=4, default=numpy_to_python)

	# 保存实验结果
	df = pd.DataFrame(results)
	output_file = "lora_weight_matrix_rank_experiments.csv"
	df.to_csv(output_file, index=False)
	print(f"\n实验结果已保存到 {output_file}")	