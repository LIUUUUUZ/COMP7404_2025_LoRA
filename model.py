from peft import (
	get_peft_model,
	LoraConfig,
	TaskType,
	PeftType,
	PeftModel
)
import os
import time
import torch
import pandas as pd
import numpy as np
import random
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	get_linear_schedule_with_warmup,
	DataCollatorWithPadding
)
from adapters import AutoAdapterModel, AdapterConfig
from tqdm import tqdm


def set_seed(seed):
	"""为可重复性设置随机种子."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def load_model(args, num_labels):
	"""根据方法加载模型, 并根据需要应用LoRA或Adapter."""
	if args.method == "adapter":
		model = AutoAdapterModel.from_pretrained(args.model_name)
	else:
		model = AutoModelForSequenceClassification.from_pretrained(
			args.model_name,
			num_labels=num_labels
		)

	if args.method == "lora":

		lora_config = configure_lora(model, args)

		print("LoRA配置:")
		print(lora_config)

		# 应用LoRA配置
		model = get_peft_model(model, lora_config)

		# 打印可训练参数信息
		model.print_trainable_parameters()

	elif args.method == "adapter":
		# 使用Adapter配置
		model = configure_adapter(model, args)

	elif args.method == "full_ft":
		for param in model.parameters():
			param.requires_grad = True
		trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		all_params = sum(p.numel() for p in model.parameters())

		print(f"Full fine-tuning - 可训练参数: {trainable_params} ({trainable_params / all_params:.2%})")

	else:
		raise ValueError(f"不支持的方法 {args.method}.")

	return model


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
			lora_weight = (lora_B @ lora_A) * (
				module.scaling if hasattr(module, 'scaling') else module.lora_alpha / module.r)

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
	if torch.cuda.is_available():
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")

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
			all_labels.extend(batch["labels"].float().cpu().numpy())

	all_preds = np.array(all_preds)
	all_labels = np.array(all_labels)

	# print(predictions)
	# print(batch["labels"])

	metrics = compute_metrics(all_preds, all_labels, metric_name)
	eval_time = time.time() - start_time

	return metrics, eval_time


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
	if torch.cuda.is_available():
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")

	# 计时：模型加载到设备
	if args.time_count:
		model_load_start = time.time()

	model.to(device)

	if args.time_count:
		model_load_time = time.time() - model_load_start
		print(f"模型加载到{device}耗时: {model_load_time:.2f}秒")

	# 计算可训练参数
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	all_params = sum(p.numel() for p in model.parameters())

	print(f"可训练参数: {trainable_params} ({trainable_params / all_params:.2%} of all parameters)")

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
		print(f"Epoch {epoch + 1}/{args.num_epochs}")
		epoch_start_time = time.time()
		progress_bar = tqdm(train_dataloader, desc="训练")

		# 计时变量
		if args.time_count:
			forward_time = 0
			backward_time = 0
			optimizer_time = 0
			data_transfer_time = 0

		for batch in progress_bar:
			# 计时：数据传输
			if args.time_count:
				data_transfer_start = time.time()

			batch = {k: v.to(device) for k, v in batch.items()}

			if args.time_count:
				data_transfer_time += time.time() - data_transfer_start
				# 计时：前向传播
				forward_start = time.time()

			outputs = model(**batch)

			if args.time_count:
				forward_time += time.time() - forward_start
				# 计时：反向传播
				backward_start = time.time()

			loss = outputs.loss
			loss.backward()

			if args.time_count:
				backward_time += time.time() - backward_start
				# 计时：优化器步骤
				optimizer_start = time.time()

			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

			if args.time_count:
				optimizer_time += time.time() - optimizer_start

		# progress_bar.set_postfix({"loss": loss.item()})

		# 输出本轮次的计时详情
		if args.time_count:
			print(f"Epoch {epoch + 1} 计时详情:")
			print(f"  数据传输总耗时: {data_transfer_time:.2f}秒")
			print(f"  前向传播总耗时: {forward_time:.2f}秒")
			print(f"  反向传播总耗时: {backward_time:.2f}秒")
			print(f"  优化器步骤总耗时: {optimizer_time:.2f}秒")

		epoch_time = time.time() - epoch_start_time
		total_train_time += epoch_time
		print(f"Epoch {epoch + 1} 完成，耗时 {epoch_time:.2f} 秒。")

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


def configure_lora(model, args):
	# 配置LoRA
	target_modules = []
	if "q" in args.lora_target:
		target_modules.append("query")
	if "k" in args.lora_target:
		target_modules.append("key")
	if "v" in args.lora_target:
		target_modules.append("value")
	if "o" in args.lora_target:
		# target_modules.append("attention.output.dense")
		target_modules.append("output.dense")
	# target_modules.append("intermediate.dense")

	# target_modules.append("classifier.dense")
	# target_modules.append("classifier.out_proj")

	lora_config = LoraConfig(
		r=args.lora_rank,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		target_modules=target_modules,
		bias="none",
		task_type=TaskType.SEQ_CLS,
		inference_mode=False,
		init_lora_weights=True,
		modules_to_save=None  # 不需要特别保存模块，因为我们已经在target_modules中包含了分类器
	)
	return lora_config


def configure_adapter(model, args):
	"""配置Adapter参数"""
	model.add_classification_head(
		head_name="task_head",
		num_labels=num_labels
	)
	# 显式解冻分类头
	for param in model.heads["task_head"].parameters():
		param.requires_grad = True

	adapter_config = AdapterConfig.load(
		"houlsby",
		# 降维因子越大, 参数量越小
		reduction_factor=1024 / args.adapter_bottleneck,
		target_modules=[
			"attention.output",
			"output.output"
		],
		add_layer_norm=args.adapter_LN
	)

	model.add_adapter("adapter", adapter_config)
	model.train_adapter("adapter")
	model.set_active_adapters("adapter")
	return model
