import argparse


def get_parser():
	parser = argparse.ArgumentParser(description="COMP_7404_2025_spring_LoRA项目")

	# 数据集部分（若不完善请修改删减增补）
	parser.add_argument("--dataset", type=str, choices=["snli", "stsb", "ag_news"], default="snli",
	                    help="数据集名称")
	parser.add_argument("--max_seq_length", type=int, default=128,
	                    help="最大序列长度")
	parser.add_argument("--batch_size", type=int, default=32,
	                    help="训练Batch大小")
	parser.add_argument("--eval_batch_size", type=int, default=64,
	                    help="评估Batch大小")

	# 模型基础部分（若不完善请修改删减增补）
	parser.add_argument("--model_name", type=str, default="roberta-large", choices=["roberta-large", "roberta-base"],
	                    help="底层模型名称")
	parser.add_argument("--method", type=str, choices=["lora", "adapter", "full_ft"], default="lora",
	                    help="微调方法名称")
	parser.add_argument("--learning_rate", type=float, default=1e-4,
	                    help="训练的学习率")
	parser.add_argument("--weight_decay", type=float, default=0.01,
	                    help="正则化的权重衰减")
	parser.add_argument("--num_epochs", type=int, default=5,
	                    help="训练的轮数")
	parser.add_argument("--warmup_ratio", type=float, default=0.06,
	                    help="学习率调度器的warmup步骤比例, 本次实验中不考虑")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
	                    help="在反向传播前累积的更新步骤数, 本次实验中不考虑")
	parser.add_argument("--seed", type=int, default=42,
	                    help="随机种子")

	parser.add_argument("--time_count", action="store_true",
	                    help="是否计时(每个阶段的详细时间)")
	parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train",
	                    help="选择训练或评估模式")

	# LoRA微调参数（若不完善请修改删减增补）
	parser.add_argument("--lora_rank", type=int, default=3,
	                    help="LoRA层Rank的大小")
	parser.add_argument("--lora_alpha", type=float, default=16,
	                    help="LoRA层Alpha缩放因子的大小, 本次实验中不考虑, 默认为16")
	parser.add_argument("--lora_dropout", type=float, default=0.1,
	                    help="LoRA层Dropout概率, 本次实验中不考虑, 默认为0.1")
	parser.add_argument("--test_iters", type=int, default=100,
	                    help="测试预测时的iter数")


	# 使用方法为 python xxxx.py --lora_target q v o,必须要有至少一个, 平时默认全部使用
	parser.add_argument("--lora_target", type=str, nargs='+',
	                    default=["q", "k", "v", "o"],
	                    help="LoRA运用的attention目标矩阵的组合")

	# Adapter配置参数
	parser.add_argument("--adapter_LN", type=bool, default="true",
	                    help="是否添加LN层")
	parser.add_argument("--adapter_bottleneck", type=int, default=3,
	                    help="adapter隐藏层维度")

	# 模型储存相关（若不完善请修改删减增补）
	parser.add_argument("--output_dir", type=str, default="./outputs",
	                    help="保存输出的目录")
	parser.add_argument("--save_model", action="store_true",
	                    help="是否保存训练好的模型")

	parser.add_argument("--output_matrices", action="store_true",
	                    help="是否输出预训练矩阵和LoRA微调后的矩阵")
	
	# 实验相关
	parser.add_argument("--compare_ranks", action="store_true", help="是否对比不同 LoRA rank 的影响")
	parser.add_argument("--test_weight_matrix_comb_with_rank", action="store_true", help="lora在不同权重矩阵和秩下的结果测试")

	return parser


def parse_args():
	parser = get_parser()
	args = parser.parse_args()
	return args
