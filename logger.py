import os
import logging
from datetime import datetime


def setup_logger(args):
	"""
    设置日志记录器,同时输出到控制台和文件(如果save_log为True)
    
    Args:
        args: 解析的参数对象,包含output_dir和save_log
        
    Returns:
        logger: 配置好的日志记录器
    """
	# 创建logger对象
	logger = logging.getLogger('COMP_7404_LoRA')
	logger.setLevel(logging.INFO)

	# 创建格式化器
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

	# 总是添加控制台处理器
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)

	# 如果save_log为True，添加文件处理器
	if args.save_log:
		# 确保输出目录存在
		os.makedirs(args.output_dir, exist_ok=True)

		# 创建日志文件名，包含时间戳
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		log_filename = os.path.join(
			args.output_dir,
			f"{args.dataset}_{args.method}_rank{args.lora_rank if args.method == 'lora' else ''}_{timestamp}.log"
		)

		# 创建文件处理器
		file_handler = logging.FileHandler(log_filename, encoding='utf-8')
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

		# 记录所有参数信息到日志文件开头
		logger.info("=" * 50)
		logger.info("实验参数配置:")
		logger.info("-" * 50)
		# 获取所有参数
		args_dict = vars(args)
		# 按字母顺序排序参数
		for key in sorted(args_dict.keys()):
			logger.info(f"{key}: {args_dict[key]}")
		logger.info("=" * 50)
		logger.info("\n")  # 添加空行分隔参数和后续日志

		logger.info(f"日志文件将保存到: {log_filename}")

	return logger
