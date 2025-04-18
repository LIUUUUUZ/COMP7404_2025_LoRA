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
from tqdm import tqdm
from parser import parse_args
from peft import (
	get_peft_model,
	LoraConfig,
	TaskType,
	PeftType,
	PeftModel
)
from experiments import run_experiments


def main():
	args = parse_args()

	start_time = time.time()

	run_experiments(args)

	total_time = time.time() - start_time
	print(f"总执行时间: {total_time:.2f} 秒")


if __name__ == "__main__":
	main()
