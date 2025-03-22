from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)
from datasets import load_dataset
from torch.utils.data import DataLoader

def load_and_preprocess_data(args):
    """加载并预处理数据集."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 设置填充标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == "snli":
        dataset = load_dataset("snli")
        # SNLI有3个标签
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

    elif args.dataset == "ag_news":
        dataset = load_dataset('ag_news')
        # ag_news也类似于MNLI，具有4个标签
        num_labels = 4
        metric_name = "accuracy"

        def preprocess_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            )

    else:
        raise ValueError(f"不支持的数据集 {args.dataset}.")

    # 应用预处理
    if args.dataset == "ag_news":
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["text"])
    elif args.dataset == "stsb":
        tokenized_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["sentence1", "sentence2"])
    else:
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
            # input_columns=["idx"]

        )
    # **删除 idx 列，避免传递给模型**
    for split in tokenized_datasets.keys():
        if "idx" in tokenized_datasets[split].column_names:
            tokenized_datasets[split] = tokenized_datasets[split].remove_columns(["idx"])

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

    if args.dataset == "snli" or args.dataset == "stsb":
        eval_dataloader = DataLoader(
            tokenized_datasets["validation"],
            batch_size=args.eval_batch_size,
            collate_fn=data_collator,
        )
    elif args.dataset == "ag_news":
        eval_dataloader = DataLoader(
            tokenized_datasets["test"],
            batch_size=args.eval_batch_size,
            collate_fn=data_collator,
        )


    return train_dataloader, eval_dataloader, num_labels, metric_name