import os
import json
import wandb
import argparse

from importlib import import_module
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data_utils import WOSDataset, get_examples_from_dialogues, load_dataset, set_seed


def data_loading(args):
    # Data Loading
    train_data_file = f"{args.data_dir}/train_dials.json"
    train_data, dev_data, dev_labels = load_dataset(train_data_file)

    train_examples = get_examples_from_dialogues(
        train_data, user_first=False, dialogue_level=False
    )
    dev_examples = get_examples_from_dialogues(
        dev_data, user_first=False, dialogue_level=False
    )

    return train_examples, dev_examples

def extract_features(args, train_examples, dev_examples):
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
    
    # Define Tokenizer
    tokenizer_module = getattr(
        import_module("transformers"), f"{args.tokenizer_name}Tokenizer"
    )
    tokenizer = tokenizer_module.from_pretrained(args.model_name_or_path)

    # Define Preprocessor
    processor_module = getattr(
        import_module("preprocessor"), f"{args.dst}Preprocessor"
    )
    processor = processor_module(slot_meta, tokenizer)

    # Extracting Featrues
    train_features = processor.convert_examples_to_features(train_examples)
    dev_features = processor.convert_examples_to_features(dev_examples)

    return slot_meta, tokenizer, processor, train_features, dev_features

def get_data_loader(args, processor, train_features, dev_features):
    train_data = WOSDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=processor.collate_fn,
    )
    print("# train:", len(train_data))

    dev_data = WOSDataset(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.eval_batch_size,
        sampler=dev_sampler,
        num_workers=4,
        collate_fn=processor.collate_fn,
    )
    print("# dev:", len(dev_data))

    return train_loader, dev_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="wandb에 저장할 project name (본인 이름 or 닉네임으로 지정)",
    )
    parser.add_argument("--model_fold", type=str, required=True, help="model 폴더명")
    parser.add_argument("--data_dir", type=str, default="../input/data/train_dataset")
    parser.add_argument("--model_dir", type=str, default="../models")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-8)
    parser.add_argument(
        "--max_lr",
        type=float,
        help="Using CustomizedCosineAnnealingWarmRestarts, Limit the maximum of learning_rate",
        default=5e-6,
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
   
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Name of Optimizer (Ex. AdamW, Adam, SGD, AdamP ...)",
        default="Adam",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        help="Name of Scheduler (Ex. linear, custom, cosine, plateau ...)",
        default="custom",
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        help="Determine max_lr of Next Cycle Sequentially, When Using CustomizedCosineScheduler",
        default=0.9,
    )
    parser.add_argument(
        "--first_cycle_ratio",
        type=float,
        help="Determine Num of First Cycle Epoch When Using CustomizedCosineScheduler (first_cycle = t_total * first_cycle_ratio)",
        default=0.05,
    )

    parser.add_argument(
        "--dst",
        type=str,
        help="Model Name For DST Task (EX. TRADE, SUMBT)",
        default="TRADE",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        help="Not Using AutoTokenizer, Tokenizer Name For Loading (EX. Bert, Electra, XLMRoberta, etc..)",
        default="Electra",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Not Using AutoModel, Model Name For Loading set_subword_embedding in model.py (EX. Bert, Electra, XLMRoberta, etc..)",
        default="Electra",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        default="monologg/koelectra-base-v3-discriminator",
    )

    # Model Specific Argument
    parser.add_argument("--hidden_size", type=int, help="GRU의 hidden size", default=768)
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="vocab size, subword vocab tokenizer에 의해 특정된다",
        default=None,
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--proj_dim",
        type=int,
        help="만약 지정되면 기존의 hidden_size는 embedding dimension으로 취급되고, proj_dim이 GRU의 hidden_size로 사용됨. hidden_size보다 작아야 함.",
        default=None,
    )
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    args = parser.parse_args()
    args.dst = args.dst.upper()
    os.makedirs(f"{args.model_dir}/{args.model_fold}", exist_ok=True)

    # wandb init
    
    wandb.init(project=args.project_name)
    wandb.run.name = f"{args.model_fold}"
    wandb.config.update(args)
    
    # random seed 고정
    set_seed(args.seed)

    train_examples, dev_examples = data_loading(args)
    slot_meta, tokenizer, processor, train_features, dev_features = extract_features(args, train_examples, dev_examples)
    train_loader, dev_loader = get_data_loader(args, processor, train_features, dev_features)
    
    train_module = getattr(
        import_module(f"train_{args.dst}"), "train"
    )
    train_module(args, tokenizer, processor, slot_meta, train_loader, dev_loader)
