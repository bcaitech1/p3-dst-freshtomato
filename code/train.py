import os
import json
import wandb
import argparse

from data_utils import set_seed
from importlib import import_module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="wandb에 저장할 project name (본인 이름 or 닉네임으로 지정)"
    )
    parser.add_argument("--model_fold", type=str, required=True, help="model 폴더명")
    parser.add_argument("--data_dir", type=str, default="../input/data/train_dataset")
    parser.add_argument("--model_dir", type=str, default="../models")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument(
        "--max_lr", 
        type=float, 
        help="Using CustomizedCosineAnnealingWarmRestarts, Limit the maximum of lr", 
        default=1e-4
        )
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
   
    parser.add_argument("--optimizer", type=str, help="Name of Optimizer (AdamW, Adam, SGD, AdamP ...)", default="AdamW")
    parser.add_argument("--scheduler", type=str, help="Name of Scheduler (linear, custom, cosine, plateau ...)", default="custom")
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        help="Determine max_lr of Next Cycle Sequentially, When Using CustomizedCosineScheduler",
        default=0.8,
    )
    parser.add_argument(
        "--first_cycle_ratio",
        type=float,
        help="Determine Num of First Cycle Epoch When Using CustomizedCosineScheduler (first_cycle = t_total * first_cycle_ratio)",
        default=0.25,
    )

    parser.add_argument(
        "--dst",
        type=str,
        help="Model Name For DST Task (EX. TRADE, SUMBT)",
        default="SUMBT",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Pre-trained model name to load from HuggingFace. It also will be used for loading corresponding tokenizer.(EX. Bert, Electra, etc..)",
        default="Electra"
    )
    
    parser.add_argument(
        "--pretrained_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        default="monologg/koelectra-base-v3-discriminator",
    )

    # Model Specific Argument
    parser.add_argument("--hidden_size", type=int, help="GRU의 hidden size", default=768) # TRADER, SUMBT
    parser.add_argument("--num_rnn_layers", type=int, help="Number of GRU layers", default=1) # TRADER, SUMBT
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="vocab size, subword vocab tokenizer에 의해 특정된다",
        default=None,
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.0)
    parser.add_argument(
        "--proj_dim",
        type=int,
        help="만약 지정되면 기존의 hidden_size는 embedding dimension으로 취급되고, proj_dim이 GRU의 hidden_size로 사용됨. hidden_size보다 작아야 함.",
        default=None,
    )
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)

    # SUMBT
    parser.add_argument("--zero_init_rnn", type=bool, default=False)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--max_label_length", type=int, default=12)
    parser.add_argument("--attn_head", type=int, default=4)
    parser.add_argument("--fix_utterance_encoder", type=bool, default=False)
    parser.add_argument("--distance_metric", type=str, default="euclidean")

    args = parser.parse_args()
    args.dst = args.dst.upper()
    os.makedirs(f"{args.model_dir}/{args.model_fold}", exist_ok=True)

    # wandb init

    wandb.init(project=args.project_name)
    wandb.run.name = f"{args.model_fold}"
    wandb.config.update(args)

    # random seed 고정
    set_seed(args.seed)
    
    train_module = getattr(
        import_module(f"train_{args.dst}"), "train"
    )

    print('='*100)
    print(args)
    print('='*100)

    train_module(args)