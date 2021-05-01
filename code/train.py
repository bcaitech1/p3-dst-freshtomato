import sys

sys.path.insert(0, "CustomizedModule")
from CustomizedScheduler import CustomizedCosineAnnealingWarmRestarts
from CustomizedScheduler import get_scheduler
from CustomizedOptimizer import get_optimizer
import argparse
import json
import os
import random
import wandb
import numpy as np
from importlib import import_module

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (
    AdamW,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
)
from torch.optim import Adam, SGD

from data_utils import WOSDataset, get_examples_from_dialogues, load_dataset, set_seed
from eval_utils import DSTEvaluator
from evaluation import _evaluation
from inference import inference
from model import TRADE, masked_cross_entropy_for_value
from preprocessor import TRADEPreprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    # random seed 고정
    set_seed(args.seed)

    # Data Loading
    train_data_file = f"{args.data_dir}/train_dials.json"
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file)

    train_examples = get_examples_from_dialogues(
        train_data, user_first=False, dialogue_level=False
    )
    dev_examples = get_examples_from_dialogues(
        dev_data, user_first=False, dialogue_level=False
    )

    # Define Preprocessor
    tokenizer_module = getattr(
        import_module("transformers"), f"{args.tokenizer_name}Tokenizer"
    )
    tokenizer = tokenizer_module.from_pretrained(args.model_name_or_path)

    processor = TRADEPreprocessor(slot_meta, tokenizer)
    args.vocab_size = len(tokenizer)
    args.n_gate = len(processor.gating2id)  # gating 갯수 none, dontcare, ptr

    # Extracting Featrues
    train_features = processor.convert_examples_to_features(train_examples)
    dev_features = processor.convert_examples_to_features(dev_examples)

    # Slot Meta tokenizing for the decoder initial inputs
    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )

    # Model 선언
    model = TRADE(args, tokenized_slot_meta)
    # getattr 사용할 수 있도록 바뀐 부분
    model.set_subword_embedding(args)  # Subword Embedding 초기화
    print(f"Subword Embeddings is loaded from {args.model_name_or_path}")
    model.to(device)
    print("Model is initialized")

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

    # Optimizer 및 Scheduler 선언
    n_epochs = args.epochs
    t_total = len(train_loader) * n_epochs
    # get_optimizer 부분에서 자동으로 warmup_steps를 계산할 수 있도록 바꿨음 (아래가 원래의 code)
    # warmup_steps = int(t_total * args.warmup_ratio)
    optimizer = get_optimizer(model, args)  # get optimizer (Adam, sgd, AdamP, ..)

    scheduler = get_scheduler(optimizer, t_total, args) # get scheduler (custom, linear, cosine, ..)

    loss_fnc_1 = masked_cross_entropy_for_value  # generation
    loss_fnc_2 = nn.CrossEntropyLoss()  # gating

    os.makedirs(f"{args.model_dir}/{args.model_fold}", exist_ok=True)

    json.dump(
        vars(args),
        open(f"{args.model_dir}/{args.model_fold}/exp_config.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(
        slot_meta,
        open(f"{args.model_dir}/{args.model_fold}/slot_meta.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

    best_score, best_checkpoint = 0, 0
    for epoch in range(n_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
                b.to(device) if not isinstance(b, list) else b for b in batch
            ]

            # teacher forcing
            if (
                args.teacher_forcing_ratio > 0.0
                and random.random() < args.teacher_forcing_ratio
            ):
                tf = target_ids
            else:
                tf = None

            all_point_outputs, all_gate_outputs = model(
                input_ids, segment_ids, input_masks, target_ids.size(-1), tf
            )

            # generation loss
            loss_1 = loss_fnc_1(
                all_point_outputs.contiguous(),
                target_ids.contiguous().view(-1),
                tokenizer.pad_token_id,
            )

            # gating loss
            loss_2 = loss_fnc_2(
                all_gate_outputs.contiguous().view(-1, args.n_gate),
                gating_ids.contiguous().view(-1),
            )
            loss = loss_1 + loss_2

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            for learning_rate in scheduler.get_lr():
                wandb.log({"learning_rate": learning_rate})
            optimizer.zero_grad()

            if step % 100 == 0:
                print(
                    f"[{epoch}/{n_epochs}] [{step}/{len(train_loader)}] loss: {loss.item()} gen: {loss_1.item()} gate: {loss_2.item()}"
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "Train epoch loss": loss.item(),
                        "Train epoch gen loss": loss_1.item(),
                        "Train epoch gate loss": loss_2.item(),
                    }
                )

        predictions = inference(model, dev_loader, processor, device)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")

        if best_score < eval_result["joint_goal_accuracy"]:
            print("Update Best checkpoint!")
            best_score = eval_result["joint_goal_accuracy"]
            best_checkpoint = epoch

            wandb.log({"epoch": epoch, "Best joint goal accuracy": best_score})

        torch.save(
            model.state_dict(), f"{args.model_dir}/{args.model_fold}/model-{epoch}.bin"
        )
    print(f"Best checkpoint: {args.model_dir}/model-{best_checkpoint}.bin")
    wandb.log({"Best checkpoint": f"{args.model_dir}/model-{best_checkpoint}.bin"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="wandb에 저장할 project name (본인 이름 or 닉네임으로 지정)",
    )
    parser.add_argument("--model_fold", type=str, required=True, help="model 폴더명")
    parser.add_argument("--data_dir", type=str, default="./input/data/train_dataset")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-8)
    parser.add_argument(
        "--max_lr",
        type=float,
        help="Using CustomizedCosineAnnealingWarmRestarts, Limit the maximum of learning_rate",
        default=5e-6,
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
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
        help="Determine max_lr of Next Cycle Sequentially, When Using Customized Scheduler",
        default=0.9,
    )
    parser.add_argument(
        "--first_cycle_ratio",
        type=float,
        help="Determine Num of First Cycle Epoch When Using Customized Scheduler (first_cycle = t_total * first_cycle_ratio)",
        default=0.05,
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        help="Not Using AutoTokenizer, Tokenizer Name For Loading (EX. Bert, Electra, XLMRoberta, etc..)",
        default="Bert",
        # default="Electra",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Not Using AutoModel, Model Name For Loading set_subword_embedding in model.py (EX. Bert, Electra, XLMRoberta, etc..)",
        # default="Bert",
        default="Electra",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        # default="monologg/kobert",
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

    # wandb init
    wandb.init(project=args.project_name)
    wandb.run.name = f"{args.model_fold}"
    wandb.config.update(args)

    train(args)
