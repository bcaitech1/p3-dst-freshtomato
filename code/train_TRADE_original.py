import sys
import json
import wandb
import random
import argparse
import numpy as np
from importlib import import_module

sys.path.insert(0, "../CustomizedModule")
from CustomizedScheduler import get_scheduler
from CustomizedOptimizer import get_optimizer

import torch
import torch.nn as nn
from tqdm import tqdm

from eval_utils import DSTEvaluator
from evaluation import _evaluation
from inference import inference_TRADE
from data_utils import train_data_loading, get_data_loader

from preprocessor import TRADEPreprocessor
from model import TRADE_original
from criterions import LabelSmoothingLoss, masked_cross_entropy_for_value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    # Define Tokenizer
    tokenizer_module = getattr(
        import_module("transformers"), f"{args.model_name}Tokenizer"
    )
    tokenizer = tokenizer_module.from_pretrained(args.pretrained_name_or_path)

    slot_meta, train_examples, dev_examples, dev_labels = train_data_loading(args, isUserFirst=False, isDialogueLevel=False)
    # Define Preprocessor
    processor = TRADEPreprocessor(slot_meta, tokenizer)

    train_features = processor.convert_examples_to_features(train_examples)
    dev_features = processor.convert_examples_to_features(dev_examples)

    train_loader = get_data_loader(processor, train_features, args.train_batch_size)
    dev_loader = get_data_loader(processor, dev_features, args.eval_batch_size)

    args.vocab_size = len(tokenizer)
    args.n_gate = len(processor.gating2id)  # gating 갯수 none, dontcare, ptr
    
    # Slot Meta tokenizing for the decoder initial inputs
    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )

    # Model 선언
    model = TRADE_original(args, tokenized_slot_meta)
    model.set_subword_embedding(args)  # Subword Embedding 초기화
    print(f"Subword Embeddings is loaded from {args.pretrained_name_or_path}")
    model.to(device)
    print("Model is initialized")

    # Optimizer 및 Scheduler 선언
    n_epochs = args.epochs
    t_total = len(train_loader) * n_epochs
    # get_optimizer 부분에서 자동으로 warmup_steps를 계산할 수 있도록 바꿨음 (아래가 원래의 code)
    # warmup_steps = int(t_total * args.warmup_ratio)
    optimizer = get_optimizer(model, args)  # get optimizer (Adam, sgd, AdamP, ..)

    scheduler = get_scheduler(
        optimizer, t_total, args
    )  # get scheduler (custom, linear, cosine, ..)

    loss_fnc_1 = masked_cross_entropy_for_value  # generation - # classes: vocab_size
    loss_fnc_2 = nn.CrossEntropyLoss()

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

        predictions = inference_TRADE(model, dev_loader, processor, device)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")

        if best_score < eval_result["joint_goal_accuracy"]:
            print("Update Best checkpoint!")
            best_score = eval_result["joint_goal_accuracy"]
            best_checkpoint = epoch

            wandb.log({
                "epoch": epoch, 
                "Best joint goal accuracy": best_score, 
                "Best turn slot accuracy": eval_result['turn_slot_accuracy'],
                "Best turn slot f1": eval_result['turn_slot_f1']
            })
        if args.logging_accuracy_per_domain_slot:
            wandb.log({k:v for k,v in eval_result.items() if k not in ("joint_goal_accuracy",'turn_slot_accuracy','turn_slot_f1')})
                 
        torch.save(
            model.state_dict(), f"{args.model_dir}/{args.model_fold}/model-{epoch}.bin"
        )
    
    print(f"Best checkpoint: {args.model_dir}/model-{best_checkpoint}.bin")
    wandb.log({"Best checkpoint": f"{args.model_dir}/model-{best_checkpoint}.bin"})
