import os
import sys
import json
import wandb
import random
import numpy as np
from importlib import import_module
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import defaultdict

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
from model import TRADE
from criterions import LabelSmoothingLoss, masked_cross_entropy_for_value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_informations(args):
    # Define Tokenizer
    tokenizer_module = getattr(
        import_module("transformers"), f"{args.model_name}Tokenizer"
    )
    tokenizer = tokenizer_module.from_pretrained(args.pretrained_name_or_path)

    slot_meta, train_examples, train_labels = train_data_loading(args, isUserFirst=False, isDialogueLevel=False)

    # Define Preprocessor
    processor = TRADEPreprocessor(slot_meta, tokenizer)

    # Extract Features
    train_features = processor.convert_examples_to_features(train_examples)

    # Slot Meta tokenizing for the decoder initial inputs
    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )

    args.vocab_size = len(tokenizer)
    args.n_gate = len(processor.gating2id)  # gating 갯수 none, dontcare, ptr

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
    return tokenizer, processor, slot_meta, tokenized_slot_meta, train_features, train_labels


def select_kfold_or_full(args, tokenizer, processor, slot_meta, tokenized_slot_meta, features, labels):
    domain_group = {
        '관광_식당':0,
        '관광':1,
        '지하철':2,
        '택시':3,
        '식당_택시':4,
        '숙소_택시':5,
        '식당':6,
        '숙소_식당':7,
        '숙소':8,
        '관광_택시':9,
        '관광_숙소_식당':10,
        '관광_숙소':11,
        '숙소_식당_택시':12,
        '관광_식당_택시':13,
        '관광_숙소_택시':14
    }

    features = np.array(features)
    dialogue_features, dialogue_labels, domain_labels = defaultdict(list), defaultdict(list), []
    for f in features:
        dialogue = '-'.join(f.guid.split('-')[:-1])
        dialogue_features[dialogue].append(f)

    for k, v in dialogue_features.items():
        feature_domain = '_'.join(sorted(v[0].domain))
        if '지하철' in feature_domain:
            feature_domain = '지하철'
        domain_labels.append(domain_group[feature_domain])

    for k, v in labels.items():
        dialogue_labels['-'.join(k.split('-')[:-1])].append([k, v])

    if args.isKfold:
        kf = StratifiedKFold(n_splits=args.fold_num, random_state=args.seed, shuffle=True)
        fold_idx = 1
        
        for train_index, dev_index in kf.split(features, domain_labels):
            os.makedirs(f'{args.model_dir}/{args.model_fold}/{fold_idx}-fold', exist_ok=True)

            train_dialogue_features, dev_dialogue_features = np.array(list(dialogue_features.items()))[train_index.astype(int)], np.array(list(dialogue_features.items()))[dev_index.astype(int)]
            
            train_features, dev_features = [], []
            [train_features.extend(t[1]) for t in train_dialogue_features]
            [dev_features.extend(t[1]) for t in dev_dialogue_features]

            dev_dialogue_labels = np.array(list(dialogue_labels.items()))[dev_index.astype(int)]
            dev_labels = {t[0]:t[1] for turn in dev_dialogue_labels[:, 1] for t in turn}

            train_loader = get_data_loader(processor, train_features, args.train_batch_size)
            dev_loader = get_data_loader(processor, dev_features, args.eval_batch_size)

            print(f"========= {fold_idx} fold =========")
            train_model(args, tokenizer, processor, slot_meta, tokenized_slot_meta, fold_idx, train_loader, dev_loader, dev_labels)
            fold_idx += 1
        
    else:
        fold_idx = None
        train_index, dev_index = train_test_split(np.array(range(len(dialogue_features))), test_size=0.1, random_state=args.seed, stratify=domain_labels)
        
        train_dialogue_features, dev_dialogue_features = np.array(list(dialogue_features.items()))[train_index.astype(int)], np.array(list(dialogue_features.items()))[dev_index.astype(int)]

        train_features, dev_features = [], []
        [train_features.extend(t[1]) for t in train_dialogue_features]
        [dev_features.extend(t[1]) for t in dev_dialogue_features]

        dev_dialogue_labels = np.array(list(dialogue_labels.items()))[dev_index.astype(int)]
        dev_labels = {t[0]:t[1] for turn in dev_dialogue_labels[:, 1] for t in turn}

        train_loader = get_data_loader(processor, train_features, args.train_batch_size)
        dev_loader = get_data_loader(processor, dev_features, args.eval_batch_size)

        train_model(args, tokenizer, processor, slot_meta, tokenized_slot_meta, fold_idx, train_loader, dev_loader, dev_labels)
    return train_loader, dev_loader
    

def train_model(args, tokenizer, processor, slot_meta, tokenized_slot_meta, fold_idx, train_loader, dev_loader, dev_labels):
    # Model 선언
    model = TRADE(args, tokenized_slot_meta)
    model.set_subword_embedding(args)  # Subword Embedding 초기화
    print(f"Subword Embeddings is loaded from {args.pretrained_name_or_path}")
    model.to(device)
    print("Model is initialized")

    # Optimizer 및 Scheduler 선언
    n_epochs = args.epochs
    t_total = len(train_loader) * n_epochs
    # get_optimizer 부분에서 자동으로 warmup_steps를 계산할 수 있도록 변경 (아래가 원래의 code)
    # warmup_steps = int(t_total * args.warmup_ratio)
    optimizer = get_optimizer(model, args)  # get optimizer (Adam, sgd, AdamP, ..)

    scheduler = get_scheduler(
        optimizer, t_total, args
    )  # get scheduler (custom, linear, cosine, ..)

    loss_fnc_1 = masked_cross_entropy_for_value  # generation - # classes: vocab_size
    loss_fnc_2 = nn.CrossEntropyLoss()  # gating - # classes: 3
    # loss_fnc_2 = LabelSmoothingLoss(classes=model.decoder.n_gate)

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

        if args.isKfold:
            torch.save(
                model.state_dict(), f"{args.model_dir}/{args.model_fold}/{fold_idx}-fold'/model-{epoch}.bin"
            )
        else:
            torch.save(
                model.state_dict(), f"{args.model_dir}/{args.model_fold}/model-{epoch}.bin"
            )

    print(f"Best checkpoint: {args.model_dir}/model-{best_checkpoint}.bin")
    wandb.log({"Best checkpoint": f"{args.model_dir}/model-{best_checkpoint}.bin"})


def train(args):
    tokenizer, processor, slot_meta, tokenized_slot_meta, train_features, train_labels = get_informations(args)
    select_kfold_or_full(args, tokenizer, processor, slot_meta, tokenized_slot_meta, train_features, train_labels)
