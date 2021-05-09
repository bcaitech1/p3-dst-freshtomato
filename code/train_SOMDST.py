import os
import sys
import json
from tqdm import tqdm
import random
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import wandb
from importlib import import_module

sys.path.insert(0, "./CustomizedModule")
from CustomizedScheduler import get_scheduler
from CustomizedOptimizer import get_optimizer
from eval_utils import DSTEvaluator
from evaluation import _evaluation
from inference import inference_TRADE
from data_utils import WOSDataset, load_dataset
from som_dst_utils import (
    get_somdst_examples_from_dialogues,
    load_somdst_dataset,
    NULL_TOKEN,
    OP_SET,
    SLOT_TOKEN,
    DOMAIN2ID,
    get_domain_nums,
    model_evaluation,
)
from preprocessor import SomDSTPreprocessor
from model import SomDST
from criterions import masked_cross_entropy_for_value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    # Define Tokenizer
    tokenizer_module = getattr(
        import_module("transformers"), f"{args.model_name}Tokenizer"
    )
    tokenizer = tokenizer_module.from_pretrained(args.pretrained_name_or_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [NULL_TOKEN, SLOT_TOKEN]}
    )
    args.vocab_size = len(tokenizer)

    op2id = OP_SET[args.op_code]

    slot_meta = json.load(open(os.path.join(args.data_dir, "slot_meta.json")))
    train_data, dev_data, dev_labels = load_somdst_dataset(
        os.path.join(args.data_dir, "train_dials.json")
    )

    train_examples = get_somdst_examples_from_dialogues(
        data=train_data, slot_meta=slot_meta, tokenizer=tokenizer, op_code=args.op_code
    )

    dev_examples = get_somdst_examples_from_dialogues(
        data=dev_data, slot_meta=slot_meta, tokenizer=tokenizer, op_code=args.op_code
    )

    preprocessor = SomDSTPreprocessor(
        slot_meta=slot_meta, tokenizer=tokenizer, op_code=args.op_code
    )

    # for test
    train_features = [
        preprocessor._convert_example_to_feature(train_examples[i]) for i in range(50)
    ]
    dev_features = [
        preprocessor._convert_example_to_feature(dev_examples[i]) for i in range(50)
    ]

    # train_features = preprocessor.convert_examples_to_features(train_examples)
    # dev_features = preprocessor.convert_examples_to_features(dev_examples)

    train_dataset = WOSDataset(features=train_features)
    dev_dataset = WOSDataset(features=dev_features)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.batch_size,
        collate_fn=preprocessor.collate_fn,
        num_workers=config.num_workers,
    )

    # Model 선언
    model = SomDST(
        args, n_op=args.n_op, n_domain=args.n_domain, update_id=args.update_id
    )
    model.resize_token_embeddings(len(tokenizer))

    # re-initialize added special tokens ([SLOT], [NULL], [EOS])
    model.encoder.bert.embeddings.word_embeddings.weight.data[1].normal_(
        mean=0.0, std=0.02
    )
    model.encoder.bert.embeddings.word_embeddings.weight.data[2].normal_(
        mean=0.0, std=0.02
    )
    model.encoder.bert.embeddings.word_embeddings.weight.data[3].normal_(
        mean=0.0, std=0.02
    )

    model.to(device)
    print("Model is initialized")

    num_train_steps = int(len(train_features) / args.batch_size * args.epochs)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    enc_param_optimizer = list(model.encoder.named_parameters())
    enc_optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=args.enc_lr)
    enc_scheduler = get_linear_schedule_with_warmup(
        optimizer=enc_optimizer,
        num_warmup_steps=int(num_train_steps * args.enc_warmup),
        num_training_steps=num_train_steps,
    )

    dec_param_optimizer = list(model.decoder.parameters())
    dec_optimizer = AdamW(dec_param_optimizer, lr=args.dec_lr)
    dec_scheduler = get_linear_schedule_with_warmup(
        optimizer=dec_optimizer,
        num_warmup_steps=int(num_train_steps * args.dec_warmup),
        num_training_steps=num_train_steps,
    )

    criterion = nn.CrossEntropyLoss()
    rng = random.Random(args.seed)

    # json.dump(
    #     vars(args),
    #     open(f"{args.model_dir}/{args.model_fold}/exp_config.json", "w"),
    #     indent=2,
    #     ensure_ascii=False,
    # )
    # json.dump(
    #     slot_meta,
    #     open(f"{args.model_dir}/{args.model_fold}/slot_meta.json", "w"),
    #     indent=2,
    #     ensure_ascii=False,
    # )

    best_score = {"epoch": 0, "joint_acc": 0, "op_acc": 0, "final_slot_f1": 0}

    for epoch in range(args.epochs):
        batch_loss = []
        model.train()

        for step, batch in enumerate(train_dataloader):
            batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
            (
                input_ids,
                input_mask,
                segment_ids,
                state_position_ids,
                op_ids,
                domain_ids,
                gen_ids,
                max_value,
                max_update,
            ) = batch

            if rng.random() < args.teacher_forcing_ratio:  # decoder teacher forcing
                teacher = gen_ids
            else:
                teacher = None

            domain_scores, state_scores, gen_scores = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                state_positions=state_position_ids,
                attention_mask=input_mask,
                max_value=max_value,
                op_ids=op_ids,
                max_update=max_update,
                teacher=teacher,
            )

            loss_s = criterion(state_scores.view(-1, len(op2id)), op_ids.view(-1))
            loss_g = masked_cross_entropy_for_value(
                logits=gen_scores.contiguous(),
                target=gen_ids.contiguous(),
                pad_idx=tokenizer.pad_token_id,
            )

            loss = loss_s + loss_g
            if args.exclude_domain is not True:
                loss_d = criterion(
                    domain_scores.view(-1, args.n_domain), domain_ids.view(-1)
                )
                loss = loss + loss_d
            batch_loss.append(loss.item())

            loss.backward()
            enc_optimizer.step()
            enc_scheduler.step()
            dec_optimizer.step()
            dec_scheduler.step()
            model.zero_grad()

            if step % 100 == 0:
                if args.exclude_domain is not True:
                    print(
                        f"[{epoch+1}/{args.epochs}] [{step}/{len(train_dataloader)}] mean_loss : {np.mean(batch_loss):.3f}, state_loss : {loss_s.item():.3f}, gen_loss : {loss_g.item():.3f}, dom_loss : {loss_d.item():.3f}"
                    )
                else:
                    print(
                        f"[{epoch+1}/{args.epochs}] [{step}/{len(train_dataloader)}] mean_loss : {np.mean(batch_loss):.3f}, state_loss : {loss_s.item():.3f}, gen_loss : {loss_g.item():.3f}"
                    )
                batch_loss = []

        eval_res = model_evaluation(
            model, dev_features, tokenizer, slot_meta, epoch + 1, args.op_code
        )
        if eval_res["joint_acc"] > best_score["joint_acc"]:
            best_score = eval_res
            model_to_save = model.module if hasattr(model, "module") else model
            save_path = os.path.join(args.save_dir, "model_best.bin")
            torch.save(model_to_save.state_dict(), save_path)
        print("Best Score : ", best_score)
        print("\n")


if __name__ == "__main__":
    from transformers import BertConfig

    op_code = "4"
    n_op = len(OP_SET[op_code])
    n_domain = get_domain_nums(DOMAIN2ID)
    update_id = OP_SET[op_code]["update"]

    config = BertConfig.from_pretrained("dsksd/bert-ko-small-minimal")
    config.dropout = 0.1
    config.op_code = "4"
    config.n_op = n_op
    config.n_domain = n_domain
    config.update_id = update_id
    config.model_name = "Bert"
    config.pretrained_name_or_path = "dsksd/bert-ko-small-minimal"
    config.data_dir = "./input/data/train_dataset"
    config.batch_size = 4
    config.num_workers = 4
    config.epochs = 1
    config.enc_lr = 4e-5
    config.dec_lr = 1e-4
    config.enc_warmup = 0.1
    config.dec_warmup = 0.1
    config.teacher_forcing_ratio = 0.5
    config.seed = 42
    config.exclude_domain = False
    config.op_code = "4"

    print(config)

    train(config)
