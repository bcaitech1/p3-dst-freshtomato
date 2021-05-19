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
from transformers import AdamW, get_linear_schedule_with_warmup, BertConfig, BertTokenizer, BertModel, ElectraConfig
from importlib import import_module
import wandb

from eval_utils import DSTEvaluator
from evaluation import _evaluation
from inference import inference_TRADE
from data_utils import WOSDataset, load_dataset, set_seed
from som_dst_utils import (
    get_somdst_examples_from_dialogues,
    load_somdst_dataset,
    NULL_TOKEN,
    SLOT_TOKEN,
    EOS_TOKEN,
    OP_SET,
    DOMAIN2ID,
    get_domain_nums,
    model_evaluation,
)
from preprocessor import SomDSTPreprocessor
from model import SomDST
from criterions import masked_cross_entropy_for_value
from config import CFG

device = CFG.Device
MAX_GRAD_NORM = 4.

def train(args):
    # operation encoder domain encoder
    op2id = OP_SET[args.op_code]
    domain2id = DOMAIN2ID
    args.n_op = len(op2id)
    args.n_domain = get_domain_nums(domain2id)
    args.update_id = op2id["update"]

    # initialize tokenizer
    tokenizer_module = getattr(
        import_module("transformers"), f"{args.model_name}Tokenizer"
    )
    tokenizer = tokenizer_module.from_pretrained(args.pretrained_name_or_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [NULL_TOKEN, SLOT_TOKEN], 'eos_token': EOS_TOKEN}
    )
    args.vocab_size = len(tokenizer)

    # load data
    slot_meta = json.load(open(os.path.join(args.data_dir, "slot_meta.json")))
    train_data, dev_data, dev_labels = load_somdst_dataset(
        os.path.join(args.data_dir, "train_dials.json")
    )

    train_examples = get_somdst_examples_from_dialogues(
        data=train_data, n_history=args.n_history
    )

    dev_examples = get_somdst_examples_from_dialogues(
        data=dev_data, n_history=args.n_history
    )

    # preprocessing
    preprocessor = SomDSTPreprocessor(
        slot_meta=slot_meta, 
        src_tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        word_dropout=0.1,
        domain2id=domain2id
    )

    train_features = preprocessor.convert_examples_to_features(train_examples, word_dropout=args.word_dropout)
    dev_features = preprocessor.convert_examples_to_features(dev_examples, word_dropout=0.0)

    train_dataset = WOSDataset(features=train_features)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=preprocessor.collate_fn,
        num_workers=args.num_workers,
    )

    # initialize model: update embedding size & initailize weights
    model_config = BertConfig.from_pretrained(args.pretrained_name_or_path)
    model_config.dropout = 0.1
    model_config.vocab_size = len(tokenizer)
  
    model = SomDST(model_config, n_op=args.n_op, n_domain=args.n_domain, update_id=args.update_id)
    # model.encoder.bert.embeddings.word_embeddings.weight.data[1].normal_(mean=0.0, std=0.02)
    # model.encoder.bert.embeddings.word_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
    # model.encoder.bert.embeddings.word_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)
    model.to(device)
    print("Model is initialized")

    # num_train_steps = int(len(train_features) / args.train_batch_size * args.epochs)
    num_train_steps = len(train_dataloader) * args.epochs

    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # enc_param_optimizer = list(model.encoder.named_parameters())
    # enc_optimizer_grouped_parameters = [
    #     {
    #         "params": [
    #             p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay)
    #         ],
    #         "weight_decay": 0.01,
    #     },
    #     {
    #         "params": [
    #             p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay)
    #         ],
    #         "weight_decay": 0.0,
    #     },
    # ]


    # initialize optimizer & scheduler
    # enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=args.enc_lr, eps=args.adam_epsilon)
    
    enc_param_optimizer = list(model.encoder.parameters())
    enc_optimizer = AdamW(enc_param_optimizer, lr=args.enc_lr, eps=args.adam_epsilon)
    enc_scheduler = get_linear_schedule_with_warmup(
        optimizer=enc_optimizer,
        num_warmup_steps=0.1,
        num_training_steps=num_train_steps
    )

    dec_param_optimizer = list(model.decoder.parameters())
    dec_optimizer = AdamW(dec_param_optimizer, lr=args.dec_lr, eps=args.adam_epsilon)
    dec_scheduler = get_linear_schedule_with_warmup(
        optimizer=dec_optimizer,
        num_warmup_steps=0.1,
        num_training_steps=num_train_steps
    )

    criterion = nn.CrossEntropyLoss()
    rng = random.Random(args.seed)

    # save experiment settings
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

    best_score = {
        "epoch": 0,
        "joint_acc": 0, # 
        "op_acc": 0,
        "slot_acc": 0,
        "slot_f1": 0,
        "op_acc": 0,
        "op_f1": 0,
    }

    for epoch in range(args.epochs):
        print(f'Epoch #{epoch}')
        batch_loss = []
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader), desc='[Step]'):
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

            # teacher forcing for generation(decoder)
            teacher = gen_ids if rng.random() < args.teacher_forcing_ratio else None

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

            for learning_rate in enc_scheduler.get_lr():
                wandb.log({"encoder_learning_rate": learning_rate})

            for learning_rate in dec_scheduler.get_lr():
                wandb.log({"decoder_learning_rate": learning_rate})

            if step % 100 == 0:
                if args.exclude_domain is not True:
                    print(
                        f"[{epoch+1}/{args.epochs}] [{step}/{len(train_dataloader)}] mean_loss : {np.mean(batch_loss):.3f}, state_loss : {loss_s.item():.3f}, gen_loss : {loss_g.item():.3f}, dom_loss : {loss_d.item():.3f}"
                    )
                    wandb.log(
                        {
                            "epoch": epoch,
                            "Train epoch loss": np.mean(batch_loss),
                            "Train epoch state loss": loss_s.item(),
                            "Train epoch generation loss": loss_g.item(),
                            "Train epoch domain loss": loss_d.item(),
                        }
                    )
                else:
                    print(
                        f"[{epoch+1}/{args.epochs}] [{step}/{len(train_dataloader)}] mean_loss : {np.mean(batch_loss):.3f}, state_loss : {loss_s.item():.3f}, gen_loss : {loss_g.item():.3f}"
                    )
                    wandb.log(
                        {
                            "epoch": epoch,
                            "Train epoch loss": np.mean(batch_loss),
                            "Train epoch state loss": loss_s.item(),
                            "Train epoch generation loss": loss_g.item(),
                        }
                    )

                batch_loss = []

        # evaluation for each epoch
        eval_res = model_evaluation(
            model,
            dev_features,
            tokenizer,
            slot_meta,
            domain2id,
            epoch + 1,
            args.op_code,
        )

        if eval_res["joint_acc"] > best_score["joint_acc"]:
            print("Update Best checkpoint!")
            best_score = eval_res
            best_checkpoint = best_score["epoch"]
            wandb.log(
                {
                    "epoch": best_score["epoch"],
                    "Best joint goal accuracy": best_score["joint_acc"],
                    "Best turn slot accuracy": best_score["slot_acc"],
                    "Best turn slot f1": best_score["slot_f1"],
                    "Best operation accucay": best_score["op_acc"],
                    "Best operation f1": best_score["op_f1"],
                }
            )

            # save phase
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(
                model_to_save.state_dict(),
                f"{args.model_dir}/{args.model_fold}/model-best.bin",
            )
            print("Best Score : ", best_score)
            print("\n")

    print(f"Best checkpoint: {args.model_dir}/model-best.bin")