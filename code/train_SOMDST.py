import sys
import json
import wandb
import random
import argparse
import numpy as np
from importlib import import_module

sys.path.insert(0, "./CustomizedModule")
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


def train(args):
    # Define Tokenizer
    tokenizer_module = getattr(
        import_module("transformers"), f"{args.model_name}Tokenizer"
    )
    tokenizer = tokenizer_module.from_pretrained(args.pretrained_name_or_path)

    slot_meta, train_examples, dev_examples, dev_labels = data_loading(args, isUserFirst=False, isDialogueLevel=False)
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
    model = TRADE(args, tokenized_slot_meta)