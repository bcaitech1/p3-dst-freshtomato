import argparse
import os
import json

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils import WOSDataset, get_examples_from_dialogues
from model import TRADE
from preprocessor import TRADEPreprocessor
from config import CFG


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def postprocess_state(state):
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
    return state


def inference_TRADE(model, eval_loader, processor, device):
    model.eval()
    predictions = {}
    for batch in tqdm(eval_loader):
        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
            b.to(device) if not isinstance(b, list) else b for b in batch
        ]

        with torch.no_grad():
            o, g = model(input_ids, segment_ids, input_masks, 9)

            _, generated_ids = o.max(-1)
            _, gated_ids = g.max(-1)

        for guid, gate, gen in zip(guids, gated_ids.tolist(), generated_ids.tolist()):
            prediction = processor.recover_state(gate, gen)
            prediction = postprocess_state(prediction)
            predictions[guid] = prediction
    return predictions

def inference_SUMBT(model, eval_loader, processor, device):
    model.eval()
    predictions = {}
    for batch in tqdm(eval_loader):
        input_ids, segment_ids, input_masks, target_ids, num_turns, guids = \
        [b.to(device) if not isinstance(b, list) else b for b in batch]

        with torch.no_grad():
            _, pred_slot = model(
                input_ids, segment_ids, input_masks, labels=None, n_gpu=1
            )
        
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            guid = guids[i]
            states = processor.recover_state(pred_slot.tolist()[i], num_turns[i])
            for tid, state in enumerate(states):
                predictions[f"{guid}-{tid}"] = state
    return predictions



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_fold", type=str, required=True, help="model 폴더명")
    parser.add_argument("--chkpt_idx", type=int, required=True, help="model check point")

    parser.add_argument("--data_dir", type=str, default=CFG.Test)
    parser.add_argument("--model_dir", type=str, default='../models')
    parser.add_argument("--output_dir", type=str, default=CFG.Output)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        default="monologg/koelectra-base-v3-discriminator",
    )
    args = parser.parse_args()

    eval_data = json.load(open(f"{args.data_dir}/eval_dials.json", "r"))
    config = json.load(open(f"{args.model_dir}/{args.model_fold}/exp_config.json", "r"))
    config = argparse.Namespace(**config)
    slot_meta = json.load(open(f"{args.model_dir}/{args.model_fold}/slot_meta.json", "r"))

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    processor = TRADEPreprocessor(slot_meta, tokenizer)

    eval_examples = get_examples_from_dialogues(
        eval_data, user_first=False, dialogue_level=False
    )

    # Extracting Featrues
    eval_features = processor.convert_examples_to_features(eval_examples)
    eval_data = WOSDataset(eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_loader = DataLoader(
        eval_data,
        batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# eval:", len(eval_data))

    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )

    model = TRADE(config, tokenized_slot_meta)
    ckpt = torch.load(f'{args.model_dir}/{args.model_fold}/model-{args.chkpt_idx}.bin', map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    print("Model is loaded")

    predictions = inference(model, eval_loader, processor, device)

    os.makedirs(args.output_dir, exist_ok=True)

    json.dump(
        predictions,
        open(f"{args.output_dir}/{args.model_fold}-predictions.csv", "w"),
        indent=2,
        ensure_ascii=False,
    )
