import argparse
import os
import json

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from importlib import import_module
from data_utils import WOSDataset, get_examples_from_dialogues, test_data_loading, get_data_loader, tokenize_ontology
from preprocessor import TRADEPreprocessor, SUMBTPreprocessor

from model import TRADE, SUMBT
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


def main_inference(args, config):
    slot_meta = json.load(open(f"{args.model_dir}/{args.model_fold}/slot_meta.json", "r"))
    ontology = json.load(open(f"{CFG.TrainOntology}", "r"))

    # Define Tokenizer
    tokenizer_module = getattr(
        import_module("transformers"), f"{args.model_name}Tokenizer"
    )
    tokenizer = tokenizer_module.from_pretrained(config.pretrained_name_or_path)

    # Extracting Featrues
    if args.dst == 'TRADE':
        eval_examples = test_data_loading(args, isUserFirst=False, isDialogueLevel=False)
        processor = TRADEPreprocessor(slot_meta, tokenizer)

        tokenized_slot_meta = []
        for slot in slot_meta:
            tokenized_slot_meta.append(
                tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
            )
        
        # Model 선언
        model = TRADE(config, tokenized_slot_meta)
        model.set_subword_embedding(config)  # Subword Embedding 초기화

    elif config.dst == 'SUMBT':
        eval_examples = test_data_loading(args, isUserFirst=True, isDialogueLevel=True)
        max_turn = max([len(e)*2 for e in eval_examples])
        processor = SUMBTPreprocessor(slot_meta,
                                    tokenizer,
                                    ontology=ontology,  # predefined ontology
                                    max_seq_length=config.max_seq_length,  # 각 turn마다 최대 길이
                                    max_turn_length=max_turn)  # 각 dialogue의 최대 turn 길이

        slot_type_ids, slot_values_ids = tokenize_ontology(ontology, tokenizer, config.max_label_length)

        # Model 선언
        num_labels = [len(s) for s in slot_values_ids] # 각 Slot 별 후보 Values의 갯수

        model = SUMBT(config, num_labels, device)
        model.initialize_slot_value_lookup(slot_values_ids, slot_type_ids)  # Tokenized Ontology의 Pre-encoding using BERT_SV

    eval_features = processor.convert_examples_to_features(eval_examples)
    eval_loader = get_data_loader(processor, eval_features, config.eval_batch_size)
    print("# eval:", len(eval_loader))

    ckpt = torch.load(f'{args.model_dir}/{args.model_fold}/model-{args.chkpt_idx}.bin', map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    print("Model is loaded")

    inference_module = getattr(
        import_module("inference"), f"inference_{config.dst}"
    )
    predictions = inference_module(model, eval_loader, processor, device)

    os.makedirs(args.output_dir, exist_ok=True)

    json.dump(
        predictions,
        open(f"{args.output_dir}/{args.model_fold}-predictions.csv", "w"),
        indent=2,
        ensure_ascii=False,
    )

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_fold", type=str, required=True, help="model 폴더명")
    parser.add_argument("--chkpt_idx", type=int, required=True, help="model check point")
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
    parser.add_argument("--data_dir", type=str, default=CFG.Test)
    parser.add_argument("--model_dir", type=str, default=CFG.Models)
    parser.add_argument("--output_dir", type=str, default=CFG.Output)
    parser.add_argument("--eval_batch_size", type=int, default=32)

    '''
    parser.add_argument("--hidden_size", type=int, help="GRU의 hidden size", default=768) # TRADER, SUMBT
    parser.add_argument("--num_rnn_layers", type=int, help="Number of GRU layers", default=1) # TRADER, SUMBT
    parser.add_argument("--zero_init_rnn", type=bool, default=False)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--max_label_length", type=int, default=12)
    parser.add_argument("--attn_head", type=int, default=4)
    parser.add_argument("--fix_utterance_encoder", type=bool, default=False)
    parser.add_argument("--distance_metric", type=str, default="euclidean")
    '''

    parser.add_argument(
        "--pretrained_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        default="monologg/koelectra-base-v3-discriminator",
    )
    
    args = parser.parse_args()
    args.dst = args.dst.upper()

    config_files = json.load(open(f"{args.model_dir}/{args.model_fold}/exp_config.json", "r"))
    config = argparse.Namespace(**config_files)

    main_inference(args, config)
