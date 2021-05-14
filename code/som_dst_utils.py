import sys
import json
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import *
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from data_utils import DSTInputExample, DSTPreprocessor
from config import CFG
from utils import save_json


# 아래의 케이스 중 operation class를 선택하는 데 활용
OP_SET = {
    "2": {"update": 0, "carryover": 1},
    "3-1": {"update": 0, "carryover": 1, "dontcare": 2},
    "3-2": {"update": 0, "carryover": 1, "delete": 2},
    "4": {"delete": 0, "update": 1, "dontcare": 2, "carryover": 3},
    "6": {"delete": 0, "update": 1, "dontcare": 2, "carryover": 3, "yes": 4, "no": 5},
}

# WoS 데이터셋의 각 turn은 도메인이 따로 부여되어 있지 않음
# 때문에, 해당 turn에서 등장한 new state를 바탕으로 도메인을 맵핑

EXPERIMENT_DOMAINS = ["관광", "숙소", "식당", "지하철", "택시"]

TURN_DOMAIN_DICT = {
    "관광": 0,
    "숙소": 1,
    "식당": 2,
    "지하철": 3,
    "택시": 4,

    "관광*숙소": 5,"숙소*관광": 5,
    "관광*식당": 6,"식당*관광": 6,
    "관광*지하철": 7,"지하철*관광": 7,
    "관광*택시": 8,"택시*관광": 8,
    "숙소*식당": 9,"식당*숙소": 9,
    "숙소*지하철": 10,"지하철*숙소": 10,
    "숙소*택시": 11,"택시*숙소": 11,
    "식당*지하철": 12,"지하철*식당": 12,
    "식당*택시": 13,"택시*식당": 13,
    "지하철*택시": 14,"택시*지하철": 14,
    "관광*숙소*식당": 15,"관광*식당*숙소": 15,"숙소*관광*식당": 15,"숙소*식당*관광": 15,"식당*관광*숙소": 15,"식당*숙소*관광": 15,
    "관광*숙소*지하철": 16,"관광*지하철*숙소": 16,"숙소*관광*지하철": 16,"숙소*지하철*관광": 16,"지하철*관광*숙소": 16,"지하철*숙소*관광": 16,
    "관광*숙소*택시": 17,"관광*택시*숙소": 17,"숙소*관광*택시": 17,"숙소*택시*관광": 17,"택시*관광*숙소": 17,"택시*숙소*관광": 17,
    "관광*식당*지하철": 18,"관광*지하철*식당": 18,"식당*관광*지하철": 18,"식당*지하철*관광": 18,"지하철*관광*식당": 18,"지하철*식당*관광": 18,
    "관광*식당*택시": 19,"관광*택시*식당": 19,"식당*관광*택시": 19,"식당*택시*관광": 19,"택시*관광*식당": 19,"택시*식당*관광": 19,
    "관광*지하철*택시": 20,"관광*택시*지하철": 20,"지하철*관광*택시": 20,"지하철*택시*관광": 20,"택시*관광*지하철": 20,"택시*지하철*관광": 20,
    "숙소*식당*지하철": 21,"숙소*지하철*식당": 21,"식당*숙소*지하철": 21,"식당*지하철*숙소": 21,"지하철*숙소*식당": 21,"지하철*식당*숙소": 21,
    "숙소*식당*택시": 22,"숙소*택시*식당": 22,"식당*숙소*택시": 22,"식당*택시*숙소": 22,"택시*숙소*식당": 22,"택시*식당*숙소": 22,
    "숙소*지하철*택시": 23,"숙소*택시*지하철": 23,"지하철*숙소*택시": 23,"지하철*택시*숙소": 23,"택시*숙소*지하철": 23,"택시*지하철*숙소": 23,
    "식당*지하철*택시": 24,"식당*택시*지하철": 24,"지하철*식당*택시": 24,"지하철*택시*식당": 24,"택시*식당*지하철": 24,"택시*지하철*식당": 24,
    "": 25 # 도메인 없음
}
# DOMAIN2ID = TURN_DOMAIN_DICT
DOMAIN2ID = {i: idx for idx, i in enumerate(EXPERIMENT_DOMAINS)}

UTTR_SPLITTER = " ; "
SLOT_TOKEN = '[SLOT]'
NULL_TOKEN = '[NULL]'
EOS_TOKEN = '[EOS]'
flatten = lambda x: [i for s in x for i in s]


@dataclass
class SomDSTInputExample(DSTInputExample):
    turn_id: int = None # 턴 번호
    turn_utter: str = None # 턴 내 발화: 시스템발화 ; 유저발화
    turn_domain: str = None
    last_dialog_state: dict = None # {'도메인-슬릇': 밸류}
    current_dialog_state: dict = None # {'도메인-슬릇': 밸류}
    dialog_history: str = None # 시스템발화 ; 유저발화 시스템발화 ; 유저발화
    dial_domains: list = None # []
    turn_domain: str = None # '식당'
    is_last_turn: bool = None

@dataclass
class SomDSTFeature:
    guid: str
    turn_domain: str
    turn_id: int
    turn_utter: str # '시스템발화 ; 유저발화'
    dialog_history: str # '시스템발화 ; 유저발화'
    last_dialog_state: Dict[str, str] # {'도메인-슬릇': 밸류}
    op_labels: List[str] # ['carryover', ...]
    generate_y: List[List[str]] # ['밸류']
    gold_state: List[str] # ['도메인-슬릇-밸류']
    gold_p_state: Dict[str, str] # {'도메인-슬릇': 밸류}
    is_last_turn: bool

    input_: str # D_{t-1} [SEP] D_{t} [SLOT] 
    input_id: List[int] # [CLS] tokenize(input_)
    segment_id: List[int] # 011111111 0000000
    input_mask: List[int] # 111111111 0000000
    slot_position: List[int] # [86, 112]
    domain_id: int
    op_ids: List[int]
    generate_ids: List[int]


def load_somdst_dataset(dataset_path: str, dev_split: float = 0.1) -> Tuple[list, list, dict]:
    data = json.load(open(dataset_path))
    num_data = len(data)
    num_dev = int(num_data * dev_split)
    if not num_dev:
        return data, []  # no dev dataset

    dom_mapper = defaultdict(list)
    for d in data:
        dom_mapper[len(d["domains"])].append(d["dialogue_idx"])

    # Dialogue별 Domain은 최소 1개부터 3개까지
    num_per_domain_trainsition = int(num_dev / 3)
    dev_idx = []
    for v in dom_mapper.values():
        idx = random.sample(v, num_per_domain_trainsition)
        dev_idx.extend(idx)

    train_data, dev_data = [], []
    for d in data:
        if d["dialogue_idx"] in dev_idx:
            dev_data.append(d)
        else:
            train_data.append(d)

    dev_labels = {}
    for dialogue in dev_data:
        d_idx = 0
        guid = dialogue["dialogue_idx"]
        for idx, turn in enumerate(dialogue["dialogue"]):
            if turn["role"] != "user":
                continue

            state = turn["state"]

            guid_t = f"{guid}-{d_idx}"
            d_idx += 1

            dev_labels[guid_t] = state

    return train_data, dev_data, dev_labels


def get_somdst_examples_from_dialogues(
    data: list, n_history: int=1
) -> List[DSTInputExample]:
    """과거 몇 번째 시점까지 가져올 것인지"""
    examples = []

    for d in tqdm(data, desc='[Getting examples]'):
        example = get_somdst_examples_from_dialogue(d, n_history=n_history)
        examples.extend(example)
    return examples


def get_somdst_examples_from_dialogue(dialogue: dict, n_history: int=1) -> List[DSTInputExample]:
    """turn 단위의 example을 생성
    n_history: 0으로 설정 시 모든 turn의 history를 가져옴
    """
    examples = [] # List[InputExample]
    guid = dialogue["dialogue_idx"] # id
    dial_domains = dialogue['domains']
    
    history = []
    dialog_history = []
    context_turns = []

    last_dialog_state = dict()

    # MultiWoZ 데이터셋의 대화 구성과 같게 변경
    dial = deepcopy(dialogue['dialogue'])
    dial.insert(0, dict(role='sys', text=''))
    dial.pop()

    num_turn_pairs = len(dial)//2
    last_turn_domain = None

    for turn_id, turn in enumerate(range(num_turn_pairs)):
        is_last_turn = True if turn_id == num_turn_pairs-1 else False
        sys = dial[2*turn_id]
        user = dial[2*turn_id+1]

        sys_utter = sys['text'].strip()
        user_utter = user['text'].strip()
        turn_utter = f"{sys_utter} ; {user_utter}"

        current_turn = (sys_utter, user_utter)
        current_dialog_state = convert_state_dict(user['state'])

        # get turn domain
        update_domains = []
        for key, value in current_dialog_state.items():
            if key in last_dialog_state:
                if value != last_dialog_state[key]:
                    dom = key.split('-')[0]
                    update_domains.append(dom)
            else:
                dom = key.split('-')[0]
                update_domains.append(dom)

        if update_domains:
            turn_domain = pd.Series(update_domains).value_counts(ascending=False).index[0]
        else:
            turn_domain = last_turn_domain if last_turn_domain is not None else random.choice(dial_domains)

        examples.append(
            SomDSTInputExample(
                guid=guid,
                current_turn=current_turn,
                context_turns=deepcopy(context_turns),
                turn_id=turn_id,
                turn_domain=turn_domain, 
                turn_utter=deepcopy(turn_utter),
                last_dialog_state=deepcopy(last_dialog_state),
                current_dialog_state=deepcopy(current_dialog_state),
                dialog_history=' '.join(dialog_history[-n_history:]),
                dial_domains=deepcopy(dial_domains),
                is_last_turn=is_last_turn
            )
        )

        context_turns.append(current_turn)
        dialog_history.append(turn_utter)
        last_dialog_state = deepcopy(current_dialog_state)
        last_turn_domain = turn_domain

    return examples


def convert_state_dict(state: list) -> dict: # [도메인-슬릇-밸류] -> {도메인-슬릇: 밸류}
    state_dict = dict()
    for s in state:
        dom, slot, val = s.split('-')
        dom_slot = '-'.join([dom, slot])
        state_dict[dom_slot] = val
    return state_dict


def model_evaluation(model, test_features, tokenizer, slot_meta, domain2id, epoch, op_code='4',
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=False, device=CFG.Device):
    model.eval()
    op2id = OP_SET[op_code]
    id2op = {v: k for k, v in op2id.items()}
    id2domain = {v: k for k, v in domain2id.items()}

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
    final_joint_acc, final_count, final_slot_F1_pred, final_slot_F1_count = 0, 0, 0, 0
    op_acc, op_F1, op_F1_count = 0, {k: 0 for k in op2id}, {k: 0 for k in op2id}
    all_op_F1_count = {k: 0 for k in op2id}

    tp_dic = {k: 0 for k in op2id}
    fn_dic = {k: 0 for k in op2id}
    fp_dic = {k: 0 for k in op2id}

    results = {}
    last_dialog_state = {}
    wall_times = []
    for feature in tqdm(test_features, desc='[Evaluation]'):
        if feature.turn_id == 0:
            last_dialog_state = {}

        if is_gt_p_state is False:
            feature.last_dialog_state = deepcopy(last_dialog_state)

        else:  # ground-truth previous dialogue state
            last_dialog_state = deepcopy(feature.gold_p_state)
            feature.last_dialog_state = deepcopy(last_dialog_state)

        input_ids = torch.LongTensor([feature.input_id]).to(device)
        input_mask = torch.LongTensor([feature.input_mask]).to(device)
        segment_ids = torch.LongTensor([feature.segment_id]).to(device)
        state_position_ids = torch.LongTensor([feature.slot_position]).to(device)

        d_gold_op, _, _ = make_turn_label(
            slot_meta=slot_meta, 
            last_dialog_state=last_dialog_state,
            turn_dialog_state=feature.gold_state,
            tokenizer=tokenizer, 
            op_code=op_code, 
            dynamic=True
            )
        gold_op_ids = torch.LongTensor([d_gold_op]).to(device) # operation 레이블

        start = time.perf_counter()
        MAX_LENGTH = 9
        with torch.no_grad():
            # ground-truth state operation
            gold_op_inputs = gold_op_ids if is_gt_op else None

            # d: 도메인에 대한 예측 - (1, 도메인 종류 수)
            # s: operation에 대한 예측 - (1, 슬릇 수, op 수)
            # g: generation에 대한 예측 - (1, ?, value의 최대 길이, vocab 사이즈)
            d, s, g = model(input_ids=input_ids,
                            token_type_ids=segment_ids,
                            state_positions=state_position_ids,
                            attention_mask=input_mask,
                            max_value=MAX_LENGTH,
                            op_ids=gold_op_inputs)

        _, op_ids = s.view(-1, len(op2id)).max(-1) # op에 대한 예측 레이블

        if g.size(1) > 0:
            generated = g.squeeze(0).max(-1)[1].tolist()
        else:
            generated = []

        if is_gt_op:
            pred_ops = [id2op[a] for a in gold_op_ids[0].tolist()]
        else:
            pred_ops = [id2op[a] for a in op_ids.tolist()]
        gold_ops = [id2op[a] for a in d_gold_op]

        if is_gt_gen:
            # ground_truth generation
            gold_gen = {
                '-'.join(dom_slot_val.split('-')[:2]): dom_slot_val.split('-')[-1] 
                for dom_slot_val in feature.gold_state
                }
        else:
            gold_gen = {}

        generated, last_dialog_state = postprocessing(
            slot_meta=slot_meta, 
            ops=pred_ops, 
            last_dialog_state=last_dialog_state,
            generated=generated,
            tokenizer=tokenizer,
            op_code=op_code,
            gold_gen=gold_gen
            )

        end = time.perf_counter()
        wall_times.append(end - start)
        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))

        if set(pred_state) == set(feature.gold_state):
            joint_acc += 1
        key = str(feature.guid) + '_' + str(feature.turn_id)
        # key = str(feature.guid)
        results[key] = [pred_state, feature.gold_state]

        # Compute prediction slot accuracy
        temp_acc = compute_acc(set(feature.gold_state), set(pred_state), slot_meta)
        slot_turn_acc += temp_acc

        # Compute prediction F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(feature.gold_state, pred_state)
        slot_F1_pred += temp_f1
        slot_F1_count += count

        # Compute operation accuracy
        temp_acc = sum([1 if p == g else 0 for p, g in zip(pred_ops, gold_ops)]) / len(pred_ops)
        op_acc += temp_acc

        if feature.is_last_turn:
            final_count += 1
            if set(pred_state) == set(feature.gold_state):
                final_joint_acc += 1
            final_slot_F1_pred += temp_f1
            final_slot_F1_count += count

        # Compute operation F1 score
        for p, g in zip(pred_ops, gold_ops):
            all_op_F1_count[g] += 1
            if p == g:
                tp_dic[g] += 1
                op_F1_count[g] += 1
            else:
                fn_dic[g] += 1
                fp_dic[p] += 1

    joint_acc_score = joint_acc / len(test_features)
    turn_acc_score = slot_turn_acc / len(test_features)
    slot_F1_score = slot_F1_pred / slot_F1_count
    op_acc_score = op_acc / len(test_features)
    final_joint_acc_score = final_joint_acc / final_count
    final_slot_F1_score = final_slot_F1_pred / final_slot_F1_count
    latency = np.mean(wall_times) * 1000
    op_F1_score = {}
    for k in op2id.keys():
        tp = tp_dic[k]
        fn = fn_dic[k]
        fp = fp_dic[k]
        precision = tp / (tp+fp) if (tp+fp) != 0 else 0
        recall = tp / (tp+fn) if (tp+fn) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        op_F1_score[k] = F1

    print("------------------------------")
    print(f'op_code: {op_code}, is_gt_op: {str(is_gt_op)}, is_gt_p_state: {str(is_gt_p_state)}, is_gt_gen: {str(is_gt_gen)}')
    print(f"Epoch {epoch} joint accuracy : {joint_acc_score:.4f}")
    print(f"Epoch {epoch} slot turn accuracy : {turn_acc_score:.4f}")
    print(f"Epoch {epoch} slot turn F1: {slot_F1_score:.4f}")
    print(f"Epoch {epoch} op accuracy : {op_acc_score:.4f}")
    print(f"Epoch {epoch} op F1 : {op_F1_score}")
    print(f"Epoch {epoch} op hit count : {op_F1_count}")
    print(f"Epoch {epoch} op all count : {all_op_F1_count}")
    print("Final Joint Accuracy : ", final_joint_acc_score)
    print("Final slot turn F1 : ", final_slot_F1_score)
    print(f"Latency Per Prediction : {latency:.4f} ms")
    print("-----------------------------\n")
    save_json(f'preds_{epoch}.json', results)
    per_domain_join_accuracy(results, slot_meta)

    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score,
              'op_acc': op_acc_score, 'op_f1': op_F1_score
              }

    return scores


def make_turn_label(slot_meta, last_dialog_state, turn_dialog_state,
                    tokenizer, op_code='4', dynamic=False):
    if dynamic:
        gold_state = turn_dialog_state
        turn_dialog_state = {}
        for x in gold_state:
            s = x.split('-')
            k = '-'.join(s[:2])
            turn_dialog_state[k] = s[2]

    op_labels = ['carryover'] * len(slot_meta)
    generate_y = []
    keys = list(turn_dialog_state.keys())
    for k in keys:
        v = turn_dialog_state[k]
        if v == 'none':
            turn_dialog_state.pop(k)
            continue
        vv = last_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv != v:
                if v == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
                    op_labels[idx] = 'dontcare'
                elif v == 'yes' and OP_SET[op_code].get('yes') is not None:
                    op_labels[idx] = 'yes'
                elif v == 'no' and OP_SET[op_code].get('no') is not None:
                    op_labels[idx] = 'no'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([tokenizer.tokenize(v) + [tokenizer.eos_token], idx])
            elif vv == v:
                op_labels[idx] = 'carryover'
        except ValueError:
            continue

    for k, v in last_dialog_state.items():
        vv = turn_dialog_state.get(k)
        try:
            idx = slot_meta.index(k)
            if vv is None:
                if OP_SET[op_code].get('delete') is not None:
                    op_labels[idx] = 'delete'
                else:
                    op_labels[idx] = 'update'
                    generate_y.append([[NULL_TOKEN, tokenizer.eos_token], idx])
        except ValueError:
            continue
    gold_state = [str(k) + '-' + str(v) for k, v in turn_dialog_state.items()]
    if len(generate_y) > 0:
        generate_y = sorted(generate_y, key=lambda lst: lst[1])
        generate_y, _ = [list(e) for e in list(zip(*generate_y))]

    if dynamic:
        op2id = OP_SET[op_code]
        generate_y = [tokenizer.convert_tokens_to_ids(y) for y in generate_y]
        op_labels = [op2id[i] for i in op_labels]

    return op_labels, generate_y, gold_state

def postprocessing(slot_meta, ops, last_dialog_state,
                   generated, tokenizer, op_code, gold_gen={}):
    gid = 0
    for st, op in zip(slot_meta, ops):
        if op == 'dontcare' and OP_SET[op_code].get('dontcare') is not None:
            last_dialog_state[st] = 'dontcare'
        elif op == 'yes' and OP_SET[op_code].get('yes') is not None:
            last_dialog_state[st] = 'yes'
        elif op == 'no' and OP_SET[op_code].get('no') is not None:
            last_dialog_state[st] = 'no'
        elif op == 'delete' and last_dialog_state.get(st) and OP_SET[op_code].get('delete') is not None:
            last_dialog_state.pop(st)
        elif op == 'update':
            g = tokenizer.convert_ids_to_tokens(generated[gid])
            gen = []
            for gg in g:
                if gg == tokenizer.eos_token:
                    break
                gen.append(gg)
            gen = ' '.join(gen).replace(' ##', '')
            gid += 1
            gen = gen.replace(' : ', ':').replace('##', '')
            if gold_gen and gold_gen.get(st) and gold_gen[st] not in ['dontcare']:
                gen = gold_gen[st]

            if gen == NULL_TOKEN and last_dialog_state.get(st) and not OP_SET[op_code].get('delete') is not None:
                last_dialog_state.pop(st)
            else:
                last_dialog_state[st] = gen
    return generated, last_dialog_state


def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count

def per_domain_join_accuracy(data, slot_temp):
    for dom in EXPERIMENT_DOMAINS:
        count = 0
        jt = 0
        acc = 0
        for k, d in data.items():
            p, g = d
            gg = [r for r in g if r.startswith(dom)]
            if len(gg) > 0:
                pp = [r for r in p if r.startswith(dom)]
                count += 1
                if set(pp) == set(gg):
                    jt += 1
                temp_acc = compute_acc(set(gg), set(pp), slot_temp)
                acc += temp_acc
        print(dom, jt / count, acc / count)


def get_turn_uttr(sys_uttr, user_uttr, splitter=UTTR_SPLITTER):
    return (sys_uttr + splitter + user_uttr).strip()


def get_turn_domain(state: list):
    """해당 turn의 domain을 추출하는 함수. 해당 turn의 발화자가 user이어야 함"""
    if state:
        unique_domain_list = list(set(map(lambda x: x.split("-")[0], state)))
        domain = "*".join(unique_domain_list)
        return domain
    else:
        return ""


def get_turn_domain_id(domain):
    return DOMAIN2ID[domain]

def get_domain_nums(domain2id):
    return len(set(domain2id.values()))