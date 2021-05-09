import json
import time
from collections import defaultdict
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import *
from tqdm import tqdm
from data_utils import DSTInputExample, convert_state_dict

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

DOMAIN2ID = TURN_DOMAIN_DICT

UTTR_SPLITTER = " ; "
SLOT_TOKEN = '[SLOT]'
NULL_TOKEN = '[NULL]'
flatten = lambda x: [i for s in x for i in s]


@dataclass
class SomDSTInputExample(DSTInputExample):
    turn_id: str = None
    turn_domain: str = None
    slot_meta: dict = None
    is_last_turn: bool = False
    max_seq_length: int = None  # 발화 최대 길이
    op2id: dict = None  # operation을 인코딩할 딕셔너리
    turn_uttr: str = None
    last_dialogue_state: List[str] = None
    turn_dialogue_state: List[str] = None
    op_labels: List[str] = None
    gold_p_state: List[str] = None  # 이전 turn에서 update한 state: List['도메인-슬릇-밸류' ]
    gold_state: List[str] = None  # 현재 turn의 update state: List['도메인-슬릇-밸류' ]
    generate_y: List[str] = None  # 현재 turn의 update state: List['밸류']


@dataclass
class SomDSTFeature:
    guid: str
    turn_id: str
    last_dialogue_state: List[str]
    gold_p_state: List[str]
    gold_state: List[str]
    input_ids: List[int]
    input_masks: List[int] # attention에 활용할 마스크 리스트(패딩은 0처리, 그 외 1처리)
    segment_ids: List[int] # 문장의 구분
    op_ids: List[int] # [carryover, ...] 등의 operation id 리스트
    slot_positions: List[int] # [SLOT] 토큰 위치
    domain_id: int # turn 도메인
    generate_ids: List[int] # 모델이 생성해야 할 id 리스트
    is_last_turn: bool


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
    data: list, slot_meta: dict, tokenizer, user_first: bool = False, n_current: int=1, op_code="4", max_seq_length: int = 256, dynamic: bool=False
) -> List[DSTInputExample]:
    examples = []

    for d in tqdm(data, desc='[Extract InputExamples from dialogues]'):
        example = get_somdst_examples_from_dialogue(d, slot_meta, tokenizer, user_first, n_current, op_code, max_seq_length, dynamic)
        examples.extend(example)

    return examples


def get_somdst_examples_from_dialogue(
    dialogue: dict,
    slot_meta: dict,
    tokenizer,
    user_first: bool = False,
    n_current: int = 1, # 0으로 설정시 모든 과거 발화를 얻음
    op_code="4",
    max_seq_length: int = 256, # TODO: 필요한지 파악한뒤, 불필요하면 제외하자
    dynamic: bool=False
) -> List[DSTInputExample]:
    guid = dialogue["dialogue_idx"]
    examples = []
    history = []
    d_idx = 0
    last_dialogue_state = []  # 직전 turn까지의 dial state

    user_first = False
    is_last_turn = False

    for idx, turn in enumerate(dialogue["dialogue"]):
        if turn["role"] != "user":
            continue

        # sys/user uttr extraction
        sys_uttr = dialogue["dialogue"][idx - 1]["text"] if idx != 0 else ""
        user_uttr = turn["text"]
        turn_uttr = get_turn_uttr(sys_uttr, user_uttr)

        turn_dialogue_state = turn.get("state")  # 현재 turn까지의 dialogue state
        # generate_y = [s for s in turn_dialogue_state if s not in last_dialogue_state]
        turn_domain = get_turn_domain(turn_dialogue_state)  # 현재 turn의 도메인

        context_turns = ' '.join(history[-n_current:])
        current_turn = [user_uttr, sys_uttr] if user_first else [sys_uttr, user_uttr]

        history += [turn_uttr]
        op_labels, generate_y, gold_state = make_turn_label(
            slot_meta, last_dialogue_state, turn_dialogue_state, tokenizer
        )

        if idx == len(dialogue["dialogue"]) - 1:
            is_last_turn = True

        example = SomDSTInputExample(
            guid=f"{guid}-{d_idx}",
            context_turns=context_turns,
            current_turn=current_turn,
            turn_id=d_idx,
            turn_domain=turn_domain,
            slot_meta=slot_meta,
            is_last_turn=is_last_turn,
            max_seq_length=max_seq_length,
            op2id=OP_SET[op_code],
            turn_uttr=turn_uttr,
            last_dialogue_state=last_dialogue_state,
            turn_dialogue_state=turn_dialogue_state,
            op_labels=op_labels,
            gold_p_state=last_dialogue_state,
            gold_state=gold_state,
            generate_y=generate_y,
            )

        examples.append(example)
        last_dialogue_state = turn_dialogue_state
        d_idx += 1

    return examples


def model_evaluation(model, test_features, tokenizer, slot_meta, domain2id, epoch, op_code='4',
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=False):
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
    for di, i in enumerate(test_features):
        if i.turn_id == 0:
            last_dialog_state = {}

        if is_gt_p_state is False:
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)
        else:  # ground-truth previous dialogue state
            last_dialog_state = deepcopy(i.gold_p_state)
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)

        input_ids = torch.LongTensor([i.input_id]).to(device)
        input_mask = torch.FloatTensor([i.input_mask]).to(device)
        segment_ids = torch.LongTensor([i.segment_id]).to(device)
        state_position_ids = torch.LongTensor([i.slot_position]).to(device)

        d_gold_op, _, _ = make_turn_label(slot_meta, last_dialog_state, i.gold_state,
                                          tokenizer, op_code, dynamic=True)
        gold_op_ids = torch.LongTensor([d_gold_op]).to(device)

        start = time.perf_counter()
        MAX_LENGTH = 9
        with torch.no_grad():
            # ground-truth state operation
            gold_op_inputs = gold_op_ids if is_gt_op else None
            d, s, g = model(input_ids=input_ids,
                            token_type_ids=segment_ids,
                            state_positions=state_position_ids,
                            attention_mask=input_mask,
                            max_value=MAX_LENGTH,
                            op_ids=gold_op_inputs)

        _, op_ids = s.view(-1, len(op2id)).max(-1)

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
            gold_gen = {'-'.join(ii.split('-')[:2]): ii.split('-')[-1] for ii in i.gold_state}
        else:
            gold_gen = {}
        generated, last_dialog_state = postprocessing(slot_meta, pred_ops, last_dialog_state,
                                                      generated, tokenizer, op_code, gold_gen)
        end = time.perf_counter()
        wall_times.append(end - start)
        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))

        if set(pred_state) == set(i.gold_state):
            joint_acc += 1
        key = str(i.id) + '_' + str(i.turn_id)
        results[key] = [pred_state, i.gold_state]

        # Compute prediction slot accuracy
        temp_acc = compute_acc(set(i.gold_state), set(pred_state), slot_meta)
        slot_turn_acc += temp_acc

        # Compute prediction F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(i.gold_state, pred_state)
        slot_F1_pred += temp_f1
        slot_F1_count += count

        # Compute operation accuracy
        temp_acc = sum([1 if p == g else 0 for p, g in zip(pred_ops, gold_ops)]) / len(pred_ops)
        op_acc += temp_acc

        if i.is_last_turn:
            final_count += 1
            if set(pred_state) == set(i.gold_state):
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
    print('op_code: %s, is_gt_op: %s, is_gt_p_state: %s, is_gt_gen: %s' % \
          (op_code, str(is_gt_op), str(is_gt_p_state), str(is_gt_gen)))
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
    print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
    print("Epoch %d op accuracy : " % epoch, op_acc_score)
    print("Epoch %d op F1 : " % epoch, op_F1_score)
    print("Epoch %d op hit count : " % epoch, op_F1_count)
    print("Epoch %d op all count : " % epoch, all_op_F1_count)
    print("Final Joint Accuracy : ", final_joint_acc_score)
    print("Final slot turn F1 : ", final_slot_F1_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    json.dump(results, open('preds_%d.json' % epoch, 'w'))
    per_domain_join_accuracy(results, slot_meta)

    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score,
              'op_acc': op_acc_score, 'op_f1': op_F1_score, 'final_slot_f1': final_slot_F1_score}
    return scores


def make_turn_label(
    slot_meta,
    last_dialog_state,
    turn_dialog_state, # {'hotel-area': 'east', 'hotel-stars': '4'}의 {'도메인-슬릇': '밸류'} 꼴
    tokenizer,
    op_code="4",
    dynamic=False,
):
    last_dialog_state_dict = convert_state_dict(last_dialog_state)
    turn_dialog_state_dict = convert_state_dict(turn_dialog_state)

    if dynamic:
        gold_state = turn_dialog_state_dict
        turn_dialog_state_dict = {}
        for x in gold_state:
            s = x.split("-")
            k = "-".join(s[:2])
            turn_dialog_state_dict[k] = s[2]

    op_labels = ["carryover"] * len(slot_meta)  # 일단 carryover이 디폴트
    generate_y = []

    keys = list(turn_dialog_state_dict.keys())

    # turn 내 DS의 각 '도메인-슬릇'에 대해 iterate
    for k in keys:
        v = turn_dialog_state_dict[k]  # value(ground truth)

        # value가 none인 경우, operation 레이블링에서 제외
        if v == "none":
            turn_dialog_state_dict.pop(k)
            continue
        vv = last_dialog_state_dict.get(k)
        try:
            idx = slot_meta.index(k)
            if vv != v:
                if v == "dontcare" and OP_SET[op_code].get("dontcare") is not None:
                    op_labels[idx] = "dontcare"
                elif v == "yes" and OP_SET[op_code].get("yes") is not None:
                    op_labels[idx] = "yes"
                elif v == "no" and OP_SET[op_code].get("no") is not None:
                    op_labels[idx] = "no"
                else:
                    op_labels[idx] = "update"
                    generate_y.append([tokenizer.tokenize(v) + ["[EOS]"], idx])
            elif vv == v:
                op_labels[idx] = "carryover"
        except ValueError:
            continue

    for k, v in last_dialog_state_dict.items():
        vv = turn_dialog_state_dict.get(k)
        try:
            idx = slot_meta.index(k)
            if vv is None:
                if OP_SET[op_code].get("delete") is not None:
                    op_labels[idx] = "delete"
                else:
                    op_labels[idx] = "update"
                    generate_y.append([["[NULL]", "[EOS]"], idx])
        except ValueError:
            continue
    gold_state = [str(k) + "-" + str(v) for k, v in turn_dialog_state_dict.items()]
    if len(generate_y) > 0:
        generate_y = sorted(generate_y, key=lambda lst: lst[1])
        generate_y, _ = [list(e) for e in list(zip(*generate_y))] # squeeze

    if dynamic:
        op2id = OP_SET[op_code]
        generate_y = [tokenizer.convert_tokens_to_ids(y) for y in generate_y]
        op_labels = [op2id[i] for i in op_labels]

    return op_labels, generate_y, gold_state


def get_turn_uttr(sys_uttr, user_uttr, splitter=UTTR_SPLITTER):
    return (sys_uttr + splitter + user_uttr).strip()


def get_turn_domain(state: list):
    """해당 turn의 domain을 추출하는 함수. 해당 turn의 발화자가 user이어야 함"""
    unique_domain_list = list(set(map(lambda x: x.split("-")[0], state)))
    domain = "*".join(unique_domain_list)
    return domain


def get_turn_domain_id(domain):
    return DOMAIN2ID[domain]

def get_domain_nums(domain2id):
    return len(set(domain2id.values()))