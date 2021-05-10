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
    gold_p_state: List[str] = None  # 아직 모르겠음
    gold_state: List[str] = None  # 현재 turn의 update state: List['도메인-슬릇-밸류' ]
    generate_y: List[str] = None  # 현재 turn의 update state: List['밸류']


@dataclass
class SomDSTFeature:
    guid: str
    input_ids: List[int]
    input_masks: List[int] # attention에 활용할 마스크 리스트(패딩은 0처리, 그 외 1처리)
    segment_ids: List[int] # 문장의 구분
    op_ids: List[int] # [carryover, ...] 등의 operation id 리스트
    slot_positions: List[int] # [SLOT] 토큰 위치
    domain_id: int # turn 도메인
    generate_ids: List[int] # 모델이 생성해야 할 id 리스트


def get_somdst_examples_from_dialogues(
    data: list, slot_meta: dict, tokenizer, user_first: bool = False, n_current: int=1, op_code="4", max_seq_length: int = 256, dynamic: bool=False
) -> List[DSTInputExample]:
    examples = []

    for d in tqdm(data):
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
    max_seq_length: int = 256,
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