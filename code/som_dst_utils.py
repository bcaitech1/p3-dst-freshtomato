from copy import deepcopy
from dataclasses import dataclass
from typing import *
from tqdm import tqdm
from data_utils import DSTInputExample

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
    '관광': 0,
    '숙소': 1,
    '식당': 2,
    '지하철': 3,
    '택시': 4,
    '관광*숙소': 5, '숙소*관광': 5,
    '관광*식당': 6, '식당*관광': 6,
    '관광*지하철': 7, '지하철*관광': 7,
    '관광*택시': 8, '택시*관광': 8,
    '숙소*식당': 9, '식당*숙소': 9,
    '숙소*지하철': 10, '지하철*숙소': 10,
    '숙소*택시': 11, '택시*숙소': 11,
    '식당*지하철': 12, '지하철*식당': 12,
    '식당*택시': 13, '택시*식당': 13,
    '지하철*택시': 14, '택시*지하철': 14,
    '관광*숙소*식당': 15, '관광*식당*숙소': 15, '숙소*관광*식당': 15, '숙소*식당*관광': 15, '식당*관광*숙소': 15, '식당*숙소*관광': 15,
    '관광*숙소*지하철': 16, '관광*지하철*숙소': 16, '숙소*관광*지하철': 16, '숙소*지하철*관광': 16, '지하철*관광*숙소': 16, '지하철*숙소*관광': 16,
    '관광*숙소*택시': 17, '관광*택시*숙소': 17, '숙소*관광*택시': 17, '숙소*택시*관광': 17, '택시*관광*숙소': 17, '택시*숙소*관광': 17,
    '관광*식당*지하철': 18, '관광*지하철*식당': 18, '식당*관광*지하철': 18, '식당*지하철*관광': 18, '지하철*관광*식당': 18, '지하철*식당*관광': 18,
    '관광*식당*택시': 19, '관광*택시*식당': 19, '식당*관광*택시': 19, '식당*택시*관광': 19, '택시*관광*식당': 19, '택시*식당*관광': 19,
    '관광*지하철*택시': 20, '관광*택시*지하철': 20, '지하철*관광*택시': 20, '지하철*택시*관광': 20, '택시*관광*지하철': 20, '택시*지하철*관광': 20,
    '숙소*식당*지하철': 21, '숙소*지하철*식당': 21, '식당*숙소*지하철': 21, '식당*지하철*숙소': 21, '지하철*숙소*식당': 21, '지하철*식당*숙소': 21,
    '숙소*식당*택시': 22, '숙소*택시*식당': 22, '식당*숙소*택시': 22, '식당*택시*숙소': 22, '택시*숙소*식당': 22, '택시*식당*숙소': 22,
    '숙소*지하철*택시': 23, '숙소*택시*지하철': 23, '지하철*숙소*택시': 23, '지하철*택시*숙소': 23, '택시*숙소*지하철': 23, '택시*지하철*숙소': 23,
    '식당*지하철*택시': 24, '식당*택시*지하철': 24, '지하철*식당*택시': 24, '지하철*택시*식당': 24, '택시*식당*지하철': 24, '택시*지하철*식당': 24
    }

domain2id = TURN_DOMAIN_DICT


@dataclass
class SomDSTFeature:
    guid: str
    input_id: List[int]
    segment_id: List[int]
    gating_id: List[int]
    target_ids: Optional[Union[List[int], List[List[int]]]]


def get_somdst_examples_from_dialogues(
    data: list, user_first: bool = False
) -> List[DSTInputExample]:
    examples = []

    for d in tqdm(data):
        example = get_somdst_examples_from_dialogue(d, user_first=user_first)
        examples.extend(example)

    return examples


def get_somdst_examples_from_dialogue(
    dialogue: dict, user_first: bool = False
) -> List[DSTInputExample]:
    guid = dialogue["dialogue_idx"]
    examples = []
    history = []
    d_idx = 0
    cumulative_state = []
    last_dialogue_state = [] # 직전 turn까지의 dial state

    user_first = False

    for idx, turn in enumerate(dialogue["dialogue"]):
        if turn["role"] != "user":
            continue

        sys_utter = dialogue["dialogue"][idx - 1]["text"] if idx != 0 else ""
        user_utter = turn["text"]
        
        turn_dialogue_state = turn.get("state") # 현재 turn까지의 dialogue state
        turn_label = [s for s in turn_dialogue_state if s not in last_dialogue_state]
        turn_domain = get_turn_domain(turn_dialogue_state) # 현재 turn의 도메인

        context = deepcopy(history)
        current_turn = [user_utter, sys_utter] if user_first else [sys_utter, user_utter]

        history += [sys_utter, user_utter]
        examples.append(
            DSTInputExample(
                guid=f"{guid}-{d_idx}",
                context_turns=context,
                current_turn=current_turn,
                last_dialogue_state=last_dialogue_state,
                turn_dialogue_state=turn_dialogue_state,
                turn_domain=turn_domain,
                turn_label=turn_label
            )
        )

        last_dialogue_state = turn_dialogue_state
        d_idx += 1

    return examples


def get_turn_domain(state: list):
    """해당 turn의 domain을 추출하는 함수. 해당 turn의 발화자가 user이어야 함"""
    unique_domain_list = list(set(map(lambda x: x.split("-")[0], state)))
    domain = "*".join(unique_domain_list)
    return domain

def get_turn_domain_id(domain):
    return domain2id[domain]
