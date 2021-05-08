from copy import deepcopy
from typing import List
from tqdm import tqdm
from data_utils import DSTInputExample

# 아래의 케이스 중 operation class를 선택하는 데 활용
OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}

# WoS 데이터셋의 각 turn은 도메인이 따로 부여되어 있지 않음
# 때문에, 해당 turn에서 등장한 new state를 바탕으로 도메인을 맵핑
EXPERIMENT_DOMAINS = [
    '관광',
    '숙소',
    '식당',
    '지하철',
    '택시',
    '관광*숙소',
    '관광*식당',
    '관광*지하철',
    '관광*택시',
    '숙소*식당',
    '숙소*지하철',
    '숙소*택시',
    '식당*지하철',
    '식당*택시',
    '지하철*택시',
    '관광*숙소*식당',
    '관광*숙소*지하철',
    '관광*숙소*택시',
    '관광*식당*지하철',
    '관광*식당*택시',
    '관광*지하철*택시',
    '숙소*식당*지하철',
    '숙소*식당*택시',
    '숙소*지하철*택시',
    '식당*지하철*택시'
    ]

domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}


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
    pre_state = []


    for idx, turn in enumerate(dialogue["dialogue"]):
        if turn["role"] != "user":
            continue

        if idx:
            sys_utter = dialogue["dialogue"][idx - 1]["text"]
        else:
            sys_utter = ""

        user_utter = turn["text"]
        state = turn.get("state")
        state = [s for s in state if s not in pre_state]

        context = deepcopy(history)
        if user_first:
            current_turn = [user_utter, sys_utter]
        else:
            current_turn = [sys_utter, user_utter]

        examples.append(
            DSTInputExample(
                guid=f"{guid}-{d_idx}",
                context_turns=context,
                current_turn=current_turn,
                pre_state=pre_state,
                label=state,
            )
        )
        pre_state = state

        history.append(sys_utter)
        history.append(user_utter)
        d_idx += 1

    return examples