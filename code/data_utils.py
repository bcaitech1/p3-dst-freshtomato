import dataclasses
import json
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class OntologyDSTFeature:
    guid: str
    input_ids: List[int]
    segment_ids: List[int]
    num_turn: int
    target_ids: Optional[List[int]]


@dataclass
class OpenVocabDSTFeature:
    guid: str
    input_id: List[int]
    segment_id: List[int]
    gating_id: List[int]
    target_ids: Optional[Union[List[int], List[List[int]]]]


class WOSDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]


def load_dataset(dataset_path, dev_split=0.1):
    """[summary]

    Args:
        dataset_path (str): Path to train data
        dev_split (float, optional): Validation set ratio. Defaults to 0.1.

    Returns:
        [type]: [description]
        train_data: 
        dev_data:
        dev_labels:
    """
    data = json.load(open(dataset_path))
    num_data = len(data)
    num_dev = int(num_data * dev_split)
    if not num_dev:
        return data, []  # no dev dataset

    # 사용 이유: domain의 갯수 별로 데이터를 나누기 위함. 한 dialogue가 가질 수 있는 도메인 수가 3가지여서 3으로 나눈 뒤 dev set을 만든 것?
    # 나누는 방식 완전 이상한듯... 바꾸자
    dom_mapper = defaultdict(list)
    for d in data:
        dom_mapper[len(d["domains"])].append(d["dialogue_idx"])

    # num_per_domain_transition = int(num_dev / 3)
    num_per_domain_transition = int(num_dev / len(dom_mapper.keys()))
    dev_idx = []
    for v in dom_mapper.values():
        idx = random.sample(v, num_per_domain_transition)
        dev_idx.extend(idx)

    # 학습셋-검증셋 분리 by 도메인 라벨 갯수
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

            state = turn.pop("state")

            guid_t = f"{guid}-{d_idx}"
            d_idx += 1

            dev_labels[guid_t] = state

    return train_data, dev_data, dev_labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def split_slot(dom_slot_value, get_domain_slot=False):
    """[summary]

    Args:
        dom_slot_value (str): "[domain]-[slot]-[value]" 형태의 state label
        get_domain_slot (bool, optional): "[domain]-[slot]", "[value]" 형태를 반환할지, "[domain]", "[slot]", "[value]"를 반환할지 결정. Defaults to False.

    Returns:
        "[domain]-[slot]", "[value]" if get_domain_slot=True else "[domain]", "[slot]", "[value]"
    """
    try:
        dom, slot, value = dom_slot_value.split("-")
    except ValueError:
        tempo = dom_slot_value.split("-")
        if len(tempo) < 2:
            return dom_slot_value, dom_slot_value, dom_slot_value
        dom, slot = tempo[0], tempo[1]
        value = dom_slot_value.replace(f"{dom}-{slot}-", "").strip()

    if get_domain_slot:
        return f"{dom}-{slot}", value
    return dom, slot, value


def build_slot_meta(data):
    slot_meta = []
    for dialog in data:
        for turn in dialog["dialogue"]:
            if not turn.get("state"):
                continue

            for dom_slot_value in turn["state"]:
                domain_slot, _ = split_slot(dom_slot_value, get_domain_slot=True)
                if domain_slot not in slot_meta:
                    slot_meta.append(domain_slot)
    return sorted(slot_meta)


def convert_state_dict(state):
    """[summary]

    Args:
        state (str): "[domain]-[slot]-[value]" 형태의 state label

    Returns:
        dic (dictionary): key="[domain]-[slot]", value="[value]" 인 딕셔너리
    """
    dic = {}
    for slot in state:
        s, v = split_slot(slot, get_domain_slot=True)
        dic[s] = v
    return dic


@dataclass
class DSTInputExample:
    """[summary]

    Variables:
        - guid: 대화의 아이디(한 dialogue 묶음 내 한 대화의 인덱스)
        - context_turns: 해당 발화 이전까지의 대화를 담은 context_turns
        - current_turn: 현재 발화 (시스템-유저 대화 쌍)
        - label: slot의 값들, state

    Returns:
        위 variable 들을 묶은 인스턴스
    """
    guid: str
    context_turns: List[str]
    current_turn: List[str]
    label: Optional[List[str]] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_examples_from_dialogue(dialogue, user_first=False):
    guid = dialogue["dialogue_idx"]
    examples = []
    history = []
    d_idx = 0
    for idx, turn in enumerate(dialogue["dialogue"]):
        if turn["role"] != "user":
            continue

        if idx:
            sys_utter = dialogue["dialogue"][idx - 1]["text"]
        else:
            sys_utter = ""

        user_utter = turn["text"]
        state = turn.get("state")
        context = deepcopy(history)
        # TODO: 이해 안감.. 왜 user_first라고 순서가 바뀌는지..?????????? 일단 False가 들어가니 스킵
        if user_first:
            current_turn = [user_utter, sys_utter]
        else:
            current_turn = [sys_utter, user_utter]
        examples.append(
            DSTInputExample(
                guid=f"{guid}-{d_idx}",
                context_turns=context,
                current_turn=current_turn,
                label=state,
            )
        )
        # TODO: 사실 user_first=True라면 이 부분도 수정되어야 하는 것 아닌지??
        history.append(sys_utter)
        history.append(user_utter)
        d_idx += 1
    return examples


def get_examples_from_dialogues(data, user_first=False, dialogue_level=False):
    examples = []
    for d in tqdm(data):
        example = get_examples_from_dialogue(d, user_first=user_first)
        if dialogue_level:
            examples.append(example)
        else:
            examples.extend(example)
    return examples


class DSTPreprocessor:
    def __init__(self, slot_meta, src_tokenizer, trg_tokenizer=None, ontology=None):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology

    def pad_ids(self, arrays, pad_idx, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays

    def pad_id_of_matrix(self, arrays, padding, max_length=-1, left=False):
        if max_length < 0:
            max_length = max([array.size(-1) for array in arrays])

        new_arrays = []
        for i, array in enumerate(arrays):
            n, l = array.size()
            pad = torch.zeros(n, (max_length - l))
            pad[
                :,
                :,
            ] = padding
            pad = pad.long()
            m = torch.cat([array, pad], -1)
            new_arrays.append(m.unsqueeze(0))

        return torch.cat(new_arrays, 0)

    def _convert_example_to_feature(self):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def recover_state(self):
        raise NotImplementedError
