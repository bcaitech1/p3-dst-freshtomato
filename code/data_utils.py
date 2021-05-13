import dataclasses
import json
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

DOMAINS = ['관광', '숙소', '식당', '지하철', '택시']


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


def load_dataset(dataset_path: str, dev_split: float = 0.1) -> Tuple[list, list, dict]:
    """Dialogue 데이터 경로를 입력 받아 train/valid/groun truth 데이터를 리턴

    Args:
        dataset_path (str): 학습에 활용할 Dialogue 데이터 경로
        dev_split (float, optional): 데이터 중 검증용 데이터에 활용할 비율. Defaults to 0.1.

    Returns:
        train_data, dev_data, dev_labels: 학습용 데이터와 검증용 데이터를 리턴하며, 각각 다음의 형태를 가짐
        - train_data: dialogue_idx, domains, dialogue의 key를 지닌 딕셔너리 형태
            [
                {'dialogue_idx': 'snowy-hat-8324:관광_식당_11',
                'domains': ['관광', '식당'],
                'dialogue': [{'role': ?, 'text': ?, 'state':?}, ... ]},
                ...
            ]
        - dev_data: 'dialogue' 내 'state'가 존재하지 않음
            [
                {'dialogue_idx': 'steep-limit-4198:식당_34',
                'domains': ['식당'],
                'dialogue': [{'role': ?, 'text': ?}, ... ]},
                ...
            ]
        - dev_label: dev_data의 각 turn별 state(label)이 나열
            {
                'steep-limit-4198:식당_34-0': ['식당-예약 명수-8'], # 'steep-limit-4198:식당_34' dialogue의 첫 user turn
                'steep-limit-4198:식당_34-1': [...],
                ...
            }
    """
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

            state = turn.pop("state")

            guid_t = f"{guid}-{d_idx}"
            d_idx += 1

            dev_labels[guid_t] = state

    return train_data, dev_data, dev_labels


def train_data_loading(args, isUserFirst, isDialogueLevel):
    # Data Loading
    train_data_file = f"{args.data_dir}/train_dials.json"
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file)

    train_examples = get_examples_from_dialogues(
        train_data, user_first=isUserFirst, dialogue_level=isDialogueLevel
    )
    dev_examples = get_examples_from_dialogues(
        dev_data, user_first=isUserFirst, dialogue_level=isDialogueLevel
    )

    return slot_meta, train_examples, dev_examples, dev_labels

def test_data_loading(args, isUserFirst, isDialogueLevel):
    eval_data = json.load(open(f"{args.data_dir}/eval_dials.json", "r"))

    eval_examples = get_examples_from_dialogues(
        eval_data, user_first=isUserFirst, dialogue_level=isDialogueLevel
    )

    return eval_examples

def get_data_loader(processor, features, batch_size):
    data = WOSDataset(features)
    sampler = RandomSampler(data)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=processor.collate_fn,
    )

    return loader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def split_slot(dom_slot_value: str, get_domain_slot=False) -> Tuple[str, str, str]:
    """도메인-슬릇-밸류 형태의 문자열을 triple로 분리하는 함수

    Args:
        dom_slot_value (str): 도메인-슬릇-밸류 문자열
        get_domain_slot (bool, optional): True시 도메인-슬릇/밸류의 형태로 분리. Defaults to False.
    """
    try:
        dom, slot, value = dom_slot_value.split("-")
    except ValueError:
        tempo = dom_slot_value.split("-")
        if len(tempo) < 2:
            return dom_slot_value, dom_slot_value, dom_slot_value # ?
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


def tokenize_ontology(ontology, tokenizer, max_seq_length=12):
    slot_types = []
    slot_values = []
    for k, v in ontology.items():
        tokens = tokenizer.encode(k)
        if len(tokens) < max_seq_length:
            gap = max_seq_length - len(tokens)
            tokens.extend([tokenizer.pad_token_id] *  gap)
        slot_types.append(tokens)
        slot_value = []
        for vv in v:
            tokens = tokenizer.encode(vv)
            if len(tokens) < max_seq_length:
                gap = max_seq_length - len(tokens)
                tokens.extend([tokenizer.pad_token_id] *  gap)
            slot_value.append(tokens)
        slot_values.append(torch.LongTensor(slot_value))
    return torch.LongTensor(slot_types), slot_values


def convert_state_dict(state):
    """
    Args:
        state ([type]): 특정 turn의 dial state

    Returns:
        dict: {도메인슬릇:밸류} 꼴의 state dict
    """
    if state:
        dic = {}
        for slot in state:
            s, v = split_slot(slot, get_domain_slot=True)
            dic[s] = v
        return dic
    else:
        return dict()


@dataclass
class DSTInputExample:
    """Dialogue State Tracking 정보를 담는 데이터 클래스. Tracking 정보는 다음의 정보를 담고 있음
    - guid: dialogue_idx + turn_idx 형태의 인덱스
    - context_turns: 현재 turn 이전까지의 dialogue context(=D_{t-1}). 현재까지의 누적 발화
    - current_turn: 현재 turn에서의 시스템/유저의 발화.
                    (system_{t}, user_{t}) 또는 (user_{t}, system_{t})의 형태
    - label: Turn t에서의 dialogue state(=B_{t})
    """

    guid: str
    current_turn: List[str]
    context_turns: List[str] = None
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


def get_examples_from_dialogue(
    dialogue: dict, user_first: bool = False
) -> List[DSTInputExample]:
    """단일의 발화 데이터로부터 DSTInputExample을 생성

    Args:
        dialogue (dict): 다음과 같은 단일 Dialogue data

                {'dialogue_idx': 'snowy-hat-8324:관광_식당_11',
                'domains': ['관광', '식당'],
                'dialogue': [{'role': 'user',...}, ...]}

        user_first (bool, optional): True시 context_turns와 current_turn이 (u_{t}, r_{t}) 형태. Defaults to False.
    Returns:
        List[DSTInputExample]: 단일 Dialogue 데이터로부터 추출된 DSTInputExample 리스트
    """
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

        history.append(sys_utter)
        history.append(user_utter)
        d_idx += 1

    return examples


def get_examples_from_dialogues(
    data: list, user_first: bool = False, dialogue_level: bool = False
) -> List[DSTInputExample]:
    """다중 발화 데이터로부터 DSTInputExample 리스트를 생성

    Args:
        data ([type]): 다음과 같은 다중 Dialogue data
            [
                {'dialogue_idx': 'snowy-hat-8324:관광_식당_11',
                'domains': ['관광', '식당'],
                'dialogue': [{'role': 'user',...}, ...]},
                {'dialogue_idx': 'snowy-hat-8324:관광_식당_11',
                'domains': ['관광', '식당'],
                'dialogue': [{'role': 'user',...}, ...]},
                ...
            ]
        user_first (bool, optional): True시 context_turns와 current_turn이 (u_{t}, r_{t}) 형태. Defaults to False.
        dialogue_level (bool, optional): True시 데이터 샘플이 dialogue 단위로 분리됨. Defaults to False.
            - True: [1번째 dialogue에 대한 DSTInputExample list, 2번째 dialogue에 대한 DSTInputExample list, ...]
            - False: [1번쟤 DSTInputExample, 2번째 DSTInputExample, ...]

    Returns:
        List[DSTInputExample]: DSTInputExample 리스트
    """
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
        """DSTInputExample을 InputFeature 형태로 변경"""
        raise NotImplementedError

    def recover_state(self):
        """모델의 출력을 prediction 포맷에 맞게 변경"""
        raise NotImplementedError