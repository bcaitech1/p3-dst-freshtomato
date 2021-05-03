from typing import List
from tqdm import tqdm
import torch
from data_utils import (
    DSTPreprocessor,
    OpenVocabDSTFeature,
    convert_state_dict,
    DSTInputExample,
    OntologyDSTFeature
)


class TRADEPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=512,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.gating2id = {"none": 0, "dontcare": 1, "ptr": 2}
        self.id2gating = {v: k for k, v in self.gating2id.items()}
        self.max_seq_length = max_seq_length

    def _convert_example_to_feature(
        self, example: DSTInputExample
    ) -> OpenVocabDSTFeature:
        """List[DSTInputExample]를 feature로 변형하는 데 사용되는 nested 함수. 다음과 같이 사용
        Examples:
            processor = TRADEPreprocessor(slot_meta, tokenizer)
            features = processor.convert_examples_to_features(examples)

        Args:
            example (DSTInputExample)

        Returns:
            [OpenVocabDSTFeature]: feature 데이터
        """

        # XLM-Robert 토크나이저 케이스 추가
        if self.src_tokenizer.special_tokens_map["sep_token"] == "</s>":
            dialogue_context = " <s> ".join(
                example.context_turns + example.current_turn
            )
        else:
            dialogue_context = " [SEP] ".join(
                example.context_turns + example.current_turn
            )

        input_id = self.src_tokenizer.encode(dialogue_context, add_special_tokens=False)

        input_id = (
            [self.src_tokenizer.cls_token_id]
            + input_id
            + [self.src_tokenizer.sep_token_id]
        )
        segment_id = [0] * len(input_id)

        target_ids = []
        gating_id = []
        if not example.label:
            example.label = []

        state = convert_state_dict(example.label)
        for slot in self.slot_meta:
            value = state.get(slot, "none")
            target_id = self.trg_tokenizer.encode(value, add_special_tokens=False) + [
                self.trg_tokenizer.sep_token_id
            ]
            target_ids.append(target_id)
            gating_id.append(self.gating2id.get(value, self.gating2id["ptr"]))
        target_ids = self.pad_ids(target_ids, self.trg_tokenizer.pad_token_id)
        return OpenVocabDSTFeature(
            example.guid, input_id, segment_id, gating_id, target_ids
        )

    def convert_examples_to_features(
        self, examples: List[DSTInputExample]
    ) -> List[OpenVocabDSTFeature]:
        """복수의 DSTInputExmple 각각을 feature로 변형하는 함수

        Args:
            examples (List[DSTInputExample]): DSTInputExample로 구성된 리스트

        Returns:
            List[OpenVocabDSTFeature]: feature로 변형된 데이터의 리스트
        """
        features = [
            self._convert_example_to_feature(e)
            for e in tqdm(examples, desc="[Conversion: Examples > Features]")
        ]
        return features

    def recover_state(self, gate_list, gen_list):
        assert len(gate_list) == len(self.slot_meta)
        assert len(gen_list) == len(self.slot_meta)

        recovered = []
        for slot, gate, value in zip(self.slot_meta, gate_list, gen_list):
            if self.id2gating[gate] == "none":
                continue

            if self.id2gating[gate] == "dontcare":
                recovered.append("%s-%s" % (slot, "dontcare"))
                continue

            token_id_list = []
            for id_ in value:
                if id_ in self.trg_tokenizer.all_special_ids:
                    break

                token_id_list.append(id_)
            value = self.trg_tokenizer.decode(token_id_list, skip_special_tokens=True)

            if value == "none":
                continue

            recovered.append("%s-%s" % (slot, value))
        return recovered

    def collate_fn(self, batch):
        guids = [b.guid for b in batch]
        input_ids = torch.LongTensor(
            self.pad_ids([b.input_id for b in batch], self.src_tokenizer.pad_token_id)
        )
        segment_ids = torch.LongTensor(
            self.pad_ids([b.segment_id for b in batch], self.src_tokenizer.pad_token_id)
        )
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)

        gating_ids = torch.LongTensor([b.gating_id for b in batch])
        target_ids = self.pad_id_of_matrix(
            [torch.LongTensor(b.target_ids) for b in batch],
            self.trg_tokenizer.pad_token_id,
        )
        return input_ids, segment_ids, input_masks, gating_ids, target_ids, guids


class SUMBTPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=64,
        max_turn_length=14,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.max_seq_length = max_seq_length
        self.max_turn_length = max_turn_length

    def _convert_example_to_feature(
        self, example: List[DSTInputExample]
    ) -> OntologyDSTFeature:
        """Dialogue 단위 내 각 turn별 DSTInputExmple을 feature로 변형하는 함수

        Args:
            example (list): 단일 dialogue. DSTInputExmple(turn)로 구성된 리스트

        Returns:
            [type]: [description]
        """
        guid = example[0].guid.rsplit("-", 1)[0]  # dialogue_idx
        turns = []
        token_types = []
        labels = []
        num_turn = None

        for turn in example[: self.max_turn_length]:
            assert len(turn.current_turn) == 2  # current turn은 시스템과 유저의 turn의 2개여야 함

            # Current turn의 utterance 토큰화
            uttrs = [
                self.src_tokenizer.encode(uttr, add_special_tokens=False)
                for uttr in turn.current_turn
            ]

            _truncate_seq_pair(uttrs[0], uttrs[1], self.max_seq_length - 3)
            tokens = (
                [self.src_tokenizer.cls_token_id]
                + uttrs[0]
                + [self.src_tokenizer.sep_token_id]
                + uttrs[1]
                + [self.src_tokenizer.sep_token_id]
            )
            token_type = [0] * (len(uttrs[0]) + 2) + [1] * (len(uttrs[1]) + 1)
            if len(tokens) < self.max_seq_length:
                gap = self.max_seq_length - len(tokens)
                tokens.extend([self.src_tokenizer.pad_token_id] * gap)
                token_type.extend([0] * gap)

            turns.append(tokens)
            token_types.append(token_type)

            # 레이블 정보 추가
            label = []
            slot_dict = convert_state_dict(turn.label) if turn.label else {}

            for slot_type in self.slot_meta:
                value = slot_dict.get(slot_type, "none")
                label_idx = ontology[slot_type].index(value)
                label.append(label_idx)

            labels.append(label)

        # Packing
        num_turn = len(turns)
        if len(turns) < self.max_turn_length:
            gap = self.max_turn_length - len(turns)
            for _ in range(gap):
                dummy_turn = [self.src_tokenizer.pad_token_id] * self.max_seq_length
                turns.append(dummy_turn)
                token_types.append(dummy_turn)
                dummy_label = [-1] * len(self.slot_meta)
                labels.append(dummy_label)

        return OntologyDSTFeature(
            guid=guid,
            input_ids=turns,
            segment_ids=token_types,
            num_turn=num_turn,
            target_ids=labels,
        )

    def convert_examples_to_features(self, examples):
        return [
            self._convert_example_to_feature(example)
            for example in tqdm(examples, desc="[Conversion: Examples > Features]")
        ]

    def recover_state(self, pred_slots, num_turn):
        """포맷에 맞게 예측값을 출력하는 함수

        Args:
            pred_slots ([type]): [description]
            num_turn ([type]): [description]

        Returns:
            [type]: [description]
        """
        states = []
        for pred_slot in pred_slots[:num_turn]:
            state = []
            for s, p in zip(self.slot_meta, pred_slot):
                v = self.ontology[s][p]
                if v != "none":
                    state.append(f"{s}-{v}")
            states.append(state)
        return states

    def collate_fn(self, batch):
        guids = [b.guid for b in batch]
        input_ids = torch.LongTensor([b.input_ids for b in batch])
        segment_ids = torch.LongTensor([b.segment_ids for b in batch])
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)
        target_ids = torch.LongTensor([b.target_ids for b in batch])
        num_turns = [b.num_turn for b in batch]
        return input_ids, segment_ids, input_masks, target_ids, num_turns, guids


if __name__ == "__main__":
    import json
    from tqdm import tqdm
    from transformers import BertTokenizer
    from data_utils import get_examples_from_dialogues, convert_state_dict, load_dataset
    from data_utils import OntologyDSTFeature, DSTPreprocessor, _truncate_seq_pair

    train_data_file = "./input/data/train_dataset/train_dials.json"
    slot_meta = json.load(open("./input/data/train_dataset/slot_meta.json"))
    ontology = json.load(open("./input/data/train_dataset/ontology.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file)

    train_examples = get_examples_from_dialogues(
        data=train_data, user_first=True, dialogue_level=True
    )

    dev_examples = get_examples_from_dialogues(
        data=dev_data, user_first=True, dialogue_level=True
    )

    max_turn = max([len(e["dialogue"]) for e in train_data])
    tokenizer = BertTokenizer.from_pretrained("dsksd/bert-ko-small-minimal")
    processor = SUMBTPreprocessor(
        slot_meta,
        tokenizer,
        ontology=ontology,  # predefined ontology
        max_seq_length=64,  # 각 turn마다 최대 길이
        max_turn_length=max_turn,
    )  # 각 dialogue의 최대 turn 길이

    train_features = processor.convert_examples_to_features(train_examples)
    dev_features = processor.convert_examples_to_features(dev_examples)

    print(len(train_features))  # 대화 level의 features
    print(len(dev_features))
