from typing import List
from tqdm import tqdm
import numpy as np
import torch
from data_utils import (
    DSTPreprocessor,
    OpenVocabDSTFeature,
    convert_state_dict,
    DSTInputExample,
    OntologyDSTFeature,
    _truncate_seq_pair,
)
from som_dst_utils import (
    convert_state_dict,
    SLOT_TOKEN,
    NULL_TOKEN,
    DOMAIN2ID,
    OP_SET,
    SomDSTFeature,
    SomDSTInputExample,
    flatten
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

    def _convert_example_to_feature(self, example):
        guid = example[0].guid.rsplit("-", 1)[0]  # dialogue_idx
        turns = []
        token_types = []
        labels = []
        num_turn = None
        for turn in example[: self.max_turn_length]:
            assert len(turn.current_turn) == 2
            uttrs = []
            for segment_idx, uttr in enumerate(turn.current_turn):
                token = self.src_tokenizer.encode(uttr, add_special_tokens=False)
                uttrs.append(token)

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
            label = []
            if turn.label:
                slot_dict = convert_state_dict(turn.label)
            else:
                slot_dict = {}
            for slot_type in self.slot_meta:
                value = slot_dict.get(slot_type, "none")
                # TODO
                # raise Exception('label_idx를 ontology에서 꺼내오는 코드를 작성하세요!')
                if value in self.ontology[slot_type]:
                    label_idx = self.ontology[slot_type].index(value)
                else:
                    label_idx = self.ontology[slot_type].index("none")
                label.append(label_idx)
            labels.append(label)
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
        return list(map(self._convert_example_to_feature, examples))

    def recover_state(self, pred_slots, num_turn):
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
        input_masks = input_ids.ne(
            self.src_tokenizer.pad_token_id
        )  # torch.ne - compute a != b
        target_ids = torch.LongTensor([b.target_ids for b in batch])
        num_turns = [b.num_turn for b in batch]
        return input_ids, segment_ids, input_masks, target_ids, num_turns, guids


class SomDSTPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta: dict,
        tokenizer,
        max_seq_length: int = 512,
        word_dropout: float = 0.1,
        slot_token: str = SLOT_TOKEN,
        domain2id: dict = None,
        op_code: str = "4",
    ):
        self.slot_meta = slot_meta
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.word_dropout = word_dropout
        self.slot_token = slot_token
        self.domain2id = DOMAIN2ID if domain2id is None else domain2id
        self.op2id = OP_SET[op_code]

    def convert_examples_to_features(self, examples):
        features = [
            self._convert_example_to_feature(example)
            for example in tqdm(examples, desc="[Conversion: Examples > Features]")
        ]
        return features

    def _convert_example_to_feature(self, example: SomDSTInputExample) -> SomDSTFeature:
        state = self._get_state_from_example(example)
        diag, segment_raw = self._get_diag_segment_from_example(example, state)
        input_ = diag + state
        segment_raw += [1] * len(state)

        slot_positions = self._get_slot_positions(input_)
        input_masks = [1] * len(input_)
        input_ids = self.tokenizer.convert_tokens_to_ids(input_)

        if len(input_masks) < self.max_seq_length:
            input_ids, segment_ids, input_masks = self._apply_padding(
                input_ids, segment_raw, input_masks
            )
        else:
            segment_ids = segment_raw

        domain_id = self.domain2id[example.turn_domain]
        op_ids = [self.op2id[a] for a in example.op_labels]
        generate_ids = [
            self.tokenizer.convert_tokens_to_ids(y) for y in example.generate_y
        ]

        return SomDSTFeature(
            guid=example.guid,
            turn_id=example.turn_id,
            last_dialogue_state=example.last_dialogue_state, # for evaluation
            gold_p_state=example.gold_p_state, # for evaluation
            gold_state=example.gold_state, # for evaluation
            input_ids=input_ids,
            input_masks=input_masks,
            segment_ids=segment_ids,
            op_ids=op_ids,
            slot_positions=slot_positions,
            domain_id=domain_id,
            generate_ids=generate_ids,
            is_last_turn=example.is_last_turn
        )


    def collate_fn(self, batch):
        input_ids = torch.LongTensor([f.input_ids for f in batch])
        input_masks = torch.LongTensor([f.input_masks for f in batch])
        segment_ids = torch.LongTensor([f.segment_ids for f in batch])

        state_position_ids = torch.LongTensor([f.slot_positions for f in batch])
        op_ids = torch.LongTensor([f.op_ids for f in batch])
        domain_ids = torch.LongTensor([f.domain_id for f in batch])
        gen_ids = [f.generate_ids for f in batch]
        max_update = max([len(b) for b in gen_ids]) # 최대 업데이트 수
        max_value = max([len(b) for b in flatten(gen_ids)]) # 생성할 레이블의 최대 길이

        for bid, b in enumerate(gen_ids):
            n_update = len(b)
            for idx, v in enumerate(b):
                b[idx] = v + [self.tokenizer.pad_token_id] * (max_value - len(v))
            gen_ids[bid] = b + [[self.tokenizer.pad_token_id] * max_value] * (max_update - n_update)
        gen_ids = torch.LongTensor(gen_ids) # (batch_size, max_update, max_value)

        return input_ids, input_masks, segment_ids, state_position_ids, op_ids, domain_ids, gen_ids, max_value, max_update



    def _get_diag_segment_from_example(self, example, state):
        diag_1 = self.tokenizer.tokenize(example.context_turns)
        diag_2 = self.tokenizer.tokenize(example.turn_uttr)
        diag_1, diag_2 = self._truncate(diag_1, diag_2, state)

        diag_1 = [self.tokenizer.cls_token] + diag_1 + [self.tokenizer.sep_token]
        diag_2 = diag_2 + [self.tokenizer.sep_token]

        segment = [0] * len(diag_1) + [1] * len(diag_2)
        diag = diag_1 + diag_2

        # word dropout
        if self.word_dropout > 0.0:
            diag = self._word_dropout(diag_1, diag_2, diag)

        return diag, segment

    def _apply_padding(self, input_id, segment_id, input_mask):
        num_pads = self.max_seq_length - len(input_mask)
        input_id += [self.tokenizer.pad_token_id] * num_pads
        segment_id += [self.tokenizer.pad_token_id] * num_pads
        input_mask += [self.tokenizer.pad_token_id] * num_pads
        return input_id, segment_id, input_mask

    def _word_dropout(self, diag_1, diag_2, diag):
        drop_mask = [0] + [1] * (len(diag_1) - 2) + [0] + [1] * (len(diag_2) - 1) + [0]
        drop_mask = np.array(drop_mask)
        word_drop = np.random.binomial(drop_mask.astype("int64"), self.word_dropout)
        diag = [w if word_drop[i] == 0 else self.tokenizer.unk_token for i, w in enumerate(diag)]
        return diag

    def _get_slot_positions(self, input_: list) -> list:
        slot_positions = [i for i, t in enumerate(input_) if t == self.slot_token]
        return slot_positions

    def _get_state_from_example(self, example):
        last_dial_state_dict = convert_state_dict(example.last_dialogue_state)
        state = []
        for s in self.slot_meta:
            state.append(self.slot_token)
            k = s.split("-")
            v = last_dial_state_dict.get(s)
            
            if v is not None: # 직전 turn까지의 state에 슬릇 s에 대한 value가 있을 경우
                k.extend(["-", v])
                t = self.tokenizer.tokenize(" ".join(k))
            else: # 직전 turn까지의 state에 슬릇 s에 대한 value가 없을 경우 => 해당 value를 [Null] 처리
                t = self.tokenizer.tokenize(" ".join(k))
                t.extend(["-", NULL_TOKEN])  # 이전 값이 없으면
            state.extend(t)
        return state

    def _truncate(self, diag_1, diag_2, state):
        avail_length_1 = self.max_seq_length - len(state) - 3
        avail_length = avail_length_1 - len(diag_2)

        if len(diag_1) > avail_length:  # truncated
            avail_length = len(diag_1) - avail_length
            diag_1 = diag_1[avail_length:]

        if len(diag_1) == 0 and len(diag_2) > avail_length_1:
            avail_length = len(diag_2) - avail_length_1
            diag_2 = diag_2[avail_length:]

        return diag_1, diag_2


if __name__ == '__main__':
    import json
    from torch.utils.data import DataLoader, RandomSampler
    from transformers import BertTokenizer
    from data_utils import load_dataset, WOSDataset
    from som_dst_utils import get_somdst_examples_from_dialogues

    train_data_file = './input/data/train_dataset/train_dials.json'
    slot_meta = json.load(open("./input/data/train_dataset/slot_meta.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file)

    tokenizer = BertTokenizer.from_pretrained('dsksd/bert-ko-small-minimal')
    tokenizer.add_special_tokens({'additional_special_tokens': [NULL_TOKEN, SLOT_TOKEN]})

    train_examples = get_somdst_examples_from_dialogues(
        train_data, slot_meta, tokenizer
    )

    dev_examples = get_somdst_examples_from_dialogues(
        dev_data, slot_meta, tokenizer
    )

    preprocessor = SomDSTPreprocessor(slot_meta=slot_meta, tokenizer=tokenizer)
    preprocessor._convert_example_to_feature(train_examples[0])

    train_features = [preprocessor._convert_example_to_feature(train_examples[i]) for i in range(10000)]
    dev_features = [preprocessor._convert_example_to_feature(dev_examples[i]) for i in range(len(dev_examples))]

    dataset = WOSDataset(features=train_features)
    sampler = RandomSampler(dataset)

    loader = DataLoader(
        dataset, 
        batch_size=11,
        sampler=sampler,
        collate_fn=preprocessor.collate_fn
    )

    for batch in loader:
        break

    print(batch)
    