from typing import *
from tqdm import tqdm
import numpy as np
import torch
from transformers import PreTrainedTokenizer
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
    flatten,
)


class TRADEPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=512,
        use_n_gate=5,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        if use_n_gate==3:
            self.gating2id = {"none": 0, "dontcare": 1, "ptr": 2}
        if use_n_gate==5:
            self.gating2id = {"none": 0, "dontcare": 1, "yes": 2, "no": 3, "ptr": 4}
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
        max_length = self.max_seq_length - 2
        if len(input_id) > max_length:
            gap = len(input_id) - max_length
            input_id = input_id[gap:]

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

            if self.id2gating[gate] in ["dontcare", "yes", "no"]:
                recovered.append("%s-%s" % (slot, self.id2gating[gate]))
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
        src_tokenizer: PreTrainedTokenizer,
        trg_tokenizer: PreTrainedTokenizer = None,
        max_seq_length: int = 512,
        word_dropout: float = 0.1,
        slot_token: str = "[SLOT]",
        domain2id: dict = None,
        op_code: str = "4",
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = src_tokenizer if trg_tokenizer is None else trg_tokenizer
        self.max_seq_length = max_seq_length
        self.word_dropout = word_dropout
        self.slot_token = slot_token
        self.domain2id = DOMAIN2ID if domain2id is None else domain2id
        self.op_code = op_code
        self.op2id = OP_SET[op_code]

        assert self.src_tokenizer.eos_token_id is not None
        return

    def convert_examples_to_features(
        self,
        examples: List[SomDSTInputExample],
        dynamic: bool = False,
        word_dropout: float = None,
    ) -> List[SomDSTFeature]:
        if word_dropout is None:
            word_dropout = self.word_dropout
        features = [
            self._convert_example_to_feature(
                example, dynamic=dynamic, word_dropout=word_dropout
            )
            for example in tqdm(examples, desc="[Examples>>>Features]")
        ]
        return features

    def collate_fn(self, batch):
        input_ids = torch.LongTensor([f.input_id for f in batch])
        input_mask = torch.LongTensor([f.input_mask for f in batch])
        segment_ids = torch.LongTensor([f.segment_id for f in batch])
        state_position_ids = torch.LongTensor([f.slot_position for f in batch])
        op_ids = torch.LongTensor([f.op_ids for f in batch])
        domain_ids = torch.LongTensor([f.domain_id for f in batch])
        gen_ids = [b.generate_ids for b in batch]
        try:
            max_update = max([len(b) for b in gen_ids]) # 최대 업데이트 수
        except:
            print('there is no slot to update')
            max_update = 1

        try:
            max_value = max([len(b) for b in flatten(gen_ids)]) # value의 최대 길이
        except:
            print('value is empty')
            max_value = 1
            
        for bid, b in enumerate(gen_ids):
            n_update = len(b)
            for idx, v in enumerate(b):
                b[idx] = v + [self.src_tokenizer.pad_token_id] * (max_value - len(v))
            gen_ids[bid] = b + [[self.src_tokenizer.pad_token_id] * max_value] * (max_update - n_update)
        gen_ids = torch.LongTensor(gen_ids)
        return (
            input_ids,
            input_mask,
            segment_ids,
            state_position_ids,
            op_ids,
            domain_ids,
            gen_ids,
            max_value,
            max_update,
        )

    def recover_state(self, last_dialog_state):
        return

    def _convert_example_to_feature(
        self, example, dynamic: bool = False, word_dropout: float = None
    ):
        op_labels, generate_y, gold_state = self._make_turn_label(
            example, dynamic=dynamic
        )
        (
            input_,
            input_id,
            segment_id,
            input_mask,
            slot_position,
            domain_id,
            op_ids,
            generate_ids,
        ) = self._get_features(
            example=example,
            op_labels=op_labels,
            generate_y=generate_y,
            word_dropout=word_dropout,
        )

        feature = SomDSTFeature(
            guid=example.guid,
            turn_domain=example.turn_domain,
            turn_id=example.turn_id,
            turn_utter=example.turn_utter,
            dialog_history=example.dialog_history,
            last_dialog_state=example.last_dialog_state,
            op_labels=op_labels,
            generate_y=generate_y,
            gold_state=gold_state,
            gold_p_state=example.last_dialog_state,
            is_last_turn=example.is_last_turn,
            input_=input_,
            input_id=input_id,
            segment_id=segment_id,
            input_mask=input_mask,
            slot_position=slot_position,
            domain_id=domain_id,
            op_ids=op_ids,
            generate_ids=generate_ids,
        )
        return feature

    def _get_features(self, example, op_labels, generate_y, word_dropout):
        state = []
        for s in self.slot_meta:
            state.append(self.slot_token)
            k = s.split("-")  # [도메인, 밸류]
            v = example.last_dialog_state.get(s)
            if v is not None:
                k.extend(["-", v])
                t = self.src_tokenizer.tokenize(" ".join(k))
            else:
                t = self.src_tokenizer.tokenize(" ".join(k))
                t.extend(["-", "[NULL]"])  # 이전 값이 없으면 value를 [NULL] 처리
            state.extend(t)

        avail_length_1 = self.max_seq_length - len(state) - 3

        diag_1 = self.src_tokenizer.tokenize(example.dialog_history)
        diag_2 = self.src_tokenizer.tokenize(example.turn_utter)
        avail_length = avail_length_1 - len(diag_2)

        if len(diag_1) > avail_length:  # truncated
            avail_length = len(diag_1) - avail_length
            diag_1 = diag_1[avail_length:]

        if len(diag_1) == 0 and len(diag_2) > avail_length_1:
            avail_length = len(diag_2) - avail_length_1
            diag_2 = diag_2[avail_length:]

        drop_mask = [0] + [1] * len(diag_1) + [0] + [1] * len(diag_2) + [0]
        diag_1 = (
            [self.src_tokenizer.cls_token] + diag_1 + [self.src_tokenizer.sep_token]
        )
        diag_2 = diag_2 + [self.src_tokenizer.sep_token]
        segment = [0] * len(diag_1) + [1] * len(diag_2)

        diag = diag_1 + diag_2

        # word dropout
        if word_dropout > 0.0:
            drop_mask = np.array(drop_mask)
            word_drop = np.random.binomial(drop_mask.astype("int64"), word_dropout)
            diag = [
                w if word_drop[i] == 0 else self.src_tokenizer.unk_token
                for i, w in enumerate(diag)
            ]
        input_ = diag + state
        segment = segment + [1] * len(state)
        input_ = input_  # X_t

        segment_id = segment
        slot_position = []
        for i, t in enumerate(input_):
            if t == self.slot_token:
                slot_position.append(i)

        input_mask = [1] * len(input_)  # 어디까지가 input이고, 어디부터가 패딩인지 표시하는거 같은데
        input_id = self.src_tokenizer.convert_tokens_to_ids(input_)

        if len(input_mask) < self.max_seq_length:
            additional_paddings = [self.src_tokenizer.pad_token_id] * (
                self.max_seq_length - len(input_id)
            )
            input_id += additional_paddings
            segment_id += additional_paddings
            input_mask += additional_paddings

        domain_id = self.domain2id[example.turn_domain]
        op_ids = [self.op2id[a] for a in op_labels]
        generate_ids = [self.src_tokenizer.convert_tokens_to_ids(y) for y in generate_y]

        return (
            input_,
            input_id,
            segment_id,
            input_mask,
            slot_position,
            domain_id,
            op_ids,
            generate_ids,
        )

    def _make_turn_label(
        self, example, dynamic=False
    ) -> Tuple[List[str], List[List[str]], List[str]]:

        if dynamic:
            gold_state = example.current_dialog_state
            example.current_dialog_state = {}
            for x in gold_state:
                s = x.split("-")
                k = "-".join(s[:2])
                example.current_dialog_state[k] = s[2]

        op_labels = ["carryover"] * len(self.slot_meta)  # 일단 carryover이 디폴트
        generate_y = []
        keys = list(
            example.current_dialog_state.keys()
        )  # 이번 turn의 slot에 대한 '도메인-슬릇' 목록

        # turn 내 DS의 각 '도메인-슬릇'에 대해 iterate
        for k in keys:
            v = example.current_dialog_state[k]  # v: 이번 turn의 한 slot에 대한 value,

            # value가 none인 경우, operation 레이블링에서 제외
            if v == "none":
                example.current_dialog_state.pop(k)
                continue
            vv = example.last_dialog_state.get(k)  # vv: 현재까지의 한 slot에 대한 value
            try:
                idx = self.slot_meta.index(k)
                if vv != v:  # abstract value 또는 특정 value로 맵핑
                    if v == "dontcare" and self.op2id.get("dontcare") is not None:
                        op_labels[idx] = "dontcare"
                    elif v == "yes" and self.op2id.get("yes") is not None:
                        op_labels[idx] = "yes"
                    elif v == "no" and self.op2id.get("no") is not None:
                        op_labels[idx] = "no"
                    else:
                        op_labels[idx] = "update"
                        generate_y.append(
                            [self.src_tokenizer.tokenize(v) + [self.src_tokenizer.eos_token], idx]
                        )
                elif vv == v:  # 같으면 그대로
                    op_labels[idx] = "carryover"
            except ValueError:
                continue

        # last_dialog_state: {'도메인-슬릇': '밸류', '''}
        for (
            k,
            v,
        ) in example.last_dialog_state.items():  # 이전 turn까지의 모든 '도메인-슬릇' ,'밸류'를 확인
            vv = example.current_dialog_state.get(k)  # 현재 turn에서 도메인-슬릇에 대한 value
            try:
                idx = self.slot_meta.index(k)  # 해당 도메인-슬릇의 위치번호
                if vv is None:  # 현재 turn에서 존재하지 않을 경우
                    print('vv is None')
                    if self.op2id.get("delete") is not None:  # delete operation이 있을 경우
                        op_labels[idx] = "delete"
                    else:  # delete operation이 없을 경우
                        op_labels[idx] = "update"
                        generate_y.append([[NULL_TOKEN, self.src_tokenizer.eos_token], idx])
            except ValueError:
                continue
        gold_state = [
            str(k) + "-" + str(v) for k, v in example.current_dialog_state.items()
        ]  # 도메인-슬릇-밸류

        if len(generate_y) > 0:
            generate_y = sorted(generate_y, key=lambda lst: lst[1])
            generate_y, _ = [list(e) for e in list(zip(*generate_y))]  # 생성해야할 것

        if dynamic:
            generate_y = [
                self.src_tokenizer.convert_tokens_to_ids(y) for y in generate_y
            ]
            op_labels = [self.op2id[i] for i in op_labels]

        return op_labels, generate_y, gold_state  # operation GT, value GT, 도메인-슬릇-밸류 GT


if __name__ == '__main__':
    import json
    from transformers import BertTokenizer
    from som_dst_utils import DOMAIN2ID, get_somdst_examples_from_dialogues

    train_data = json.load(open('./preprocessed/valid_dials.json'))
    tokenizer = BertTokenizer.from_pretrained('dsksd/bert-ko-small-minimal')
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ['[NULL]', '[SLOT]'], 'eos_token': '[EOS]'}
        )

    slot_meta = json.load(open("./input/data/train_dataset/slot_meta.json"))
    domain2id = DOMAIN2ID
    id2domain = {idx: value for value, idx in domain2id.items()}

    train_examples = get_somdst_examples_from_dialogues(train_data, n_history=1)

    preprocessor = SomDSTPreprocessor(
        slot_meta=slot_meta,
        src_tokenizer=tokenizer,
        max_seq_length=512,
        word_dropout=0.0
    )
    train_features = preprocessor.convert_examples_to_features(train_examples)