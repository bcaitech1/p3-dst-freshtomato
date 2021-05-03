from typing import List
from tqdm import tqdm
import torch
from data_utils import (
    DSTPreprocessor,
    OpenVocabDSTFeature,
    convert_state_dict,
    DSTInputExample,
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
        if self.src_tokenizer.special_tokens_map['sep_token'] == '</s>':
            dialogue_context = " <s> ".join(example.context_turns + example.current_turn)
        else:
            dialogue_context = " [SEP] ".join(example.context_turns + example.current_turn)

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
