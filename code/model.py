import argparse
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineEmbeddingLoss, CrossEntropyLoss
from importlib import import_module
from transformers import BertModel, BertPreTrainedModel, ElectraModel, ElectraPreTrainedModel


class TRADE_original(nn.Module):
    def __init__(self, config, tokenized_slot_meta, pad_idx=0):
        super(TRADE_original, self).__init__()
        self.encoder = GRUEncoder(
            config.vocab_size,
            config.hidden_size,
            config.num_rnn_layers,
            config.hidden_dropout_prob,
            config.proj_dim,
            pad_idx,
        )

        self.decoder = SlotGenerator(
            config.vocab_size,
            config.hidden_size,
            config.hidden_dropout_prob,
            config.n_gate,
            config.proj_dim,
            pad_idx,
        )

        self.decoder.set_slot_idx(tokenized_slot_meta)
        self.tie_weight()

    def set_subword_embedding(self, config):  # args 전체를 input으로 받는 것으로 바뀌었음
        model_module = getattr(
            import_module("transformers"), f"{config.model_name}Model"
        )
        model = model_module.from_pretrained(config.pretrained_name_or_path)
        self.encoder.embed.weight = model.embeddings.word_embeddings.weight
        self.tie_weight()

    def tie_weight(self):
        self.decoder.embed.weight = self.encoder.embed.weight
        if self.decoder.proj_layer:
            self.decoder.proj_layer.weight = self.encoder.proj_layer.weight

    def forward(
        self, input_ids, token_type_ids, attention_mask=None, max_len=10, teacher=None
    ):

        encoder_outputs, pooled_output = self.encoder(input_ids=input_ids)
        all_point_outputs, all_gate_outputs = self.decoder(
            input_ids,
            encoder_outputs,
            pooled_output.unsqueeze(0),
            attention_mask,
            max_len,
            teacher,
        )

        return all_point_outputs, all_gate_outputs


class PLMEncoder(nn.Module):
    def __init__(self, config):
        super(PLMEncoder, self).__init__()
        plm_module = getattr(
                            import_module("transformers"), f"{config.model_name}Model"
                        )
        self.plm = plm_module.from_pretrained(config.pretrained_name_or_path)
    
    def forward(self, input_ids):
        if self.plm.pooler: # Bert의 경우 pooler layer 존재
            return self.plm(input_ids)
        else:
            output = self.plm(input_ids=input_ids)[0] # (B, T, D)
            pooled_output = output[:, 0, :] # 0 번째 CLS 토큰에 대한 것들
            return output, pooled_output


class TRADE(nn.Module):
    def __init__(self, config, tokenized_slot_meta, pad_idx=0):
        super(TRADE, self).__init__()
        self.encoder = PLMEncoder(config)    
        self.decoder = SlotGenerator(
            config.vocab_size,
            config.hidden_size,
            config.hidden_dropout_prob,
            config.n_gate,
            config.proj_dim,
            pad_idx,
        )

        self.decoder.set_slot_idx(tokenized_slot_meta)
        self.tie_weight()

    def set_subword_embedding(self, config):  # args 전체를 input으로 받는 것으로 바뀌었음
    #     model_module = getattr(
    #         import_module("transformers"), f"{config.model_name}Model"
    #     )
    #     model = model_module.from_pretrained(config.pretrained_name_or_path)
    #     self.encoder.embed.weight = model.embeddings.word_embeddings.weight
        self.tie_weight()

    def tie_weight(self):
        self.decoder.embed.weight = self.encoder.embeddings.word_embeddings.weight
    # def tie_weight(self):
    #     self.decoder.embed.weight = self.encoder.embed.weight
    #     if self.decoder.proj_layer:
    #         self.decoder.proj_layer.weight = self.encoder.proj_layer.weight

    def forward(
        self, input_ids, token_type_ids, attention_mask=None, max_len=10, teacher=None
    ):

        encoder_outputs, pooled_output = self.encoder(input_ids=input_ids)
        all_point_outputs, all_gate_outputs = self.decoder(
            input_ids,
            encoder_outputs,
            pooled_output.unsqueeze(0),
            attention_mask,
            max_len,
            teacher,
        )

        return all_point_outputs, all_gate_outputs


class SUMBT(nn.Module):
    def __init__(self, args, num_labels, device):
        super(SUMBT, self).__init__()

        self.hidden_dim = args.hidden_size
        self.rnn_num_layers = args.num_rnn_layers
        self.zero_init_rnn = args.zero_init_rnn
        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length
        self.num_labels = num_labels
        self.num_slots = len(num_labels)
        self.attn_head = args.attn_head
        self.device = device

        ### Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
            args.pretrained_name_or_path
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.bert.pooler.parameters():
                p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(
            args.pretrained_name_or_path
        )
        # os.path.join(args.bert_dir, 'bert-base-uncased.model'))
        for p in self.sv_encoder.bert.parameters():
            p.requires_grad = False

        self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim)
        self.value_lookup = nn.ModuleList(
            [nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels]
        )

        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0)

        ### RNN Belief Tracker
        self.nbt = nn.GRU(
            input_size=self.bert_output_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.rnn_num_layers,
            dropout=self.hidden_dropout_prob,
            batch_first=True,
        )
        self.init_parameter(self.nbt)

        if not self.zero_init_rnn:
            self.rnn_init_linear = nn.Sequential(
                nn.Linear(self.bert_output_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.hidden_dropout_prob),
            )

        self.linear = nn.Linear(self.hidden_dim, self.bert_output_dim)
        self.layer_norm = nn.LayerNorm(self.bert_output_dim)

        ### Measure
        self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        ### Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def initialize_slot_value_lookup(self, label_ids, slot_ids):

        self.sv_encoder.eval()

        # Slot encoding
        slot_type_ids = torch.zeros(slot_ids.size(), dtype=torch.long).to(
            slot_ids.device
        )
        slot_mask = slot_ids > 0
        hid_slot, _ = self.sv_encoder(
            slot_ids.view(-1, self.max_label_length),
            slot_type_ids.view(-1, self.max_label_length),
            slot_mask.view(-1, self.max_label_length),
        )
        hid_slot = hid_slot[:, 0, :]
        hid_slot = hid_slot.detach()
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)

        for s, label_id in enumerate(label_ids):
            label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(
                label_id.device
            )
            label_mask = label_id > 0
            hid_label, _ = self.sv_encoder(
                label_id.view(-1, self.max_label_length),
                label_type_ids.view(-1, self.max_label_length),
                label_mask.view(-1, self.max_label_length),
            )
            hid_label = hid_label[:, 0, :]
            hid_label = hid_label.detach()
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1

        print("Complete initialization of slot and value lookup")
        self.sv_encoder = None

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels=None,
        n_gpu=1,
        target_slot=None,
    ):
        # input_ids: [B, M, N]
        # token_type_ids: [B, M, N]
        # attention_mask: [B, M, N]
        # labels: [B, M, J]

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # Batch size (B)
        ts = input_ids.size(1)  # Max turn size (M)
        bs = ds * ts
        slot_dim = len(target_slot)  # J

        # Utterance encoding
        hidden, _ = self.utterance_encoder(
            input_ids.view(-1, self.max_seq_length),
            token_type_ids.view(-1, self.max_seq_length),
            attention_mask.view(-1, self.max_seq_length),
        )
        hidden = torch.mul(
            hidden,
            attention_mask.view(-1, self.max_seq_length, 1)
            .expand(hidden.size())
            .float(),
        )
        hidden = hidden.repeat(slot_dim, 1, 1)  # [J*M*B, N, H]

        hid_slot = self.slot_lookup.weight[
            target_slot, :
        ]  # Select target slot embedding
        hid_slot = hid_slot.repeat(1, bs).view(bs * slot_dim, -1)  # [J*M*B, N, H]

        # Attended utterance vector
        hidden = self.attn(
            hid_slot,  # q^s  [J*M*B, N, H]
            hidden,  # U [J*M*B, N, H]
            hidden,  # U [J*M*B, N, H]
            mask=attention_mask.view(-1, 1, self.max_seq_length).repeat(slot_dim, 1, 1),
        )
        hidden = hidden.squeeze()  # h [J*M*B, H] Aggregated Slot Context
        hidden = hidden.view(slot_dim, ds, ts, -1).view(
            -1, ts, self.bert_output_dim
        )  # [J*B, M, H]

        # NBT
        if self.zero_init_rnn:
            h = torch.zeros(
                self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim
            ).to(
                self.device
            )  # [1, slot_dim*ds, hidden]
        else:
            h = hidden[:, 0, :].unsqueeze(0).repeat(self.rnn_num_layers, 1, 1)
            h = self.rnn_init_linear(h)

        if isinstance(self.nbt, nn.GRU):
            rnn_out, _ = self.nbt(hidden, h)  # [J*B, M, H_GRU]
        elif isinstance(self.nbt, nn.LSTM):
            c = torch.zeros(
                self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim
            ).to(
                self.device
            )  # [1, slot_dim*ds, hidden]
            rnn_out, _ = self.nbt(hidden, (h, c))  # [slot_dim*ds, turn, hidden]
        rnn_out = self.layer_norm(self.linear(self.dropout(rnn_out)))

        hidden = rnn_out.view(slot_dim, ds, ts, -1)  # [J, B, M, H_GRU]

        # Label (slot-value) encoding
        loss = 0
        loss_slot = []
        pred_slot = []
        output = []
        for s, slot_id in enumerate(target_slot):  ## note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight
            num_slot_labels = hid_label.size(0)

            _hid_label = (
                hid_label.unsqueeze(0)
                .unsqueeze(0)
                .repeat(ds, ts, 1, 1)
                .view(ds * ts * num_slot_labels, -1)
            )
            _hidden = (
                hidden[s, :, :, :]
                .unsqueeze(2)
                .repeat(1, 1, num_slot_labels, 1)
                .view(ds * ts * num_slot_labels, -1)
            )
            _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)
            _dist = -_dist
            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(ds, ts, 1))
            output.append(_dist)

            if labels is not None:
                _loss = self.nll(_dist.view(ds * ts, -1), labels[:, :, s].view(-1))
                loss_slot.append(_loss.item())
                loss += _loss

        pred_slot = torch.cat(pred_slot, 2)
        if labels is None:
            return output, pred_slot

        # calculate joint accuracy
        accuracy = (pred_slot == labels).view(-1, slot_dim)
        acc_slot = (
            torch.sum(accuracy, 0).float()
            / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        )
        acc = (
            sum(torch.sum(accuracy, 1) / slot_dim).float()
            / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()
        )  # joint accuracy

        if n_gpu == 1:
            return loss, loss_slot, acc, acc_slot, pred_slot
        else:
            return (
                loss.unsqueeze(0),
                None,
                acc.unsqueeze(0),
                acc_slot.unsqueeze(0),
                pred_slot.unsqueeze(0),
            )

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)


class SomDST(BertPreTrainedModel):
    def __init__(
        self,
        config,
        n_op: int,
        n_domain: int,
        update_id: int,
        exclude_domain: bool = False,
    ):
        super(SomDST, self).__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.encoder = SomDSTEncoder(config, n_op, n_domain, update_id, exclude_domain)
        self.decoder = SomDSTDecoder(
            config, self.encoder.bert.embeddings.word_embeddings.weight
        )
        self.init_weights()

    def forward(
        self,
        input_ids,  # X_t의 input_id
        token_type_ids,  # X_t의 token_type_id
        state_positions,  # X_t의 position embeddings
        attention_mask,
        max_value,
        op_ids=None,
        max_update=None,
        teacher=None,
    ):

        # encoder phase
        enc_outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            state_positions=state_positions,
            attention_mask=attention_mask,
            op_ids=op_ids,
            max_update=max_update,
        )

        (
            domain_scores,
            state_scores,
            decoder_inputs,
            sequence_output,
            pooled_output,
        ) = enc_outputs

        # decoder phase
        gen_scores = self.decoder(
            x=input_ids,
            decoder_input=decoder_inputs,
            encoder_output=sequence_output,
            hidden=pooled_output,
            max_len=max_value,
            teacher=teacher,
        )

        return domain_scores, state_scores, gen_scores

# SOP
class SomDSTEncoder(nn.Module):
    """State Operation Predictor + Domain Predictor

    Args:
        nn ([type]): [description]
    """

    def __init__(self, config, n_op, n_domain, update_id, exclude_domain=False):
        super(SomDSTEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.exclude_domain = exclude_domain
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.action_cls = nn.Linear(config.hidden_size, n_op)

        if self.exclude_domain is not True:
            self.domain_cls = nn.Linear(config.hidden_size, n_domain)

        self.n_op = n_op
        self.n_domain = n_domain
        self.update_id = update_id

    def forward(self, input_ids, token_type_ids, state_positions, attention_mask, op_ids=None, max_update=None):
        bert_outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output, pooled_output = bert_outputs[:2]
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))
        state_output = torch.gather(sequence_output, 1, state_pos)
        state_scores = self.action_cls(self.dropout(state_output))  # B,J,4

        if self.exclude_domain:
            domain_scores = torch.zeros(1, device=input_ids.device)  # dummy
        else:
            domain_scores = self.domain_cls(self.dropout(pooled_output))

        batch_size = state_scores.size(0)
        if op_ids is None:
            op_ids = state_scores.view(-1, self.n_op).max(-1)[-1].view(batch_size, -1)
        if max_update is None:
            max_update = op_ids.eq(self.update_id).sum(-1).max().item()

        gathered = []
        for b, a in zip(state_output, op_ids.eq(self.update_id)):  # update
            if a.sum().item() != 0:
                v = b.masked_select(a.unsqueeze(-1)).view(1, -1, self.hidden_size)
                n = v.size(1)
                gap = max_update - n
                if gap > 0:
                    zeros = torch.zeros(1, 1*gap, self.hidden_size, device=input_ids.device)
                    v = torch.cat([v, zeros], 1)
            else:
                v = torch.zeros(1, max_update, self.hidden_size, device=input_ids.device)
            gathered.append(v)
        decoder_inputs = torch.cat(gathered)
        return domain_scores, state_scores, decoder_inputs, sequence_output, pooled_output.unsqueeze(0)

# SVG
class SomDSTDecoder(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(SomDSTDecoder, self).__init__()
        self.pad_idx = 0
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.pad_idx)
        self.embed.weight = bert_model_embedding_weights
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, 1, batch_first=True)
        self.w_gen = nn.Linear(config.hidden_size*3, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.dropout)

        # for n, p in self.gru.named_parameters():
        #     if 'weight' in n:
        #         p.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, x, decoder_input, encoder_output, hidden, max_len, teacher=None):
        mask = x.eq(self.pad_idx)
        batch_size, n_update, _ = decoder_input.size()  # B,J',5 # long
        state_in = decoder_input
        all_point_outputs = torch.zeros(n_update, batch_size, max_len, self.vocab_size).to(x.device)
        result_dict = {}
        for j in range(n_update):
            w = state_in[:, j].unsqueeze(1)  # B,1,D
            slot_value = []
            for k in range(max_len):
                w = self.dropout(w)
                _, hidden = self.gru(w, hidden)  # 1,B,D
                # B,T,D * B,D,1 => B,T
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
                attn_e = attn_e.squeeze(-1).masked_fill(mask, -1e9)
                attn_history = nn.functional.softmax(attn_e, -1)  # B,T

                # B,D * D,V => B,V
                attn_v = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B,V
                attn_vocab = nn.functional.softmax(attn_v, -1)

                # B,1,T * B,T,D => B,1,D
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D

                p_gen = self.sigmoid(self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1)))  # B,1
                p_gen = p_gen.squeeze(-1)

                p_context_ptr = torch.zeros_like(attn_vocab).to(x.device)
                p_context_ptr.scatter_add_(1, x, attn_history)  # copy B,V
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
                _, w_idx = p_final.max(-1)
                slot_value.append([ww.tolist() for ww in w_idx])
                
                if teacher is not None:
                    w = self.embed(teacher[:, j, k]).unsqueeze(1)
                else:
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D

                all_point_outputs[j, :, k, :] = p_final

        return all_point_outputs.transpose(0, 1)




class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, dropout, proj_dim=None, pad_idx=0):
        super(GRUEncoder, self).__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        if proj_dim:
            self.proj_layer = nn.Linear(d_model, proj_dim, bias=False)
        else:
            self.proj_layer = None

        self.d_model = proj_dim if proj_dim else d_model
        self.gru = nn.GRU(
            self.d_model,
            self.d_model,
            n_layer,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        mask = input_ids.eq(self.pad_idx).unsqueeze(-1)
        x = self.embed(input_ids)
        if self.proj_layer:
            x = self.proj_layer(x)
        x = self.dropout(x)
        o, h = self.gru(x)
        o = o.masked_fill(mask, 0.0)
        output = o[:, :, : self.d_model] + o[:, :, self.d_model :]
        hidden = h[0] + h[1]  # n_layer 고려
        return output, hidden


class SlotGenerator(nn.Module):
    def __init__(
        self, vocab_size, hidden_size, dropout, n_gate, proj_dim=None, pad_idx=0
    ):
        super(SlotGenerator, self).__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_idx
        )  # shared with encoder

        if proj_dim:
            self.proj_layer = nn.Linear(hidden_size, proj_dim, bias=False)
        else:
            self.proj_layer = None
        self.hidden_size = proj_dim if proj_dim else hidden_size

        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, 1, dropout=dropout, batch_first=True
        )
        self.n_gate = n_gate
        self.dropout = nn.Dropout(dropout)
        self.w_gen = nn.Linear(self.hidden_size * 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.w_gate = nn.Linear(self.hidden_size, n_gate)

    def set_slot_idx(self, slot_vocab_idx):
        whole = []
        max_length = max(map(len, slot_vocab_idx))
        for idx in slot_vocab_idx:
            if len(idx) < max_length:
                gap = max_length - len(idx)
                idx.extend([self.pad_idx] * gap)
            whole.append(idx)
        self.slot_embed_idx = whole  # torch.LongTensor(whole)

    def embedding(self, x):
        x = self.embed(x)
        if self.proj_layer:
            x = self.proj_layer(x)
        return x

    def forward(
        self, input_ids, encoder_output, hidden, input_masks, max_len, teacher=None
    ):
        input_masks = input_masks.ne(1)
        # J, slot_meta : key : [domain, slot] ex> LongTensor([1,2])
        # J,2
        batch_size = encoder_output.size(0)
        slot = torch.LongTensor(self.slot_embed_idx).to(input_ids.device)  ##
        slot_e = torch.sum(self.embedding(slot), 1)  # J,d
        J = slot_e.size(0)

        all_point_outputs = torch.zeros(batch_size, J, max_len, self.vocab_size).to(
            input_ids.device
        )

        # Parallel Decoding
        w = slot_e.repeat(batch_size, 1).unsqueeze(1)
        hidden = hidden.repeat_interleave(J, dim=1)
        encoder_output = encoder_output.repeat_interleave(J, dim=0)
        input_ids = input_ids.repeat_interleave(J, dim=0)
        input_masks = input_masks.repeat_interleave(J, dim=0)
        for k in range(max_len):
            w = self.dropout(w)
            _, hidden = self.gru(w, hidden)  # 1,B,D

            # B,T,D * B,D,1 => B,T
            attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
            attn_e = attn_e.squeeze(-1).masked_fill(input_masks, -1e9)
            attn_history = F.softmax(attn_e, -1)  # B,T

            if self.proj_layer:
                hidden_proj = torch.matmul(hidden, self.proj_layer.weight)
            else:
                hidden_proj = hidden

            # B,D * D,V => B,V
            attn_v = torch.matmul(
                hidden_proj.squeeze(0), self.embed.weight.transpose(0, 1)
            )  # B,V
            attn_vocab = F.softmax(attn_v, -1)

            # B,1,T * B,T,D => B,1,D
            context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D
            p_gen = self.sigmoid(
                self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1))
            )  # B,1
            p_gen = p_gen.squeeze(-1)

            p_context_ptr = torch.zeros_like(attn_vocab).to(input_ids.device)
            p_context_ptr.scatter_add_(1, input_ids, attn_history)  # copy B,V
            p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
            _, w_idx = p_final.max(-1)

            if teacher is not None:
                w = (
                    self.embedding(teacher[:, :, k])
                    .transpose(0, 1)
                    .reshape(batch_size * J, 1, -1)
                )
            else:
                w = self.embedding(w_idx).unsqueeze(1)  # B,1,D
            if k == 0:
                gated_logit = self.w_gate(context.squeeze(1))  # B,3
                all_gate_outputs = gated_logit.view(batch_size, J, self.n_gate)
            all_point_outputs[:, :, k, :] = p_final.view(batch_size, J, self.vocab_size)

        return all_point_outputs, all_gate_outputs


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForUtteranceEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores
