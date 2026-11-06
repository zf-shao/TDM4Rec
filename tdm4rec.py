import torch
import math
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from mamba_ssm import Mamba
import torch.nn.functional as fn
import numpy as np


class TDMamba4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(TDMamba4Rec, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]

        self.initializer_range = config["initializer_range"]

        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        self.hidden_dropout_prob = config["hidden_dropout_prob"]

        self.ratio = config['ratio']

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.ts_embedding = nn.Embedding(self.n_time_span_ids, self.hidden_size, padding_idx=0)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,             
                dropout=self.hidden_dropout_prob,
                ratio=self.ratio,   
                num_layers=self.num_layers,   
            ) for _ in range(self.num_layers)
        ])

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def make_embedding(self, sequence, ts_seq):
        item_embeddings = self.item_embedding(sequence)
        ts_embeddings = self.ts_embedding(ts_seq) 
        sequence_embeddings = item_embeddings + ts_embeddings
        sequence_embeddings = self.LayerNorm(sequence_embeddings)    
        sequence_embeddings = self.dropout(sequence_embeddings) 
        return sequence_embeddings


    def forward(self, item_seq, ts_seq, item_seq_len):
        item_emb = self.make_embedding(item_seq, ts_seq)

        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)
        
        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)

        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        ts_seq = interaction[self.TIME_SPAN_ID_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, ts_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        ts_seq = interaction[self.TIME_SPAN_ID_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, ts_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        ts_seq = interaction[self.TIME_SPAN_ID_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, ts_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores

class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, ratio, num_layers):
        super().__init__()
        self.num_layers = num_layers        
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )         
        self.dropout = nn.Dropout(dropout)        
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.filter_layer = FilterLayer(hidden_size=d_model, dropout_prob=dropout, ratio=ratio)
        self.ffn = FeedForward(d_model, d_model * 4, dropout)

    def forward(self, input_tensor):
        x = self.filter_layer(input_tensor)
        hidden_states = self.mamba(x)
        if self.num_layers == 1:        # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))        
        else:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + x)
        return self.ffn(hidden_states)
        
class FilterLayer(nn.Module):
    def __init__(self, hidden_size, dropout_prob, ratio):
        super(FilterLayer, self).__init__()
        self.out_dropout = nn.Dropout(dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ratio = ratio

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.abs(torch.fft.rfftfreq(input_tensor.shape[-2])).to(input_tensor.device)
        select_weight = weight < weight.quantile(q=self.ratio).item()
        ones = torch.ones(hidden).to(input_tensor.device) > 0
        select_weight = torch.outer(select_weight, ones)
        select_weight = torch.unsqueeze(select_weight, dim=0)
        x = x * select_weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.intermediate_act_fn = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, 1e-12)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states





