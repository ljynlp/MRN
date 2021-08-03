import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.nn.init as init
from transformers import BertModel
import numpy as np
import math


class ConvAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pre_channels, channels, groups, dropout=0.1):
        super(ConvAttentionLayer, self).__init__()
        assert hid_dim % n_heads == 0
        self.n_heads = n_heads
        input_channels = hid_dim * 2 + pre_channels
        self.groups = groups

        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.linear1 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.linear2 = nn.Linear(hid_dim, hid_dim, bias=False)

        self.conv = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_channels, channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.score_layer = nn.Conv2d(channels, n_heads, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, pre_conv=None, mask=None, residual=True, self_loop=True):
        ori_x, ori_y = x, y

        B, M, _ = x.size()
        B, N, _ = y.size()

        fea_map = torch.cat([x.unsqueeze(2).repeat_interleave(N, 2), y.unsqueeze(1).repeat_interleave(M, 1)],
                            -1).permute(0, 3, 1, 2).contiguous()
        if pre_conv is not None:
            fea_map = torch.cat([fea_map, pre_conv], 1)
        fea_map = self.conv(fea_map)

        scores = self.activation(self.score_layer(fea_map))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask.eq(0), -9e10)

        x = self.linear1(self.dropout(x))
        y = self.linear2(self.dropout(y))

        out_x = torch.matmul(F.softmax(scores, -1), y.view(B, N, self.n_heads, -1).transpose(1, 2))
        out_x = out_x.transpose(1, 2).contiguous().view(B, M, -1)
        out_y = torch.matmul(F.softmax(scores.transpose(2, 3), -1), x.view(B, M, self.n_heads, -1).transpose(1, 2))
        out_y = out_y.transpose(1, 2).contiguous().view(B, N, -1)

        if self_loop:
            out_x = out_x + x
            out_y = out_y + y

        out_x = self.activation(out_x)
        out_y = self.activation(out_y)

        if residual:
            out_x = out_x + ori_x
            out_y = out_y + ori_y
        return out_x, out_y, fea_map


class ConvAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, pre_channels, channels, layers, groups, dropout):
        super(ConvAttention, self).__init__()

        self.layers = nn.ModuleList([ConvAttentionLayer(hid_dim, n_heads, pre_channels if i == 0 else channels,
                                                        channels, groups, dropout=dropout) for i in range(layers)])

    def forward(self, x, y, fea_map=None, mask=None, residual=True, self_loop=True):
        fea_list = []
        for layer in self.layers:
            x, y, fea_map = layer(x, y, fea_map, mask, residual, self_loop)
            fea_list.append(fea_map)

        return x, y, fea_map.permute(0, 2, 3, 1).contiguous()


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class MRN(nn.Module):
    def __init__(self, vocab_size, tok_emb_size, ner_emb_size, dis_emb_size, hid_size,
                 channels, layers, chunk, dropout1, dropout2, embeddings=None):
        super(MRN, self).__init__()
        self.chunk = chunk

        if embeddings is None:
            self.word_emb = nn.Embedding(vocab_size, tok_emb_size)
        else:
            self.word_emb = nn.Embedding.from_pretrained(embeddings, freeze=True)

        self.ner_embs = nn.Embedding(7, ner_emb_size)
        self.dis_embs = nn.Embedding(20, dis_emb_size)
        self.ref_embs = nn.Embedding(3, dis_emb_size)
        emb_size = tok_emb_size + ner_emb_size # + ref_emb_size

        self.encoder = nn.LSTM(emb_size, hid_size // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

        self.men2men_conv_att = ConvAttention(hid_size, 1, dis_emb_size, channels,
                                              groups=1, layers=layers, dropout=dropout1)

        self.mlp_sub = MLP(n_in=hid_size, n_out=hid_size, dropout=dropout2)
        self.mlp_obj = MLP(n_in=hid_size, n_out=hid_size, dropout=dropout2)
        self.biaffine = Biaffine(n_in=hid_size, n_out=97, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, channels, dropout=dropout2)
        self.linear = nn.Linear(channels, 97)

    def forward(self, doc_inputs, ner_inputs, dis_inputs, ref_inputs,
                doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask):
        length = doc_inputs.ne(0).sum(dim=-1)

        tok_embs = self.word_emb(doc_inputs)
        ner_embs = self.ner_embs(ner_inputs)

        tok_embs = self.dropout1(torch.cat([tok_embs, ner_embs], dim=-1))
        packed_embs = pack_padded_sequence(tok_embs, length, batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        outs, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=length.max())

        max_e = doc2ent_mask.size(1)
        max_m = doc2men_mask.size(1)

        min_value = torch.min(outs).item()

        _outs = outs.unsqueeze(1).expand(-1, max_m, -1, -1)
        _outs = torch.masked_fill(_outs, doc2men_mask.eq(0).unsqueeze(-1), min_value)
        men_reps, _ = torch.max(_outs, dim=2)

        dis_emb = self.dis_embs(dis_inputs).permute(0, 3, 1, 2)
        # ref_emb = self.ref_embs(ref_inputs).permute(0, 3, 1, 2)
        # pre_fea_maps = torch.cat([dis_emb, ref_emb], 1)
        x, y, fea_maps = self.men2men_conv_att(men_reps, men_reps, dis_emb, men2men_mask)

        min_x_value = torch.min(x).item()
        x = x.unsqueeze(1).expand(-1, max_e, -1, -1)
        x = torch.masked_fill(x, men2ent_mask.eq(0).unsqueeze(-1), min_x_value)
        x, _ = torch.max(x, dim=2)

        min_y_value = torch.min(y).item()
        y = y.unsqueeze(1).expand(-1, max_e, -1, -1)
        y = torch.masked_fill(y, men2ent_mask.eq(0).unsqueeze(-1), min_y_value)
        y, _ = torch.max(y, dim=2)

        min_f_value = torch.min(fea_maps).item()

        fea_list = []
        chunk = self.chunk
        fea_maps = torch.split(fea_maps, chunk, dim=0)
        m2e_mask2 = torch.split(men2ent_mask, chunk, dim=0)
        for fea_map, m2e_mask in zip(fea_maps, m2e_mask2):
            fea_map = fea_map.unsqueeze(1).repeat(1, max_e, 1, 1, 1)
            fea_map = torch.masked_fill(fea_map, m2e_mask.eq(0)[:, :, :, None, None], min_f_value)
            fea_map, _ = torch.max(fea_map, dim=2)

            fea_map = fea_map.unsqueeze(1).repeat(1, max_e, 1, 1, 1)
            fea_map = torch.masked_fill(fea_map, m2e_mask.eq(0)[:, :, None, :, None], min_f_value)
            fea_map, _ = torch.max(fea_map, dim=3)
            fea_list.append(fea_map)
        fea_maps = torch.cat(fea_list, dim=0)

        ent_sub = self.dropout2(self.mlp_sub(x))
        ent_obj = self.dropout2(self.mlp_obj(y))

        rel_outputs1 = self.biaffine(ent_sub, ent_obj)

        fea_maps = self.dropout2(self.mlp_rel(fea_maps))
        rel_outputs2 = self.linear(fea_maps)
        return rel_outputs1 + rel_outputs2
