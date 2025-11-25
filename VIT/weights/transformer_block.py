import copy
from typing import Optional, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor, einsum

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class ISA_Block(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dropout=0.1,
            block_size=[12, 4], block_num=[3, 3],
    ):
        """
        d_model: 每一个单词本来的词向量长度
        """
        super().__init__()
        self.self_attn_short = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_long = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.block_size = block_size
        self.block_num = block_num

    def forward(self, q, k, v, pos):
        B, N, C = q.shape
        h_b, w_b = self.block_size
        bn_h, bn_w = self.block_num
        pos = pos.repeat(B, 1, 1)


        q = q.view(B, bn_h, h_b, bn_w, w_b, C).permute(1, 3, 0, 2, 4, 5).contiguous()
        q = q.view(bn_h * bn_w, B * h_b * w_b, C)

        k = k.view(B, bn_h, h_b, bn_w, w_b, C).permute(1, 3, 0, 2, 4, 5).contiguous()
        k = k.view(bn_h * bn_w, B * h_b * w_b, C)

        v = v.view(B, bn_h, h_b, bn_w, w_b, C).permute(1, 3, 0, 2, 4, 5).contiguous()
        v = v.view(bn_h * bn_w, B * h_b * w_b, C)

        pos = pos.view(B, bn_h, h_b, bn_w, w_b, C).permute(1, 3, 0, 2, 4, 5).contiguous()
        pos = pos.view(bn_h * bn_w, B * h_b * w_b, C)

        # print(q.shape, pos.shape, k.shape, v.shape)
        v = self.self_attn_long(q + pos, k + pos, v)[0]


        v = v.view(B, bn_h, h_b, bn_w, w_b, C).permute(2, 4, 0, 1, 3, 5).contiguous()
        v = v.view(h_b * w_b, B * bn_h * bn_w, C)

        pos = pos.view(B, bn_h, h_b, bn_w, w_b, C).permute(2, 4, 0, 1, 3, 5).contiguous()
        pos = pos.view(h_b * w_b, B * bn_h * bn_w, C)

        out = self.self_attn_short(v + pos, v + pos, v)[0]

        out = out.view(h_b, w_b, B, bn_h, bn_w, C).permute(2, 0, 3, 1, 4, 5).contiguous()
        out = out.view(B, N, C)

        return out

# class ISA_Block(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         nhead,
#         dropout=0.1,
#         block_size=[12, 4], block_num = [3, 3],
#     ):
#         """
#         d_model: 每一个单词本来的词向量长度
#         """
#         super().__init__()
#         self.self_attn_short = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.self_attn_long = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.block_size = block_size
#         self.block_num = block_num
#
#     def forward(self, q, k, v, pos):
#         B, N, C = q.shape
#         h_b, w_b = self.block_size
#         bn_h, bn_w = self.block_num
#         print(q.shape)
#         B_p, N_p, C_p = pos.shape
#
#         # 拆分，防止cls_token 影响分组
#         q_cls = q[:, 0:1, :]
#         q_ex = q[:, 1:, :]
#
#         k_cls = k[:, 0:1, :]
#         k_ex = k[:, 1:, :]
#
#         v_cls = v[:, 0:1, :]
#         v_ex = v[:, 1:, :]
#
#         pos_cls = pos[:, 0:1, :]
#         pos_ex = pos[:, 1:, :]
#
#         q_ex = q_ex.view(B, bn_h, h_b, bn_w, w_b, C).permute(1, 3, 0, 2, 4, 5).contiguous()
#         q_ex = q_ex.view(bn_h * bn_w, B * h_b * w_b, C)
#         q = torch.cat((q_cls, q_ex), dim=1)
#
#         k_ex = k_ex.view(B, bn_h, h_b, bn_w, w_b, C).permute(1, 3, 0, 2, 4, 5).contiguous()
#         k_ex = k_ex.view(bn_h * bn_w, B * h_b * w_b, C)
#         k = torch.cat((k_cls, k_ex), dim=1)
#
#         v_ex = v_ex.view(B, bn_h, h_b, bn_w, w_b, C).permute(1, 3, 0, 2, 4, 5).contiguous()
#         v_ex = v_ex.view(bn_h * bn_w, B * h_b * w_b, C)
#         v = torch.cat((v_cls, v_ex), dim=1)
#
#         pos_ex = pos_ex.view(B_p, bn_h, h_b, bn_w, w_b, C).permute(1, 3, 0, 2, 4, 5).contiguous()
#         pos_ex = pos_ex.view(bn_h * bn_w, B_p * h_b * w_b, C)
#         pos = torch.cat((pos_cls, pos_ex), dim=1)
#
#         # print(q.shape, pos.shape, k.shape, v.shape)
#         v = self.self_attn_long(q + pos, k + pos, v)[0]
#
#         v_cls = v[:, 0:1, :]
#         v_ex = v[:, 1:, :]
#
#         pos_cls = pos[:, 0:1, :]
#         pos_ex = pos[:, 1:, :]
#
#         v_ex = v_ex.view(B, bn_h, h_b, bn_w, w_b, C).permute(2, 4, 0, 1, 3, 5).contiguous()
#         v_ex = v_ex.view(h_b * w_b, B * bn_h * bn_w, C)
#         v = torch.cat((v_cls, v_ex), dim=1)
#
#         pos_ex = pos_ex.view(B_p, bn_h, h_b, bn_w, w_b, C).permute(2, 4, 0, 1, 3, 5).contiguous()
#         pos_ex = pos_ex.view(h_b * w_b, B_p * bn_h * bn_w, C)
#         pos = torch.cat((pos_cls, pos_ex), dim=1)
#
#         out = self.self_attn_short(v + pos, v + pos, v)[0]
#
#         out = out.view(h_b, w_b, B, bn_h, bn_w, C).permute(2, 0, 3, 1, 4, 5).contiguous()
#         out = out.view(B, N, C)
#
#         return out
    
# class ISA_Block_CA(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         nhead,
#         dropout=0.1,
#         block_size=[16, 16],
#     ):
#         super().__init__()
#         self.self_attn_short = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.self_attn_long = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.block_size = block_size
#
#     def forward(self, q, k, v, pos_q, pos_k):
#         n, c, h, w = q.shape
#         h_b, w_b = self.block_size
#         bn_h, bn_w = h // h_b, w // w_b
#
#         q = q.view(n, c, bn_h, h_b, bn_w, w_b).permute(2, 3, 0, 4, 5, 1).contiguous()
#         q = q.view(bn_h * bn_w, n * h_b * w_b, c)
#         k = k.view(n, c, bn_h, h_b, bn_w, w_b).permute(2, 3, 0, 4, 5, 1).contiguous()
#         k = k.view(bn_h * bn_w, n * h_b * w_b, c)
#         v = v.view(n, c, bn_h, h_b, bn_w, w_b).permute(2, 3, 0, 4, 5, 1).contiguous()
#         v = v.view(bn_h * bn_w, n * h_b * w_b, c)
#         pos_q = pos_q.view(n, c, bn_h, h_b, bn_w, w_b).permute(2, 3, 0, 4, 5, 1).contiguous()
#         pos_q = pos_q.view(bn_h * bn_w, n * h_b * w_b, c)
#         pos_k = pos_k.view(n, c, bn_h, h_b, bn_w, w_b).permute(2, 3, 0, 4, 5, 1).contiguous()
#         pos_k = pos_k.view(bn_h * bn_w, n * h_b * w_b, c)
#         v = self.self_attn_long(q + pos_q, k + pos_k, v)[0]
#
#         v = v.view(bn_h, bn_w, n, h_b, w_b, c).permute(0, 4, 3, 1, 2, 5).contiguous()
#         v = v.view(h_b * w_b, n * bn_h * bn_w, c)
#         pos_q = pos_q.view(bn_h, bn_w, n, h_b, w_b, c).permute(0, 4, 3, 1, 2, 5).contiguous()
#         pos_q = pos_q.view(h_b * w_b, n * bn_h * bn_w, c)
#         out = self.self_attn_short(v + pos_q, v + pos_q, v)[0]
#
#         out = out.view(h_b, w_b, n, bn_h, bn_w, c).permute(5, 0, 3, 1, 4, 2).contiguous()
#         out = out.view(n, c, h, w)
#
#         return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", block_size=[12, 4], block_num = [3, 3], mlp_ratio=4.):
        super().__init__()
        self.self_attn = ISA_Block(d_model, nhead, dropout=dropout, block_size=block_size, block_num = block_num)
        self.mlp_hidden_dim = int(d_model * mlp_ratio)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, self.mlp_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.mlp_hidden_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        B, N, C  = src.shape
        src2 = self.self_attn(src, src, src, pos)
        src = src.view(B, N, C).permute(0, 1, 2).contiguous()
        src2 = src2.view(B, N, C).permute(0, 1, 2).contiguous()

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.permute(0, 2, 1).contiguous().view(B, N, C)

        return src

# class TransformerDecoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", block_size=[16, 16]):
#         super().__init__()
#         self.self_attn = ISA_Block(d_model, nhead, dropout=dropout, block_size=block_size)
#         self.multihead_attn = ISA_Block_CA(d_model, nhead, dropout=dropout, block_size=block_size)
#         self.block_size = block_size
#
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#
#         self.activation = _get_activation_fn(activation)
#
#     def forward(self, tgt, memory, pos, query_pos):
#         n, c, h, w = tgt.shape
#         h_b, w_b = self.block_size
#         bn_h, bn_w = h // h_b, w // w_b
#
#         tgt2 = self.self_attn(tgt, tgt, tgt, query_pos)
#         tgt = tgt.view(n, c, h * w).permute(0, 2, 1).contiguous()
#         tgt2 = tgt2.view(n, c, h * w).permute(0, 2, 1).contiguous()
#
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)
#         tgt = tgt.permute(0, 2, 1).contiguous().view(n, c, h, w)
#
#         tgt2 = self.multihead_attn(tgt, memory, memory, query_pos, pos)
#         tgt = tgt.view(n, c, h * w).permute(0, 2, 1).contiguous()
#         tgt2 = tgt2.view(n, c, h * w).permute(0, 2, 1).contiguous()
#
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt)
#
#         tgt = tgt.permute(0, 2, 1).contiguous().view(n, c, h, w)
#
#         return tgt

class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, dropout=0.0):
        super().__init__()

        self.to_q = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Linear(dim, dim_head, bias=False),
            nn.Linear(dim_head, dim_head, bias=False),
        )
        self.to_k = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Linear(dim, dim_head, bias=False),
            nn.Linear(dim_head, dim_head, bias=False),
        )
        self.to_v = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Linear(dim, dim_head, bias=False),
            nn.Linear(dim_head, dim_head, bias=False),
        )

        self.to_out = nn.Sequential(nn.Linear(dim_head, dim), nn.Dropout(dropout))

    def forward(self, query, key, value):
        b, n_q, c = query.shape
        _, n_k, _ = key.shape

        query = self.to_q(query)
        key = self.to_k(key)
        value = self.to_v(value)

        dots = einsum('b i d, b j d -> b i j', query, key)
        attn = dots.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, value)
        out = self.to_out(out)

        return out, attn

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, pos, query_pos):
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                pos=pos,
                query_pos=query_pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerHELayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attn = Attention(dim=d_model, dim_head=dim_feedforward, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, query, key, value):

        n, c, h, w = query.shape
        query = query.view(n, c, h * w).permute(0, 2, 1).contiguous()

        query2, attn = self.attn(query=query, key=key, value=value)
        query = query + self.dropout(query2)
        query = self.norm1(query)

        query2 = self.linear2(self.dropout1(self.activation(self.linear1(query))))
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query = query.permute(0, 2, 1).contiguous().view(n, c, h, w)

        return query

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")