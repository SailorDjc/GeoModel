import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import MultiHeadSpatialLayer


class GNN(nn.Module):
    def __init__(self, coors, in_feats, out_feats, num_head, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(MultiHeadSpatialLayer(coors, in_feats, out_feats, num_head))
        for l in range(1, n_layers):
            self.layers.append(
                MultiHeadSpatialLayer(coors, out_feats, out_feats, num_head))
        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, x):
        h = x
        layer_outputs = []
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.leaky_relu(h)
                h = self.dropout(h)
            layer_outputs.append(h[:blocks[-1].number_of_dst_nodes()])
        h = torch.cat(layer_outputs, dim=1)
        # h = self.last_layer(h)
        return h


class SelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, attn_pdrop=0.5, resid_drop=0.5):
        super(SelfAttention, self).__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.att_head_size = int(out_dim / num_heads)
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim, bias=True)
            self.K = nn.Linear(in_dim, out_dim, bias=True)
            self.V = nn.Linear(in_dim, out_dim, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim, bias=False)
            self.K = nn.Linear(in_dim, out_dim, bias=False)
            self.V = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_pdrop)

        self.dense = nn.Linear(out_dim, out_dim)
        self.pro_x = nn.Linear(in_dim, out_dim)
        self.LayerNorm = nn.LayerNorm(out_dim, out_dim)
        self.resid_drop = nn.Dropout(resid_drop)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.att_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        x = self.pro_x(x)
        q_layer = self.transpose_for_scores(q)
        k_layer = self.transpose_for_scores(k)
        v_layer = self.transpose_for_scores(v)

        att_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        att_scores = att_scores / math.sqrt(self.att_head_size)

        att_probs = nn.Softmax(dim=-1)(att_scores)
        att_probs = self.attn_drop(att_probs)
        context_layer = torch.matmul(att_probs, v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.out_dim,)
        context_layer = context_layer.view(new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.resid_drop(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + x)
        return hidden_states


class GraphTransfomer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embd_pdrop)
        self.gnn = GNN(config.coors, config.in_size, config.n_embd, config.gnn_n_head, config.gnn_n_layer)

        self.blocks = nn.Sequential()
        self.blocks.add_module('at_1', SelfAttention(config.n_embd * config.gnn_n_layer
                                                     , int(config.n_embd * config.gnn_n_layer / 2)
                                                     , config.n_head, use_bias=True))
        self.blocks.add_module('at_2', SelfAttention(int(config.n_embd * config.gnn_n_layer / 2)
                                                     , config.n_embd, config.n_head
                                                     , use_bias=True))
        self.blocks.add_module('at_3', SelfAttention(config.n_embd, config.n_embd
                                                     , config.n_head, use_bias=True))

        self.p_layer = nn.Linear(config.n_embd, config.out_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, blocks, x):
        gh = self.gnn(blocks, x)  # [512, gnn_n_layer * 512]
        gh = gh.unsqueeze(1)
        h = self.blocks(gh)
        logits = h.squeeze(1)
        logits_result = self.p_layer(logits)
        return logits_result
