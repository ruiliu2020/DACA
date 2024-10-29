import torch
import torch.nn as nn
import copy
from transformers import BertModel, BertForSequenceClassification
import math 
import torch.nn.functional as F
from torch.autograd import Variable, Function
from typing import Any, Optional, Tuple

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class DACA(nn.Module):
    def __init__(self, bert, opt):
        super(DACA, self).__init__()
        self.bert = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.layernorm = LayerNorm(opt.bert_dim)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.fp_dense = nn.Linear(opt.mem_dim, opt.polarities_dim)
        self.fc_dense = nn.Linear(opt.mem_dim, opt.polarities_dim)
        
        self.fp = FP_nn(opt)
        self.fc = FC_nn(opt)

    def forward(self, inputs, lp=1.0):
        text_bert_indices, bert_segments_ids, src_mask, aspect_mask = inputs[0], inputs[1], inputs[2], inputs[3]
        src_mask = src_mask.unsqueeze(-2)

        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.mem_dim) 

        sequence_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)[0]
        pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)[1]
        
        label_embedding = pooled_output
        sequence_output = self.layernorm(sequence_output)
        att_inputs = self.dropout(sequence_output)

        pooled_output = self.dropout(pooled_output)

        fp_outputs, fp_adj = self.fp(att_inputs, pooled_output, src_mask)
        outputs_fp = fp_outputs.sum(dim=1)
        fc_outputs, fc_adj = self.fc(att_inputs, pooled_output, src_mask)
        outputs_fc = (fc_outputs * aspect_mask).sum(dim=1) / asp_wn

        #OPL
        fp_x = Proj(outputs_fp,outputs_fc)
        fp_y = Proj(outputs_fp,outputs_fp-fp_x)

        #GRL
        fc_y = GradientReverseFunction.apply(outputs_fc,lp)

        logits_p = self.fp_dense(fp_y)
        logits_c = self.fc_dense(fc_y)

        return logits_p, logits_c, fp_y, label_embedding

class FP_nn(nn.Module):
    def __init__(self,opt):
        super(FP_nn,self).__init__()
        self.opt = opt
        self.fp_weight_list = nn.ModuleList()
        self.layers = opt.num_layers
        self.mem_dim = opt.mem_dim
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # gcn layer
        self.fp_W = nn.ModuleList()
    
        for layer in range(self.layers):
            fp_input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.fp_W.append(nn.Linear(fp_input_dim, self.mem_dim))

        self.fp_attn = MultiHeadAttention(opt.attention_heads, opt.bert_dim)

        for j in range(self.layers):
            fp_input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.fp_weight_list.append(nn.Linear(fp_input_dim, self.mem_dim))

    def forward(self, att_inputs, sent_em, src_mask): 
        fp_attn_tensor = self.fp_attn(att_inputs, att_inputs, src_mask)
        fp_attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(fp_attn_tensor, 1, dim=1)]
        multi_head_list = []
        fp_adj = None

         # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if fp_adj is None:
                fp_adj = fp_attn_adj_list[i]
            else:
                fp_adj += fp_attn_adj_list[i]
        # fp_adj /= self.attention_heads
        fp_adj = fp_adj / self.attention_heads

        for j in range(fp_adj.size(0)):
            fp_adj[j] = fp_adj[j] - torch.diag(torch.diag(fp_adj[j]))
            fp_adj[j] = fp_adj[j] + torch.eye(fp_adj[j].size(0)).to(self.opt.device)
        fp_adj = src_mask.transpose(1, 2) * fp_adj

        fp_denom = fp_adj.sum(2).unsqueeze(2) + 1
        fp_outputs = att_inputs
        for l in range(self.layers):
            fp_Ax = fp_adj.bmm(fp_outputs)
            fp_AxW = self.fp_weight_list[l](fp_Ax)
            fp_AxW = fp_AxW / fp_denom
            fp_gAxW = F.relu(fp_AxW)
            fp_outputs = self.gcn_drop(fp_gAxW) if l < self.layers - 1 else fp_gAxW
        
        return fp_outputs, fp_adj

class FC_nn(nn.Module):
    def __init__(self,opt):
        super(FC_nn,self).__init__()
        self.opt = opt
        self.fc_weight_list = nn.ModuleList()
        self.layers = opt.num_layers
        self.mem_dim = opt.mem_dim
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        
        # gcn layer
        self.fc_W = nn.ModuleList()

        for layer in range(self.layers):
            fc_input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.fc_W.append(nn.Linear(fc_input_dim, self.mem_dim))

        self.fc_attn = MultiHeadAttention(opt.attention_heads, opt.bert_dim)

        for j in range(self.layers):
            fc_input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.fc_weight_list.append(nn.Linear(fc_input_dim, self.mem_dim))

    def forward(self,att_inputs,sent_em,src_mask):
        fc_attn_tensor = self.fc_attn(att_inputs, att_inputs, src_mask)
        fc_attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(fc_attn_tensor, 1, dim=1)]
        multi_head_list = []
        fc_adj = None

         # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if fc_adj is None:
                fc_adj = fc_attn_adj_list[i]
            else:
                fc_adj += fc_attn_adj_list[i]
        # fc_adj /= self.attention_heads
        fc_adj = fc_adj / self.attention_heads

        for j in range(fc_adj.size(0)):
            fc_adj[j] = fc_adj[j] - torch.diag(torch.diag(fc_adj[j]))
            fc_adj[j] = fc_adj[j] + torch.eye(fc_adj[j].size(0)).to(self.opt.device)
        fc_adj = src_mask.transpose(1, 2) * fc_adj

        fc_denom = fc_adj.sum(2).unsqueeze(2) + 1
        fc_outputs = att_inputs
        for l in range(self.layers):
            fc_Ax = fc_adj.bmm(fc_outputs)
            fc_AxW = self.fc_weight_list[l](fc_Ax)
            fc_AxW = fc_AxW / fc_denom
            fc_gAxW = F.relu(fc_AxW)
            fc_outputs = self.gcn_drop(fc_gAxW) if l < self.layers - 1 else fc_gAxW
        
        return fc_outputs, fc_adj

def Proj(a,b):
    b_mo = torch.sqrt(torch.sum(b*b,dim=-1).float())  
    axb = torch.sum(a.mul(b),dim=-1)          
    d = (axb/b_mo).unsqueeze(-1).mul(F.normalize(b.float(),p=2,dim=-1))    
    return d

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class weighting(nn.Module):
    def __init__(self, opt):
        super(weighting, self).__init__()
        self.opt = opt
        self.weight_branch = nn.Sequential(
            nn.Linear(opt.bert_dim, opt.bert_dim // 2),
            nn.Linear(opt.bert_dim // 2, opt.bert_dim // 4),
            nn.Linear(opt.bert_dim // 4, opt.polarities_dim))
    
    def forward(self, feature, label):
        normed_label_feats = F.normalize(feature, dim=-1)
        # normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=label.reshape(-1, 1, 1).expand(-1, 1, normed_label_feats.size(-1))).squeeze(1)
        weight_logit = self.weight_branch(normed_label_feats)#.to(self.opt.device)
        return weight_logit
    
