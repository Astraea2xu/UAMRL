import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_add_pool as gap
from transformers import BertTokenizer, BertModel
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch import nn, einsum
import torch.nn.functional as F
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

device = torch.device('cuda')

class Sequence_Model(nn.Module):
    def __init__(self, in_channel, embedding_channel, med_channel, out_channel, kernel_size=3, stride=1, padding=1, relative_position=False, Heads=None, use_residue=False):
        super(Sequence_Model, self).__init__()
        self.in_channel = in_channel
        self.med_channel = med_channel
        self.out_channel = out_channel
        self.residue_in_channel = 64
        self.dim = '1d'
        self.dropout = 0.1
        self.relative_position = relative_position
        self.use_residue = use_residue
        
        self.emb = nn.Linear(in_channel, embedding_channel)
        self.dropout = nn.Dropout(self.dropout)
        
        self.layers = nn.Sequential(
            nn.Conv1d(embedding_channel, med_channel[1], kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(med_channel[1]),
            nn.LeakyReLU(),
            nn.Conv1d(med_channel[1], med_channel[2], kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(med_channel[2]),
            nn.LeakyReLU(),
            nn.Conv1d(med_channel[2], out_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(out_channel),
        )
        self.logits = nn.Sequential(nn.AdaptiveMaxPool1d(1))
    def forward(self, x):
        x = self.dropout(self.emb(x))
        x = self.layers(x.permute(0, 2, 1))
        x1 = self.logits(x).view(-1, 256)
        x2 = x.permute(0, 2, 1)
        return x1,x2
        
class Flat_Model(nn.Module):
    def __init__(self, in_channel, med_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Flat_Model, self).__init__()
        self.in_channel = in_channel
        self.med_channel = med_channel
        self.out_channel = out_channel
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, med_channel[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(med_channel[1]),
            nn.LeakyReLU(),
            nn.Conv2d(med_channel[1], med_channel[2], kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(med_channel[2]),
            nn.LeakyReLU(),
            nn.Conv2d(med_channel[2], out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            SELayer(out_channel),
            nn.AdaptiveMaxPool2d(1)
        )
        
    def forward(self, x):
        x = self.layers(x).view(-1, 256)
        return x

class GraphConv(nn.Module):
    def __init__(self, feature_dim, emb_dim, hidden_dim=32, output_dim=256, dropout=0.1):
        super(GraphConv, self).__init__()
        self.dropout = dropout
        self.hidden = hidden_dim
        self.emb = nn.Linear(feature_dim, emb_dim)
        self.cconv1 = SAGEConv(emb_dim, hidden_dim, aggr='sum')
        self.cconv2 = SAGEConv(hidden_dim, hidden_dim * 2, aggr='sum')
        self.cconv3 = SAGEConv(hidden_dim * 2, hidden_dim * 4, aggr='sum')
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.flat = nn.Linear(hidden_dim * 4, output_dim)

    
    def forward(self, data):
        # 获取小分子和蛋白质输入的结构信息
        compound_feature, compound_index, compound_batch = data.x, data.edge_index, data.batch
        # 对小分子进行卷积操作
        compound_feature = self.dropout(self.emb(compound_feature))
        compound_feature = self.cconv1(compound_feature, compound_index)
        compound_feature = self.relu(compound_feature)
        compound_feature = self.cconv2(compound_feature, compound_index)
        compound_feature = self.relu(compound_feature)
        compound_feature = self.cconv3(compound_feature, compound_index)

        # 对卷积后的小分子进行图的池化操作
        compound_feature = gap(compound_feature, compound_batch)
        compound_feature = self.flat(compound_feature)
        return compound_feature
       
class Multimodal_Affinity(nn.Module):
    def __init__(self, compound_sequence_channel, protein_sequence_channel, med_channel, hidden_size, embedding_dim = 128, output_size=1):
        super(Multimodal_Affinity, self).__init__()
        
        self.compound_sequence = Sequence_Model(compound_sequence_channel, embedding_dim, med_channel, hidden_size, kernel_size=3, padding=1)
        self.protein_sequence = Sequence_Model(protein_sequence_channel, embedding_dim, med_channel, hidden_size, kernel_size=3, padding=1)
        self.compound_stru = GraphConv(27, embedding_dim)
        self.protein_stru = Flat_Model(1, med_channel, hidden_size)

        self.shared_layer = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 512))
        self.ss_layer = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 512))
        self.gg_layer = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 512))
        self.sg_layer = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 512))
        self.gs_layer = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 512))

        self.ss_layers_mu = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.ss_layers_v = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.ss_layers_alpha = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.ss_layers_beta = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))

        self.sg_layers_mu = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.sg_layers_v = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.sg_layers_alpha = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.sg_layers_beta = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))

        self.gs_layers_mu = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.gs_layers_v = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.gs_layers_alpha = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.gs_layers_beta = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))

        self.gg_layers_mu = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.gg_layers_v = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.gg_layers_alpha = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.gg_layers_beta = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))

        self.ss_private_mu = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.ss_private_v = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.ss_private_alpha = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.ss_private_beta = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))

        self.sg_private_mu = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.sg_private_v = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.sg_private_alpha = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.sg_private_beta = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))

        self.gs_private_mu = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.gs_private_v = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.gs_private_alpha = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.gs_private_beta = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))

        self.gg_private_mu = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.gg_private_v = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.gg_private_alpha = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.gg_private_beta = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))

        self.loss_cmd = CMD()
        self.loss_diff = DiffLoss()
        self.latents = nn.Parameter(torch.randn(512, 256))
        self.layers = nn.ModuleList([])
        depth = 6
        for i in range(depth):
            cross_attn_layers =[]
            self_attn_layers = nn.ModuleList([])
            for j in range(2):
                cross_attn_layers.append(Attention(256, 256, heads = 1, dim_head = 64, dropout = 0.))
                cross_attn_layers.append(FeedForward(256, dropout = 0.))
            self_attn_layers.append(Attention(256, 256, heads = 8, dim_head = 32, dropout = 0.))
            self_attn_layers.append(FeedForward(256, dropout = 0.))
            self.layers.append(nn.ModuleList([*cross_attn_layers, self_attn_layers]))
        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(256),
            nn.Linear(256, 512)
        )
        self.Q_mu = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.Q_v = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.Q_alpha = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
        self.Q_beta = nn.Sequential( nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, output_size))
    def evidence(self, x):
        # 用于将输入通过 softplus 函数激活，确保输出为正值。这用于计算 NIG 分布中的参数，如方差、alpha、beta等。
        # return tf.exp(x)
        return F.softplus(x)
    def split(self, mu, logv, logalpha, logbeta):
        # 用于将预测的输出分解为 mu（均值）、v（方差）、alpha 和 beta，并通过 evidence() 函数保证它们为正值。
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta    
    def get_cmdloss(self, x1, x2):
        return self.loss_cmd(x1, x2, 5)
    def get_diffloss(self, x1, x2):
        return self.loss_diff(x1, x2)
    
    def forward(self, compound_sequence, compound_graph, protein_sequence, protein_graph):

        c_sequence_feature, cs = self.compound_sequence(compound_sequence)
        c_graph_feature = self.compound_stru(compound_graph)
        p_sequence_feature, ps = self.protein_sequence(protein_sequence)
        p_graph_feature = self.protein_stru(protein_graph)

        tensors = [cs,ps]
        Q = repeat(self.latents, 'n d -> b n d', b = c_sequence_feature.shape[0]) # note: batch dim should be identical across modalities
        for layer in self.layers:
            for i in range(2):
                Q = layer[i*2](Q, context = tensors[i]) + Q
                Q = layer[i*2+1](Q) + Q
            self_attn, self_ff = layer[-1]
            Q = self_attn(Q) + Q
            Q = self_ff(Q) + Q
        Q = self.to_logits(Q)

        sequence_feature = torch.cat((c_sequence_feature, p_sequence_feature), dim=1)
        graph_feature = torch.cat((c_graph_feature, p_graph_feature), dim=1)
        sequence_graph = torch.cat((c_sequence_feature, p_graph_feature), dim=1)
        graph_sequence = torch.cat((c_graph_feature, p_sequence_feature), dim=1)

        ss_shared = self.shared_layer(sequence_feature)
        gg_shared = self.shared_layer(graph_feature)
        sg_shared = self.shared_layer(sequence_graph)
        gs_shared = self.shared_layer(graph_sequence)

        ss_private = self.ss_layer(sequence_feature)
        gg_private = self.gg_layer(graph_feature)
        sg_private = self.sg_layer(sequence_graph)
        gs_private = self.gs_layer(graph_sequence)

        cmd_loss = self.get_cmdloss(ss_shared,gg_shared)
        cmd_loss += self.get_cmdloss(ss_shared,sg_shared)
        cmd_loss += self.get_cmdloss(ss_shared,gs_shared)
        cmd_loss += self.get_cmdloss(gg_shared,sg_shared)
        cmd_loss += self.get_cmdloss(gg_shared,gs_shared)
        cmd_loss += self.get_cmdloss(sg_shared,gs_shared)

        diff_loss = self.get_diffloss(ss_shared,ss_private)
        diff_loss += self.get_diffloss(gg_shared,gg_private)
        diff_loss += self.get_diffloss(sg_shared,sg_private)
        diff_loss += self.get_diffloss(gs_shared,gs_private)
        diff_loss += self.get_diffloss(ss_private,gg_private)
        diff_loss += self.get_diffloss(ss_private,sg_private)
        diff_loss += self.get_diffloss(ss_private,gs_private)
        diff_loss += self.get_diffloss(gg_private,sg_private)
        diff_loss += self.get_diffloss(gg_private,gs_private)
        diff_loss += self.get_diffloss(sg_private,gs_private)

        ss_mu = self.ss_layers_mu(ss_shared)
        ss_v = self.ss_layers_v(ss_shared)
        ss_alpha = self.ss_layers_alpha(ss_shared)
        ss_beta = self.ss_layers_beta(ss_shared)

        gg_mu = self.gg_layers_mu(gg_shared)
        gg_v = self.gg_layers_v(gg_shared)
        gg_alpha = self.gg_layers_alpha(gg_shared)
        gg_beta = self.gg_layers_beta(gg_shared)

        sg_mu = self.sg_layers_mu(sg_shared)
        sg_v = self.sg_layers_v(sg_shared)
        sg_alpha = self.sg_layers_alpha(sg_shared)
        sg_beta = self.sg_layers_beta(sg_shared)

        gs_mu = self.gs_layers_mu(gs_shared)
        gs_v = self.gs_layers_v(gs_shared)
        gs_alpha = self.gs_layers_alpha(gs_shared)
        gs_beta = self.gs_layers_beta(gs_shared)

        ss_private_mu = self.ss_private_mu(ss_private)
        ss_private_v = self.ss_private_v(ss_private)
        ss_private_alpha = self.ss_private_alpha(ss_private)
        ss_private_beta = self.ss_private_beta(ss_private)

        gg_private_mu = self.gg_private_mu(gg_private)
        gg_private_v = self.gg_private_v(gg_private)
        gg_private_alpha = self.gg_private_alpha(gg_private)
        gg_private_beta = self.gg_private_beta(gg_private)

        sg_private_mu = self.sg_private_mu(sg_private)
        sg_private_v = self.sg_private_v(sg_private)
        sg_private_alpha = self.sg_private_alpha(sg_private)
        sg_private_beta = self.sg_private_beta(sg_private)

        gs_private_mu = self.gs_private_mu(gs_private)
        gs_private_v = self.gs_private_v(gs_private)
        gs_private_alpha = self.gs_private_alpha(gs_private)
        gs_private_beta = self.gs_private_beta(gs_private)


        mu_ss, v_ss, alpha_ss, beta_ss = self.split(ss_mu, ss_v, ss_alpha, ss_beta)
        mu_gg, v_gg, alpha_gg, beta_gg = self.split(gg_mu, gg_v, gg_alpha, gg_beta)
        mu_sg, v_sg, alpha_sg, beta_sg = self.split(sg_mu, sg_v, sg_alpha, sg_beta)        
        mu_gs, v_gs, alpha_gs, beta_gs = self.split(gs_mu, gs_v, gs_alpha, gs_beta)    

        mu_ss_private, v_ss_private, alpha_ss_private, beta_ss_private = self.split(ss_private_mu, ss_private_v, ss_private_alpha, ss_private_beta)
        mu_gg_private, v_gg_private, alpha_gg_private, beta_gg_private = self.split(gg_private_mu, gg_private_v, gg_private_alpha, gg_private_beta)
        mu_sg_private, v_sg_private, alpha_sg_private, beta_sg_private = self.split(sg_private_mu, sg_private_v, sg_private_alpha, sg_private_beta)        
        mu_gs_private, v_gs_private, alpha_gs_private, beta_gs_private = self.split(gs_private_mu, gs_private_v, gs_private_alpha, gs_private_beta)   

        Q_mu = self.Q_mu(Q)
        Q_v = self.Q_v(Q)
        Q_alpha = self.Q_alpha(Q)
        Q_beta = self.Q_beta(Q)

        mu_Q, v_Q, alpha_Q, beta_Q = self.split(Q_mu, Q_v, Q_alpha, Q_beta)  
        return mu_ss, v_ss, alpha_ss, beta_ss, mu_gg, v_gg, alpha_gg, beta_gg, mu_sg, v_sg, alpha_sg, beta_sg, mu_gs, v_gs, alpha_gs, beta_gs, cmd_loss / 6 ,\
                mu_ss_private, v_ss_private, alpha_ss_private, beta_ss_private, mu_gg_private, v_gg_private, alpha_gg_private, beta_gg_private, mu_sg_private, v_sg_private, alpha_sg_private, beta_sg_private, mu_gs_private, v_gs_private, alpha_gs_private, beta_gs_private, diff_loss / 10 ,\
                mu_Q, v_Q, alpha_Q, beta_Q

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
    
class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        # add leaky relu
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.LeakyReLU(negative_slope=1e-2)
        )

        self.attn_weights = None
        # self._init_weights()

    def _init_weights(self):
    # Use He initialization for Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # Initialize bias to zero if there's any
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        # attn = sim.softmax(dim = -1)
        attn = temperature_softmax(sim, temperature=0.5, dim=-1)
        self.attn_weights = attn
        attn = self.dropout(attn)


        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.LeakyReLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

def temperature_softmax(logits, temperature=1.0, dim=-1):
    """
    Temperature scaled softmax
    Args:
        logits:
        temperature:
        dim:

    Returns:
    """
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=dim)

def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        初始化 SE layer，接收的参数有：
        - channel: 输入的通道数
        - reduction: 通道缩减比例，用于控制第一个全连接层的输出通道数
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应全局平均池化层，输出大小为 (batch_size, channel, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 压缩通道数
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 恢复到原始通道数
            nn.Sigmoid()  # 激活函数，将输出映射到 (0, 1) 区间
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入的 batch 大小和通道数
        y_pool = self.avg_pool(x).view(b, c)  # 对输入进行全局平均池化并展平
        y = self.fc(y_pool).view(b, c, 1, 1)  # 通过全连接层计算权重，并调整形状以便广播
        return x * y.expand_as(x)  # 将原始输入乘以 SE 模块生成的权重