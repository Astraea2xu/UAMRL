import os, sys
import numpy as np
import torch
import torch.nn as nn
from util import *
from torch_geometric.nn import SAGEConv, global_add_pool as gap
import torch.nn.functional as F
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import esm as ESM
import models.ESMDBP as ft_model
import pandas as pd
device = torch.device('cuda')
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM


class Sequence_Model(nn.Module):
    def __init__(self, in_channel, embedding_channel, med_channel, hidden_size, kernel_size=3, stride=1, padding=1, relative_position=False, Heads=None, use_residue=False):
        super(Sequence_Model, self).__init__()
        self.in_channel = in_channel
        self.med_channel = med_channel
        self.hidden_size = hidden_size
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
            nn.Conv1d(med_channel[2], hidden_size, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, x):
        x1 = self.emb(x)
        x2 = self.dropout(x1)
        x3 = self.layers(x2.permute(0, 2, 1)).view(-1, 256)
        return x3
        
class Flat_Model(nn.Module):
    def __init__(self, in_channel, med_channel, hidden_size, kernel_size=3, stride=1, padding=1):
        super(Flat_Model, self).__init__()
        self.in_channel = in_channel
        self.med_channel = med_channel
        self.hidden_size = hidden_size
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, med_channel[1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(med_channel[1]),
            nn.LeakyReLU(),
            nn.Conv2d(med_channel[1], med_channel[2], kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(med_channel[2]),
            nn.LeakyReLU(),
            nn.Conv2d(med_channel[2], hidden_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(hidden_size),
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

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):

        super(CrossAttentionFusion, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        self.query_p = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_p = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_p = nn.Linear(embed_dim, embed_dim, bias=False)

        self.query_d = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_d = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_d = nn.Linear(embed_dim, embed_dim, bias=False)
    def apply_heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def forward(self, protein, drug):
        protein = protein.unsqueeze(dim=1)
        drug = drug.unsqueeze(dim=1)
        # Compute queries, keys, values for both protein and drug after grouping
        query_prot = self.apply_heads(self.query_p(protein), self.num_heads, self.head_size)
        key_prot = self.apply_heads(self.key_p(protein), self.num_heads, self.head_size)
        value_prot = self.apply_heads(self.value_p(protein), self.num_heads, self.head_size)

        query_drug = self.apply_heads(self.query_d(drug), self.num_heads, self.head_size)
        key_drug = self.apply_heads(self.key_d(drug), self.num_heads, self.head_size)
        value_drug = self.apply_heads(self.value_d(drug), self.num_heads, self.head_size)
        
        # Compute attention scores
        logits_pp = torch.softmax(torch.einsum('blhd, bkhd->blkh', query_prot, key_prot),dim=2)
        logits_pd = torch.softmax(torch.einsum('blhd, bkhd->blkh', query_prot, key_drug),dim=2)
        logits_dp = torch.softmax(torch.einsum('blhd, bkhd->blkh', query_drug, key_prot),dim=2)
        logits_dd = torch.softmax(torch.einsum('blhd, bkhd->blkh', query_drug, key_drug),dim=2)
        
        prot_embedding = (torch.einsum('blkh, bkhd->blhd', logits_pp, value_prot).flatten(-2) + torch.einsum('blkh, bkhd->blhd', logits_pd, value_drug).flatten(-2)) / 2
        drug_embedding = (torch.einsum('blkh, bkhd->blhd', logits_dp, value_prot).flatten(-2) + torch.einsum('blkh, bkhd->blhd', logits_dd, value_drug).flatten(-2)) / 2

        prot_embed = prot_embedding.mean(1)  # query : [batch_size, hidden]
        drug_embed = drug_embedding.mean(1)  # query : [batch_size, hidden]

        return prot_embed, drug_embed

class ESM_DBP(nn.Module):
    def __init__(self, ):
        super(ESM_DBP, self).__init__()
        esm = ESM.ESM2()
        self.esm_model = torch.nn.DataParallel(esm)
        self.esm_model.load_state_dict(torch.load(f'{curPath}/ESM-DBP.model', map_location=lambda storage, loc: storage, weights_only=True))
        self.alphabet = ESM.data.Alphabet.from_architecture("ESM-1b")

    def get_one_prediction_res(self, batch_tokens):
        results = self.esm_model(tokens=batch_tokens, repr_layers=[33], return_contacts=False) # logits: 1, num + 2, 33; representations: 33: 1, num + 2, 1280
        token_representations = torch.squeeze(results["representations"][33]) # 1, num + 2, 1280
        fea_represent = token_representations[1:-1] # num, 1280
        c_fea_represent=torch.mean(fea_represent,dim=0).unsqueeze(dim=0) # 1, 1280
        return c_fea_represent

    def forward(self, data):
        feature_list = torch.tensor([]).reshape(0, 1280).to(device)
        for data_id in data.id:
            for seq_recoder in SeqIO.parse(f'{rootPath}/train_set/target_fasta/{data_id}.fasta', 'fasta'):
                seq = seq_recoder.seq
                datazip = [(data_id, seq)]
                batch_converter = self.alphabet.get_batch_converter()
                batch_labels, batch_strs, batch_tokens = batch_converter(datazip) # 1, num + 2
                feature = self.get_one_prediction_res(batch_tokens)
                feature_list = torch.cat((feature_list, feature), dim = 0)
        return feature_list
class Smiles_Bert(nn.Module):
    
    def __init__(self, ):
        super(Smiles_Bert, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("JuIm/SMILES_BERT")
        self.smil_model = AutoModelForMaskedLM.from_pretrained("JuIm/SMILES_BERT")

    def forward(self, data):
        feature_list = torch.Tensor().to(device)
        for data_id in data.id:
            seq = pd.read_csv(f'{rootPath}/train_set/drug_smiles/{data_id}.smi', header=None).to_numpy().tolist()[0][0].split('\t')[0]
            inputs = self.tokenizer(seq, return_tensors='pt', padding=True).to(device)
            outputs = self.smil_model(**inputs)
            feature = torch.mean(outputs.logits, dim=1)
            feature_list =  torch.cat((feature_list, feature), dim = 0)
        return feature_list
    
class Multimodal_Affinity(nn.Module):
    def __init__(self, compound_sequence_channel, protein_sequence_channel, med_channel, hidden_size, output_size=1):
        super(Multimodal_Affinity, self).__init__()
        self.embedding_dim = 128 
        self.compound_sequence = Sequence_Model(compound_sequence_channel, self.embedding_dim, med_channel, hidden_size, kernel_size=3, padding=1) # 65, 128, [32,64,128], 256
        # self.protein_sequence = Sequence_Model(protein_sequence_channel, self.embedding_dim, med_channel, hidden_size, kernel_size=3, padding=1) # 21, 128, [32,64,128], 256
        self.compound_stru = GraphConv(27, self.embedding_dim) # 27, 128
        self.protein_stru = Flat_Model(1, med_channel, hidden_size) # 1, [32,64,128], 256
        self.esm = ESM_DBP()
        self.smi = Smiles_Bert()
        self.protein_sequence = nn.Sequential()
        self.protein_sequence.add_module('protein_sequence_1', nn.Linear(in_features=1280, out_features=1000))
        self.protein_sequence.add_module('protein_sequence_1_BatchNorm1d', nn.BatchNorm1d(1000))
        self.protein_sequence.add_module('protein_sequence_1_Dropout', nn.Dropout(0.5))
        self.protein_sequence.add_module('protein_sequence_1_activation', nn.LeakyReLU())
        self.protein_sequence.add_module('protein_sequence_2', nn.Linear(in_features=1000, out_features=500))
        self.protein_sequence.add_module('protein_sequence_2_BatchNorm1d', nn.BatchNorm1d(500))
        self.protein_sequence.add_module('protein_sequence_2_Dropout', nn.Dropout(0.5))
        self.protein_sequence.add_module('protein_sequence_2_activation', nn.LeakyReLU())
        self.protein_sequence.add_module('protein_sequence_3', nn.Linear(in_features=500, out_features=hidden_size))

        # self.compound_sequence = nn.Sequential()
        # self.compound_sequence.add_module('compound_sequence_1', nn.Linear(in_features=52000,  out_features=1024))
        # self.compound_sequence.add_module('compound_sequence_3_BatchNorm1d', nn.BatchNorm1d(1024))
        # self.compound_sequence.add_module('compound_sequence_3_Dropout', nn.Dropout(0.5))
        # self.compound_sequence.add_module('compound_sequence_3_activation', nn.LeakyReLU())
        # self.compound_sequence.add_module('compound_sequence_4', nn.Linear(in_features=1024, out_features=512))
        # self.compound_sequence.add_module('compound_sequence_4_BatchNorm1d', nn.BatchNorm1d(512))
        # self.compound_sequence.add_module('compound_sequence_4_Dropout', nn.Dropout(0.5))
        # self.compound_sequence.add_module('compound_sequence_4_activation', nn.LeakyReLU())
        # self.compound_sequence.add_module('compound_sequence_5', nn.Linear(in_features=512, out_features=hidden_size))    

        self.early_layers_mu = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.early_layers_v = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.early_layers_alpha = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.early_layers_beta = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))

        self.ss_layers_mu = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.ss_layers_v = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.ss_layers_alpha = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.ss_layers_beta = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))

        self.sg_layers_mu = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.sg_layers_v = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.sg_layers_alpha = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.sg_layers_beta = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))

        self.gs_layers_mu = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.gs_layers_v = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.gs_layers_alpha = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.gs_layers_beta = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))

        self.gg_layers_mu = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.gg_layers_v = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.gg_layers_alpha = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))
        self.gg_layers_beta = nn.Sequential( nn.Linear(512, 1024), nn.LeakyReLU(), nn.Linear(1024, 256), nn.LeakyReLU(), nn.Linear(256, output_size))


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

    def forward(self, data, compound_sequence, protein_graph): 
        with torch.no_grad():
            protein_sequence = self.esm(data) #batch, 1280
        # compound_sequence = self.smi(data) #batch, 52000
        c_sequence_feature = self.compound_sequence(compound_sequence)      # batch_size, hide_size
        c_graph_feature = self.compound_stru(data)                          # batch_size, hide_size
        p_sequence_feature = self.protein_sequence(protein_sequence)        # batch_size, hide_size
        p_graph_feature = self.protein_stru(protein_graph)                  # batch_size, hide_size   

        ss_f = torch.cat((c_sequence_feature, p_sequence_feature), dim=1)
        ss_mu = self.ss_layers_mu(ss_f)
        ss_v = self.ss_layers_v(ss_f)
        ss_alpha = self.ss_layers_alpha(ss_f)
        ss_beta = self.ss_layers_beta(ss_f)

        gg_f = torch.cat((c_sequence_feature, p_sequence_feature), dim=1)
        gg_mu = self.gg_layers_mu(gg_f)
        gg_v = self.gg_layers_v(gg_f)
        gg_alpha = self.gg_layers_alpha(gg_f)
        gg_beta = self.gg_layers_beta(gg_f)

        sg_f = torch.cat((c_sequence_feature, p_sequence_feature), dim=1)
        sg_mu = self.sg_layers_mu(sg_f)
        sg_v = self.sg_layers_v(sg_f)
        sg_alpha = self.sg_layers_alpha(sg_f)
        sg_beta = self.sg_layers_beta(sg_f)

        gs_f = torch.cat((c_sequence_feature, p_sequence_feature), dim=1)
        gs_mu = self.gs_layers_mu(gs_f)
        gs_v = self.gs_layers_v(gs_f)
        gs_alpha = self.gs_layers_alpha(gs_f)
        gs_beta = self.gs_layers_beta(gs_f)

        # early_f = torch.cat((c_sequence_feature + c_graph_feature, p_sequence_feature + p_graph_feature), dim=1)
        # early_mu = self.ss_layers_mu(early_f)
        # early_v = self.ss_layers_v(early_f)
        # early_alpha = self.ss_layers_alpha(early_f)
        # early_beta = self.ss_layers_beta(early_f)   

        mu_ss, v_ss, alpha_ss, beta_ss = self.split(ss_mu, ss_v, ss_alpha, ss_beta)
        mu_gg, v_gg, alpha_gg, beta_gg = self.split(gg_mu, gg_v, gg_alpha, gg_beta)
        mu_sg, v_sg, alpha_sg, beta_sg = self.split(sg_mu, sg_v, sg_alpha, sg_beta)        
        mu_gs, v_gs, alpha_gs, beta_gs = self.split(gs_mu, gs_v, gs_alpha, gs_beta)        

        return mu_ss, v_ss, alpha_ss, beta_ss, mu_gg, v_gg, alpha_gg, beta_gg, mu_sg, v_sg, alpha_sg, beta_sg, mu_gs, v_gs, alpha_gs, beta_gs

if __name__ == '__main__':
    model = Sequence_Model(65, [128, 256], 256, use_residue=True)
    # print(model)
