import h5py, math, os, torch
import pandas as pd
import numpy as np
import cv2
from Bio import SeqIO
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
import torch.nn as nn
import torch.nn.functional as F

smiles_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

amino_acids = ['PAD','A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

atom_list = ['C', 'H', 'O', 'N', 'F', 'S', 'P', 'I', 'Cl', 'As', 'Se', 'Br', 'B', 'Pt', 'V', 'Fe', 'Hg', 'Rh', 'Mg', 'Be', 'Si', 'Ru', 'Sb', 'Cu', 'Re', 'Ir', 'Os']

def moe_nig(u1, la1, alpha1, beta1, u2, la2, alpha2, beta2):
    
    # 实现了多模态的 NIG（正态逆伽玛分布）融合。它通过加权平均的方式将来自不同模态（如文本、音频）的输出进行融合，得到最终的均值 u 和分布参数 la, alpha, beta。
    # Eq. 9
    u = (la1 * u1 + u2 * la2) / (la1 + la2)
    la = la1 + la2
    alpha = alpha1 + alpha2 + 0.5
    beta = beta1 + beta2 + 0.5 * (la1 * (u1 - u) ** 2 + la2 * (u2 - u) ** 2)
    return u, la, alpha, beta


def criterion_nig(u, la, alpha, beta, y):
    # our loss function
    om = 2 * beta * (1 + la)
    loss = sum(
        0.5 * torch.log(np.pi / la) - alpha * torch.log(om) + (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om) + torch.lgamma(alpha) - torch.lgamma(alpha+0.5)) / len(u)
    lossr = 0.01 * sum(torch.abs(u - y) * (2 * la + alpha)) / len(u)
    loss = loss + lossr
    return loss

def smiles2onehot(pdbid):
    seq = pd.read_csv('train_set/drug_smiles/' + pdbid + '.smi', header=None).to_numpy().tolist()[0][0].split('\t')[0]
    integer_encoder = []
    onehot_encoder = []
    for item in seq:
        integer_encoder.append(smiles_dict[item])
    for index in integer_encoder:
        temp = [0 for _ in range(len(smiles_dict) + 1)]
        temp[index] = 1
        onehot_encoder.append(temp)
    return onehot_encoder


def protein2onehot(pdbid):
    for seq_recoder in SeqIO.parse('train_set/target_fasta/' + pdbid + '.fasta', 'fasta'):
        seq = seq_recoder.seq
    protein_to_int = dict((c, i) for i, c in enumerate(amino_acids))
    integer_encoded = [protein_to_int[char] for char in seq]
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0 for _ in range(len(amino_acids))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded

def _to_onehot(data, max_len):
    feature_list = []
    for seq in data:
        if max_len == 1000:
            feature = protein2onehot(seq)
            if len(feature) > 1000:
                feature = feature[:1000]
            feature_list.append(feature)
        elif max_len == 150:
            feature = smiles2onehot(seq)
            if len(feature) > 150:
                feature = feature[:150]
            feature_list.append(feature)
        else:
            print('max length error!')
    for i in range(len(feature_list)):
        if len(feature_list[i]) != max_len:
            for j in range(max_len - len(feature_list[i])):
                if max_len == 1000:
                    temp = [0] * 21
                    temp[0] = 1
                elif max_len == 150:
                    temp = [0] * 65
                    temp[0] = 1
                feature_list[i].append(temp)
    return torch.from_numpy(np.array(feature_list, dtype=np.float32))

def img_resize(data):
    data_list = []
    for id in data:
        img = np.load('train_set/distance_matrix/' + id + '.npz')['map']
        img_resize = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
        data_list.append(img_resize)
    return np.array(data_list)

class CompoundDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset=None , compound=None, protein=None, affinity=None, transform=None, pre_transform=None, compound_graph=None, protein_graph=None):
        super(CompoundDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processd data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processd data not found: {}, doing pre-processing ...'.format(self.processed_paths[0]))
            self.process(compound, affinity, compound_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        pass
    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']
    
    def download(self):
        # download_url(url='', self.raw_dir)
        pass
    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def process(self, compound, affinity, compound_graph):##com--testid
        assert (len(compound) == len(affinity)), '这两个列表必须是相同的长度!'
        data_list = []
        data_len = len(compound)
        for i in range(data_len):
            print('将分子格式转换为图结构：{}/{}'.format(i + 1, data_len))
            smiles = compound[i]
            # target = protein[i]
            label = affinity[i]
            print(smiles)
            # print(target)
            print(label)

            size, features, edge_index = compound_graph[i][smiles]
            GCNCompound = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(-1, 0), y=torch.FloatTensor([label]), id=smiles)
            GCNCompound.__setitem__('size', torch.LongTensor([size]))
            data_list.append(GCNCompound)
            # data_list.append(GCNProtein)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('将构建完的图信息保存到文件中')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        # 计算均值
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        
        # 计算中心化向量
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        
        # 对中心化向量进行标准化处理
        sx1_std = torch.std(sx1, dim=0) + 1e-8
        sx2_std = torch.std(sx2, dim=0) + 1e-8
        sx1 = sx1 / sx1_std
        sx2 = sx2 / sx2_std

        # 计算均值差异
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

# def get_CMD(x1, x2, n_moments=5):
#     mean_x1 = torch.mean(x1, dim=0)
#     mean_x2 = torch.mean(x2, dim=0)

#     # 初始为两个分布均值的 L2 范数差
#     diff = mean_x1 - mean_x2
#     loss = torch.norm(diff, p=2)

#     # 计算高阶中心矩的差异
#     for k in range(2, n_moments + 1):
#         moment_diff = torch.mean((x1 - mean_x1) ** k, dim=0) - torch.mean((x2 - mean_x2) ** k, dim=0)
#         loss = loss + torch.norm(moment_diff, p=2)  # 用非in-place操作累加loss

#     return loss


def get_DiffLoss(input1, input2):

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

def get_MSE(pred, real):
    diffs = torch.add(real, -pred)
    n = torch.numel(diffs.data)
    mse = torch.sum(diffs.pow(2)) / n
    return mse


def get_KL(x1, x2):
    # 将 x1 和 x2 转换为有效的概率分布（通过 softmax）
    p = F.softmax(x1, dim=1)  # 假设 x1 是 P 分布
    q = F.softmax(x2, dim=1)  # 假设 x2 是 Q 分布
    # 计算每个样本的 KL 散度
    kl_div = F.kl_div(q.log(), p, reduction='batchmean')
    return kl_div
    
def get_JS(x1, x2):
    # 将 x1 和 x2 转换为有效的概率分布（通过 softmax）
    p = F.softmax(x1, dim=1)  # 假设 x1 是 P 分布
    q = F.softmax(x2, dim=1)  # 假设 x2 是 Q 分布
    # 计算平均分布 M
    m = 0.5 * (p + q)
    
    # 计算 KL 散度 D_KL(P || M) 和 D_KL(Q || M)
    kl_pm = F.kl_div(m.log(), p, reduction='batchmean')
    kl_qm = F.kl_div(m.log(), q, reduction='batchmean')
    
    # JS 散度是 KL 散度的平均值
    js_div = 0.5 * (kl_pm + kl_qm)
    
    return js_div
    


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def get_MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
        source: 源域数据（n * len(x))
        target: 目标域数据（m * len(y))
        kernel_mul:
        kernel_num: 取不同高斯核的数量
        fix_sigma: 不同高斯核的sigma值
    Return:
        loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#

class MutualDistillationLoss(nn.Module):

    def __init__(self, temp=4., lambda_hyperparam=.1):

        super(MutualDistillationLoss, self).__init__()
        self.temp = temp
        self.kl_div = nn.KLDivLoss(reduction='none')
        self.lambda_hyperparam = lambda_hyperparam


    def forward(self, multi_view_logits, single_view_logits, targets):

        averaged_single_logits = torch.mean(single_view_logits, dim=1)
        q = torch.softmax(averaged_single_logits / self.temp, dim=1)

        try:
            max_q, pred_q = torch.max(q, dim=1)
            q_correct = pred_q == targets
            q_correct = q_correct.float().mean().item()
            max_q = max_q.mean().item()
        except RuntimeError:
            q_correct = 0.
            max_q = 0.

        p = torch.softmax(multi_view_logits / self.temp, dim=1)
        max_p, _ = torch.max(p, dim=1)
        max_p = max_p.mean().item()

        log_q = torch.log_softmax(averaged_single_logits / self.temp, dim=1)
        log_p = torch.log_softmax(multi_view_logits / self.temp, dim=1)

        loss = (1/2) * (self.kl_div(log_p, q.detach()).sum(dim=1).mean() + self.kl_div(log_q, p.detach()).sum(dim=1).mean())
        loss_weighted = loss * (self.temp ** 2) * self.lambda_hyperparam

        return loss_weighted
    
    