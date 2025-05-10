import esm
import torch
import numpy as np
import models.ESMDBP as ft_model
import os,sys
import pandas as pd


esm_model_dict_dir='models'
fasta_path='example.fasta'
result_dir='result'
device='cuda:0'

esm_model=esm.ESM2()
alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

esm_model = torch.nn.DataParallel(esm_model)
esm_model.load_state_dict(torch.load(esm_model_dict_dir+os.sep+'ESM-DBP.model', map_location=lambda storage, loc: storage, weights_only=True))
esm_model.to(device)
esm_model.eval()

dbp_model=ft_model.SimpleFC(1280)
dbp_model.load_state_dict(torch.load(esm_model_dict_dir+os.sep+'ESM-DBP-DBP.model',map_location=lambda storage, loc: storage, weights_only=True))
dbp_model.to(device)
dbp_model.eval()

tf_model=ft_model.SimpleFC(1280)
tf_model.load_state_dict(torch.load(esm_model_dict_dir+os.sep+'ESM-DBP-TF.model',map_location=lambda storage, loc: storage, weights_only=True))
tf_model.to(device)
tf_model.eval()

dbs_model=ft_model.BiLstmDBP(input_size=1280,hidden_size=100,num_layers=2)
dbs_model.load_state_dict(torch.load(esm_model_dict_dir+os.sep+'ESM-DBP-DBS.model',map_location=lambda storage, loc: storage, weights_only=True))
dbs_model.to(device)
dbs_model.eval()


def get_one_protein_esm_fea(protein_name,seq):
    data = [(protein_name, seq)]
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data) # 1, num + 2
    # Extract per-residue representations (on device)
    with torch.no_grad():
        batch_tokens=batch_tokens.to(device)
        #batch*seq_len*fea_dim
        results = esm_model(tokens=batch_tokens, repr_layers=[33], return_contacts=False) # logits:1, num + 2, 33; representations: 33: 1, num + 2, 1280

        token_representations = torch.squeeze(results["representations"][33])
        return token_representations[1:-1]


def readfastaAndSeq(file_path):
    fi=open(file_path,'r')
    dicts={}
    while True:
        oneLine=fi.readline()
        twoLine=fi.readline()
        if not oneLine:break
        dicts[oneLine[1:-1]]=twoLine.replace('\n','')
    fi.close()
    return dicts


def get_one_prediction_res(pro_name,seq):
    with torch.no_grad():
        fea_represent=get_one_protein_esm_fea(pro_name,seq) # num, 1280
        c_fea_represent=torch.mean(fea_represent,dim=0).unsqueeze(dim=0) # 1, 1280

        dbs_pro = dbs_model(fea_represent) # num, 2
        dbs_out = torch.softmax(dbs_pro, dim=1).squeeze()[:, 1]  # num

        dbp_pro = dbp_model(c_fea_represent) # 1, 2
        dbp_out=torch.softmax(dbp_pro,dim=1).squeeze() # 2

        tf_pro = tf_model(c_fea_represent) # 1, 2
        tf_out = torch.softmax(tf_pro, dim=1).squeeze() # 2

        return dbs_out,dbp_out[1],tf_out[1]

if __name__=='__main__':
    fi=open(result_dir+os.sep+'DBP_TF_prediction.res','w')
    fi.write('Protein Name\tDBP prediction probability\tDBP prediction result\tTF prediction probability\tTF prediction result\n')
    pro_dicts=readfastaAndSeq(fasta_path)

    for pro,seq in pro_dicts.items():

        dbs_pro,dbp_out,tf_out=get_one_prediction_res(pro,seq)
        fi.write(pro + '\t' + str(round(dbp_out.item(),4)) + '\t' + str(dbp_out.item()>=0.5) + '\t' + str(round(tf_out.item(),4)) + '\t' + str(tf_out.item()>=0.5) + '\n')
        np_dbs_pro=dbs_pro.to('cpu').detach().numpy()
        dicts={'AA':list(seq),'DBS prediction probability':np_dbs_pro,'DBS prediction result':np_dbs_pro>=0.4}
        data = pd.DataFrame(dicts)
        data.to_csv(result_dir+os.sep+pro+'_DBS_prediction.csv')
    fi.close()