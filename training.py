import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from util import *
from util import _to_onehot
from config import get_config
from models.UAMRL import Multimodal_Affinity
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from evaluate_metrics import *
import prettytable as pt
import networkx as nx 
import matplotlib.pyplot as plt



device = torch.device('cuda:0')

writer = SummaryWriter()


def training(model, train_loader, optimizer, epoch, epochs):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), colour='red', ncols=75)
    training_loss = 0.0
    for batch, data in loop:
        if len(data) < 2:
            break
        compound_sequence = _to_onehot(data.id, 150).to(device)
        protein_img = torch.from_numpy(img_resize(data.id)).unsqueeze(1).to(torch.float).to(device)
        mu_ss, v_ss, alpha_ss, beta_ss, mu_gg, v_gg, alpha_gg, beta_gg, mu_sg, v_sg, alpha_sg, beta_sg, mu_gs, v_gs, alpha_gs, beta_gs = model(data.to(device), compound_sequence, protein_img)
        mu_ss_gg, v_ss_gg, alpha_ss_gg, beta_ss_gg = moe_nig(mu_ss, v_ss, alpha_ss, beta_ss, mu_gg, v_gg, alpha_gg, beta_gg)
        mu_ss_gg_sg, v_ss_gg_sg, alpha_ss_gg_sg, beta_ss_gg_sg = moe_nig(mu_ss_gg, v_ss_gg, alpha_ss_gg, beta_ss_gg, mu_sg, v_sg, alpha_sg, beta_sg)
        mu_all, v_all, alpha_all, beta_all = moe_nig(mu_ss_gg_sg, v_ss_gg_sg, alpha_ss_gg_sg, beta_ss_gg_sg, mu_gs, v_gs, alpha_gs, beta_gs)

        raw_loss = criterion_nig(mu_ss, v_ss, alpha_ss, beta_ss, data.y.view(-1, 1).to(torch.float).to(device)) + \
                   criterion_nig(mu_gg, v_gg, alpha_gg, beta_gg, data.y.view(-1, 1).to(torch.float).to(device)) + \
                   criterion_nig(mu_sg, v_sg, alpha_sg, beta_sg, data.y.view(-1, 1).to(torch.float).to(device)) + \
                   criterion_nig(mu_gs, v_gs, alpha_gs, beta_gs, data.y.view(-1, 1).to(torch.float).to(device)) + \
                   criterion_nig(mu_all, v_all, alpha_all, beta_all, data.y.view(-1, 1).to(torch.float).to(device))
        combined_loss = raw_loss

        optimizer.zero_grad()
        raw_loss.backward()
        optimizer.step()
        training_loss += combined_loss.item()
        loop.set_postfix(loss=combined_loss.item())

    print(f"cls loss:{np.mean(training_loss):.4f}")

def validation(model, loader, epoch=1, epochs=1):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        loop = tqdm(enumerate(loader), total=len(loader), colour='blue', ncols=75)
        # for batch, data in enumerate(loader):
        for batch, data in loop:
            compound_sequence = _to_onehot(data.id, 150).to(device)
            protein_img = torch.from_numpy(img_resize(data.id)).unsqueeze(1).to(torch.float).to(device)
            mu_ss, v_ss, alpha_ss, beta_ss, mu_gg, v_gg, alpha_gg, beta_gg, mu_sg, v_sg, alpha_sg, beta_sg, mu_gs, v_gs, alpha_gs, beta_gs = model(data.to(device), compound_sequence, protein_img)
            mu_ss_gg, v_ss_gg, alpha_ss_gg, beta_ss_gg = moe_nig(mu_ss, v_ss, alpha_ss, beta_ss, mu_gg, v_gg, alpha_gg, beta_gg)
            mu_ss_gg_sg, v_ss_gg_sg, alpha_ss_gg_sg, beta_ss_gg_sg = moe_nig(mu_ss_gg, v_ss_gg, alpha_ss_gg, beta_ss_gg, mu_sg, v_sg, alpha_sg, beta_sg)
            output, v_all, alpha_all, beta_all = moe_nig(mu_ss_gg_sg, v_ss_gg_sg, alpha_ss_gg_sg, beta_ss_gg_sg, mu_gs, v_gs, alpha_gs, beta_gs)
            total_preds = torch.cat((total_preds, output.detach().cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()
    return total_labels, total_preds

def graph_showing(data):
    G = nx.Graph()
    edge_index = data.edge_index.t().numpy()
    G.add_edges_from(edge_index)
    nx.draw(G)
    plt.savefig('aa.png')



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
    

if __name__ == '__main__':

    config = get_config()

    # load dataset
    batch_size = config.batch_size
    train_data = CompoundDataset(root='train_set', dataset='train_set')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = CompoundDataset(root='train_set', dataset='test_data')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # val_data = CompoundDataset(root='data', dataset='val_data')
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    epochs = config.epochs
    compound_sequence_dim = config.compound_sequence_dim
    protein_sequence_dim = config.protein_sequence_dim

    learning_rate = config.learning_rate

    patience = config.patience
    curr_patience = patience

    model = Multimodal_Affinity(compound_sequence_dim, protein_sequence_dim, [32, 64, 128] , config.hidden_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    best_result = [10.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    for epoch in range(1, epochs + 1):
        # model.load_state_dict(torch.load(f'data/best_model/model_{config.name}.pt', weights_only=True))
        training(model, train_loader, optimizer, epoch, epochs)
        test_labels, test_preds = validation(model, test_loader, epoch, epochs)
        test_result = [mae(test_labels, test_preds), rmse(test_labels, test_preds), pearson(test_labels, test_preds), spearman(test_labels, test_preds), ci(test_labels, test_preds), r_squared(test_labels, test_preds)]
       
        # val_labels, val_preds = validation(model, val_loader,epoch, epochs)
        # val_result = [mae(val_labels, val_preds), rmse(val_labels, val_preds), pearson(val_labels, val_preds), spearman(val_labels, val_preds), r_squared(val_labels, val_preds)]
        # tb.add_row(['{} / {}'.format(epoch, epochs), 'Validation', val_result[0], val_result[1], val_result[2], val_result[3], val_result[-1]])

        tb = pt.PrettyTable()
        tb.field_names = ['Epoch / curr_p ', 'Set', 'MAE', 'RMSE', 'Pearson', 'Spearman', 'CI', 'R-Squared']
        tb.add_row([f'{epoch} / {curr_patience}', 'Test', f'{test_result[0]:.4f}', f'{test_result[1]:.4f}', f'{test_result[2]:.4f}', f'{test_result[3]:.4f}', f'{test_result[4]:.4f}', f'{test_result[-1]:.4f}'])
        tb.add_row([f'best_result', 'Test',f'{best_result[0]:.4f}', f'{best_result[1]:.4f}', f'{best_result[2]:.4f}', f'{best_result[3]:.4f}', f'{best_result[4]:.4f}', f'{best_result[-1]:.4f}'])
        print(tb)
        writer.add_scalar('RMSE/Test RMSE', test_result[1], epoch) # 之后可视化RMSE
        with open(f'result/{config.name}.txt', 'a') as write:
            write.writelines(str(tb) + '\n')
        if test_result[1] < best_result[1]:
            best_result = test_result
            curr_patience = patience
            print("Found new best model on dev set!")
            # 修改优化器的学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate  # 修改学习率
            print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
            torch.save(model.state_dict(), f'data/best_model/model_{config.name}.pt')
        else:
            curr_patience -= 1
            if curr_patience <= -1:
                print("Update learning rate.")
                curr_patience = patience
                # model.load_state_dict(torch.load(f'data/best_model/model_{config.name}.pt', weights_only=True))
                scheduler.step()
                print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

    
