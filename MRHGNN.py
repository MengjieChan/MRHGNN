# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:28:11 2023

@author: Mengjie Chen
"""

import pandas as pd
import numpy as np
import Data_Process
import Synergy_Models
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import mlflow.pytorch
import Config
args = Config.parse()


experiment = mlflow.set_experiment("MRHGNN" + args.dataset)



def metrics(labels, predictions, epoch, type):

    auc = roc_auc_score(labels, predictions) 

    aupr = average_precision_score(labels, predictions)

    binary_predictions = (np.array(predictions) > 0.5).astype(int)

    binary_labels = np.array(labels).astype(int)

    f1 = f1_score(binary_labels, binary_predictions)

    accuracy = accuracy_score(binary_labels, binary_predictions)

    mlflow.log_metric(key=f"auc-{type}", value=float(auc), step=epoch)
    mlflow.log_metric(key=f"aupr-{type}", value=float(aupr), step=epoch)
    mlflow.log_metric(key=f"f1-{type}", value=float(f1), step=epoch)
    mlflow.log_metric(key=f"accuracy-{type}", value=float(accuracy), step=epoch)
    return auc, aupr, f1, accuracy



def test(edges, labels):
    Model.eval()
    with torch.no_grad():
        preds, _ = Model(Drug_Features, Cell_Line_Feature, edges)
        loss = 0
        pred = []
        real = []
        for idx in range(Data.numDrug, Data.numNode):
            pred.extend(torch.sigmoid(preds[idx]).cpu().detach().numpy())
            real.extend(labels[idx].cpu().detach().numpy())
            
            each_preds = preds[idx]
            each_labels = labels[idx]

            criterion = nn.BCEWithLogitsLoss()
            each_loss = criterion(each_preds, each_labels)
            loss += each_loss
        loss = loss / Data.CellsCount
    return loss, pred, real  
 
    
    
def train(Drug_Features, Data, epochs):
    
    Model.train()
    best_val_loss = 0
    best_model_state = None

    for epoch in range(epochs):

        preds, protein_embedding = Model(Drug_Features, Cell_Line_Feature, Data.train_edges)
        loss_train = 0
        train_pred = []
        train_real = []
        # compute train loss function
        for idx in range(Data.numDrug, Data.numNode):

            train_real.extend(Data.train_labels[idx].cpu().detach().numpy())
            each_labels = Data.train_labels[idx]
            each_preds = (torch.add(preds[idx][:len(each_labels)], preds[idx][len(each_labels):])) / 2
            train_pred.extend(torch.sigmoid(each_preds).cpu().detach().numpy())

            each_pos_weight = Data.pos_weights[idx]
            criterion = nn.BCEWithLogitsLoss(pos_weight=each_pos_weight)
            each_loss = criterion(each_preds, each_labels)

            loss_train += each_loss

        loss_ssl = args.alpha * SSL_agent.make_loss(protein_embedding)
        loss = loss_train / Data.CellsCount + loss_ssl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_results = metrics(train_real, train_pred, epoch, 'train')
        mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

        loss_valid, valid_pred, valid_real = test(Data.valid_edges, Data.valid_labels)
        valid_results = metrics(valid_real, valid_pred, epoch, 'valid')     
        mlflow.log_metric(key="valid loss", value=float(loss_valid), step=epoch)
        
        if valid_results[3] > best_val_loss:
            best_val_loss = valid_results[3]
            best_model_state = Model.state_dict()
            best_epoch = epoch
           

        if epoch % 20 == 0:
            print('Epoch: ', epoch, 'loss_train: {:.6f},'.format(loss),
                          'AUC: {:.6f},'.format(train_results[0]), 'AUPR: {:.6f},'.format(train_results[1]),
                          'F1: {:.6f},'.format(train_results[2]), 'ACC: {:.6f}'.format(train_results[3])
                          )
            print('Epoch: ', epoch, 'loss_val: {:.6f},'.format(loss_valid),
                          'AUC: {:.6f},'.format(valid_results[0]), 'AUPR: {:.6f},'.format(valid_results[1]),
                          'F1: {:.6f},'.format(valid_results[2]), 'ACC: {:.6f}'.format(valid_results[3])
                          )

    return best_model_state, best_epoch



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)   



if __name__ == '__main__':
                
    mlflow.log_param("Dataset_Name", args.dataset)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("weight_decay", args.weight_decay)
    mlflow.log_param("alpha", args.alpha)
    mlflow.log_param("device", args.device)
    mlflow.log_param("epochs", args.epochs)

    
    realFolds, Protein_Adjacency_Matrix, Drug_Adjacency_Matrix, numNode1, Drug_Features, Cell_Line_Feature, _, _, _, V1, E1, edge_num1, edge_length1, degV1 = Data_Process.process_data(args.dataset)
    
    
    Drug_Features = torch.tensor(Drug_Features).float().to(args.device)
    Cell_Line_Feature = torch.tensor(Cell_Line_Feature).float().to(args.device)
    V1 = torch.tensor(V1).long().to(args.device)
    E1 = torch.tensor(E1).long().to(args.device)

    for i in range(len(degV1)):
        degV1[i] = torch.tensor(degV1[i]).float().to(args.device)
    

    results = []
    
    for fold in range(args.k_fold):
        mlflow.start_run(nested=True)
        torch.manual_seed(args.seed)
        Data = realFolds[fold]
        Data = Data_Process.torch_from_numpy(Data, args.device)

        Model = Synergy_Models.Synergy(Data.numDrug, 
                                Synergy_Models.BioEncoder(Drug_Features.shape[1], Cell_Line_Feature.shape[1], Data.CellsCount, numNode1 - Data.numDrug, 512, device = args.device), 
                                Synergy_Models.RHGNN(V1, E1, edge_num1, edge_length1, degV1, 512, 256, 256, num_edge_types = 4, dropout = 0.4),
                                Synergy_Models.RHGNN(Data.V, Data.E, Data.hypergraph_edge_num, Data.edge_length, Data.degV_dict, 512, 128, 256, num_edge_types = 2, dropout = 0),
                                Synergy_Models.ChannelAttention(emb_size = 256),
                                Synergy_Models.BilinearDecoder(feature_dim = 256, numDrug = Data.numDrug, cellscount = Data.CellsCount)).to(args.device)

        SSL_agent = Synergy_Models.EdgeMask(Protein_Adjacency_Matrix, device = args.device, nhid = 256, mask_ratio = 0.6)

        optimizer = torch.optim.Adam(list(Model.parameters()) + list(SSL_agent.linear.parameters()), lr = args.learning_rate, weight_decay = args.weight_decay)

        mlflow.log_param("num_params", count_parameters(Model))
        best_model_state, best_epoch = train(Drug_Features, Data, args.epochs)
        Model.load_state_dict(best_model_state)
        mlflow.pytorch.log_model(Model, "model")
        
        loss_test, test_pred, test_real = test(Data.test_edges, Data.test_labels)
        test_results = metrics(test_real, test_pred, best_epoch, 'test')
        
        results.append(list(test_results))
        
        print('loss_test: {:.6f},'.format(loss_test),
                        'AUC: {:.6f},'.format(test_results[0]), 'AUPR: {:.6f},'.format(test_results[1]),
                        'F1: {:.6f},'.format(test_results[2]), 'ACC: {:.6f}'.format(test_results[3])
                        )
        print(best_epoch)
        mlflow.end_run()
        mlflow.log_metric(key="results", value=float(test_results[1]), step=fold)
    results = pd.DataFrame(results).to_numpy()
    print('Mean: ', np.mean(results, axis=0))
    print('Std: ', np.std(results, axis=0))
    with open("output_MRHGNN_" + args.dataset, "w") as file:
        for item in results:
            file.write(str(item) + "\n")

      