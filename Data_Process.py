# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:28:59 2023

@author: Mengjie Chen
"""

import Config
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem, Descriptors
import random
import multiprocessing
from multiprocessing import Process, Queue, Value, Manager
import time
import torch
import torch_sparse
from itertools import combinations

params = Config.parse()



def Incidence_matrix(edge_list, num_node):
    # Construct sparse matrix
    row_indices = []
    col_indices = []
    values = []
    for index in range(len(edge_list)):
        for i, drug_id in enumerate(edge_list[index]):
            row_indices.append(drug_id)
            col_indices.append(index)
            values.append(1)
    
    sparse_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(num_node, len(edge_list)))
    return sparse_matrix



def Hyperedge_Index(Hyperedges):
    hyperedge_index = [[],[]]
    for i in range(len(Hyperedges)):
        length = len(Hyperedges[i])
        hyperedge_index[0].extend(Hyperedges[i])
        hyperedge_index[1].extend([i] * length)
    return hyperedge_index



def Construct_Hypergraph(hyperedge1, hyperedge2, node_num):

    G = hyperedge1 + hyperedge2
    degV = {}
    for i, g in enumerate([hyperedge1, hyperedge2]):
        h = Incidence_matrix(g, node_num)
        degv = torch.from_numpy(h.sum(1)).view(-1, 1).float().pow(-1)
        degv[degv.isinf()] = 1 # when not added self-loop, some nodes might not be connected with any edge
        degV[i] = degv.numpy()

    H = Incidence_matrix(G, node_num)

    (row, col), _ = torch_sparse.from_scipy(H)
    V, E = row, col

    return V.numpy(), E.numpy(), degV



def producer(queue, datum, event):
    for data in datum:
        Interaction_Score, numDrug, numCellline, iFold = data
        train_edges = {}
        train_labels = {}
        test_edges = {}
        test_labels = {}
        valid_edges = {}
        valid_labels = {}
        pos_weights = {}
        
        Synergistic_Graph = []
        Antagonistic_Graph = []
        
        Cell_Line_Specific = Interaction_Score.groupby('CellLine')
        
        for Cell_Line, Interaction in Cell_Line_Specific:
            for idx, (a, b) in Interaction[['DrugA', 'DrugB']].iterrows():
                Interaction.at[idx, 'DrugA'], Interaction.at[idx, 'DrugB'] = sorted([a, b])
            Positive_Interaction = Interaction[Interaction.iloc[:,3] >= params.threshold]
            Negitive_Interaction = Interaction[Interaction.iloc[:,3] < 0]
            
            
            Positive_Interaction = sorted(Positive_Interaction[['DrugA', 'DrugB']].values.tolist())
            Negitive_Interaction = sorted(Negitive_Interaction[['DrugA', 'DrugB']].values.tolist())
            
            random.seed(params.seed)
            random.shuffle(Positive_Interaction)
            random.shuffle(Negitive_Interaction)
        
            nSize = len(Positive_Interaction)
            foldSize = int(nSize / params.k_fold)
            startTest = iFold * foldSize
            endTest = (iFold + 1) * foldSize
            if endTest > nSize:
                endTest = nSize

            if iFold == params.k_fold - 1:
                startValid = 0
            else:
                startValid = endTest

            endValid = startValid + foldSize
            
            test_pos = Positive_Interaction[startTest:endTest]
            valid_pos = Positive_Interaction[startValid:endValid]
            

            test_neg = Negitive_Interaction[startTest:endTest]
            valid_neg = Negitive_Interaction[startValid:endValid]
            
            train_pos = [ x for x in Positive_Interaction if x not in test_pos + valid_pos ]
            train_neg = [ x for x in Negitive_Interaction if x not in test_neg + valid_neg ]
            
            Synergistic_Graph.extend([sorted(row + [Cell_Line]) for row in train_pos])
            Antagonistic_Graph.extend([sorted(row + [Cell_Line]) for row in train_neg])
            
            
            test = np.concatenate([test_pos, test_neg])
            y_test = [1] * len(test_pos) + [0] * len(test_neg)
            
            valid = np.concatenate([valid_pos, valid_neg])
            y_valid = [1] * len(valid_pos) + [0] * len(valid_neg)
            
            train = np.concatenate([train_pos, train_neg])
            y_train = [1] * len(train_pos) + [0] * len(train_neg)
            train = np.concatenate([train, [ [x[1],x[0]] for x in train ] ])
            
            pos_weight = len(train_neg) / len(train_pos)
            
            train_edges[Cell_Line] = train
            train_labels[Cell_Line] = y_train
            test_edges[Cell_Line] = test
            test_labels[Cell_Line] = y_test
            valid_edges[Cell_Line] = valid
            valid_labels[Cell_Line] = y_valid
            pos_weights[Cell_Line] = pos_weight
        hypergraph2 = Synergistic_Graph + Antagonistic_Graph
        
        edge_length = [len(Synergistic_Graph), len(Synergistic_Graph + Antagonistic_Graph)]
        node_num = numDrug + numCellline

        V, E, degV = Construct_Hypergraph(Synergistic_Graph, Antagonistic_Graph, node_num)

        realFold = RealFoldData(train_edges, train_labels, test_edges, test_labels, valid_edges, valid_labels, pos_weights)
        realFold.iFold = iFold
        realFold.CellsCount = numCellline
        realFold.numDrug = numDrug
        realFold.numNode = numDrug + numCellline
        realFold.V = V
        realFold.E = E
        realFold.edge_length = edge_length
        realFold.degV_dict = degV
        realFold.hypergraph_edge_num = len(hypergraph2)

        queue.put(realFold)
        event.set() # Notify consumer that new data is available


def consumer(queue, counter, realFolds, event):
    while True:
        data = queue.get()
        if data is None:
            break
       
        iFold = data.iFold
        event.clear()  # Reset event before processing data
        realFolds[iFold] = data
        # Update counter
        with counter.get_lock():
            counter.value += 1
            print("Fold: ", iFold, "Total: ", counter.value)
        event.set()


class RealFoldData:
    def __init__(self, train_edges, train_labels, test_edges, test_labels, valid_edges, valid_labels, pos_weights):
        self.train_edges = train_edges
        self.train_labels = train_labels
        self.test_edges = test_edges
        self.test_labels = test_labels
        self.valid_edges = valid_edges
        self.valid_labels = valid_labels
        self.pos_weights = pos_weights

def train_valid_test_split(Dataset_Name, DrugToID, numDrug):
    path = "%s/%s/" % (params.data_file, Dataset_Name)
    Interaction_Score = pd.read_csv(path + Dataset_Name + '_SCORE.csv',encoding='UTF-8')

    if params.Use_Cell_Line_Feature:
        Cell_Line = pd.read_csv(path + Dataset_Name + '_CELL_LINE_EXPRESSION.csv',encoding='UTF-8')
        CellLineToID = Cell_Line[['Cell_Line']].copy()
        CellLineToID['ID'] = range(numDrug, numDrug + len(CellLineToID))
    else:
        CellLineToID = Interaction_Score[['Cell_Line']].copy().drop_duplicates()
        CellLineToID['ID'] = range(numDrug, numDrug + len(CellLineToID))
    Cell_Line_Feature = Cell_Line.iloc[:,1:].to_numpy()

    print('Cell_Line_num, Features_dim: ', Cell_Line_Feature.shape)
    
    Cell_Line_Feature = (Cell_Line_Feature - np.mean(Cell_Line_Feature, axis=0)) / np.std(Cell_Line_Feature, axis=0)


    Interaction_Score = (
    pd.merge(Interaction_Score, DrugToID.iloc[:,1:], left_on='Drug_A', right_on='PubChem_CID', how='inner')
    .merge(DrugToID.iloc[:,1:], left_on='Drug_B', right_on='PubChem_CID', how='inner')
    .merge(CellLineToID, left_on='Cell_Line', right_on='Cell_Line', how='inner')
    )

    Interaction_Score = Interaction_Score[['Drug_ID_x','Drug_ID_y','ID','Score']]
    Interaction_Score = Interaction_Score.rename(columns={'Drug_ID_x':'DrugA','Drug_ID_y':'DrugB','ID':'CellLine'})
    
    numCellline = len(CellLineToID)
    

    # multiprocessing.set_start_method('fork')
    producers = []
    consumers = []
    queue = Queue(params.k_fold + params.n_data_worker)
    counter = Value('i', 0)
    
    # Create a dictionary that can be shared between multiple processes
    realFolds = Manager().dict()

    foldPerWorker = int(params.k_fold / params.n_data_worker)

    event = multiprocessing.Event()

    for i in range(params.n_data_worker):
        startFold = i * foldPerWorker
        endFold = (i + 1) * foldPerWorker
        endFold = min(endFold, params.k_fold)
        datums = []
        for iFold in range(startFold, endFold):
            data = Interaction_Score, numDrug, numCellline, iFold
            datums.append(data)
        producers.append(Process(target=producer, args=(queue, datums, event)))

    for _ in range(params.n_data_worker):
        p = Process(target=consumer, args=(queue, counter, realFolds, event))
        p.daemon = True
        consumers.append(p)
    print("Start Producers...")
    for p in producers:
        p.start()
    print("Start Consumers...")
    for p in consumers:
        p.start()

    for p in producers:
        p.join()
    print("Finish Producers")
    event.wait()  # Wait for all consumer processes to complete.
    while counter.value < params.k_fold:
        time.sleep(1)
        continue
    for _ in range(params.n_data_worker):
        queue.put(None)
    print("Finish Consumers")

    return realFolds, Cell_Line_Feature, CellLineToID




def process_data(Dataset_Name):
    
    path = "%s/%s/" % (params.data_file, Dataset_Name)
    
    Drug_Information = pd.read_csv(path + Dataset_Name + '_DRUG.csv',encoding='UTF-8').drop_duplicates(subset = ['PubChem_CID'])
    
    Drug_CID = Drug_Information['PubChem_CID'].tolist() # list
    
    Drug_Target = pd.read_csv(params.data_file +'/Chemical_Target_Interaction.csv',encoding='UTF-8')
    
    Drug_Target = Drug_Target[Drug_Target['PubChem_CID'].isin(Drug_CID)]
    
    # print('Dataset: ', Dataset_Name)
    # print('# Drug: ', len(Drug_Target['PubChem_CID'].drop_duplicates()))
    # print('# Protein: ', len(Drug_Target['Entry_ID'].drop_duplicates()))
    # print('# Drug_Protein_Interaction: ', len(Drug_Target))
    Drug_Information['Drug_ID'] = range(len(Drug_Information))
    

    DrugToID = Drug_Information[['Name','PubChem_CID','Drug_ID']]
    
    Drug_Target = pd.merge(Drug_Target, DrugToID.iloc[:,1:], left_on = 'PubChem_CID', right_on = 'PubChem_CID', how = 'inner')
    
   
    Target_List = Drug_Target['Entry_ID'].drop_duplicates().tolist()
    Protein_Interaction = pd.read_csv(params.data_file +'/Protein_Protein_Interaction.csv',encoding='UTF-8')
    
    Protein_Interaction = Protein_Interaction[Protein_Interaction['Protein_A'].isin(Target_List) & Protein_Interaction['Protein_B'].isin(Target_List)]
    
    Target = pd.Series(Target_List, index = [index + len(DrugToID) for index in range(len(Target_List))])
    TargetToID = pd.DataFrame({'Target': Target.values, 'Target_ID': Target.index})
    
    Drug_Target = pd.merge(Drug_Target, TargetToID, left_on = 'Entry_ID', right_on = 'Target', how = 'inner')
    Drug_Target = pd.DataFrame(Drug_Target[['Drug_ID','Target_ID']])
    
    Protein_Interaction = (
    pd.merge(Protein_Interaction, TargetToID, left_on='Protein_A', right_on='Target', how='inner')
    .merge(TargetToID, left_on='Protein_B', right_on='Target', how='inner')
    )
    
    Protein_Interaction = pd.DataFrame(Protein_Interaction[['Target_ID_x','Target_ID_y']])
    Protein_Interaction = Protein_Interaction.rename(columns={'Target_ID_x': 'Protein_A', 'Target_ID_y': 'Protein_B'})
    
    
    for idx, (a, b) in Protein_Interaction.iterrows():
        Protein_Interaction.at[idx, 'Protein_A'], Protein_Interaction.at[idx, 'Protein_B'] = sorted([a, b])
    Protein_Interaction = Protein_Interaction.drop_duplicates()
    

    Drug_Interaction = pd.read_csv(params.data_file +'/Drug_Drug_Interaction.csv',encoding='UTF-8')
    
    Drug_Interaction = (
    pd.merge(Drug_Interaction, DrugToID, left_on='DrugA', right_on='PubChem_CID', how='inner')
    .merge(DrugToID, left_on='DrugB', right_on='PubChem_CID', how='inner')
    )
    
    Drug_Interaction = pd.DataFrame(Drug_Interaction[['Drug_ID_x','Drug_ID_y']])
    Drug_Interaction = Drug_Interaction.rename(columns={'Drug_ID_x': 'Drug_A', 'Drug_ID_y': 'Drug_B'})

    
    Grouped_Drug = Drug_Target.groupby('Drug_ID')['Target_ID'].apply(list).reset_index()
    Grouped_Target = Drug_Target.groupby('Target_ID')['Drug_ID'].apply(list).reset_index()
    
    M1 = []
    M2 = []
    M3 = []
    M4 = []
    for _, (drug, target_list) in Grouped_Drug.iterrows():
        
        combination = pd.DataFrame(list(combinations(sorted(target_list), 2)), columns=['a', 'b'])
        combination = pd.merge(combination, Protein_Interaction, left_on=['a', 'b'], right_on=['Protein_A', 'Protein_B'], how='left')
       
        m1 = combination[~(combination['Protein_A'].isna() & combination['Protein_B'].isna())]
        m4 = combination[combination['Protein_A'].isna() & combination['Protein_B'].isna()]
        
        m1_list = [sorted([drug] + row) for row in m1[['a', 'b']].values.tolist()]
        m4_list = [sorted([drug] + row) for row in m4[['a', 'b']].values.tolist()]
        M1.extend(m1_list)
        M4.extend(m4_list)
        
    
    for _, (target, drug_list) in Grouped_Target.iterrows():
        
        combination = pd.DataFrame(list(combinations(sorted(drug_list), 2)), columns=['a', 'b'])
        combination = pd.merge(combination, Drug_Interaction, left_on=['a', 'b'], right_on=['Drug_A', 'Drug_B'], how='left')
       
        m2 = combination[~(combination['Drug_A'].isna() & combination['Drug_B'].isna())]
        m3 = combination[combination['Drug_A'].isna() & combination['Drug_B'].isna()]
        m2_list = [sorted(row + [target]) for row in m2[['a', 'b']].values.tolist()]
        m3_list = [sorted(row + [target]) for row in m3[['a', 'b']].values.tolist()]
        M2.extend(m2_list)
        M3.extend(m3_list)

    Protein_Interaction_Double = pd.concat([Protein_Interaction, Protein_Interaction[['Protein_B', 'Protein_A']].rename(columns={'Protein_B': 'Protein_A', 'Protein_A': 'Protein_B'})], axis=0)
    Drug_Interaction_Double = pd.concat([Drug_Interaction, Drug_Interaction[['Drug_B', 'Drug_A']].rename(columns={'Drug_B': 'Drug_A', 'Drug_A': 'Drug_B'})], axis=0)
      
    Hyperedge = M1 + M2 + M3 + M4
      
    numDrug = len(DrugToID)
    numTarget = len(TargetToID)
    numNode1 = numDrug + numTarget

    edge_length = [len(M1), len(M1 + M2), len(M1 + M2 + M3), len(Hyperedge)]
    
    edge_num1 = len(Hyperedge)
    degV1 = {}
    for i, g in enumerate([M1, M2, M3, M4]):
        h = Incidence_matrix(g, numNode1)
        degv = torch.from_numpy(h.sum(1)).view(-1, 1).float().pow(-1)
        degv[degv.isinf()] = 1 # when not added self-loop, some nodes might not be connected with any edge
        degV1[i] = degv.numpy()

    H = Incidence_matrix(Hyperedge, numNode1)

    (row, col), _ = torch_sparse.from_scipy(H)
    V1, E1 = row, col

    Protein_Interaction_Index = Protein_Interaction_Double-len(DrugToID)
    Protein_Adjacency_Matrix = coo_matrix((np.ones(len(Protein_Interaction_Index)), (Protein_Interaction_Index['Protein_A'], Protein_Interaction_Index['Protein_B'])), shape=(len(TargetToID), len(TargetToID)))
    Drug_Adjacency_Matrix = coo_matrix((np.ones(len(Drug_Interaction_Double)), (Drug_Interaction_Double['Drug_A'], Drug_Interaction_Double['Drug_B'])), shape=(len(DrugToID), len(DrugToID)))

    
    # Drug feature matrix
    Fingerprints = []
    descriptors = []
    for tup in zip(Drug_Information['Drug_ID'], Drug_Information['SMILES']):
        mol = Chem.MolFromSmiles(tup[1])
        # mol_f = MACCSkeys.GenMACCSKeys(mol)
        mol_f = AllChem.GetMorganFingerprintAsBitVect(mol, 6, nBits=300)
        Fingerprints.append((str(tup[0]), mol_f.ToList()))
        descriptor = Descriptors.CalcMolDescriptors(mol, missingVal=None, silent=True)
        descriptors.append((str(tup[0]), descriptor))
    
    Fingerprints = pd.DataFrame(dict(Fingerprints)).transpose().to_numpy() # DataFrame row: drug; col: feature(gene)
    descriptors = pd.DataFrame(dict(descriptors)).transpose().to_numpy()
    Drug_Features = np.hstack((Fingerprints, descriptors))
    
    print('the columns with zero variance: ', np.where(np.var(Drug_Features, axis=0) == 0))
    print('the columns with nan value: ', np.where(np.isnan(Drug_Features).any(axis=0)))
    Drug_Features = np.delete(Drug_Features, np.where(np.var(Drug_Features, axis=0) == 0)[0], axis=1)
    Drug_Features = Drug_Features[:, ~np.isnan(Drug_Features).any(axis=0)]
    print('Drug_num, Features_dim: ', Drug_Features.shape)
    
    Drug_Features = (Drug_Features - np.mean(Drug_Features, axis=0)) / np.std(Drug_Features, axis=0)

    
    realFolds, Cell_Line_Feature, CellLineToID = train_valid_test_split(Dataset_Name, DrugToID, numDrug)

    
    return realFolds, Protein_Adjacency_Matrix, Drug_Adjacency_Matrix, numNode1, Drug_Features, Cell_Line_Feature, DrugToID, TargetToID, CellLineToID, V1.numpy(), E1.numpy(), edge_num1, edge_length, degV1
    


def torch_from_numpy(Data, Device):
    for idx in range(Data.numDrug, Data.numNode):
        Data.train_edges[idx] = torch.tensor(Data.train_edges[idx]).long().to(Device)
        Data.train_labels[idx] = torch.tensor(Data.train_labels[idx]).float().to(Device)
        Data.valid_edges[idx] = torch.tensor(Data.valid_edges[idx]).long().to(Device)
        Data.valid_labels[idx] = torch.tensor(Data.valid_labels[idx]).float().to(Device)
        Data.test_edges[idx] = torch.tensor(Data.test_edges[idx]).long().to(Device)
        Data.test_labels[idx] = torch.tensor(Data.test_labels[idx]).float().to(Device)
        Data.pos_weights[idx] = torch.tensor(Data.pos_weights[idx]).float().to(Device)

    for i in range(len(Data.degV_dict)):
        Data.degV_dict[i] = torch.tensor(Data.degV_dict[i]).float().to(Device)
    Data.V = torch.tensor(Data.V).long().to(Device)
    Data.E = torch.tensor(Data.E).long().to(Device)

    return Data



