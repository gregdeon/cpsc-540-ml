import pandas as pd
from tqdm import tqdm
import numpy as np 

import torch
import torch.nn as nn

from dataset import ChessDataset
from models import *

def predict(model, dataset, order):
    # Put model into evaluate mode 
    model.eval()

    # Compute model outputs
    softmax = nn.Softmax(dim=0)
    p_model_list = [None] * len(dataset)
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc='Examples'):
            inputs, _ = dataset[i]
            output = model(inputs)
            p_model = softmax(output)
            p_model_list[order[i]] = p_model
    # print(p_model_list[0])
    return np.concatenate(p_model_list)

# Folder names
if __name__ == "__main__":
    model_list = [
        'random', 
        'stockfish_score', 
        'linear_moves', 
        'nn_moves', 
        'transformer_moves', 
        'nn_moves_no_stockfish',
        'transformer_moves_no_stockfish',
        'nn_board',
        'transformer_board',
        'nn_all',
        'transformer_all'
    ]

    data_csv = '../data/test.csv'
    data_cached = '../data/test_cached.pt'

    dataset = torch.load(data_cached)
    df = pd.read_csv(data_csv)

    # Pandas doesn't preserve the order of the groups when we load a dataset
    # Need this hack to figure out what order to put the predictions in
    groups, order = np.unique(df.groupby('id_board', sort=True).ngroup(), return_index=True)
    board_order = groups[np.argsort(np.argsort(order))]
    # print(board_order)
    # print(order)

    for model_name in model_list:
        print('Predicting with model %s...' % model_name)
        print('- Loading...')
        model = torch.load('models/%s/best.pt' % model_name)
        
        print('- Predicting...')
        predictions = predict(model, dataset, board_order)
        
        print('- Saving...')
        df_model = df.copy()
        df_model['p_model'] = predictions
        df_model.to_csv('../data/test_results_%s.csv' % model_name)
        print()