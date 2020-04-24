import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_dataloader

class PreferBackwardMoves(nn.Module):
	def __init__(self):
		super(PreferBackwardMoves, self).__init__()
		self.softmax = nn.LogSoftmax(dim=0)

	def forward(self, x):
		(board, sf_eval, moves) = x
		return -moves['move_dy']

class StockfishScoreModel(nn.Module):
	def __init__(self):
		super(StockfishScoreModel, self).__init__()
		self.scale = nn.Parameter(torch.tensor([0.1]))

	def forward(self, x):
		(board, sf_eval, moves) = x
		stockfish_scores = moves['move_stockfish_eval']	
		logits = self.scale * stockfish_scores
		return logits

if __name__ == "__main__":
    print('loading...')
    loader = get_dataloader('../../data/sample_dataset_subset.csv')

    (board, sf_eval, moves), label = next(iter(loader))

    model_1 = PreferBackwardMoves()
    print(model_1((board, sf_eval, moves)))

    model_2 = StockfishScoreModel()
    output = model_2((board, sf_eval, moves))
    print(output)
    print(torch.argmax(output))
    print(label)