"""
Model definitions.

Models take a tuple of (board features, stockfish evaluation features, move features) as inputs:
- Board features: details about the game, the current board, and the previous move
- Stockfish evaluation: breakdown of Stockfish's static evaluation of the current position
- Move features: list of details about each move, including Stockfish analysis of variation

Models output a list of logits (higher = more likely), with one value for each move.
In other words, the output 
"""
import numpy as np

import torch
import torch.nn as nn

class PreferBackwardMoves(nn.Module):
	"""
	Silly example model. Predicts that backward moves are more likely.
	"""
	def __init__(self):
		super(PreferBackwardMoves, self).__init__()

	def forward(self, x):
		(board, sf_eval, moves) = x
		return -torch.FloatTensor(1.0*moves['move_dy'])

class StockfishScoreModel(nn.Module):
	"""
	Simple model only based on Stockfish analysis. 
	
	Single parameter s interpolates between playing randomly (s=0) and playing Stockfish's selected move (s = infty).

	Output is logit[move] = s * stockfish_eval[move].
	"""
	def __init__(self):
		super(StockfishScoreModel, self).__init__()
		self.scale = nn.Parameter(torch.tensor([1e-3]))

	def forward(self, x):
		(board, sf_eval, moves) = x
		stockfish_scores = moves['move_stockfish_eval']	
		logits = self.scale * stockfish_scores
		return logits


if __name__ == "__main__":
    print('loading...')
	from dataset import get_dataloader
    loader = get_dataloader('../../data/sample_dataset_subset.csv')

    (board, sf_eval, moves), label = next(iter(loader))

    model_1 = PreferBackwardMoves()
    print(model_1((board, sf_eval, moves)))

    model_2 = StockfishScoreModel()
    output = model_2((board, sf_eval, moves))
    print(output)
    print(torch.argmax(output))
    print(label)