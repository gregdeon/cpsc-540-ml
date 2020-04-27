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
    def __init__(self, move_dy_index=5):
        super(PreferBackwardMoves, self).__init__()
        self.move_dy_index = move_dy_index

    def forward(self, x):
        (board, sf_eval, moves) = x
        return -torch.FloatTensor(1.0*moves[:, self.move_dy_index])

class StockfishScoreModel(nn.Module):
    """
    Simple model only based on Stockfish analysis. 
    
    Single parameter s interpolates between playing randomly (s=0) and playing Stockfish's selected move (s = infty).

    Output is logit[move] = s * stockfish_eval[move].
    """
    def __init__(self, initial_scale=1e-3, stockfish_score_idx=-1):
        super(StockfishScoreModel, self).__init__()
        self.scale = nn.Parameter(torch.tensor([initial_scale]))
        self.stockfish_score_idx = stockfish_score_idx

    def forward(self, x):
        (board, sf_eval, moves) = x
        stockfish_scores = moves[:, self.stockfish_score_idx] 
        logits = self.scale * stockfish_scores
        return logits

class LinearMovesModel(nn.Module):
    """
    Linear model based on move features alone.
    """
    def __init__(self, num_features):
        super(LinearMovesModel, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        (board, sf_eval, moves) = x
        logits = self.linear(moves).squeeze(dim=1)
        return logits

class NeuralNet(nn.Module):
    """
    Deeper neural net.

    TODO: document...
    """
    def __init__(self, num_features_board, num_features_moves, hidden_units, activation):
        super(NeuralNet, self).__init__()
        self.linear_board = nn.Linear(num_features_board, hidden_units)
        self.linear_moves = nn.Linear(num_features_moves, hidden_units)
        self.linear_output = nn.Linear(hidden_units, 1)
        self.activation = activation

    def forward(self, x):
        (board, sf_eval, moves) = x

        hidden = self.activation(self.linear_board(board) + self.linear_moves(moves))
        logits = self.linear_output(hidden).squeeze(dim=1)
        return logits



if __name__ == "__main__":
    print('loading...')
    from dataset import ChessDataset
    dataset = ChessDataset('../data/dataset_subset.csv')
    feature_names = dataset.get_column_names()

    (board, sf_eval, moves), label = dataset[10]

    move_dy_index = feature_names['move'].index('move_dy')
    model_1 = PreferBackwardMoves(move_dy_index)
    print(model_1((board, sf_eval, moves)))

    stockfish_score_index = feature_names['move'].index('move_stockfish_eval')
    model_2 = StockfishScoreModel()
    output = model_2((board, sf_eval, moves))

    num_move_features = len(feature_names['move'])
    model_3 = LinearMovesModel(num_move_features)
    print(model_3((board, sf_eval, moves)))

    num_board_features = len(feature_names['board'])
    model = NeuralNet(num_board_features, num_move_features, 8, nn.ReLU())
    print(model((board, sf_eval, moves)))
    # print(output)
    # print(torch.argmax(output))
    # print(label)