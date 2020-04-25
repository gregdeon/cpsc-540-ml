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

class LinearMovesModel(nn.Module):
    """

    """
    def __init__(self, num_features):
        super(LinearMovesModel, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        (board, sf_eval, moves) = x
        # TODO: get move features automatically instead of by name
        move_features = [
            'move_from_x',
            'move_from_y',         
            'move_to_x',           
            'move_to_y',           
            'move_dx',              
            'move_dy',              
            'move_piece_pawn',      
            'move_piece_knight',   
            'move_piece_bishop',   
            'move_piece_rook',     
            'move_piece_queen',    
            'move_piece_king',     
            'move_is_capture',      
            'move_is_threatened',   
            'move_is_defended',     
            'move_stockfish_eval',
        ]
        inputs = torch.stack([(moves[feature].float()) for feature in move_features]).transpose(0, 2).transpose(0, 1)
        logits = self.linear(inputs).squeeze(dim=2)
        return logits

if __name__ == "__main__":
    print('loading...')
    from dataset import get_dataloader
    loader = get_dataloader('../data/dataset_subset.csv')

    (board, sf_eval, moves), label = next(iter(loader))

    model_1 = PreferBackwardMoves()
    print(model_1((board, sf_eval, moves)))

    model_2 = StockfishScoreModel()
    output = model_2((board, sf_eval, moves))

    # TODO: read number of move features from example datapoint
    model_3 = LinearMovesModel(16)
    print(model_3((board, sf_eval, moves)))
    # print(output)
    # print(torch.argmax(output))
    # print(label)