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

class PureRandomModel(nn.Module):
    """
    Simple baseline. Output uniform distribution over legal moves.
    """
    def __init__(self):
        super(PureRandomModel, self).__init__()
        self.dummy_param = nn.Parameter(torch.tensor([0.0]))
        
    def forward(self, x):
        (board, sf_eval, moves) = x
        return 0.0 * moves[:, 0]


class StockfishScoreModel(nn.Module):
    """
    Simple model only based on Stockfish analysis. 
    
    Single parameter s interpolates between playing randomly (s=0) and playing Stockfish's selected move (s = infty).

    Output is logit[move] = s * stockfish_eval[move].
    """
    def __init__(self, initial_scale=1e-3, feature_idx=-1):
        super(StockfishScoreModel, self).__init__()
        self.scale = nn.Parameter(torch.tensor([initial_scale]))
        self.stockfish_eval_idx = feature_idx

    def forward(self, x):
        (board, sf_eval, moves) = x
        stockfish_scores = moves[:, self.stockfish_eval_idx] 
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

class NeuralNetMoves(nn.Module):
    """
    2-layer neural net, only considering move features
    """
    def __init__(self, num_features_moves, hidden_units_1, hidden_units_2, activation):
        super(NeuralNetMoves, self).__init__()
        self.linear_moves = nn.Linear(num_features_moves, hidden_units_1)
        self.linear_hidden = nn.Linear(hidden_units_1, hidden_units_2)
        self.linear_output = nn.Linear(hidden_units_2, 1)
        self.activation = activation
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        (board, sf_eval, moves) = x

        hidden_1 = self.activation(self.linear_moves(moves))
        hidden_2 = self.activation(self.linear_hidden(self.dropout(hidden_1)))
        logits = self.linear_output(hidden_2).squeeze(dim=1)
        return logits

class NeuralNetMovesNoStockfish(nn.Module):
    """
    2-layer neural net, only considering move features
    """
    def __init__(self, allowed_features, hidden_units, activation):
        super(NeuralNetMovesNoStockfish, self).__init__()
        num_features = len(allowed_features)
        self.allowed_features = allowed_features
        self.linear_moves = nn.Linear(num_features, hidden_units)
        self.linear_output = nn.Linear(hidden_units, 1)
        self.activation = activation
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        (board, sf_eval, moves) = x

        inputs = moves[:, self.allowed_features]
        hidden = self.activation(self.linear_moves(inputs))
        logits = self.linear_output(self.dropout(hidden)).squeeze(dim=1)
        return logits

class NeuralNetBoard(nn.Module):
    """
    Deeper neural net. Considers board and move features.
    """
    def __init__(self, num_features_board, num_features_moves, hidden_units_1, hidden_units_2, activation):
        super(NeuralNetBoard, self).__init__()
        self.linear_board = nn.Linear(num_features_board, hidden_units_1)
        self.linear_moves = nn.Linear(num_features_moves, hidden_units_1)
        self.linear_hidden = nn.Linear(hidden_units_1, hidden_units_2)
        self.linear_output = nn.Linear(hidden_units_2, 1)
        self.activation = activation
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        (board, sf_eval, moves) = x

        hidden_1 = self.activation(self.linear_board(board) + self.linear_moves(moves))
        hidden_2 = self.activation(self.linear_hidden(self.dropout(hidden_1)))
        logits = self.linear_output(self.dropout(hidden_2)).squeeze(dim=1)
        return logits

class NeuralNetAll(nn.Module):
    """
    Deeper neural net. Considers all board, Stockfish, and move features.
    """
    def __init__(self, num_features_board, num_features_stockfish, num_features_moves, hidden_units_1, hidden_units_2, activation):
        super(NeuralNetAll, self).__init__()
        self.linear_board     = nn.Linear(num_features_board    , hidden_units_1)
        self.linear_stockfish = nn.Linear(num_features_stockfish, hidden_units_1)
        self.linear_moves     = nn.Linear(num_features_moves    , hidden_units_1)
        self.linear_hidden = nn.Linear(hidden_units_1, hidden_units_2)
        self.linear_output = nn.Linear(hidden_units_2, 1)
        self.activation = activation
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        (board, sf_eval, moves) = x

        hidden_1 = self.activation(self.linear_board(board) + self.linear_stockfish(sf_eval) + self.linear_moves(moves))
        hidden_2 = self.activation(self.linear_hidden(self.dropout(hidden_1)))
        logits = self.linear_output(self.dropout(hidden_2)).squeeze(dim=1)
        return logits

class TransformerMoves(nn.Module):
    def __init__(self, num_features_moves, hidden_units):
        super(TransformerMoves, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(num_features_moves, 1, hidden_units)
        self.linear_output = nn.Linear(num_features_moves, 1)

    def forward(self, x):
        (board, sf_eval, moves) = x
        # print(moves.size)
        hidden = self.transformer(moves.unsqueeze(dim=1))
        logits = self.linear_output(hidden).squeeze(dim=1).squeeze(dim=1)
        return logits

class TransformerMovesNoStockfish(nn.Module):
    def __init__(self, allowed_features, hidden_units):
        super(TransformerMovesNoStockfish, self).__init__()
        num_features = len(allowed_features)
        self.allowed_features = allowed_features
        self.transformer = nn.TransformerEncoderLayer(num_features, 1, hidden_units)
        self.linear_output = nn.Linear(num_features, 1)

    def forward(self, x):
        (board, sf_eval, moves) = x
        # print(moves.size)
        hidden = self.transformer(moves[:, self.allowed_features].unsqueeze(dim=1))
        logits = self.linear_output(hidden).squeeze(dim=1).squeeze(dim=1)
        return logits

class TransformerBoard(nn.Module):
    def __init__(self, num_features_board, num_features_moves, hidden_units):
        super(TransformerBoard, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(num_features_moves, 1, hidden_units)
        self.linear_board = nn.Linear(num_features_board, num_features_moves)
        self.linear_output = nn.Linear(num_features_moves, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        (board, sf_eval, moves) = x
        hidden = self.transformer(moves.unsqueeze(dim=1)) + self.activation(self.linear_board(board))
        logits = self.linear_output(hidden).squeeze(dim=1).squeeze(dim=1)
        return logits

class TransformerAll(nn.Module):
    def __init__(self, num_features_board, num_features_sf, num_features_moves, hidden_units):
        super(TransformerAll, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(num_features_moves, 1, hidden_units)
        self.linear_board = nn.Linear(num_features_board, num_features_moves)
        self.linear_sf = nn.Linear(num_features_sf, num_features_moves)
        self.linear_output = nn.Linear(num_features_moves, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        (board, sf_eval, moves) = x
        hidden = self.transformer(moves.unsqueeze(dim=1)) + self.activation(self.linear_board(board) + self.linear_sf(sf_eval))
        logits = self.linear_output(hidden).squeeze(dim=1).squeeze(dim=1)
        return logits

class AttentionMoves(nn.Module):
    def __init__(self, num_features_moves, hidden_units_1, hidden_units_2):
        super(AttentionMoves, self).__init__()
        self.self_attention = nn.MultiheadAttention(num_features_moves, 1)
        self.linear_hidden_1 = nn.Linear(num_features_moves, hidden_units_1)
        self.linear_hidden_2 = nn.Linear(hidden_units_1, hidden_units_2)
        self.linear_output = nn.Linear(hidden_units_2, 1)

    def forward(self, x):
        (board, sf_eval, moves) = x
        inputs = moves.unsqueeze(dim=1)
        attn_output, _ = self.self_attention(inputs, inputs, inputs)
        hidden_1 = inputs + attn_output
        hidden_2 = self.linear_hidden_1(hidden_1)
        hidden_3 = self.linear_hidden_2(hidden_2)
        logits = self.linear_output(hidden_3).squeeze(dim=1).squeeze(dim=1)
        return logits 
    

def build_model(model_type, feature_names):
    """
    Helper function: create a model object
    """

    num_board_features = len(feature_names['board'])
    num_stockfish_features  = len(feature_names['stockfish'])
    num_move_features  = len(feature_names['move'])    

    if model_type == 'random':
        model = PureRandomModel()

    elif model_type == 'stockfish_score':
        stockfish_idx = feature_names['move'].index('move_stockfish_eval')
        model = StockfishScoreModel(feature_idx=stockfish_idx)

    elif model_type == 'linear_moves':
        model = LinearMovesModel(num_move_features)

    elif model_type == 'nn_moves':
        model = NeuralNetMoves(num_move_features, 32, 16, nn.ReLU())

    elif model_type == 'nn_moves_no_stockfish':
        allowed_features = [i for i in range(len(feature_names['move'])) if 'stockfish' not in feature_names['move'][i]]
        model = NeuralNetMovesNoStockfish(allowed_features, 16, nn.ReLU())

    elif model_type == 'nn_board':
        model = NeuralNetBoard(num_board_features, num_move_features, 32, 16, nn.ReLU())

    elif model_type == 'nn_all':
        model = NeuralNetAll(num_board_features, num_stockfish_features, num_move_features, 32, 16, nn.ReLU())

    elif model_type == 'transformer_moves':
        print(num_move_features)
        model = TransformerMoves(num_move_features, 16)

    elif model_type == 'transformer_moves_no_stockfish':
        allowed_features = [i for i in range(len(feature_names['move'])) if 'stockfish' not in feature_names['move'][i]]
        model = TransformerMovesNoStockfish(allowed_features, 16)

    elif model_type == 'attention_moves':
        print(num_move_features)
        model = AttentionMoves(num_move_features, 32, 16)

    elif model_type == 'transformer_board':
        model = TransformerBoard(num_board_features, num_move_features, 16)

    elif model_type == 'transformer_all':
        model = TransformerAll(num_board_features, num_stockfish_features, num_move_features, 16)

    else: 
        raise ValueError('Unrecognized model type %s' % model_type)

    return model

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
    model = NeuralNet(num_board_features, num_move_features, 2, nn.ReLU())
    print(model((board, sf_eval, moves)))
    # print(output)
    # print(torch.argmax(output))
    # print(label)