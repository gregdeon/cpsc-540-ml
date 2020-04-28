import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

def get_feature_names(column_names):
    """
    Split column names into IDs, board features, move features, and labels
    
    TODO: keep move_uci and move_piece, but replace strings with numerical values
    - move_uci: maybe from_x, from_y, to_x, to_y?
    - move_piece: maybe 1-hot?
    """

    columns_drop = [c for c in column_names if c in ['id_game', 'id_board', 'board_fen', 'move_uci']]
    columns_board = [c for c in column_names if (c.startswith('game_') or c.startswith('board_') or c.startswith('prev_move_')) and c != 'board_fen']
    columns_stockfish = [c for c in column_names if c.startswith('stockfish_')]
    columns_move = [c for c in column_names if c.startswith('move_') and c not in ['move_uci', 'move_piece', 'move_played']]
    columns_label = ['move_played']

    return (columns_drop, columns_board, columns_stockfish, columns_move, columns_label)

def normalize(df, df_standard=None, skip_columns = [], ):
    """
    Normalize each column to a z-score. Take means and standard deviations from df_standard (or df, if no df_standard provided)
    """

    if df_standard is None:
        df_standard = df

    result = df.copy()
    for feature_name in df.columns:
        if feature_name in skip_columns:
            result[feature_name] = df[feature_name]
        else:
            col_mean = df_standard[feature_name].mean()
            col_std = df_standard[feature_name].std()
            if col_std < 1e-6:
                result[feature_name] = 0
            else:
                result[feature_name] = (df[feature_name] - col_mean) / (col_std)
    return result


class ChessDataset(Dataset):
    """
    Dataset for working with parsed chess games.

    Example usage:
        dataset = ChessDataset('path/to/data.csv')
        (board, stockfish_eval, moves), correct_move = dataset[0]
    """
    def __init__(self, csv_file, csv_file_standard, verbose=False):
        # Read and split column names into semantic groups
        if verbose:
            print('Reading CSV...')
        df = pd.read_csv(csv_file)
        (columns_drop, columns_board, columns_stockfish, columns_move, columns_label) = get_feature_names(df.columns)

        df_standard = pd.read_csv(csv_file_standard)
        df = normalize(df, df_standard, skip_columns=['id_game', 'id_board', 'board_fen', 'move_uci', 'move_piece', 'move_played'])

        df_per_board = df.dropna()
        grouped = df.groupby('id_board')

        # Split each board into features
        if verbose:
            print('Separating features...')
            print('- Board features')
        self.board_features = torch.Tensor(df_per_board[columns_board].values)
        self.board_feature_names = columns_board

        if verbose:
            print('- Stockfish evaluation features')
        self.stockfish_features = torch.Tensor(df_per_board[columns_stockfish].values) 
        self.stockfish_feature_names = columns_stockfish

        if verbose:
            print('- Per-move features')
        # TODO: this is the new bottleneck. I don't know any nice ways to speed it up
        self.move_features = grouped.apply(lambda x: torch.Tensor(x[columns_move + columns_label].values)).values

        if verbose:
            print('- Moves played')
        # Extract label ('move_played') and remove from move features
        self.correct_moves = [f[:, -1].argmax() for f in self.move_features]
        for i in range(len(self.move_features)):
            self.move_features[i] = self.move_features[i][:, :-1]
        self.move_feature_names = columns_move

    def __len__(self):
        return len(self.board_features)

    def __getitem__(self, idx):
        x = (self.board_features[idx], self.stockfish_features[idx], self.move_features[idx])
        y = self.correct_moves[idx] 
        return (x, y)

    def get_column_names(self):
        """
        Return a dict of column names for each of the feature tensors
        """
        return {
            'board': self.board_feature_names,
            'stockfish': self.stockfish_feature_names,
            'move': self.move_feature_names,
        }

if __name__ == "__main__":
    # Sample code for loading and reading one individual board
    print('loading...')
    chess_dataset = ChessDataset('../data/val.csv', '../data/train.csv', verbose=True)
    
    print('fetching...')
    (board, stockfish_eval, moves), label = chess_dataset[0]
    print(board)
    print(stockfish_eval)
    print(moves)
    print(label)
    (board, stockfish_eval, moves), label = chess_dataset[2]
    print(board)

    print('Iterating through dataset...')
    for ((board, sf_eval, moves), label) in chess_dataset:
        pass

    print('Columns:')
    print(chess_dataset.get_column_names())

    print('Caching datasets...')
    train_dataset = ChessDataset('../data/train.csv', '../data/train.csv', verbose=True)
    torch.save(train_dataset, '../data/train_cached.pt')
    val_dataset   = ChessDataset('../data/val.csv'  , '../data/train.csv', verbose=True)
    torch.save(val_dataset  , '../data/val_cached.pt')
    test_dataset  = ChessDataset('../data/test.csv' , '../data/train.csv', verbose=True)
    torch.save(test_dataset , '../data/test_cached.pt')