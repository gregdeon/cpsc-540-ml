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
    columns_move = [c for c in column_names if c.startswith('move_') and c not in ['move_uci', 'move_piece']]
    columns_label = ['move_played']

    return (columns_drop, columns_board, columns_stockfish, columns_move, columns_label)


class ChessDataset(Dataset):
    """
    Dataset for working with parsed chess games.

    Example usage:
        dataset = ChessDataset('path/to/data.csv')
        (board, stockfish_eval, moves), correct_move = dataset[0]
    """
    def __init__(self, csv_file, verbose=False):
        # Read and split column names into semantic groups
        if verbose:
            print('Reading CSV...')
        df = pd.read_csv(csv_file)
        (columns_drop, columns_board, columns_stockfish, columns_move, _) = get_feature_names(df.columns)

        df_per_board = df.dropna()
        grouped = df.groupby('id_board')

        # Split each board into features
        if verbose:
            print('Separating features...')
            print('- Board features')
        self.board_features = df_per_board[columns_board].to_dict('records')

        if verbose:
            print('- Stockfish evaluation features')
        self.stockfish_features = df_per_board[columns_stockfish].to_dict('records') 

        if verbose:
            print('- Per-move features')
        # TODO: this is the new bottleneck. I don't know any nice ways to speed it up
        self.move_features = grouped.apply(lambda x: {feature: np.array(x[feature].values) for feature in columns_move}).values

        if verbose:
            print('- Moves played')
        self.correct_moves = [f['move_played'].argmax() for f in self.move_features]
        for f in self.move_features:
            del f['move_played']

    def __len__(self):
        return len(self.board_features)

    def __getitem__(self, idx):
        x = (self.board_features[idx], self.stockfish_features[idx], self.move_features[idx])
        y = self.correct_moves[idx] 
        return (x, y)

def get_dataloader(csv_path):
    """
    Set up a dataloader for a chess dataset.

    TODO: add config?
    Might help to add shuffle=True...
    """
    dataset = ChessDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=1)
    return dataloader


# TODO: using a DataLoader is hard because our move lists have varying lengths...
# Need a custom collate function, like this, but it also needs to convert variables to tensors...
# Until we figure this out, need a batch size of 1
def collate_boards(batch):
    x = [b[0] for b in batch]
    y = [b[1] for b in batch]
    return (x, y)

if __name__ == "__main__":
    # Sample code for loading and reading one individual board
    print('loading...')
    chess_dataset = ChessDataset('../data/dataset.csv', verbose=True)
    
    print('fetching...')
    (board, stockfish_eval, moves), label = chess_dataset[3]
    print(board)
    print(moves)
    print(label)

    # Example of iterating through dataset
    # for (i, ((board, sf_eval, moves), label)) in enumerate(chess_dataset):
    #     print(i, len(moves['move_stockfish_eval']), max(moves['move_stockfish_eval']))

    dl = DataLoader(chess_dataset, batch_size=1) #, collate_fn=collate_boards)
    (board, sf_eval, moves), label  = next(iter(dl))
    print(board)
    print(moves)
    print(label)
