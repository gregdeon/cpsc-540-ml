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
    columns_move = [c for c in column_names if c.startswith('move_') and c not in ['move_played', 'move_uci', 'move_piece']]
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

        # Split rows into boards
        if verbose:
            print('Splitting into boards...')
        dfs = [board_df.drop(columns=columns_drop).reset_index(drop=True) for (_, board_df) in df.groupby('id_board')]
        
        # Split each board into features
        if verbose:
            print('Separating features...')
        self.board_features = [df[columns_board].iloc[0].to_dict() for df in dfs]
        self.stockfish_features = [df[columns_stockfish].iloc[0].to_dict() for df in dfs]
        self.move_features = [df[columns_move].to_dict('list') for df in dfs]
        self.correct_moves = [df['move_played'].idxmax() for df in dfs]

        # Convert move features to np arrays
        for i in range(len(self.move_features)):
            self.move_features[i] = {k: np.array(self.move_features[i][k]) for k in self.move_features[i]}

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
    chess_dataset = ChessDataset('../data/dataset_subset.csv', verbose=True)
    
    print('fetching...')
    (board, stockfish_eval, moves), label = chess_dataset[3]
    print(board)
    print(moves)
    print(label)

    # Example of iterating through dataset
    for (i, ((board, sf_eval, moves), label)) in enumerate(chess_dataset):
        print(i, len(moves['move_stockfish_eval']), max(moves['move_stockfish_eval']))

    dl = DataLoader(chess_dataset, batch_size=1) #, collate_fn=collate_boards)
    (board, sf_eval, moves), label  = next(iter(dl))
    print(board)
    print(moves)
    print(label)
