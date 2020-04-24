import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

def get_feature_names(column_names):
    """
    Split column names into IDs, board features, move features, and labels
    
    TODO: set up column names to make this more readable instead of hard-coding...
    """

    columns_id = ['game_id', 'board_id']

    columns_board = [
        'time_control_initial',                 # Initial seconds in game
        'time_control_increment',               # Increment per move in game
        'ELO_self',                             # ELO of player to move
        'ELO_opponent',                         # ELO of opponent
        'fen',                                  # Current game state as a FEN string
        'full_moves',                           # Move number (where 1 move consists of a white move, then a black move)
        'white_to_move',                        # True if it's white's turn
        'clock_self',                           # Seconds left at the start of this move
        'clock_opponent',                       # Opponent's seconds left at the start of this move
        'clock_used',                           # Seconds that the player spent on this move
        'check',                                # True if the king is in check
        'can_kingside_castle',                  # True if the player has kingside castling rights
        'can_queenside_castle',                 # True if the player has queenside castling rights
        'number_of_legal_moves',                # Number of possible moves that can be played
        'attacked_pieces',                      # Number of player's pieces that are under attack
        'attacking_pieces',                     # Number of player's pieces that are attacking an opposing piece
        'undefended_pieces',                    # Number of player's pieces that are hanging (no defender)
        'material',                             # Material balance score, normalized to [-1, 1]
        'prev_move_is_capture',                 # True if the previous move captured a piece
        'prev_move_clock_used',                 # Number of seconds the opponent used on the previous move
        'prev_move_threat_on_undefended_piece', # True if the previous move threatened a hanging piece
    ]
    columns_stockfish = [c for c in column_names if c.startswith('stockfish_')]
    columns_move = [c for c in column_names if c.startswith('move_') and c != 'move_played']
    columns_label = ['move_played']

    return (columns_id, columns_board, columns_stockfish, columns_move, columns_label)


class ChessDataset(Dataset):
    """
    Dataset for working with parsed chess games.

    Example usage:
        dataset = ChessDataset('path/to/data.csv')
        (board, stockfish_eval, moves, correct_move) = dataset[0]
    """
    def __init__(self, csv_file):
        # Read and split column names into semantic groups
        df = pd.read_csv(csv_file)
        (columns_id, columns_board, columns_stockfish, columns_move, _) = get_feature_names(df.columns)

        # Split rows into boards
        dfs = [board_df.drop(columns=columns_id).reset_index(drop=True) for (_, board_df) in df.groupby('board_id')]
        
        # Split each board into features
        self.board_features = [df[columns_board].iloc[0].to_dict() for df in dfs]
        self.stockfish_features = [df[columns_stockfish].iloc[0].to_dict() for df in dfs]
        self.move_features = [df[columns_move].to_dict('records') for df in dfs]
        self.correct_moves = [df['move_played'].idxmax() for df in dfs]
        
    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, idx):
        return (self.board_features[idx], self.stockfish_features[idx], self.move_features[idx], self.correct_moves[idx])


if __name__ == "__main__":
    # Sample code for loading and reading one individual board
    print('loading...')
    chess_dataset = ChessDataset('../../data/sample_dataset.csv')
    print('fetching...')
    (board, stockfish_eval, moves, label) = chess_dataset[0]
    print(board)
    print(moves)
    print(label)
