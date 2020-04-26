"""
Randomly split the dataset into training, validation, and test sets.
"""

import pandas as pd
import random
import math

path_dataset = r'../data/dataset.csv'
path_train = r'../data/train.csv'
path_val   = r'../data/val.csv'
path_test  = r'../data/test.csv'

def split_dataset(dataset_file, train_file, val_file, test_file, frac_train, frac_val):
    print('Reading data...')
    df = pd.read_csv(dataset_file)
    grouped = df.groupby('id_board')

    print('Shuffling boards...')
    board_dfs = [board_df for (_, board_df) in grouped]
    random.shuffle(board_dfs)

    print('Splitting into separate datasets...')
    num_boards = len(board_dfs)
    idx_end_train = math.floor(frac_train * num_boards)
    idx_end_valid = math.floor((frac_train + frac_val) * num_boards)
    
    train_dfs = board_dfs[:idx_end_train]
    print('- Training  : %d/%d (%.2f%%)' % (len(train_dfs), num_boards, 100*len(train_dfs)/num_boards))
    valid_dfs = board_dfs[idx_end_train:idx_end_valid]
    print('- Validation: %d/%d (%.2f%%)' % (len(valid_dfs), num_boards, 100*len(valid_dfs)/num_boards))
    test_dfs = board_dfs[idx_end_valid:]
    print('- Test      : %d/%d (%.2f%%)' % (len(test_dfs), num_boards, 100*len(test_dfs)/num_boards))

    print('Writing files...')
    print('- Training...')
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    train_df.to_csv(train_file)
    print('- Validation...')
    valid_df = pd.concat(valid_dfs).reset_index(drop=True)
    valid_df.to_csv(val_file)
    print('- Testing...')
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    test_df.to_csv(test_file)

if __name__ == "__main__":
    split_dataset(
        dataset_file = path_dataset,
        train_file   = path_train,
        val_file     = path_val,
        test_file    = path_test,
        frac_train = 0.6,
        frac_val   = 0.2
    )