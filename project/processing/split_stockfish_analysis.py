"""
Split the Stockfish analysis score feature.

TODO: ideally, this would happen in create_dataset.py... this is a quick hack for now.
"""

import pandas as pd

def splitStockfishColumn(df_in):
	df = df_in.copy()
	col = df['move_stockfish_eval']
	df['move_stockfish_cp'] = col
	df['move_stockfish_mate_winning'] = 0
	df['move_stockfish_mate_losing' ] = 0

	rows_winning = col > 9900
	rows_losing = col < -9900
	mate_scores_winning = 10000 - col[rows_winning] 
	mate_scores_losing  = 10000 + col[rows_losing]

	df.loc[rows_winning, 'move_stockfish_cp'] = 0
	df.loc[rows_losing,  'move_stockfish_cp'] = 0
	df.loc[rows_winning, 'move_stockfish_mate_winning'] = mate_scores_winning
	df.loc[rows_losing,  'move_stockfish_mate_losing' ] = mate_scores_losing
	return df

if __name__ == "__main__":
	files = {
		'train': ('../data/train_old.csv', '../data/train.csv'),
		'val'  : ('../data/val_old.csv'  , '../data/val.csv'  ),
		'test' : ('../data/test_old.csv' , '../data/test.csv' ),
	}
	for k in files:
		print(k)
		(old, new) = files[k]
		df = pd.read_csv(old)
		df.drop('Unnamed: 0', axis='columns')
		df_split = splitStockfishColumn(df)
		df_split.to_csv(new)
