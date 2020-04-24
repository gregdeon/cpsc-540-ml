import torch
import torch.nn as nn

from dataset import ChessDataset

class PreferBishopMoves(nn.Module):
	def __init__(self):
		super(PreferBishopMoves, self).__init__()
		self.softmax = nn.Softmax(dim=0)

	def forward(self, x):
		is_bishop_move = torch.FloatTensor([move['move_piece'].lower() == 'b' for move in x])
		y = self.softmax(is_bishop_move)
		return y


if __name__ == "__main__":
    print('loading...')
    chess_dataset = ChessDataset('../../data/sample_dataset.csv')
    model = PreferBishopMoves()

    (board, stockfish, moves, label) = chess_dataset[10]
    print(model(moves))
    print(label)