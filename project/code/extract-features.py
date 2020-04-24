import chess, chess.pgn, chess.svg, chess.engine
import csv


def extract_features(board):
    """
    Return a list of features about the board.

    :param board: chess.Board object
    :return: dictionary of features
    """

    # Piece counts
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    piece_counts = {
        color: {piece: len(board.pieces(piece, color)) for piece in piece_types} 
    for color in [chess.WHITE, chess.BLACK]}

    # Material counts
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9, 
    }
    material_counts = {
        color: sum([piece_values[piece] * piece_counts[color][piece] for piece in piece_types]) 
    for color in [chess.WHITE, chess.BLACK]}

    return {
        'white_pawns'  : piece_counts[chess.WHITE][chess.PAWN  ],
        'white_knights': piece_counts[chess.WHITE][chess.KNIGHT],
        'white_bishops': piece_counts[chess.WHITE][chess.BISHOP],
        'white_rooks'  : piece_counts[chess.WHITE][chess.ROOK  ],
        'white_queens' : piece_counts[chess.WHITE][chess.QUEEN ],
        'black_pawns'  : piece_counts[chess.BLACK][chess.PAWN  ],
        'black_knights': piece_counts[chess.BLACK][chess.KNIGHT],
        'black_bishops': piece_counts[chess.BLACK][chess.BISHOP],
        'black_rooks'  : piece_counts[chess.BLACK][chess.ROOK  ],
        'black_queens' : piece_counts[chess.BLACK][chess.QUEEN ],
        'white_material': material_counts[chess.WHITE],
        'black_material': material_counts[chess.BLACK],
    }

if __name__ == "__main__":
    with open(r'../data/positions_raw.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]

    board = chess.Board(rows[6]['fen'])
    features = extract_features(board)
    print(features)