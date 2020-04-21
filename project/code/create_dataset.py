"""
Code for creating dataset
"""

import csv
import math
import random

import chess.pgn

DATASET_FIELDS = [
    # Unique IDs
    'game_id',
    'board_id',

    # Game features
    'time_control_initial',   # Initial seconds in game
    'time_control_increment', # Increment per move in game
    'ELO_self',               # ELO of player to move
    'ELO_opponent',           # ELO of opponent

    # Board features
    'fen',                   # Current game state as a FEN string
    'full_moves',            # Move number (where 1 move consists of a white move, then a black move)
    'white_to_move',         # True if it's white's turn
    'clock_self',            # Seconds left at the start of this move
    'clock_opponent',        # Opponent's seconds left at the start of this move
    'clock_used',            # Seconds that the player spent on this move
    'check',                 # True if the king is in check
    'can_kingside_castle',   # True if the player has kingside castling rights
    'can_queenside_castle',  # True if the player has queenside castling rights
    'number_of_legal_moves', # Number of possible moves that can be played
    'attacked_pieces',       # Number of player's pieces that are under attack
    'attacking_pieces',      # Number of player's pieces that are attacking an opposing piece
    'undefended_pieces',     # Number of player's pieces that are hanging (no defender)
    'material',              # Material balance score, normalized to [-1, 1]

    # (previous move features)
    'prev_move_is_capture',                 # True if the previous move captured a piece
    'prev_move_clock_used',                 # Number of seconds the opponent used on the previous move
    'prev_move_threat_on_undefended_piece', # True if the previous move threatened a hanging piece

    # (TODO: stockfish static analysis features?)

    # Move features
    'move_uci',            # Notation
    'move_piece',          # One of {P, N, B, R, Q, K} 
    'move_dx',             # Signed move distance (horizontal)
    'move_dy',             # Signed move distance (vertical)
    'move_is_capture',     # True if this move would capture a piece
    'move_is_threatened',  # True if this piece is under attack
    'move_is_defended',    # True if this piece has a defender of any piece type
    'move_stockfish_eval', # TODO: Stockfish evaluation of this move
    'move_played',         # True if the human player chose this move
]

def parseElo(header):
    white_elo = header['WhiteElo']
    black_elo = header['BlackElo']
    return {'white_elo': white_elo, 'black_elo': black_elo}

def parseTimeControl(header):
    time_control = header['TimeControl']
    [game_time, move_time] = [int(s) for s in time_control.split('+')]
    return [game_time, move_time]

def parseTime(time_str):
    [h, m, s] = [int(s) for s in time_str.split(':')]
    return 60*60*h + 60*m + s

def parseComment(comment_str):
    clk_position = comment_str.find('%clk')
    if clk_position == -1:
        return None
    else:
        return parseTime(comment_str[clk_position+5:clk_position+12])

def scale_score(cp):
    # [-1,1]
    return (2/(1+math.exp(-0.004 * cp)) - 1)

def materialEval(board):
    eval = 0.0
    eval += len(board.pieces(chess.QUEEN, chess.WHITE)) * 9.0
    eval += len(board.pieces(chess.ROOK, chess.WHITE)) * 5.0
    eval += len(board.pieces(chess.BISHOP, chess.WHITE)) * 3.25
    eval += len(board.pieces(chess.KNIGHT, chess.WHITE)) * 3.0
    eval += len(board.pieces(chess.PAWN, chess.WHITE)) * 1.0
    eval -= len(board.pieces(chess.QUEEN, chess.BLACK)) * 9.0
    eval -= len(board.pieces(chess.ROOK, chess.BLACK)) * 5.0
    eval -= len(board.pieces(chess.BISHOP, chess.BLACK)) * 3.25
    eval -= len(board.pieces(chess.KNIGHT, chess.BLACK)) * 3.0
    eval -= len(board.pieces(chess.PAWN, chess.BLACK)) * 1.0
    eval += random.uniform(0.1, -0.1)
    moves = list(board.pseudo_legal_moves)
    move_val = len(moves)/50.0
    if board.turn:
        return scale_score(eval+move_val)
    else:
        return scale_score(-eval+move_val)

def getNumberofAttackedPieces(board):
    # attack depends on turn
    attackers = 0
    piece_map = board.piece_map() # get all the piece locations
    for square_index, piece in piece_map.items(): 
        if board.turn and piece.symbol().isupper():   # if white turn then find all attacks on white
            attackers += len(board.attackers(chess.BLACK, square_index))
            #print(square_index, ":", piece) 
        elif not board.turn and piece.symbol().islower():  # similar fo black
            attackers += len(board.attackers(chess.WHITE, square_index))
            #print(square_index, ":", piece) 
    return attackers

def getNumberofAttackingPieces(board):
    # attack depends on turn
    attacking_pieces = 0
    piece_map = board.piece_map()
    for square_index, piece in piece_map.items():
        if board.turn and piece.symbol().islower():  # for all white pieces attacking pieces on black occupied sqaures
            attacking_pieces += len(board.attackers(chess.WHITE,square_index))
        elif not board.turn and piece.symbol().isupper(): # for all black pieces attacking pieces on white occupied sqaures
            attacking_pieces += len(board.attackers(chess.BLACK,square_index))
    return attacking_pieces

def getNumberofUndefendedPieces(board):
    undefended_pieces = 0  # assume all are defended LOL :P
    piece_map = board.piece_map()
    for square_index,piece in piece_map.items():
        if board.turn and piece.symbol().isupper():
            undefended_pieces += not board.is_attacked_by(chess.WHITE,square_index) # for all white occupied sqaures check it is attacked by a white piece
        elif not board.turn and piece.symbol().islower():
            undefended_pieces += not board.is_attacked_by(chess.BLACK,square_index)
    return undefended_pieces

def getPreviousMoveClockUsed(game):
    # need to go 4 moves back to calculate the time 
    if game is None or game.comment is None:
        return 0
    if game.parent is None or game.parent.comment is None:
        return 0
    if game.parent.parent is None or game.parent.parent.comment is None:
        return 0
    if game.parent.parent.parent is None or game.parent.parent.parent.comment is None:
        return 0
    if game.parent.parent.parent.parent is None or game.parent.parent.parent.parent.comment is None:
        return 0
    
    parent_time= parseComment(game.parent.parent.comment) # this can also be none
    grandparent_time = parseComment(game.parent.parent.parent.parent.comment) # this can also be none in the begining
    if grandparent_time is None or parent_time is None:
        return 0
    
    if game.parent.parent.board().turn != game.parent.parent.parent.parent.board().turn:
        print("turns are not matching....")
    return grandparent_time-parent_time

# approach - find the sqaure to whcih the piece is moved. Then, find all the sqaures that are attacked from this sqaure. 
# from this set count the number of sqaures which belong to opponent 
def isPreviousMoveThreat(game):
    move = game.move                      # get the move which got us to this board
    board = game.board()
    if game is None or move is None:
        return False
    to_square = move.to_square           # select the sqaure to which the piece is moved
    attacks = board.attacks(to_square)   #find all the sqaures which are attacked by this sqaure
    piece_map = board.piece_map()
    threatened_pieces = 0
    for square_index,piece in piece_map.items(): 
        if board.turn and piece.symbol().isupper(): #if now its white turn, then check attacks on white
            if square_index in attacks: 
                threatened_pieces+=1
        elif not board.turn and piece.symbol().islower(): # if now its black turn, then check if black sqaure belongs to attacks set
            if square_index in attacks: 
                threatened_pieces+=1
        
    if threatened_pieces > 0:
        return True
    return False
    
def getMoveFeatures(board, move_played):
    """
    Return a list of objects describing features about every legal move.

    :param board: chess.Board object
    :param move: chess.Move object representing the move played in game
    :return: list of objects with move features. same length as board.legal_moves
    """
    ret = []
    
    for move in board.legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        ret.append({
            'move_uci': move.uci(),
            'move_piece': board.piece_at(from_square),
            'move_dx': chess.square_file(to_square) - chess.square_file(from_square),
            'move_dy': chess.square_rank(to_square) - chess.square_rank(from_square),
            'move_is_capture': board.is_capture(move),
            'move_is_threatened': board.is_attacked_by(not board.turn, from_square),
            'move_is_defended': board.is_attacked_by(board.turn, from_square),
            'move_stockfish_eval': 'TODO',
            'move_played': (move_played.uci() == move.uci()) if move_played is not None else None, # true/false
        })
    return ret

def parseGame(game):
    game_features = []

    # Game setting
    players_elo = parseElo(game.headers)
    [time_control_initial, time_control_increment] = parseTimeControl(game.headers)
    
    # Game starts with full time on clock
    white_clock = time_control_initial
    black_clock = time_control_initial

    # Time used by previous move
    clock_used_previous = 0

    # Game status
    game = game.root() # start from the root
    while not game.is_end():
        board = game.board()

        # Start with game features
        features = {
            "time_control_initial": time_control_initial,
            "time_control_increment": time_control_increment,
            "ELO_self": players_elo['white_elo'] if board.turn else players_elo['black_elo'],
            "ELO_opponent": players_elo['black_elo'] if board.turn else players_elo['white_elo']
        }

        # time for player and opponent
        clock_start = white_clock if board.turn else black_clock
        clock_end = parseComment(game.variations[0].comment) # note_node
        clock_used = clock_start - clock_end
        clock_opponent = black_clock if board.turn else white_clock
        
        features['clock_self'] =  clock_start
        features['clock_opponent']= clock_opponent
        features['clock_used'] = clock_used
        features['prev_move_clock_used'] = clock_used_previous
        
        # Update clocks for next move
        if board.turn:
            white_clock = clock_end
        else:
            black_clock = clock_end

        move = game.move # get the move from the game
        
        # Previous move features - NOT SURE IF THIS IS DONE CORRECTLY
        if game.parent is not None:
            game_node_parent = game.parent
            parent_board = game_node_parent.board()
            prev_game_clock_used = parseComment(game.comment) - parseComment(game_node_parent.comment) if parseComment(game_node_parent.comment) is not None else 0
            features['prev_move_is_capture'] = parent_board.is_capture(move) # check with previous board current move?
            features['prev_move_threat_on_undefended_piece'] = isPreviousMoveThreat(game) # passed currrent game since move inside this game is previous move actually
        else:
            features['prev_move_is_capture'] = False
            features['prev_move_threat_on_undefended_piece'] = False
            
        # Board features
        features["fen"] = board.fen()
        features['number_of_legal_moves'] = board.legal_moves.count()
        features["full_moves"] = board.fullmove_number    #Counts move pairs. Starts at 1 and is incremented after every move of the black side.
        features["check"] = board.is_check()
        features["material"] = materialEval(board)
        features["attacked_pieces"] = getNumberofAttackedPieces(board)
        features["attacking_pieces"] = getNumberofAttackingPieces(board)
        features["undefended_pieces"] = getNumberofUndefendedPieces(board)
        features['white_to_move'] = board.turn

        # Castling rights
        features["can_kingside_castle"] = board.has_kingside_castling_rights(board.turn) 
        features["can_queenside_castle"] = board.has_kingside_castling_rights(board.turn) 
        
        # Get per-move features
        next_move = game.variations[0].move
        features['move_features'] = getMoveFeatures(board, next_move)

        game_features.append(features)
        
        # Go to next node
        game = game.variations[0]
        clock_used_previous = clock_used
    return game_features

def parseDataset(pgn_fname, num_games, output_fname, report_every=1):
    """
    Convert a PGN into an ML-ready CSV.

    :param pgn_fname: path to PGN with many games
    :param num_games: number of games to read from PGN
    :param output_fname: path of CSV file to write
    """

    # Set up output file
    with open(output_fname, 'w') as f_out:
        writer = csv.DictWriter(f_out, DATASET_FIELDS)
        writer.writeheader()

        with open(pgn_fname, 'r') as f_in:
            game_num = 0
            board_num = 0
            while game_num < num_games:
                game = chess.pgn.read_game(f_in)

                game_features_list = parseGame(game)

                for game_features in game_features_list:
                    # Write separate row for each move
                    move_features_list = game_features['move_features']
                    del game_features['move_features']
                    for move_features in move_features_list:
                        writer.writerow({
                            'game_id': game_num, 
                            'board_id': board_num, 
                            **game_features, 
                            **move_features
                        })
                    board_num += 1
                game_num += 1
                if game_num % report_every == 0:
                    print('Parsed game: %d' % (game_num))


if __name__ == "__main__":
    parseDataset('../data/lichess_db_head.pgn', 30, '../data/sample_dataset.csv', report_every=5)

