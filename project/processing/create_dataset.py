"""
Code for creating dataset
"""

import csv
import math
import random

import chess.pgn

from evaluate_position import load_engine, get_static_evaluation_keys, get_static_evaluation, get_analysis

DATASET_FIELDS = [
    # Unique IDs
    'id_game',
    'id_board',

    # Game features
    'game_time_control_initial',   # Initial seconds in game
    'game_time_control_increment', # Increment per move in game
    'game_ELO_self',               # ELO of player to move
    'game_ELO_opponent',           # ELO of opponent

    # Board features
    'board_fen',                   # Current game state as a FEN string
    'board_full_moves',            # Move number (where 1 move consists of a white move, then a black move)
    'board_white_to_move',         # True if it's white's turn
    'board_clock_self',            # Seconds left at the start of this move
    'board_clock_opponent',        # Opponent's seconds left at the start of this move
    'board_clock_used',            # Seconds that the player spent on this move
    'board_check',                 # True if the king is in check
    'board_can_kingside_castle',   # True if the player has kingside castling rights
    'board_can_queenside_castle',  # True if the player has queenside castling rights
    'board_number_of_legal_moves', # Number of possible moves that can be played
    'board_attacked_pieces',       # Number of player's pieces that are under attack
    'board_attacking_pieces',      # Number of player's pieces that are attacking an opposing piece
    'board_undefended_pieces',     # Number of player's pieces that are hanging (no defender)
    'board_material',              # Material balance score, normalized to [-1, 1]

    # (previous move features)
    'prev_move_is_capture',                 # True if the previous move captured a piece
    'prev_move_clock_used',                 # Number of seconds the opponent used on the previous move
    'prev_move_threat_on_undefended_piece', # True if the previous move threatened a hanging piece

    # Move features
    'move_uci',            # Notation
    'move_from_x',         # Initial file
    'move_from_y',         # Initial rank
    'move_to_x',           # Ending file
    'move_to_y',           # Ending rank
    'move_dx',             # Signed move distance (horizontal)
    'move_dy',             # Signed move distance (vertical)
    'move_piece',          # One of {P, N, B, R, Q, K}, or lowercase for black
    'move_piece_pawn',     # One-hot encoding of piece types
    'move_piece_knight',   #
    'move_piece_bishop',   #
    'move_piece_rook',     #
    'move_piece_queen',    #
    'move_piece_king',     #
    'move_is_capture',     # True if this move would capture a piece
    'move_is_threatened',  # True if this piece is under attack
    'move_is_defended',    # True if this piece has a defender of any piece type
    'move_stockfish_eval', # TODO: Stockfish evaluation of this move
    'move_played',         # True if the human player chose this move
] + [
    # Static evaluation features
    'stockfish_%s' % k for k in get_static_evaluation_keys()
]

# For Stockfish evaluation: equivalent score in centipawns when evaluation gives "mate in N"
MATE_CENTIPAWNS = 100*100 # 100 pawns

def parseElo(header):
    white_elo = header['WhiteElo']
    black_elo = header['BlackElo']
    return {'white_elo': white_elo, 'black_elo': black_elo}

def parseTimeControl(header):
    """
    Get the time control from a game header.

    :param header: dict with game headers from PGN
    :return: list of [initial time, increment], or None (for games with no time control)
    """
    time_control = header['TimeControl']
    if time_control == '-':
        return None
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
    
def getMoveFeatures(board, move_played, engine, analysis_limit):
    """
    Return a list of objects describing features about every legal move.

    :param board: chess.Board object
    :param move: chess.Move object representing the move played in game
    :param engine: chess.Engine object with Stockfish engine
    :return: list of objects with move features. same length as board.legal_moves
    """

    # Run analysis
    analysis = get_analysis(engine, board, analysis_limit)
    
    # Build list of moves
    ret = []
    for move in board.legal_moves:
        from_square = move.from_square
        to_square = move.to_square
        piece = board.piece_at(from_square)
        piece_string = str(piece)

        ret.append({
            'move_uci': move.uci(),
            'move_from_x': chess.square_file(from_square),
            'move_from_y': chess.square_rank(from_square),
            'move_to_x'  : chess.square_file(to_square),
            'move_to_y'  : chess.square_rank(to_square),
            'move_dx'    : chess.square_file(to_square) - chess.square_file(from_square),
            'move_dy'    : chess.square_rank(to_square) - chess.square_rank(from_square),
            'move_piece'       : piece,
            'move_piece_pawn'  : int(piece_string.lower() == 'p'),
            'move_piece_knight': int(piece_string.lower() == 'n'),
            'move_piece_bishop': int(piece_string.lower() == 'b'),
            'move_piece_rook'  : int(piece_string.lower() == 'r'),
            'move_piece_queen' : int(piece_string.lower() == 'q'),
            'move_piece_king'  : int(piece_string.lower() == 'k'),
            'move_is_capture'   : int(board.is_capture(move)),
            'move_is_threatened': int(board.is_attacked_by(not board.turn, from_square)),
            'move_is_defended'  : int(board.is_attacked_by(board.turn, from_square)),
            'move_stockfish_eval': analysis[move].pov(board.turn).score(mate_score=MATE_CENTIPAWNS),
            'move_played': int(move_played.uci() == move.uci()) if move_played is not None else None, # true/false
        })
    return ret

def parseGame(game, engine, analysis_limit):
    game_features = []

    # Game setting
    players_elo = parseElo(game.headers)
    time_control = parseTimeControl(game.headers)

    # Filter out correspondence games
    if time_control is None:
        return []

    # Filter out fast games and low/high ELOs
    min_elo = min(int(players_elo['white_elo']), int(players_elo['black_elo']))
    max_elo = max(int(players_elo['white_elo']), int(players_elo['black_elo']))
    [time_control_initial, time_control_increment] = time_control
    if time_control_initial < 180 or min_elo < 1300 or max_elo > 1800:
        return []
    
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
            "game_time_control_initial": time_control_initial,
            "game_time_control_increment": time_control_increment,
            "game_ELO_self": players_elo['white_elo'] if board.turn else players_elo['black_elo'],
            "game_ELO_opponent": players_elo['black_elo'] if board.turn else players_elo['white_elo']
        }

        # time for player and opponent
        clock_start = white_clock if board.turn else black_clock
        clock_end = parseComment(game.variations[0].comment) # note_node
        clock_used = clock_start - clock_end
        clock_opponent = black_clock if board.turn else white_clock
        
        features['board_clock_self'] =  clock_start
        features['board_clock_opponent']= clock_opponent
        features['board_clock_used'] = clock_used
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
            features['prev_move_is_capture'] = int(parent_board.is_capture(move)) # check with previous board current move?
            features['prev_move_threat_on_undefended_piece'] = int(isPreviousMoveThreat(game)) # passed currrent game since move inside this game is previous move actually
        else:
            features['prev_move_is_capture'] = int(False)
            features['prev_move_threat_on_undefended_piece'] = int(False)
            
        # Board features
        features["board_fen"] = board.fen()
        features['board_number_of_legal_moves'] = board.legal_moves.count()
        features["board_full_moves"] = board.fullmove_number    #Counts move pairs. Starts at 1 and is incremented after every move of the black side.
        features["board_check"] = int(board.is_check())
        features["board_material"] = materialEval(board)
        features["board_attacked_pieces"] = getNumberofAttackedPieces(board)
        features["board_attacking_pieces"] = getNumberofAttackingPieces(board)
        features["board_undefended_pieces"] = getNumberofUndefendedPieces(board)
        features['board_white_to_move'] = int(board.turn)

        # Castling rights
        features["board_can_kingside_castle"]  = int(board.has_kingside_castling_rights(board.turn)) 
        features["board_can_queenside_castle"] = int(board.has_kingside_castling_rights(board.turn)) 
        
        # Get per-move features
        next_move = game.variations[0].move

        # print('starting move features')
        features['move_features'] = getMoveFeatures(board, next_move, engine, analysis_limit)
        # print('ending move features')

        # Get static evaluation features
        static_eval = get_static_evaluation(engine, board)
        for k in static_eval:
            features['stockfish_%s' % k] = static_eval[k]

        game_features.append(features)
        
        # Go to next node
        game = game.variations[0]
        clock_used_previous = clock_used
    return game_features

def parseDataset(pgn_fname, output_fname, engine, analysis_limit, num_games, start_from=None, prev_board_num=0, report_every=1):
    """
    Convert a PGN into an ML-ready CSV.

    :param pgn_fname: path to PGN with many games
    :param output_fname: path of CSV file to write
    :param engine: Stockfish engine object (from load_engine)
    :param analysis_limit: chess.engine.Limit object for analysis
    :param num_games: number of games to read from PGN
    :param start_from: for restarting, skip start_from games
    :param prev_board_num: for restarting, last board number in previous CSV file
    """

    # Set up output file
    file_mode = 'w' if start_from is None else 'a'
    with open(output_fname, file_mode) as f_out:
        writer = csv.DictWriter(f_out, DATASET_FIELDS)

        if start_from is None:
            writer.writeheader()

        # Read games
        with open(pgn_fname, 'r') as f_in:
            game_num = 0
            board_num = prev_board_num+1
            num_filtered_games = 0

            # Catch up if skipping games
            if start_from is not None:
                for i in range(start_from):
                    chess.pgn.read_game(f_in)
                game_num = start_from

            # Parse games
            while game_num < num_games:
                game = chess.pgn.read_game(f_in)

                game_features_list = parseGame(game, engine, analysis_limit)

                if len(game_features_list) == 0:
                    num_filtered_games += 1

                for game_features in game_features_list:
                    # Write separate row for each move
                    move_features_list = game_features['move_features']
                    del game_features['move_features']
                    for (move_number, move_features) in enumerate(move_features_list):
                        row = {
                            'id_game': game_num, 
                            'id_board': board_num, 
                            **move_features
                        }

                        # Only write board features for first move
                        if move_number == 0:
                            row.update(game_features)
                        writer.writerow(row)
                    board_num += 1
                game_num += 1
                if game_num % report_every == 0:
                    print('Parsed game: %d (excluded: %d)' % (game_num, num_filtered_games))


if __name__ == "__main__":
    engine = load_engine()
    parseDataset(
        '../data/lichess_db_head.pgn', 
        '../data/dataset.csv', 
        engine, 
        chess.engine.Limit(depth=10),
        1000, 
        80,
        2156,
        report_every=10
    )
    engine.close()
