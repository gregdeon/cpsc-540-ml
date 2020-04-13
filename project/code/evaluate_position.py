"""
Functions for evaluating positions with Stockfish.

Stockfish can give two types of evaluations: 
- Static: Describes which player has the advantage in the current position, without reading out 
          any variations. Broken down into components (like material, space, etc...). Fast. 
- Analysis: Describes which player has the advantage after reading a couple of moves ahead. Assumes
            near-optimal play by both sides. Slow: requires a few seconds for a good evaluation.
"""


import chess.engine

ENGINE_PATH = '/usr/games/stockfish'

def load_engine(engine_path=ENGINE_PATH):
    """
    Run and connect to Stockfish locally.

    :param engine_path: path to a Stockfish engine executable
    :return: a SimpleEngine object

    Note: make sure you call .close() on this object when you're finished with it!
    """
    return chess.engine.SimpleEngine.popen_uci(engine_path)


class EvalCommand(chess.engine.BaseCommand[chess.engine.UciProtocol, None]):
    """
    Helper class for sending static evaluation messages to Stockfish.
    """
    def start(self, engine) -> None:
        self.eval_strings = []
        engine.send_line("eval")

    def line_received(self, engine: chess.engine.UciProtocol, line: str) -> None:
        self.eval_strings.append(line)
        if line.startswith('Total Evaluation'):
            self.result.set_result(self.eval_strings)
            self.set_finished()

def parse_static_evaluation(eval_strings):
    """
    Helper function: parse static evaluation table into dictionary.

    A typical Stockfish eval output looks like:

          Eval term |    White    |    Black    |    Total    
                    |   MG    EG  |   MG    EG  |   MG    EG  
    ----------------+-------------+-------------+-------------
           Material |   ---   --- |   ---   --- | -2.45 -3.00 
          Imbalance |   ---   --- |   ---   --- |  0.35  0.35 
              Pawns |   ---   --- |   ---   --- |  0.15  0.12 
            Knights |  0.26  0.06 |  0.08  0.02 |  0.17  0.04 
             Bishop |  0.00  0.00 |  0.00  0.00 |  0.00  0.00 
              Rooks |  0.18  0.08 |  0.18  0.08 |  0.00  0.00 
             Queens |  0.00  0.00 |  0.00  0.00 |  0.00  0.00 
           Mobility |  0.36  0.87 |  0.04  0.68 |  0.33  0.19 
        King safety |  0.11 -0.13 |  0.04 -0.06 |  0.06 -0.06 
            Threats |  0.53  0.60 |  0.34  0.52 |  0.19  0.08 
       Passed pawns |  0.00  0.00 |  0.95  0.96 | -0.95 -0.96 
              Space |  0.01  0.00 |  0.02  0.00 | -0.02  0.00 
    ----------------+-------------+-------------+-------------
              Total |   ---   --- |   ---   --- | -2.16 -3.07 
    
    This function reads off each of the numbers in this table.

    :param eval_strings: list of strings -- one per line of eval output
    :return: dict of evaluation outputs. Keys are (eval_term, color, phase) tuples, with values:
        - eval_term: 'material', 'imbalance', ..., 'space', 'total'
        - color: 'white', 'black', 'total'
        - phase: 'midgame', 'endgame'
    """

    ret = {}
    
    # Only parse lines with scores
    eval_lines = list(range(3, 15)) + [16]
    eval_colors = 'white', 'black', 'total'
    eval_phases = 'midgame', 'endgame'

    for i_term in eval_lines:
        # Each line looks like: 'Pawns |   ---   --- |   ---   --- |  0.02  0.00',
        eval_line_split = eval_strings[i_term].split('|')
        eval_term = eval_line_split[0].strip().lower().replace(' ', '_')
        
        for i_color in range(len(eval_colors)):
            eval_line_color = eval_line_split[i_color + 1].strip().split()
            for i_phase in range(len(eval_phases)):
                value_string = eval_line_color[i_phase]
                if value_string == '---':
                    continue
                else:
                    value = float(value_string)
                    ret[(eval_term, eval_colors[i_color], eval_phases[i_phase])] = value
    
    return ret

def get_static_evaluation(engine, board):
    """
    Run and parse static evaluation.

    :param engine: a SimpleEngine object, probably from load_engine()
    :param board: a python-chess Board object
    :return: dict of evaluation outputs. See parse_static_evaluation for description
    """

    # Warning: this uses undocumented features of chess.engine.UciProtocol
    engine.protocol._position(board)
    eval_strings = engine.communicate(EvalCommand)
    return parse_static_evaluation(eval_strings)

def get_raw_analysis(engine, board, analysis_limit):
    """
    Analyze a position with Stockfish.

    :param engine: a SimpleEngine object, probably from load_engine()
    :param board: a python-chess Board object
    :param analysis_limit: a chess.engine.Limit object describing how deeply to read
    :return: raw output from engine analysis: list of dict objects like
    {
        'depth': 14,
        'seldepth': 27,
        'multipv': 1,
        'score': PovScore(Cp(-20), WHITE),
        'nodes': 3702548,
        'nps': 1161765,
        'tbhits': 0,
        'time': 3.187,
        'pv': [Move.from_uci('b1c3'), ... Move.from_uci('g1f1')]
    }

    Note: analysis limit can be in seconds:
        chess.engine.Limit(time=3)
    or in search depth:
        chess.engine.Limit(depth=14)
    """
    
    # Run analysis for all possible variations
    info = engine.analyse(board, analysis_limit, multipv=500)
    return info

def get_analysis(engine, board, analysis_limit):
    """
    Get a simplified version of Stockfish's analysis output.

    :param engine: a SimpleEngine object, probably from load_engine()
    :param board: a python-chess Board object
    :param analysis_limit: a chess.engine.Limit object describing how deeply to read
    :return: list of (first move, score)
    """
    info = get_raw_analysis(engine, board, analysis_limit)
    return [(i['pv'][0], i['score']) for i in info]


if __name__ == "__main__":
    fen = 'rnbqkb1r/pppp1ppp/5n2/8/4Pp2/5N2/PPPP2PP/RNBQKB1R w KQkq - 2 4'
    board = chess.Board(fen)
    engine = load_engine()
    
    static_evaluation = get_static_evaluation(engine, board)
    print(static_evaluation[('material', 'total', 'midgame')])
    print(static_evaluation[('total', 'total', 'midgame')])

    analysis = get_analysis(engine, board, chess.engine.Limit(depth=14))
    print(analysis)

    engine.close()