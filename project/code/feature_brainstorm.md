Game setting
- Time control: maybe 2 numbers, like (base time, increment) in seconds
- Player strengths: ELO of self and opponent 

Game status
- Current board
	- Move number
	- Time left: seconds left for self and opponent.
	- Color to move: white or black. Unclear if this will help, since it's encoded in the legal moves...
	- Number of legal moves
	- Check: whether the king is currently in check
	- Castling rights: 4 yes/no values. Each player might have kingside and/or queenside castling rights.
	- Material: total counts for self and opponent. Maybe use textbook material count, like 1 for pawns, 3 for knights/bishops, 5 for rooks, and 9 for queens.
	- Number of attacking pieces
	- Number of attacked pieces
	- Number of undefended pieces
	- Stockfish evaluation: static evaluation of position. Would be helpful to get this broken down into components...
- Previous move
	- Time taken: in seconds
	- Was capture: yes/no. 
	- Is threat on undefended piece: yes/no

Legal moves
- Stockfish eval: evaluation after reading out variation starting with move. Maybe consider reading to a few depths?
- Piece: type of piece being moved
- dx, dy: how far piece is being moved
- Is capture: yes/no
- Is threatened: yes/no
- Is defended: yes/no
