import main
import chess
import zobrist
from repetition import RepetitionTable

ILLEGAL_MOVE_MESSAGE = """
put a legal move you donut.
uci notation is written as  (from square)(to square)(+promotion piece if applicable)
for example:
a2a4 --> a2 to a4
c7c8r --> pawn goes from c7 to c8, promoting to a rook
"""

ILLEGAL_FEN_MESSAGE = """
Illegal FEN detected. Please make sure you pasted it right.
"""
def play_game(bot_thinking_time):

    initial_state = input("do you want to initialize the board state from a predetermined FEN? (y) yes (n) no ")
    if initial_state == "y":
        while True:
            try:
                initial_state = input("Enter your FEN: ")
                chessboard = chess.Board(initial_state)
                break
            except Exception:
                print(ILLEGAL_FEN_MESSAGE)
    else:
        initial_state = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        chessboard = chess.Board(initial_state)

    rep = RepetitionTable()
    key = zobrist.zobrist_key(chessboard)
    rep.increment(key)

    player_color = input("do you want to play as white? (y) yes (n) no ")
    if player_color == "y" and chessboard.turn == chess.WHITE:
        while True:
            try:
                player_move = chess.Move.from_uci(input("ok, you start. enter a move (uci format): "))
                break
            except chess.InvalidMoveError:
                print(ILLEGAL_MOVE_MESSAGE)
        key = zobrist.update_zobrist_key(chessboard, key, player_move)
        rep.increment(key)
    print(chessboard.fen())


    while True:
        bot_move = (main.iterative_deepening(bot_thinking_time, chessboard.fen(), rep, key))
        key = zobrist.update_zobrist_key(chessboard, key, bot_move)
        rep.increment(key)

        if chessboard.is_game_over():
            print("game over")
            print("result: ", chessboard.result())
            return

        while True:
            try:
                player_move = chess.Move.from_uci(input("your turn. enter a move (uci format): "))
                break
            except chess.InvalidMoveError:
                print(ILLEGAL_MOVE_MESSAGE)
        key = zobrist.update_zobrist_key(chessboard, key, player_move)
        rep.increment(key)

        if chessboard.is_game_over():
            print("game over")
            print("result: ", chessboard.result())
            return

play_game(9)