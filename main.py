import cython_chess
import chess
import time
# Minimax search done
# Alpha beta pruning done
# Quiescent Search done
# piece square tables done

# TODO: piece immobility penalties, MVV LVA move ordering, endgame logic, threefold repetition logic

start_time = time.time()

PIECE_VALUES = {
    "P": 100,
    "N": 300,
    "B": 325,
    "R": 500,
    "Q": 900,
    "K": 0,
    ".": 0,
    "k": 0,
    "p": -100,
    "n": -300,
    "b": -325,
    "r": -500,
    "q": -900,
}

piece_square_tables = {
    "P": [
         0,   0,   0,   0,   0,   0,   0,   0,
        25,  20,  15,  10,  10,  15,  20,  25,
        15,  10,   5,   0,   0,   5,  10,  15,
        10,   0,   0,  20,  20,   0,   0,  10,
         5,   5,  10,  35,  35,  10,   5,   5,
        10,  10,  20,  30,  30, -15,  10,  10,
        10,  10,  10,  10,  10,  10,  10,  10,
         0,   0,   0,   0,   0,   0,   0,   0
    ],
    "N": [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   0,   5,   0,   0,   5,   0, -10,
        -10,   5,  10,  10,  10,  10,   5, -10,
        -10,   0,  10,  20,  20,  10,   0, -10,
        -10,   0,  10,  20,  20,  10,   0, -10,
        -10,   5,  10,  10,  10,  10,   5, -10,
        -10,   0,   5,   0,   0,   5,   0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ],
    "B": [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,  15,  10,  10,  15,   0, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ],
    "R": [
          0,   0,   0,   5,   5,   0,   0,   0,
         15,  15,  15,  20,  20,  15,  15,  15,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
          5,  10,  10,  10,  10,  10,  10,   5,
          0,   0,   0,  10,  10,   0,   0,   0
    ],
    "Q": [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,   5,   5,   5,   0, -10,
         -5,   0,   5,   5,   5,   5,   0,  -5,
          0,   0,   5,   5,   5,   5,   0,  -5,
        -10,   5,   5,   5,   5,   5,   0, -10,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20
    ],
    "K": [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
         20,  20,   0,   0,   0,   0,  20,  20,
         20,  30,  10,   0,   0,  10,  30,  20
    ],
    "KE": [
        -50, -40, -20, -10, -10, -20, -40, -50,
        -40, -10,   5,  10,  10,   5, -10, -40,
        -20,   5,  15,  25,  25,  15,   5, -20,
        -10,  10,  25,  30,  30,  25,  10, -10,
        -10,  10,  25,  30,  30,  25,  10, -10,
        -20,   5,  15,  25,  25,  15,   5, -20,
        -40, -10,   5,  10,  10,   5, -10, -40,
        -50, -40, -20, -10, -10, -20, -40, -50
    ]
}

PROMOTION_BONUS = {
    "QUEEN" : 1000,
    "ROOK" : 500,
    "KNIGHT" : 200,
    "BİSHOP" : 100
}

FEN = input("input FEN: ")
if FEN == "f":
    FEN = "rnb1kbnr/ppp1pppp/8/8/8/2N5/PPPB1PPP/R2QKBNR b KQkq - 0 4"
board = chess.Board(FEN)


def flatten_board(chessboard):
    flattened =  [board.piece_at(sq) for sq in chess.SQUARES]
    readable = []
    for piece in flattened:
        if piece == None:
            readable.append(".")
            continue
        readable.append(piece.symbol())
    return readable


EVALUATED = 0

def endgame_ratio(board):
    # to determine the endgame, amount of pieces (EXCLUDING KINGS AND PAWNS) need to be summed
    pieces = 0

    for i in range(64):
        piece = board.piece_at(i)
        if piece == None:
            continue
        piece_symbol = piece.symbol()
        if piece_symbol.lower() not in ["k", "p"]:
            pieces += 1

    # formula: 8+ pieces is 0 endgameness
    # <=4 is 1 endgameness
    # interpolate between
    return max(0, min(1, ((8-pieces) / 4)))


def evaluate(board):
    global EVALUATED

    evaluation = 0
    endgameness = endgame_ratio(board)

    for i in range(64):
        piece = board.piece_at(i)
        if piece == None:
            continue
        piece_symbol = piece.symbol()

        piece_value = PIECE_VALUES[piece_symbol.upper()]
        position_coord = 56 - ((i // 8) * 8) + (i % 8)
        # This formula converts coordinates from the cython-chess system so that they are compatible with the piece_squares_table system
        # Couldn't bother redoing the piece square tables so... this will do

        if not piece_symbol.isupper():
            position_coord = (63 - position_coord) # flipping coords for black pieces

        if piece_symbol.lower() == "k":
            positional_value = (1-endgameness) * piece_square_tables["K"][position_coord] + endgameness * piece_square_tables["KE"][position_coord]
        else:
            positional_value = piece_square_tables[piece_symbol.upper()][position_coord]


        total_value = positional_value + piece_value
        if not piece_symbol.isupper():
            total_value *= -1

        evaluation += total_value


    EVALUATED += 1
    if EVALUATED % 1000 == 0: print(EVALUATED)
    return evaluation

def order_moves(moves, board: chess.Board):
    scored = []

    for move in moves:
        score = 0

        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUES[attacker.piece_type]

        # MVV LVA
        if board.is_capture(move):
            if board.is_en_passant(move):
                victim_piece_type = chess.PAWN
            else:
                victim = board.piece_at(move.to_square)
                victim_piece_type = victim.piece_type

            victim_value = PIECE_VALUES[victim_piece_type]
            score += (victim_value - attacker_value)

        # promotion bonus
        if move.promotion is not None:
            score += PROMOTION_BONUS.get(move.promotion, 0)

        scored.append((score, move))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]
def captured_piece(board: chess.Board, move: chess.Move):
    if not board.is_capture(move):
        return None

    if board.is_en_passant(move):
        ep_square = chess.square(
            chess.square_file(move.to_square),
            chess.square_rank(move.from_square)
        )
        return board.piece_at(ep_square)

    return board.piece_at(move.to_square)
def qui_search(chessboard, alpha, beta):
    static_evaluation = evaluate(chessboard)
    best_eval = static_evaluation
    best_line = []

    MAXIMIZING_PLAYER = board.turn
    if MAXIMIZING_PLAYER:
        if static_evaluation > beta:
            return beta, []
        alpha = max(alpha, static_evaluation) # alpha beta pruning

        if static_evaluation + 1000 < alpha: # delta pruning, stop looking if position is hopelessly bad
            return alpha, []
    else:
        if static_evaluation < alpha:
            return alpha, []
        beta = min(beta, static_evaluation)

        if static_evaluation - 1000 > beta: # stop looking? position is too bad?
            return beta, []

    capture_moves = [move for move in board.legal_moves if board.is_capture(move)]
    for capture in capture_moves:
        captured = captured_piece(board, capture)
        captured_value = PIECE_VALUES[captured.symbol()]
        board.push(capture)

        if MAXIMIZING_PLAYER:
            if static_evaluation + captured_value + 300 < alpha:
                # dont bother capturing if the capture doesnt even change the best move you can force
                board.pop()
                continue
        else:
            if static_evaluation - captured_value - 300 > beta:
                board.pop()
                continue

        evaluation, child_line = qui_search(chessboard, alpha, beta)
        board.pop()
        full_line = [capture] + child_line
        if MAXIMIZING_PLAYER:
            if evaluation > best_eval:
                best_eval = evaluation
                best_line = full_line
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
        else:
            if evaluation < best_eval:
                best_eval = evaluation
                best_line = full_line
                beta = min(beta,evaluation)
                if beta <= alpha:
                    break

    return best_eval, best_line


MATE_SCORE = 10000000
def search(chessboard, top_depth = 6, depth = 6, alpha = -MATE_SCORE, beta = MATE_SCORE, best_line = []):

    if depth == 0:
        evalu, best_line = qui_search(chessboard, alpha, beta)
        return evalu, best_line
    MAXIMIZING_PLAYER = board.turn
    moves = list(chessboard.legal_moves)

    best_eval = 99999 # For black
    if MAXIMIZING_PLAYER: # for white
        best_eval = -99999
    best_line = []

    if not any(board.legal_moves):
        # No legal moves: either checkmate or stalemate.
        if board.is_check():
            checkmate_score = MATE_SCORE - (top_depth - depth) # Prefering earlier checkmates
            if board.turn == chess.WHITE:
                checkmate_score = -checkmate_score
            return checkmate_score, []
        else:
            # Stalemate
            return 0, []

    for move in moves:
        board.push(move)
        evalu, child_line = search(chessboard, top_depth, depth - 1, alpha, beta)
        board.pop()

        if MAXIMIZING_PLAYER:
            if evalu > best_eval:
                best_eval = evalu
                best_line = [move] + child_line
            alpha = max(alpha, best_eval)
        else:
            if evalu < best_eval:
                best_eval = evalu
                best_line = [move] + child_line
            beta = min(beta, best_eval)

        if alpha >= beta:
            break

    return best_eval, best_line

print(search(board, 4, 4))
print("time elapsed:", round((time.time() - start_time), 2), "s")