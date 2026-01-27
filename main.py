import cython_chess
import chess
import time
from cProfile import Profile
from transposition import TranspositionTable
from pstats import Stats
import zobrist


# Minimax search done
# Alpha beta pruning done
# Quiescent Search done
# piece square tables done
# endgame logic done
# mvv lva move ordering done
# TODO: piece immobility penalties, threefold repetition logic, transposition table


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
        50,  20,  15,  10,  10,  15,  20,  50,
        35,  10,   5,   0,   0,   5,  10,  35,
        25,   0,   0,  20,  20,   0,   0,  25,
        15,   5,  10,  35,  35,  10,   5,   15,
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
    chess.QUEEN: 1000,
    chess.ROOK: 500,
    chess.KNIGHT: 200,
    chess.BISHOP: 100,
}

FEN = input("input FEN: ")
if FEN == "f":
    FEN = "rnb1kbnr/ppp1pppp/8/8/8/2N5/PPPB1PPP/R2QKBNR b KQkq - 0 4"
board = chess.Board(FEN)
start_time = time.time()



EVALUATED = 0

def endgame_ratio(chessboard):
    # to determine the endgame, amount of pieces (EXCLUDING KINGS AND PAWNS) need to be summed
    pieces = 0

    for i in range(64):
        piece = chessboard.piece_at(i)
        if piece == None:
            continue
        piece_symbol = piece.symbol()
        if piece_symbol.lower() not in ["k", "p"]:
            pieces += 1

    # formula: 8+ pieces is 0 endgameness
    # <=4 is 1 endgameness
    # interpolate between
    return max(0, min(1, ((8-pieces) / 4)))

def is_into_enemy_pawn_attack(chessboard: chess.Board, move: chess.Move):
    enemy = not chessboard.turn
    enemy_pawns = chessboard.pieces(chess.PAWN, enemy)
    enemy_pawn_attackers = chessboard.attackers(enemy, move.to_square) & enemy_pawns
    return enemy_pawn_attackers != 0
def evaluate(chessboard):
    global EVALUATED

    evaluation = 0
    endgameness = endgame_ratio(chessboard)

    for i in range(64):
        piece = chessboard.piece_at(i)
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

def order_moves(moves, chessboard: chess.Board):
    scored = []
    CHECK_BONUS = 75
    PAWN_ATTACK_PENALTY_FACTOR = .75

    for move in moves:
        score = 0

        attacker = chessboard.piece_at(move.from_square)
        attacker_value = PIECE_VALUES[attacker.symbol().upper()]

        # MVV LVA
        if chessboard.is_capture(move):
            if chessboard.is_en_passant(move):
                victim_piece_type = "P"
            else:
                victim = chessboard.piece_at(move.to_square)
                victim_piece_type = victim.symbol().upper()

            victim_value = PIECE_VALUES[victim_piece_type]
            score += (victim_value - attacker_value)

        # promotion bonus
        if move.promotion is not None:
            score += PROMOTION_BONUS.get(move.promotion, 0)

        # check bonus
        if chessboard.gives_check(move):
            score += CHECK_BONUS

        # penalty for moving into pawn attack
        if is_into_enemy_pawn_attack(chessboard, move):
            if not board.is_capture(move):
                score -= attacker_value * PAWN_ATTACK_PENALTY_FACTOR

        scored.append((score, move))

    scored.sort(key=lambda x: x[0], reverse=True)
    #print(scored)
    return [m for _, m in scored]
def captured_piece(chessboard: chess.Board, move: chess.Move):
    if not chessboard.is_capture(move):
        return None

    if chessboard.is_en_passant(move):
        ep_square = chess.square(
            chess.square_file(move.to_square),
            chess.square_rank(move.from_square)
        )
        return chessboard.piece_at(ep_square)

    return chessboard.piece_at(move.to_square)
def qui_search(chessboard, alpha, beta):
    static_evaluation = evaluate(chessboard)
    best_eval = static_evaluation
    best_line = []

    MAXIMIZING_PLAYER = chessboard.turn
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

    all_moves = cython_chess.generate_legal_moves(chessboard, chess.BB_ALL, chess.BB_ALL)
    unordered_capture_moves = [move for move in all_moves if chessboard.is_capture(move)]
    capture_moves = order_moves(unordered_capture_moves, chessboard)
    for capture in capture_moves:
        captured = captured_piece(chessboard, capture)
        captured_value = PIECE_VALUES[captured.symbol().upper()]
        chessboard.push(capture)

        if MAXIMIZING_PLAYER:
            if static_evaluation + captured_value + 300 < alpha:
                # dont bother capturing if the capture doesnt even change the best move you can force
                chessboard.pop()
                continue
        else:
            if static_evaluation - captured_value - 300 > beta:
                chessboard.pop()
                continue

        evaluation, child_line = qui_search(chessboard, alpha, beta)
        chessboard.pop()
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
def search(
    chessboard: chess.Board,
    transposition_table,
    key: int,
    top_depth=6,
    depth=6,
    alpha=-MATE_SCORE,
    beta=MATE_SCORE,
):
    alpha0 = alpha
    beta0 = beta

    # TT lookup uses current key
    tt_entry = transposition_table.lookup(key)
    if tt_entry and tt_entry["depth"] >= depth:
        if tt_entry["bound"] == "EXACT":
            return tt_entry["eval"], tt_entry["best"]
        if tt_entry["bound"] == "LOWER":
            alpha = max(alpha, tt_entry["eval"])
        elif tt_entry["bound"] == "UPPER":
            beta = min(beta, tt_entry["eval"])
        if alpha >= beta:
            return tt_entry["eval"], tt_entry["best"]

    if depth == 0:
        evalu, best_line = qui_search(chessboard, alpha, beta)
        return evalu, best_line

    maximizing = chessboard.turn
    moves = list(cython_chess.generate_legal_moves(chessboard, chess.BB_ALL, chess.BB_ALL))

    if not moves:
        if chessboard.is_check():
            mate = MATE_SCORE - (top_depth - depth)
            if chessboard.turn == chess.WHITE:
                mate = -mate
            return mate, []
        return 0, []

    moves = order_moves(moves, chessboard)

    best_eval = -MATE_SCORE if maximizing else MATE_SCORE
    best_line = []

    for move in moves:
        # push + update key together
        new_key = zobrist.update_zobrist_key(chessboard, key, move)  # this PUSHES on chessboard !!!

        evalu, child_line = search(
            chessboard,
            transposition_table,
            new_key,
            top_depth=top_depth,
            depth=depth - 1,
            alpha=alpha,
            beta=beta,
        )

        chessboard.pop()

        if maximizing:
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

    # Determine bound using ORIGINAL window (alpha0, beta0)
    bound = "EXACT"
    if best_eval <= alpha0:
        bound = "UPPER"
    elif best_eval >= beta0:
        bound = "LOWER"

    transposition_table.store(
        key,         # store under the key of the CURRENT position
        best_eval,
        depth,
        best_line,
        bound,
    )

    return best_eval, best_line


def test():
    transposition_table = TranspositionTable()
    init_Zkey = zobrist.zobrist_key(board)
    print(search(board, transposition_table, init_Zkey, 4, 4))
    print("time elapsed:", round((time.time() - start_time), 2), "s")
    print("total TT hits:", transposition_table.total_hits)


profiler = Profile()
profiler.runcall(test)
stats = Stats(profiler)
stats.strip_dirs()
stats.sort_stats("cumulative")
stats.print_stats()
# 8.88s