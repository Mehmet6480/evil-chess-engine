import cython_chess
import chess
import time
from cProfile import Profile
from transposition import TranspositionTable
from pstats import Stats
import zobrist

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

MIDGAME_PST = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        98, 134, 61, 95, 68, 126, 34, -11,
        -6, 7, 26, 31, 65, 56, 25, -20,
        -14, 13, 6, 21, 23, 12, 17, -23,
        -27, -2, -5, 12, 17, 6, 10, -25,
        -26, -4, -4, -10, 3, 3, 33, -12,
        -35, -1, -20, -23, -15, 24, 38, -22,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.KNIGHT: [
        -167, -89, -34, -49,  61, -97, -15, -107,
         -73, -41,  72,  36,  23,  62,   7,  -17,
         -47,  60,  37,  65,  84, 129,  73,   44,
          -9,  17,  19,  53,  37,  69,  18,   22,
         -13,   4,  16,  13,  28,  19,  21,   -8,
         -23,  -9,  12,  10,  19,  17,  25,  -16,
         -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23
    ],
    chess.BISHOP: [
        -29, 4, -82, -37, -25, -42, 7, -8,
        -26, 16, -18, -13, 30, 59, 18, -47,
        -16, 37, 43, 40, 35, 50, 37, -2,
        -4, 5, 19, 50, 37, 37, 7, -2,
        -6, 13, 13, 26, 34, 12, 10, 4,
        0, 15, 15, 15, 14, 27, 18, 10,
        4, 15, 16, 0, 7, 21, 33, 1,
        -33, -3, -14, -21, -13, -12, -39, -21
    ],
    chess.ROOK: [
        32, 42, 32, 51, 63, 9, 31, 43,
        27, 32, 58, 62, 80, 67, 26, 44,
        -5, 19, 26, 36, 17, 45, 61, 16,
        -24, -11, 7, 26, 24, 35, -8, -20,
        -36, -26, -12, -1, 9, -7, 6, -23,
        -45, -25, -16, -17, 3, 0, -5, -33,
        -44, -16, -20, -9, -1, 11, -6, -71,
        -19, -13, 1, 17, 16, 7, -37, -26
    ],
    chess.QUEEN: [
        -28, 0, 29, 12, 59, 44, 43, 45,
        -24, -39, -5, 1, -16, 57, 28, 54,
        -13, -17, 7, 8, 29, 56, 47, 57,
        -27, -27, -16, -16, -1, 17, -2, 1,
        -9, -26, -9, -10, -2, -4, 3, -3,
        -14, 2, -11, -2, -5, 2, 14, 5,
        -35, -8, 11, 2, 8, 15, -3, 1,
        -1, -18, -9, 10, -15, -25, -31, -50,
    ],
    chess.KING: [
        -65, 23, 16, -15, -56, -34, 2, 13,
        29, -1, -20, -7, -8, -4, -38, -29,
        -9, 24, 2, -16, -20, 6, 22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49, -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
        1, 7, -8, -64, -43, -16, 9, 8,
        -15, 36, 12, -54, 8, -28, 24, 14,
    ]
}
ENDGAME_PST = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        178, 173, 158, 134, 147, 132, 165, 187,
        94, 100, 85, 67, 56, 53, 82, 84,
        32, 24, 13, 5, -2, 4, 17, 17,
        13, 9, -3, -7, -7, -8, 3, -1,
        4, 7, -6, 1, 0, -5, -1, -8,
        13, 8, 8, 10, 13, 0, 2, -7,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.KNIGHT: [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25, -8, -25, -2, -9, -25, -24, -52,
        -24, -20, 10, 9, -1, -9, -19, -41,
        -17, 3, 22, 22, 22, 11, 8, -18,
        -18, -6, 16, 25, 16, 17, 4, -18,
        -23, -3, -1, 15, 10, -3, -20, -22,
        -42, -20, -10, -5, -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    chess.BISHOP: [
        -14, -21, -11, -8, -7, -9, -17, -24,
        -8, -4, 7, -12, -3, -13, -4, -14,
        2, -8, 0, -1, -2, 6, 0, 4,
        -3, 9, 12, 9, 14, 10, 3, 2,
        -6, 3, 13, 19, 7, 10, -3, -9,
        -12, -3, 8, 10, 13, 3, -7, -15,
        -14, -18, -7, -1, 4, -9, -15, -27,
        -23, -9, -23, -5, -9, -16, -5, -17
    ],
    chess.ROOK: [
        13, 10, 18, 15, 12, 12, 8, 5,
        11, 13, 13, 11, -3, 3, 8, 3,
        7, 7, 7, 5, 4, -3, -5, -3,
        4, 3, 13, 1, 2, 1, -1, 2,
        3, 5, 8, 4, -5, -6, -8, -11,
        -4, 0, -5, -1, -7, -12, -8, -16,
        -6, -6, 0, 2, -9, -9, -11, -3,
        -9, 2, 3, -1, -5, -13, 4, -20,
    ],
    chess.QUEEN: [
        -9, 22, 22, 27, 27, 19, 10, 20,
        -17, 20, 32, 41, 58, 25, 30, 0,
        -20, 6, 9, 49, 47, 35, 19, 9,
        3, 22, 24, 45, 57, 40, 57, 36,
        -18, 28, 19, 47, 31, 34, 39, 23,
        -16, -27, 15, 6, 9, 17, 10, 5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43, -5, -32, -20, -41,
    ],
    chess.KING: [
        -74, -35, -18, -18, -11, 15, 4, -17,
        -12, 17, 14, 17, 17, 38, 23, 11,
        10, 17, 23, 15, 20, 45, 44, 13,
        -8, 22, 24, 27, 26, 33, 26, 3,
        -18, -4, 21, 24, 27, 23, 9, -11,
        -19, -3, 11, 21, 23, 16, 7, -9,
        -27, -11, 4, 13, 14, 4, -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43
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
    FEN = "r1b1kb1r/1pp2ppp/p2q1n2/3p4/5P2/2NBP3/PPPB2PP/R2QK2R w KQkq - 1 10"
board = chess.Board(FEN)
start_time = time.time()




def non_king_or_pawn_pieces(chessboard):
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
    return pieces

def efficient_endgameness_udpate(chessboard : chess.Board, move : chess.Move):
    global non_kp_pieces
    global endgameness
    if chessboard.is_en_passant(move):
        pass
    elif chessboard.is_capture(move):
        if chessboard.piece_at(move.to_square).piece_type != chess.PAWN:
            non_kp_pieces -= 1
    if move.promotion is not None:
        non_kp_pieces += 1
    endgameness = max(0.0, min(1.0, (8 - non_kp_pieces) / 4.0))

non_kp_pieces = non_king_or_pawn_pieces(board)
endgameness = max(0.0, min(1.0, (8 - non_kp_pieces) / 4.0))

WHITE_PST_COORDS = [56 - ((sq // 8) * 8) + (sq % 8) for sq in range(64)]
BLACK_PST_COORDS = [WHITE_PST_COORDS[chess.square_mirror(sq)] for sq in range(64)]
VALUES = [0, 100, 300, 325, 500, 900, 0]



KING_RING_SQUARES = [None] * 64

for ksq in range(64):
    bb = chess.BB_KING_ATTACKS[ksq]
    squares = []
    while bb:
        lsb = bb & -bb
        squares.append(lsb.bit_length() - 1)
        bb ^= lsb
    KING_RING_SQUARES[ksq] = tuple(squares)


ATTACK_MULTIPLIERS = [0, 0, 0.5, 0.75, 0.88, 0.94, 0.97, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# the logic is that 1 piece by itself cannot really attack and 2 pieces will be a weak attack, etc.
# king safety penalty will be multiplied by the amt of pieces attacking.
def evaluate(chessboard):

    # local copies of global values for extra speed
    phase = endgameness
    midgame_pst = MIDGAME_PST
    endgame_pst = ENDGAME_PST
    white_coords = WHITE_PST_COORDS
    black_coords = BLACK_PST_COORDS
    piece_values = VALUES
    KING_SAFETY_PENALTY = 17
    midgame_score = 0 ; endgame_score = 0
    white_king_attackers = 0 ; black_king_attackers = 0
    white_king_ring_attacked = 0 ; black_king_ring_attacked = 0

    white_king_sq = chessboard.king(chess.WHITE) ; white_king_ring = chess.BB_KING_ATTACKS[white_king_sq]
    black_king_sq = chessboard.king(chess.BLACK) ; black_king_ring = chess.BB_KING_ATTACKS[black_king_sq]

    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        base_value = piece_values[piece_type]

        mg_table = midgame_pst[piece_type]
        eg_table = endgame_pst[piece_type]

        # BITBOARDS instead of piece_map for more speed
        # 64 bit integers representing locations of each piece type
        white_bitboard = chessboard.pieces_mask(piece_type, chess.WHITE)
        while white_bitboard:
            least_significant_bit = white_bitboard & -white_bitboard
            #  This does assume two's complement representation for integers (which should be a safe bet).
            # To form a negative number under two's complement, take the positive number, invert every bit, and then add 1.
            # The inversion changes every bit (obviously). The adding 1 will cause a 1 to "ripple up" to the first bit that is a 0 (which was a 1 in the original number).
            # every bit after the LSB will be inverted, so AND with the original version will get rid of them
            square = least_significant_bit.bit_length() - 1
            midgame_score += base_value
            midgame_score += mg_table[white_coords[square]]
            endgame_score += base_value
            endgame_score += eg_table[white_coords[square]]

            white_bitboard ^= least_significant_bit # get rid of least significant bit.
            # since the least significant bit and the original value will all have the LSB index as 1, it will be turned to 0
            # other bit indexes will be untouched

            # black king safety!
            BKR_attack_intersection = chessboard.attacks_mask(square) & black_king_ring
            intersection_magnitude = BKR_attack_intersection.bit_count()
            black_king_ring_attacked += intersection_magnitude
            if intersection_magnitude > 0:
                black_king_attackers += 1

        black_bitboard = chessboard.pieces_mask(piece_type, chess.BLACK)
        while black_bitboard:
            least_significant_bit = black_bitboard & -black_bitboard
            square = least_significant_bit.bit_length() - 1
            midgame_score -= base_value
            midgame_score -= mg_table[black_coords[square]]
            endgame_score -= base_value
            endgame_score -= eg_table[black_coords[square]]

            black_bitboard ^= least_significant_bit

            WKR_attack_intersection = chessboard.attacks_mask(square) & white_king_ring
            intersection_magnitude = WKR_attack_intersection.bit_count()
            white_king_ring_attacked += intersection_magnitude
            if intersection_magnitude > 0:
                white_king_attackers += 1

    # white king safety penalty
    white_king_safety_penalty = KING_SAFETY_PENALTY * white_king_ring_attacked * ATTACK_MULTIPLIERS[white_king_attackers]
    #print("white king safety penalty: ", white_king_safety_penalty)
    midgame_score -= white_king_safety_penalty

    #black king safety penalty
    black_king_safety_penalty = KING_SAFETY_PENALTY * black_king_ring_attacked * ATTACK_MULTIPLIERS[black_king_attackers]
    #print("black king safety penalty: ", black_king_safety_penalty)
    midgame_score += black_king_safety_penalty

    evaluation = phase*endgame_score + (1-phase)*midgame_score
    return evaluation

BB_FILE_A = chess.BB_FILE_A
BB_FILE_H = chess.BB_FILE_H

def pawn_attacks_mask(pawns_bb, color):
    if color == chess.WHITE:
        left = (pawns_bb & ~BB_FILE_A) << 7 # the square a pawn is attacking are only +7/+9 or -7/-9
        right = (pawns_bb & ~BB_FILE_H) << 9
    else:
        left = (pawns_bb & ~BB_FILE_A) >> 9
        right = (pawns_bb & ~BB_FILE_H) >> 7
    return left | right

def order_moves(moves, chessboard: chess.Board, current_depth : int, ply, use_killers = True):
    INDEXED_PIECE_VALUES = [0, 100, 300, 325, 500, 900, 0]
    scored = []
    CHECK_BONUS = 75
    PAWN_ATTACK_PENALTY_FACTOR = .75
    KILLER1_BONUS = 160
    KILLER2_BONUS = 90

    enemy = not chessboard.turn
    enemy_pawns_bb = chessboard.pieces_mask(chess.PAWN, enemy)
    enemy_pawn_attacks = pawn_attacks_mask(enemy_pawns_bb, enemy)
    BB_SQUARES = chess.BB_SQUARES

    for move in moves:
        score = 0

        attacker = chessboard.piece_at(move.from_square)
        attacker_value = INDEXED_PIECE_VALUES[attacker.piece_type]
        move_to = move.to_square
        is_capture = chess.BB_SQUARES[move_to] & chessboard.occupied_co[not chessboard.turn]
        is_promotion = (move.promotion is not None)

        # MVV LVA
        if is_capture:
            if chessboard.is_en_passant(move):
                victim_piece_type = "P"
            else:
                victim = chessboard.piece_at(move_to)
                victim_piece_type = victim.symbol().upper()

            victim_value = PIECE_VALUES[victim_piece_type]
            score += (victim_value - attacker_value)

        # promotion bonus
        if is_promotion:
            score += PROMOTION_BONUS.get(move.promotion, 0)

        # check bonus
        """
        if current_depth >= 2: # gives_check is computationally expensive so avoid calling it at shallow depths.
            if chessboard.gives_check(move):
                score += CHECK_BONUS
        """


        # penalty for moving into pawn attack
        if not is_capture and (enemy_pawn_attacks & BB_SQUARES[move_to]):
            score -= attacker_value * PAWN_ATTACK_PENALTY_FACTOR

        # killer move bonus
        if use_killers and (not is_capture) and (not is_promotion):
            killer1 = killer1_list[ply]
            killer2 = killer2_list[ply]
            if move == killer1:
                score += KILLER1_BONUS
            elif move == killer2:
                score += KILLER2_BONUS

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
    global non_kp_pieces
    global endgameness

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

    unordered_capture_moves = cython_chess.generate_legal_captures(chessboard, chess.BB_ALL, chess.BB_ALL)
    capture_moves = order_moves(unordered_capture_moves, chessboard, 0, 0, False)
    for capture in capture_moves:
        captured = captured_piece(chessboard, capture)
        captured_value = PIECE_VALUES[captured.symbol().upper()]

        old_non_kp = non_kp_pieces
        old_endg = endgameness

        efficient_endgameness_udpate(chessboard, capture)

        chessboard.push(capture)
        if MAXIMIZING_PLAYER:
            if static_evaluation + captured_value + 300 < alpha:
                # dont bother capturing if the capture doesnt even change the best move you can force
                chessboard.pop()
                endgameness = old_endg
                non_kp_pieces = old_non_kp
                continue
        else:
            if static_evaluation - captured_value - 300 > beta:
                chessboard.pop()
                endgameness = old_endg
                non_kp_pieces = old_non_kp
                continue

        evaluation, child_line = qui_search(chessboard, alpha, beta)
        chessboard.pop()

        endgameness = old_endg
        non_kp_pieces = old_non_kp
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
MAX_POSSIBLE_SEARCH_DEPTH = 48
killer1_list = [None] * MAX_POSSIBLE_SEARCH_DEPTH
killer2_list = [None] * MAX_POSSIBLE_SEARCH_DEPTH
def search(
    chessboard: chess.Board,
    transposition_table,
    key: int,
    top_depth=6,
    depth=6,
    alpha=-MATE_SCORE,
    beta=MATE_SCORE,
    search_first = None,
    check_streak = 0
):
    global non_kp_pieces
    global endgameness


    alpha0 = alpha
    beta0 = beta
    ply = top_depth - depth
    #if depth+1 == top_depth:
    #    print("top depth node progress, ", time.time() - start_time)

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
            mate = MATE_SCORE - ply
            if chessboard.turn == chess.WHITE:
                mate = -mate
            return mate, []
        return 0, []

    # enforce order of the search first node
    preferred_move = search_first
    if preferred_move is None and tt_entry is not None:
        line = tt_entry["best"]
        if line:
            preferred_move = line[0]
    if preferred_move is not None:
        try:
            idx = moves.index(preferred_move)
            if idx != 0:
                moves[0], moves[idx] = moves[idx], moves[0]
            moves = [moves[0]] + order_moves(moves[1:], chessboard, depth, ply)
        except ValueError:
            moves = order_moves(moves, chessboard, depth, ply)
    else:
        moves = order_moves(moves, chessboard, depth, ply)

    best_eval = -MATE_SCORE if maximizing else MATE_SCORE
    best_line = []

    for move_index, move in enumerate(moves):

        old_non_kp = non_kp_pieces
        old_endg = endgameness
        is_capture = chessboard.is_capture(move)
        is_promotion = (move.promotion is not None)
        is_killer = (move == killer1_list[ply]) or (move == killer2_list[ply])
        gives_check = False
        if depth >= 2 and (not is_capture) and (not is_promotion):
            gives_check = chessboard.gives_check(move)

        if gives_check and check_streak < 2: # look further into checks.
            extension = 1
            check_streak += 1
        else:
            extension = 0
            check_streak = 0


        # LMR is only for STATISTICALLY PROBABLY BAD MOVES, so those metrics help determine that status.
        lmr_eligible = (depth >= LMR_MIN_DEPTH and move_index >= LMR_FULL_MOVES and (not is_capture) and (not is_promotion) and (not is_killer) and (not gives_check))


        efficient_endgameness_udpate(chessboard, move)
        # push + update key together
        new_key = zobrist.update_zobrist_key(chessboard, key, move)  # this PUSHES a move on the chessboard !!!

        if lmr_eligible:
            r = late_move_reductions(depth, move_index)
            reduced_depth = depth - 1 - r
            if reduced_depth < 0: reduced_depth = 0
            evalu, child_line = search(chessboard, transposition_table, new_key, top_depth= top_depth, depth = reduced_depth, alpha = alpha, beta = beta, search_first = None)
            # reduced depth search.
            if maximizing: needs_full_search = (evalu > alpha)
            else: needs_full_search = (evalu < beta)

            if needs_full_search and r > 0:
                evalu, child_line = search(chessboard, transposition_table, new_key, top_depth = top_depth, depth = depth - 1, alpha = alpha, beta = beta, search_first = None)
        else: # normal search
            evalu, child_line = search(chessboard, transposition_table, new_key, top_depth=top_depth, depth=depth - 1, alpha=alpha, beta=beta, search_first= None, check_streak= check_streak)


        chessboard.pop()

        non_kp_pieces =  old_non_kp
        endgameness = old_endg

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
            # record the aforementioned killer move

            if (not chessboard.is_capture(move) and (move.promotion is None)): # important to filter out captures and promotions since those are already favorable
                # additionally, captures are very position-dependent
                if killer1_list[ply] != move:
                    killer2_list[ply] = killer1_list[ply]
                    killer1_list[ply] = move

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

LMR_MIN_DEPTH = 3
LMR_FULL_MOVES = 3
LMR_MAX_REDUCTION = 3
def late_move_reductions(depth, move_index): # most quiet moves at the end of move ordering are quite bad, so dont search them deeply initally.
    if depth < LMR_MIN_DEPTH:
        return 0
    if move_index < LMR_FULL_MOVES:
        return 0

    r = 1
    if depth >= 6: # this is just a logarithmic curve.
        r += 1
    if move_index >= 8:
        r += 1
    if move_index >= 16:
        r += 1
    return min(r, LMR_MAX_REDUCTION)
def iterative_deepening(target_depth):
    transposition_table = TranspositionTable()
    init_Zkey = zobrist.zobrist_key(board)
    best_move = None
    for i in range (1, target_depth+1):
        evaluation, best_line = (search(board, transposition_table, init_Zkey, top_depth = i, depth =  i, search_first = best_move))
        best_move = best_line[0]
        print(f"== DEPTH {i} time elapsed:", round((time.time() - start_time), 2), f"s - Eval: {round((evaluation/100),2)}, best move {best_move} ==")
        if i == target_depth:
            best_line_str = ""
            for move in best_line:
                best_line_str += str(move)
                best_line_str += " "
            print(best_line_str)
        #print("total TT hits:", transposition_table.total_hits)




iterative_deepening(8)

"""
def run():
    iterative_deepening(5)
profiler = Profile()
profiler.runcall(run)
stats = Stats(profiler)
stats.strip_dirs()
stats.sort_stats("cumulative")
stats.print_stats()
stats.dump_stats(filename= "debug.prof")
"""



# fen: r1b1kb1r/1pp2ppp/p2q1n2/3p4/5P2/2NBP3/PPPB2PP/R2QK2R w KQkq - 1 10
# 8.88s