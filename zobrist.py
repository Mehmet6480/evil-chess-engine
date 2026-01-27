import cython_chess
import chess
import random

random.seed(1)
Z_PIECE_INDEX = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


PIECE_KEYS = [[random.getrandbits(64) for _ in range(64)] for _ in range(12)]
CASTLING_KEYS = [random.getrandbits(64) for _ in range(4)]   # K Q k q
EN_PASSANT_KEYS = [random.getrandbits(64) for _ in range(8)] # for every file
TURN_KEY = random.getrandbits(64)


def zobrist_key(board: chess.Board):
    key = 0

    # for pieces
    for sq, piece in board.piece_map().items():
        idx = Z_PIECE_INDEX[(piece.piece_type, piece.color)]
        key ^= PIECE_KEYS[idx][sq]

    # castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        key ^= CASTLING_KEYS[0]  # K
    if board.has_queenside_castling_rights(chess.WHITE):
        key ^= CASTLING_KEYS[1]  # Q
    if board.has_kingside_castling_rights(chess.BLACK):
        key ^= CASTLING_KEYS[2]  # k
    if board.has_queenside_castling_rights(chess.BLACK):
        key ^= CASTLING_KEYS[3]  # q

    # en passant
    ep = board.ep_square
    if ep is not None:
        if any(board.generate_legal_ep()):
            key ^= EN_PASSANT_KEYS[chess.square_file(ep)]

    # turn
    if board.turn == chess.BLACK:
        key ^= TURN_KEY

    return key


def xor_castling(k: int, b: chess.Board) -> int:
    if b.has_kingside_castling_rights(chess.WHITE):
        k ^= CASTLING_KEYS[0]
    if b.has_queenside_castling_rights(chess.WHITE):
        k ^= CASTLING_KEYS[1]
    if b.has_kingside_castling_rights(chess.BLACK):
        k ^= CASTLING_KEYS[2]
    if b.has_queenside_castling_rights(chess.BLACK):
        k ^= CASTLING_KEYS[3]
    return k
def xor_ep(k: int, b: chess.Board) -> int:
    ep = b.ep_square
    if ep is not None and any(b.generate_legal_ep()):
        k ^= EN_PASSANT_KEYS[chess.square_file(ep)]
    return k

def update_zobrist_key(board: chess.Board, key: int, move: chess.Move) -> int:

    # remove old EP and old castling rights from the key
    key = xor_ep(key, board)
    key = xor_castling(key, board)

    # moving piece (must exist)
    mover = board.piece_at(move.from_square)
    mover_idx = Z_PIECE_INDEX[(mover.piece_type, mover.color)]

    #remove mover from origin square...
    key ^= PIECE_KEYS[mover_idx][move.from_square]

    #remove captured piece
    if board.is_en_passant(move):
        cap_sq = chess.square(chess.square_file(move.to_square),
                              chess.square_rank(move.from_square))
        victim = board.piece_at(cap_sq)  # should be a pawn
        victim_idx = Z_PIECE_INDEX[(victim.piece_type, victim.color)]
        key ^= PIECE_KEYS[victim_idx][cap_sq]
    elif board.is_capture(move):
        victim = board.piece_at(move.to_square)
        victim_idx = Z_PIECE_INDEX[(victim.piece_type, victim.color)]
        key ^= PIECE_KEYS[victim_idx][move.to_square]

    #add piece on destination (promotion or normal move)
    if move.promotion is not None:
        promoted_idx = Z_PIECE_INDEX[(move.promotion, mover.color)]
        key ^= PIECE_KEYS[promoted_idx][move.to_square]
    else:
        key ^= PIECE_KEYS[mover_idx][move.to_square]

    #if castling, update rook squares too
    if board.is_castling(move):
        if move.to_square in (chess.G1, chess.G8):  # kingside
            rook_from = chess.H1 if mover.color == chess.WHITE else chess.H8
            rook_to   = chess.F1 if mover.color == chess.WHITE else chess.F8
        else:  # queenside
            rook_from = chess.A1 if mover.color == chess.WHITE else chess.A8
            rook_to   = chess.D1 if mover.color == chess.WHITE else chess.D8

        rook = board.piece_at(rook_from)
        rook_idx = Z_PIECE_INDEX[(rook.piece_type, rook.color)]
        key ^= PIECE_KEYS[rook_idx][rook_from]
        key ^= PIECE_KEYS[rook_idx][rook_to]


    key ^= TURN_KEY

    board.push(move)

    key = xor_castling(key, board)
    key = xor_ep(key, board)

    return key