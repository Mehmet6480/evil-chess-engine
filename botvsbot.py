import chess
import chess.pgn
import time
import main
from repetition import RepetitionTable
import zobrist
FEN_LIST = [
    "r4rk1/1pp2pb1/2npq1pp/4p2n/p3P3/P1PPBN1P/1P1N1PP1/R3QRK1 w - - 3 23",
    "1r2k2r/2p4p/p1b1p1pP/Pp1p4/3Pp3/2P1P3/2P2PP1/2KR1B1R w k - 1 19",
    "r2r2k1/2pb1pp1/2n1pq1p/pB1p3P/3Pn1PR/P1B1PN2/1PP1QP2/2KR4 w - - 1 20",
    "r1b2rk1/pp4pp/2p5/1P1pqp2/8/2nBP3/P2N1PPP/2RQR1K1 w - - 0 16",
    "r1q3k1/1b2bppp/pp2pn2/2n5/1PB2B2/P1N1PN2/4QPPP/2Rr2K1 w - - 0 17",
    "r2q1rk1/pp3pp1/2p2b1p/4N3/3PQ3/2P3P1/PP3PBP/R5K1 w - - 1 18",
    "r3kb1r/1q3p2/2bp1np1/p1p1p2p/Pp2P1P1/8/1PBPQP1P/RNB1K1NR w KQk - 1 22",
    "r4rk1/pbqn1pp1/1p1bp2p/4N3/2PB1P2/1P1B3P/P1Q3P1/R4R1K w - - 1 20",
    "1k1r3r/p1p2ppp/1p1b4/1N2n3/4P3/P7/1P3PPP/R1BR2K1 w - - 3 19",
    "3rr1k1/2pq1pp1/5n1p/1pQp4/1P6/n1N1P2P/5PP1/R3R1K1 w - - 0 20",
    "1br2rk1/1b3ppp/p1nq1n2/8/B2p4/2B2NP1/PP1N1P1P/R2QR1K1 w - - 0 18",
    "7r/p1k1bppp/1N6/8/3p4/3K4/PPP4P/R1B1q3 w - - 1 25",
    "3r1rk1/ppb2pp1/2p1p2p/2P3n1/1PBR4/P3PP2/1B3P1P/5RK1 w - - 1 20",
    "r2qr3/1p2npk1/pb2p1pp/3pPbN1/1P3B2/P1P3P1/4BP1P/R2Q1RK1 w - - 0 18",
    "r3r1k1/2p1b2p/2npq1p1/5p2/1pP1PpPP/1N1P1P2/P4Q2/1K1R2NR w - - 0 24",
    "r2q1rk1/1p2p1bp/p5p1/3b1p2/P2Nn3/4P3/1P1BBPPP/R2Q1RK1 w - - 1 16",
    "r1bq1rk1/p6p/1pp1p1p1/2N2p2/2PPp3/4PP2/PP1Q2PP/R4RK1 w - - 0 16",
    "2r4k/1bq1b1p1/pp1p1n1p/8/P3P3/2NQ2R1/1PP2PPP/4R1K1 w - - 4 22",
    "r2q1rk1/p3bnpp/1p3p2/8/8/8/PPQ2PPP/RNB1R1K1 w - - 0 17",
    "r5r1/p1b1kp2/2p1p3/4N3/PP6/2P3p1/5P2/R3R1K1 w - - 0 25",
    "r2qkb1r/2pb1n1p/1p3pp1/p2Pp3/2Q5/N7/PPPP1PPP/R1B1K1NR w KQkq - 1 21",
    "3r4/4k1p1/1pp2p1p/4pP2/r1P3P1/P7/1P3K1P/3R1R2 w - - 2 23",
    "3q1rk1/2p2pbp/1r4p1/p3p3/P7/2Qn3P/3NNPP1/R2R2K1 w - - 6 25",
    "1k1r3r/ppq3p1/2P1Q1p1/6bn/2pP4/2P3P1/PP1N1PKP/R4R2 w - - 0 19",
    "3r3r/1k1q1pb1/2p1p2p/pp1pP1p1/3P4/2P2PB1/PP1Q2PP/2KR3R w - - 1 19",
    "r1b2rk1/ppp1q1pp/3p4/3Pn3/4p3/3BPN2/PP3PPP/R2Q1RK1 w - - 0 13",
    "r1q2rk1/1p1bbppp/1P2pn2/p1Ppn3/P4N2/1NP1BPP1/2Q1P2P/R3KB1R w KQ - 1 18",
    "r2q1rk1/1p3ppp/4pn2/3pb3/1p1B4/1P2P3/P1P2PPP/R2QR1K1 w - - 2 17",
    "r3k2r/6pp/p3p3/3pB1q1/PbpPp3/4P2P/1P3PP1/R2Q1RK1 w kq - 0 19",
    "r1q1kb2/pp3r2/4p2p/2ppPppQ/b2Nn3/1NPBP3/PP3PPP/R3K2R w KQq - 2 16",
    "5rk1/pb2ppbp/1p3np1/4N3/3PP3/1q4P1/PB3PBP/4R1K1 w - - 0 19",
    "2r2rk1/1p2nppp/1q2p3/p7/3PR3/P2Q1N2/1P3PPP/R5K1 w - - 0 18",
    "5rk1/p3qpbp/2p5/1p4P1/4P2Q/1BPr4/PP1N1P2/2KR4 w - - 0 25",
    "r1b1q1rk/ppbn3p/2p1pp2/2Pp1p2/3P4/3BPNN1/PPQ2PPP/R4RK1 w - - 4 14",
    "r2qk2r/1p2npp1/3b4/pP1p1b2/P2Pp1n1/4P1P1/2NN2BP/R1BQ1RK1 w kq - 1 17",
    "r4rk1/2q2ppp/2p3b1/pp1pP3/1P6/P5P1/1B2PP1P/2RQ1RK1 w - - 0 19",
    "r3k2r/ppp1qppb/3bpn1p/4N3/3P2P1/2N4P/PPP1Q3/2KR1B1R w kq - 3 13",
    "2r2rk1/pp1b1ppp/1q1b1n2/4p3/3p4/PPN1PP1P/2PBNQP1/R4RK1 w - - 0 17",
    "r1b1r1k1/pp1n1pp1/4p2p/3pq3/2P5/B1PB4/P3QPPP/2R2RK1 w - - 0 16",
    "r1bq1rk1/pp2p1bp/2pp1np1/2Pn1pN1/3P1B2/1QNBP3/PP3PPP/R3K2R w KQ - 1 11",
    "r2q1rk1/p4ppp/bp2p1n1/3pP3/3P4/2P3P1/PPQB3P/2KRN2R w - - 0 18",
    "rn1qbrk1/pp2b1pp/2p1p3/3p1p2/2PPn3/1PNBP3/PB1N1PPP/R2Q1RK1 w - - 5 11",
    "1r1r2k1/5pp1/2pq1n1p/p1Np1b2/3Q4/PP4P1/4PPBP/2R2RK1 w - - 0 20",
    "B1b2rk1/p2nppbp/6p1/2q1P3/2pp4/1P5P/P4PPB/R2Q1RK1 w - - 0 19",
    "2kr2r1/ppp1bp2/4p1p1/4P3/2P1p1Qp/1P2P3/PB1q1PPP/R4RK1 w - - 1 19",
    "3k3r/ppp3b1/3p1pp1/4p2p/2PnP1P1/7P/PP3P2/R2K1B1R w - - 0 21",
    "r1bq1rk1/1pp3bp/3p1np1/p1nPppB1/2P1P3/2NB1P2/PP1QN1PP/R3K2R w KQ - 4 11",
    "r5kr/pQ4pp/3qpn2/3p1b2/N2P4/8/PP2PPPP/3K1B1R w - - 0 18",
    "r2q1rk1/pbpn3p/1p1b2p1/3p4/3Pn3/P1NB4/1PQ1N1PP/R1B2RK1 w - - 0 14",
    "r1bqr1k1/pppnbpp1/3p1n1p/3P4/4Pp2/2PBB1N1/PP1N2PP/R2Q1RK1 w - - 0 13",
    "rn1q2k1/pbppp1b1/1p3rpp/8/2PPp3/2NB1N2/PP3PPP/R2QK2R w KQ - 0 11",
    "r1bq1rk1/pp6/n1pp1bpp/4pp2/2PP4/P1N1P1P1/1PQ1NPBP/R3K2R w KQ - 0 12",
    "r4rk1/p1p3pp/2p1ppq1/5b2/3P4/2N5/PPP2PPP/R2QR1K1 w - - 3 16",
    "rn1q1rk1/pbp4p/1p1pp3/5pp1/2PP4/B1PBP3/P1Qn1PPP/1R3RK1 w - - 0 13",
    "rn4k1/pp4p1/2p3pp/3p1b2/3P1q1P/2P2N2/P1QR1PP1/2K4R w - - 2 17",
    "r2r2k1/pbq2pp1/1ppbp2p/3nN3/3P4/PP2PB2/1BQ2PPP/2R2RK1 w - - 5 17",
    "r3k2r/pp1q1ppb/2Nbpn1p/1Q6/3P2P1/2N4P/PPP5/2KR1B1R w kq - 1 15",
    "2rq1rk1/1b3pbp/p3p1p1/1p1n4/3B4/2NBPP2/PP2Q1PP/2R2RK1 w - - 4 18",
    "2r3k1/6p1/pp2p2p/8/PP2pP2/1R2P1qP/6P1/3Q2K1 w - - 0 27",
    "r2q1rk1/4bpp1/p3p3/1p1pB1P1/2pPn3/P1N1P2P/1PPQ4/R3K2R w KQ - 1 17",
    "2rqr1k1/3nbpp1/p1p1pn1p/1p6/3Pp3/2P2BBP/PPQN1PP1/2KRR3 w - - 0 17",
    "2r1k2r/p4p1p/4pp2/8/8/1R6/P1P2PPP/5RK1 w - - 0 22",
    "r1bq1rk1/p1p2ppp/2p1p1n1/3pB3/3P1QP1/7P/PPP1PP2/R3KB1R w KQ - 3 16",
    "r1b2rk1/5ppp/3np3/1pqpN1P1/p4P2/3B1P2/PP2Q2P/1K1R3R w - - 0 20",
    "r4rk1/pb3ppp/1p2pn2/3pN3/3PnP2/qP1BP3/P2N2PP/R2Q1RK1 w - - 0 16",
    "r3r1k1/1b3pp1/p1pb1q1p/3p4/1P1P4/P1N2N1P/5PP1/R2Q1RK1 w - - 1 17",
    "r1bq2k1/pp4pp/2p1pr2/3p1p2/4n3/2PBPN1P/PPQ2PP1/3R1RK1 w - - 4 16",
    "r4rk1/p2q1ppp/pbp1p3/4Rn2/1P1P4/2Q2N2/PBP2PPP/R5K1 w - - 3 18",
    "r3r1k1/3q1pbp/p4Bp1/3b4/2Pp4/5Q2/PP3PPP/2RR2K1 w - - 0 23",
    "r2q1rk1/ppp2pp1/2n2pbp/8/1b1PP3/2NB4/PPP1N1PP/R2Q1RK1 w - - 7 11",
    "r3kb1r/2p2pp1/p3p2p/2p1P3/3B2n1/2N2P2/PPP3PP/3RK2R w Kkq - 0 14",
    "r4rk1/1ppq2pp/2nb1p2/5b2/p2P4/P3BNP1/1P3PBP/R2QR1K1 w - - 2 17",
    "3rr1k1/2q1b1p1/p1pp1pnp/4pb2/2P1P3/BPN2PP1/P2Q2KP/3R1R2 w - - 0 22",
    "4r3/pp3Rpk/2bp3p/8/3b1B2/3P4/PPP3PP/R5K1 w - - 4 23",
    "rn1q1rk1/p1p1bpp1/1p2p2p/6P1/3Pp2B/1QP1PN1P/PP3P2/R3K2R w KQ - 0 16",
    "r4rk1/pp3pp1/2p2n1p/P3qb2/2p5/2N3P1/1PQ1PPBP/R2R2K1 w - - 1 16",
    "r1b2rk1/pp2b1pp/2n1pn2/q2p1p2/3P4/2N1PNP1/PP3PBP/R1BQR1K1 w - - 4 11",
    "1r1qk2r/Q1pbbppp/4p3/8/3P4/2P2N2/P5PP/R1B2RK1 w k - 2 14",
    "5rk1/Q6p/2pp1qpb/4p3/8/3P3N/P2r1PPP/R4RK1 w - - 2 22",
    "r5r1/pp5p/1b2k3/6p1/7P/2PPpPP1/PP2K3/R6R w - - 1 23",
    "r1b1qrk1/ppp3bp/3p1np1/2nPpp2/2P5/P1N1PN2/1P1BBPPP/R2QK2R w KQ e6 0 11",
    "2rqr1k1/1b2ppbp/p4np1/3p4/1PnP4/2NBPNB1/R2Q1PPP/4K2R w K - 3 17",
    "r1b1k2r/4ppbp/pq1p1np1/2pPn3/4P3/1BN5/PP2NPPP/R1BQK2R w KQkq - 2 11",
    "r1b2rk1/ppp1npb1/n2q3p/3Pp1p1/2P5/1Q3P2/PP1BN1PP/RN2KB1R w KQ - 3 11",
    "r4rk1/pp1b2pp/2p1pq2/3p1p2/2PP4/2PBPP2/P1Q3PP/R4RK1 w - - 1 15",
    "r2qk2r/ppp2p2/2nb1npp/3p2Q1/3PP3/2P2N1P/PP1N1P2/R1B2RK1 w kq - 0 16",
    "r4rk1/5ppp/3qpn2/pp1p4/3p2B1/2P1P3/PP3PPP/R2Q1RK1 w - - 0 18",
    "3q1r2/1p2ppbk/p2pb1pp/8/P6N/5Q1P/1PP1B3/2B2RK1 w - - 1 23",
    "3r1rk1/p4ppp/1p2p3/2q1QP2/3p4/2P3P1/PP4KP/4RR2 w - - 0 20",
    "2r2rk1/pp1nnpbp/1q2p1p1/3pP3/3p2P1/1NP4P/PP3PB1/R1BQR1K1 w - - 0 16",
    "rnb2rk1/2q1ppbp/p2p1np1/P1pP4/4P3/2N2N2/1P3PPP/R1BQKB1R w KQ - 2 11",
    "r1b1qr1k/ppp1p1bp/3p1np1/5p2/1nPP1B2/2NBPN2/PPQ2PPP/3R1RK1 w - - 6 11",
    "rnbq1r2/2p1npbk/p2pp1pp/1p6/2BPP3/P1N1BN2/1PPQ1PPP/3R1RK1 w - - 0 11",
    "2kr1b1r/pp1b1pp1/1qn1p2p/3pPn2/3p1P2/2P2N2/PPQ1BBPP/RN3RK1 w - - 0 12",
    "r1bq1r2/ppp3bk/3p1np1/3Pp2p/1PP1Pp2/2NQ1P2/P3N1PP/1RB2RK1 w - - 1 14",
    "r3r1k1/ppqb1pp1/1n2p2p/8/8/BBP5/PQ3PPP/2R1R1K1 w - - 5 20",
    "r4rk1/pp3p1p/2b1p1p1/2q5/3bP3/5QP1/PP2NP1P/R2R2K1 w - - 0 21",
    "r5k1/ppp1nr1p/6p1/3p4/3P4/2NP4/PP4PP/R4RK1 w - - 0 21",
    "r2qk2r/1pp2pp1/p1p2n1p/8/3PP1b1/P4N2/1P1B1PPP/R2QK2R w KQkq - 1 11",
    "r1b3k1/ppb3pp/2p1p2r/2PpPp2/P3n2q/1P1BP2P/1B1N1PP1/R2Q1RK1 w - - 1 16",
]
RESULT_LABEL = {"1-0": "white", "0-1": "black", "1/2-1/2": "draw"}

def self_play_pgn(thinking_time: float, start_fen: str = chess.STARTING_FEN, max_plies: int = 1000):
    board = chess.Board(start_fen)
    game = chess.pgn.Game()

    if start_fen != chess.STARTING_FEN:
        game.setup(board)
        game.headers["SetUp"] = "1"
        game.headers["FEN"] = start_fen

    node = game
    ply = 0

    rep = RepetitionTable()
    key = zobrist.zobrist_key(board)
    rep.increment(key)

    while not board.is_game_over(claim_draw=False) and ply < max_plies:
        print("PLY: ", ply+1)
        move = main.iterative_deepening(thinking_time, board.fen(), rep, key)

        if move not in board.legal_moves:
            raise ValueError(f"Illegal move returned: {move.uci()} in FEN {board.fen()}")

        key = zobrist.update_zobrist_key(board, key, move)
        rep.increment(key)

        node = node.add_main_variation(move)
        ply += 1

    if not board.is_game_over(claim_draw=False):
        game.headers["Result"] = "1/2-1/2"
        return "1/2-1/2", board.fen(), game

    game.headers["Result"] = board.result(claim_draw=False)
    return game.headers["Result"], board.fen(), game

def run_batch_selfplay(fens, thinking_time: float, max_plies: int = 1000, print_pgn: bool = False):
    white_wins = 0
    black_wins = 0
    draws = 0

    for i, fen in enumerate(fens, start=1):
        print(f"\n=== GAME {i}/{len(fens)} ===")
        t0 = time.time()

        result, final_fen, game = self_play_pgn(thinking_time, fen, max_plies=max_plies)
        who = RESULT_LABEL.get(result, "draw")

        if who == "white":
            white_wins += 1
            print("white won")
        elif who == "black":
            black_wins += 1
            print("black won")
        else:
            draws += 1
            print("draw")

        if print_pgn:
            print(game)

        elapsed = time.time() - t0
        print("final result:", result)
        print("final fen:", final_fen)
        print(f"running total -> white: {white_wins} | draw: {draws} | black: {black_wins}")
        print("game time:", round(elapsed, 2), "s")

    print("\n=== FINAL TOTALS ===")
    print(f"white: {white_wins} | draw: {draws} | black: {black_wins}")

if __name__ == "__main__":
    run_batch_selfplay(FEN_LIST, thinking_time=3.0, max_plies=1000, print_pgn=True)

# first 30 games done
