// `bench`: fixed-depth searches over a fixed position set; the node count is the build's signature.

use crate::chess::Board;
use crate::search::{Engine, Limits};
use std::time::Instant;

pub const BENCH_DEPTH: u32 = 13;

const FENS: [&str; 40] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2QKB1R w KQ - 0 8",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
    "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
    "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
    "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
    "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
    "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
    "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
    "r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
    "4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
    "3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
    "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
    "8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1",
    "r1b1k1nr/1p1pqppp/2n5/p1p1p3/P7/1N1P3P/1PP1PPPR/R1BQKBN1 w Qkq - 1 9",
    "8/1p2Qp2/q1k1b3/3pB3/8/P3P3/3K1P2/8 b - - 6 41",
    "8/8/5k2/8/3r1PK1/8/5N2/5B2 b - - 36 101",
    "4rr1k/4q2p/3p4/pbp3P1/3pP2P/8/PP1QB3/3K1R1R b - - 0 24",
    "6k1/3pb3/2p1p1q1/p1P1Pn1p/1r1PNPp1/1P1Q2P1/5B1K/4R3 b - - 8 39",
    "6R1/1p2r3/1R6/3rk1p1/P1Kp4/P4PP1/8/8 b - - 0 41",
    "r2rqb1k/ppn2bp1/7p/2pPp3/4P1Q1/PP2N1PB/1B5P/4RRK1 b - - 7 32",
    "r1bq1rk1/pp4pp/2n1p3/2bn1p2/5B2/3BPN1P/PP2NPP1/R2Q1RK1 w - - 0 15",
    "2kr1b1r/2pbn3/pp3p2/8/4PN2/BPN4p/P1P2KRP/4R3 w - - 0 23",
    "1N1k1b2/2p4p/4p3/4Pp1n/5P2/5K1P/5B2/8 w - - 29 53",
    "1nb2rk1/r4p2/2p3pp/p7/Pp2RN1P/6P1/1PPR2B1/2K5 w - - 0 27",
    "5rk1/1p2np2/pqb1pBp1/5n1r/1PP4P/3P1N2/4QPB1/1RR3K1 w - - 1 26",
    "4r2r/pB1b1pk1/n5p1/8/1P1N1R2/P3n1Pp/3NP2P/R5K1 b - - 0 22",
    "r3kb2/2p1qp2/1p3n1r/p3p2P/2P1Pnp1/1P2N1Nb/PBQ1BP2/3RK2R b Kq - 3 20",
    "8/5k2/8/7R/2K5/5r1P/8/8 b - - 30 154",
    "8/8/1p4p1/1P1pR3/2k3r1/4PK2/8/8 b - - 3 55",
    "rnbq1rk1/pppp1pp1/3bpn2/7p/1P2P3/P4P2/2PP1BPP/RN1QKBNR b KQ - 2 7",
    "r3kb1r/pp1b4/1qn1p2n/2pp1p2/8/P2P1NP1/1PP1P1BP/RNBQK2R w KQkq - 1 11",
    "2kr4/2pp3N/2b2pR1/p6n/3bP3/P1N2P2/1PPB3r/2KR4 w - - 0 22",
    "5k2/8/2q3p1/6P1/5P2/6K1/1Q6/8 w - - 9 78",
    "8/8/5p2/5k2/2PR3P/7r/1K6/8 b - - 0 51",
    "8/8/2B5/3Pk1n1/2K5/2P5/8/8 b - - 1 75",
    "8/1p2k2p/p7/P4Pp1/1P1BKb2/7P/7r/3R4 b - - 10 43",
    "1q6/2p1pk2/4bn2/2P2p1p/1r3P2/3N2PB/2P5/Q2R2K1 b - - 0 29",
];

/// Each position starts from a cleared engine, so the total is the same on every run and machine.
pub fn run(depth: u32) {
    let mut engine = Engine::new();
    let mut warm = Board::default();
    let _ = engine.search(&mut warm, Limits::depth(1));
    let (mut nodes, mut secs) = (0u64, 0f64);
    for fen in FENS {
        engine.reset();
        let mut board = Board::from_fen(fen).expect("bench fen");
        let t = Instant::now();
        let (_, _, n, _) = engine.search(&mut board, Limits::depth(depth));
        secs += t.elapsed().as_secs_f64();
        nodes += n;
    }
    println!("{nodes} nodes {} nps", (nodes as f64 / secs.max(1e-9)) as u64);
}
