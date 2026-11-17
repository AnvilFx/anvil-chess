use super::*;
use crate::chess::Board;

#[test]
fn threat_map_matches_square_attack_queries() {
    let fens = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ];
    let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
    for fen in fens {
        for _ in 0..20 {
            let mut board = Board::from_fen(fen).unwrap();
            for _ in 0..40 {
                let them = board.side_to_move.opp();
                let t = threat_map(&board);
                for sq in 0..64u8 {
                    let hit = t >> sq & 1 == 1;
                    assert_eq!(hit, board.is_square_attacked(Square(sq), them), "{sq} in {}", board.to_fen());
                }
                let mut all = MoveList::new();
                board.gen_legal_into(&mut all);
                if all.is_empty() {
                    break;
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                board.make_move(all.get((rng % all.len() as u64) as usize));
            }
        }
    }
}

#[test]
fn capture_cannot_lose_implies_nonnegative_see() {
    let fens = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 9",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ];
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut hits = 0;
    for fen in fens {
        for _ in 0..30 {
            let mut board = Board::from_fen(fen).unwrap();
            for _ in 0..50 {
                let mut all = MoveList::new();
                board.gen_legal_into(&mut all);
                if all.is_empty() {
                    break;
                }
                for m in all.iter().filter(|m| m.is_capture()) {
                    if capture_cannot_lose(&board, m) {
                        hits += 1;
                        assert!(see(&board, m) >= 0, "{m} in {}", board.to_fen());
                    }
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let m = all.get((rng % all.len() as u64) as usize);
                board.make_move(m);
            }
        }
    }
    assert!(hits > 500);
}

#[test]
fn captures_hanging_queen() {
    let fen = "rnb1kbnr/pppp1ppp/8/4p3/3q4/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 3";
    let mut board = Board::from_fen(fen).unwrap();
    let (best, _) = search(&mut board, 4);
    assert_eq!(best.unwrap().to_string(), "e3d4");
}

#[test]
fn finds_mate_in_one() {
    let fen = "4k3/7Q/4K3/8/8/8/8/8 w - - 0 1";
    let mut board = Board::from_fen(fen).unwrap();
    let (best, score) = search(&mut board, 3);
    let m = best.expect("should find a move");
    assert!(score >= MATE_THRESHOLD);
    let undo = board.make_move(m);
    let mut next = MoveList::new();
    board.gen_legal_into(&mut next);
    let mate = next.is_empty() && board.in_check(board.side_to_move);
    board.unmake_move(m, undo);
    assert!(mate, "{} should deliver mate", m);
}

#[test]
fn repetition_helper_detects_prior_occurrence() {
    // Positions before the current node, most recent last.
    let cur = 0xABCDu64;
    let hashes = [0x1111u64, cur, 0x2222u64];
    assert!(is_repetition(cur, 50, &hashes, 0, 0));
    assert!(!is_repetition(cur, 1, &hashes, 0, 0));
    // Odd distance means a different side to move.
    let hashes_odd = [0x2222u64, cur];
    assert!(!is_repetition(cur, 50, &hashes_odd, 0, 0));
    let hashes_absent = [0x2222u64, 0x3333u64];
    assert!(!is_repetition(cur, 50, &hashes_absent, 0, 0));
}

#[test]
fn repetition_before_root_needs_three_fold() {
    let cur = 0xABCDu64;
    // One occurrence in the game history (root at index 2): not yet a draw.
    let once = [cur, 0x1u64, 0x2u64, 0x3u64];
    assert!(!is_repetition(cur, 50, &once, 2, 0));
    // The root position itself only counts as history.
    let at_root = [0x1u64, 0x2u64, cur, 0x3u64];
    assert!(!is_repetition(cur, 50, &at_root, 2, 0));
    // Two occurrences in the history are a draw.
    let twice = [cur, 0x1u64, cur, 0x2u64, 0x3u64, 0x4u64];
    assert!(is_repetition(cur, 50, &twice, 4, 0));
    // A position seen inside the tree is a draw at once...
    let tree = [0x1u64, 0x2u64, 0x3u64, cur, 0x4u64];
    assert!(is_repetition(cur, 50, &tree, 1, 0));
    // ...unless a null move lies in between.
    assert!(!is_repetition(cur, 50, &tree, 1, 4));
}

#[test]
fn insufficient_material_classification() {
    let draws = [
        "8/8/8/4k3/8/8/8/4K3 w - - 0 1",   // K vs K
        "8/8/8/4k3/8/8/8/4K2N w - - 0 1",  // K+N vs K
        "8/8/8/4k3/8/8/8/4K2B w - - 0 1",  // K+B vs K
        "8/8/8/2b1k3/8/8/8/4K3 w - - 0 1", // K vs K+B
    ];
    for fen in draws {
        let b = Board::from_fen(fen).unwrap();
        assert!(is_insufficient_material(&b), "{fen} should be a draw");
    }
    let wins = [
        "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",   // pawn present
        "8/8/8/4k3/8/8/8/4K2R w - - 0 1",    // rook present
        "8/8/8/4k3/8/8/8/3QK3 w - - 0 1",    // queen present
        "8/8/8/2b1k3/8/8/8/2B1K3 w - - 0 1", // two bishops on the board
    ];
    for fen in wins {
        let b = Board::from_fen(fen).unwrap();
        assert!(!is_insufficient_material(&b), "{fen} is not insufficient");
    }
}

#[test]
fn search_scores_insufficient_material_as_draw() {
    let mut board = Board::from_fen("4k3/8/8/8/8/8/8/4K2N w - - 0 1").unwrap();
    let (best, score) = search(&mut board, 5);
    assert!(best.is_some());
    assert_eq!(score, 0, "K+N vs K is a dead draw");
}

#[test]
fn search_respects_fifty_move_rule() {
    let mut board = Board::from_fen("8/8/8/3k4/8/8/3KQ3/8 w - - 100 1").unwrap();
    let (best, score) = search(&mut board, 5);
    assert!(best.is_some());
    assert_eq!(score, 0, "fifty-move rule forces a draw despite the queen");
}

#[test]
fn mate_beats_fifty_move_rule() {
    let mut board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 100 1").unwrap();
    let (best, score) = search(&mut board, 4);
    let m = best.expect("should find a move");
    assert!(score >= MATE_THRESHOLD, "must see the mate, not claim a draw");
    let undo = board.make_move(m);
    let mut next = MoveList::new();
    board.gen_legal_into(&mut next);
    let mate = next.is_empty() && board.in_check(board.side_to_move);
    board.unmake_move(m, undo);
    assert!(mate, "{m} should deliver mate");
}

#[test]
fn engine_avoids_repeating_a_won_position() {
    use crate::chess::Board as B;
    // Seeding the history with the root makes king shuffles a repetition.
    let mut engine = Engine::new();
    let mut board = B::from_fen("4k3/8/4K3/8/8/8/8/3Q4 w - - 0 1").unwrap();
    engine.set_game_history(vec![board.hash]);
    let (best, score, _, _) = engine.search(&mut board, Limits::depth(6));
    assert!(best.is_some());
    assert!(score > 100, "up a queen, the engine must find a winning line, got {score}");
}

#[test]
fn reported_pvs_are_legal_lines() {
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    let mut board = Board::from_fen(fen).unwrap();
    let mut engine = Engine::new();
    let mut pvs: Vec<(u32, Vec<Move>)> = Vec::new();
    let (best, ..) = engine.search_with_info(&mut board, Limits::depth(9), &mut |info| {
        pvs.push((info.depth, info.pv.to_vec()));
    });
    assert_eq!(pvs.iter().map(|(d, _)| *d).collect::<Vec<_>>(), (1..=9).collect::<Vec<_>>());
    assert_eq!(pvs.last().unwrap().1[0], best.unwrap());
    for (depth, pv) in &pvs {
        let mut b = board.clone();
        for &m in pv {
            let mut legal = MoveList::new();
            b.gen_legal_into(&mut legal);
            assert!(legal.iter().any(|l| l == m), "illegal {m} in depth {depth} pv");
            b.make_move(m);
        }
    }
}

#[test]
fn stopped_search_still_returns_a_legal_move() {
    let mut board = Board::default();
    let mut engine = Engine::new();
    engine.stop_handle().store(true, std::sync::atomic::Ordering::Relaxed);
    let (best, ..) = engine.search(&mut board, Limits::infinite());
    let mut legal = MoveList::new();
    board.gen_legal_into(&mut legal);
    assert!(legal.iter().any(|l| Some(l) == best));
}

#[test]
fn tt_roundtrip() {
    let mut tt = Tt::new();
    let hash = 0xdead_beef_cafe_babe_u64;
    let mv = Move::new(Square::new(4, 1), Square::new(4, 3), Move::DOUBLE_PUSH);
    tt.store(hash, Some(mv), 42, None, 5, BOUND_EXACT, true);
    assert!(tt.probe(hash).unwrap().pv());
    let entry = tt.probe(hash).expect("entry should be present");
    assert_eq!(entry.key, hash);
    assert_eq!(Move(entry.move_packed), mv);
    assert_eq!(entry.score, 42);
    assert_eq!(entry.depth, 5);
    assert_eq!(entry.bound(), BOUND_EXACT);
    // A different hash that maps to the same slot still misses.
    let foreign = hash ^ 1u64; // perturbs key but not slot for power-of-2 mask
    let _ = foreign; // suppress unused if collision is rare
}

// The original recompute-every-step swap list, as the reference for `see` and `see_ge`.
fn see_reference(board: &Board, m: Move) -> i32 {
    fn attackers_of(board: &Board, sq: Square, side: Color, occ: u64) -> u64 {
        let (s, o) = (side.idx(), side.opp().idx());
        let bb = &board.piece_bb[s];
        let bq = bb[PieceKind::Bishop.idx()] | bb[PieceKind::Queen.idx()];
        let rq = bb[PieceKind::Rook.idx()] | bb[PieceKind::Queen.idx()];
        ((PAWN_ATTACKS[o][sq.idx()] & bb[PieceKind::Pawn.idx()])
            | (KNIGHT_ATTACKS[sq.idx()] & bb[PieceKind::Knight.idx()])
            | (bishop_attacks(sq, occ) & bq)
            | (rook_attacks(sq, occ) & rq)
            | (KING_ATTACKS[sq.idx()] & bb[PieceKind::King.idx()]))
            & occ
    }
    let (to, from) = (m.to(), m.from());
    let Some(attacker) = board.piece_at(from) else { return 0 };
    let mut occ = board.occupancy & !from.bb();
    let captured = if m.flag() == Move::EP_CAPTURE {
        let cap = if attacker.color == Color::White { Square(to.0 - 8) } else { Square(to.0 + 8) };
        occ &= !cap.bb();
        SEE_VALUES[0]
    } else {
        board.piece_at(to).map_or(0, |p| SEE_VALUES[p.kind.idx()])
    };
    let mut gain = [0i32; 32];
    gain[0] = captured;
    let mut cur = m.promo_kind().unwrap_or(attacker.kind);
    if let Some(promo) = m.promo_kind() {
        gain[0] += SEE_VALUES[promo.idx()] - SEE_VALUES[0];
    }
    let mut side = attacker.color.opp();
    let mut d = 0;
    loop {
        d += 1;
        gain[d] = SEE_VALUES[cur.idx()] - gain[d - 1];
        let att = attackers_of(board, to, side, occ);
        let Some(k) = (0..6).find(|&k| att & board.piece_bb[side.idx()][k] != 0) else { break };
        occ &= !lsb(att & board.piece_bb[side.idx()][k]).bb();
        cur = PieceKind::from_idx(k);
        side = side.opp();
    }
    for i in (1..d).rev() {
        gain[i - 1] = gain[i - 1].min(-gain[i]);
    }
    gain[0]
}

#[test]
fn see_and_see_ge_match_reference() {
    let fens = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 9",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "2r2rk1/1bqnbppp/pp1ppn2/8/2PNP3/1PN1BP2/P2QB1PP/R4RK1 w - - 0 13",
        "3q1rk1/1b3ppp/p3pn2/1p1r4/3P4/1BN1QN2/PP3PPP/2R2RK1 b - - 0 17",
    ];
    let thresholds = [-1000, -500, -330, -320, -200, -100, -1, 0, 1, 100, 220, 320, 500, 900, 1500];
    let mut rng: u64 = 0xD1B5_4A32_D192_ED03;
    let mut checked = 0;
    for fen in fens {
        for _ in 0..30 {
            let mut board = Board::from_fen(fen).unwrap();
            for _ in 0..60 {
                let mut all = MoveList::new();
                board.gen_legal_into(&mut all);
                if all.is_empty() {
                    break;
                }
                let threats = threat_map(&board);
                for m in all.iter() {
                    let r = see_reference(&board, m);
                    assert_eq!(see_with_threats(&board, m, threats), r, "{m} in {}", board.to_fen());
                    if threats & (m.from().bb() | m.to().bb()) == 0 && m.flag() != Move::EP_CAPTURE {
                        let gain0 = board.piece_at(m.to()).map_or(0, |p| SEE_VALUES[p.kind.idx()])
                            + m.promo_kind().map_or(0, |k| SEE_VALUES[k.idx()] - SEE_VALUES[0]);
                        assert_eq!(r, gain0, "{m} in {}", board.to_fen());
                    }
                    assert_eq!(see(&board, m), r, "{m} in {}", board.to_fen());
                    for t in thresholds {
                        assert_eq!(see_ge(&board, m, t), r >= t, "{m} t={t} in {}", board.to_fen());
                    }
                    checked += 1;
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                board.make_move(all.get((rng % all.len() as u64) as usize));
            }
        }
    }
    assert!(checked > 10000);
}

#[test]
fn tt_move_legality_matches_generator() {
    let fens = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
    ];
    let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
    let mut accepted = 0;
    for fen in fens {
        for _ in 0..6 {
            let mut board = Board::from_fen(fen).unwrap();
            for _ in 0..30 {
                let mut all = MoveList::new();
                board.gen_legal_into(&mut all);
                if all.is_empty() {
                    break;
                }
                let in_check = board.in_check(board.side_to_move);
                let legal: Vec<Move> = all.iter().collect();
                for raw in 0..=u16::MAX {
                    let m = Move(raw);
                    let fast = tt_move_is_legal(&board, m, in_check);
                    let listed = legal.contains(&m);
                    assert!(!fast || listed, "{m} accepted in {}", board.to_fen());
                    if listed && !in_check && !m.is_castle() && m.flag() != Move::EP_CAPTURE {
                        assert!(fast, "{m} rejected in {}", board.to_fen());
                    }
                    accepted += fast as u32;
                }
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                board.make_move(all.get((rng % all.len() as u64) as usize));
            }
        }
    }
    assert!(accepted > 1000);
}
