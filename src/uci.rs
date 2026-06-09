// UCI protocol front-end. Reads commands on stdin, drives the search, and
// emits info/bestmove on stdout. Synchronous: a `go` blocks until the search
// finishes (fine for fixed depth/movetime/clock play; no `stop` for infinite).

use crate::core::{Board, Color, Move, MoveList, PieceKind, Square};
use crate::search::{Engine, Limits, MATE_SCORE, MATE_THRESHOLD, PARAM_SPECS};
use crate::timeman::{self, GoLimits};
use std::io::{self, BufRead, Write};
use std::time::Instant;

pub fn run() {
    let stdin = io::stdin();
    let mut board = Board::default();
    let mut engine = Engine::new();

    // Pay the one-time lazy-init cost (NNUE net parsing, magic attack tables) at
    // startup rather than on the first `go`. It runs during the GUI handshake,
    // off any move clock, so the opening move does not absorb a few hundred ms
    // of warm-up and flag under a tight movetime budget.
    warm_up(&mut engine);

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        let cmd = line.split_whitespace().next().unwrap_or("");

        match cmd {
            "uci" => {
                println!("id name Anvil");
                println!("id author Anvil contributors");
                for s in PARAM_SPECS {
                    println!(
                        "option name {} type spin default {} min {} max {}",
                        s.name, s.default, s.min, s.max
                    );
                }
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "setoption" => set_option(&mut engine, line),
            "ucinewgame" => {
                board = Board::default();
                engine.reset();
            }
            "position" => {
                let history = parse_position(&mut board, line);
                engine.set_game_history(history);
            }
            "go" => handle_go(&mut engine, &mut board, line),
            "quit" => break,
            _ => {}
        }
        let _ = io::stdout().flush();
    }
}

/// Force the lazy statics (NNUE network, magic sliding-attack tables) to
/// initialize via a tiny throwaway search, then clear the hash so it does not
/// carry into real play. Only the first call pays the real cost.
fn warm_up(engine: &mut Engine) {
    let mut b = Board::default();
    let _ = engine.search(&mut b, Limits::depth(6));
    engine.reset();
}

/// `position startpos [moves ...]` or `position fen <6 fields> [moves ...]`.
///
/// Returns the Zobrist hashes of every position reached before the final
/// (current) one: the game history used for repetition detection in search.
/// Empty when no moves are given.
fn parse_position(board: &mut Board, line: &str) -> Vec<u64> {
    let rest = line.strip_prefix("position").unwrap_or("").trim();
    let (setup, moves) = match rest.find("moves") {
        Some(i) => (rest[..i].trim(), Some(rest[i + "moves".len()..].trim())),
        None => (rest, None),
    };

    if setup.starts_with("startpos") {
        *board = Board::default();
    } else if let Some(fen) = setup.strip_prefix("fen") {
        match Board::from_fen(fen.trim()) {
            Ok(b) => *board = b,
            Err(_) => return Vec::new(),
        }
    } else {
        return Vec::new();
    }

    let mut history = Vec::new();
    if let Some(moves) = moves {
        for tok in moves.split_whitespace() {
            match find_move(board, tok) {
                Some(m) => {
                    // Record the position before this move, then advance.
                    history.push(board.hash);
                    board.make_move(m);
                }
                None => break,
            }
        }
    }
    history
}

/// Resolve a UCI move string (e.g. "e2e4", "e7e8q", "e1g1" castle) to a legal
/// Move on the current board, or None if it isn't legal here.
fn find_move(board: &Board, uci: &str) -> Option<Move> {
    if uci.len() < 4 {
        return None;
    }
    let from = Square::from_algebraic(&uci[0..2])?;
    let to = Square::from_algebraic(&uci[2..4])?;
    let promo = uci.as_bytes().get(4).map(|&b| b as char);

    let mut moves = MoveList::new();
    board.gen_legal_into(&mut moves);
    for m in moves.iter() {
        if m.from() != from || m.to() != to {
            continue;
        }
        match (promo, m.promo_kind()) {
            (None, None) => return Some(m),
            (Some(c), Some(k)) => {
                let want = match k {
                    PieceKind::Knight => 'n',
                    PieceKind::Bishop => 'b',
                    PieceKind::Rook => 'r',
                    PieceKind::Queen => 'q',
                    _ => ' ',
                };
                if c == want {
                    return Some(m);
                }
            }
            _ => {}
        }
    }
    None
}

/// `setoption name <Name> value <Value>`. Only integer `spin` options exist (the
/// tunable search parameters, see `PARAM_SPECS`); unknown names are ignored per
/// UCI tolerance. Option names contain no spaces, so the parse is a simple split.
fn set_option(engine: &mut Engine, line: &str) {
    let after = match line.find(" name ") {
        Some(i) => &line[i + " name ".len()..],
        None => return,
    };
    let (name, value) = match after.find(" value ") {
        Some(i) => (after[..i].trim(), after[i + " value ".len()..].trim()),
        None => return,
    };
    if let Ok(v) = value.parse::<i32>() {
        engine.set_param(name, v);
    }
}

fn handle_go(engine: &mut Engine, board: &mut Board, line: &str) {
    let limits = parse_go(line, board.side_to_move);
    let t0 = Instant::now();
    let (best, score, nodes, depth) = engine.search(board, limits);
    let ms = t0.elapsed().as_millis().max(1) as u64;
    let nps = nodes * 1000 / ms;

    let pv = best.map(|m| m.to_string()).unwrap_or_default();
    println!(
        "info depth {} score {} nodes {} time {} nps {} pv {}",
        depth,
        format_score(score),
        nodes,
        ms,
        nps,
        pv
    );
    match best {
        Some(m) => println!("bestmove {}", m),
        None => println!("bestmove 0000"),
    }
}

/// Map a `go` command to a search limit. Fixed depth/nodes are honored directly;
/// `movetime` and clock play route through `timeman` for a soft/hard budget.
fn parse_go(line: &str, stm: Color) -> Limits {
    let go = parse_go_limits(line);

    if let Some(d) = go.depth {
        return Limits::depth(d);
    }
    if let Some(n) = go.nodes {
        return Limits::nodes(n);
    }
    if go.infinite {
        // No async `stop` command yet, so bound infinite to a deep fixed depth.
        return Limits::depth(64);
    }
    if go.movetime.is_some() || go.wtime.is_some() || go.btime.is_some() {
        return Limits::budget(timeman::allocate(&go, stm));
    }

    // No constraints given: a sane bounded default rather than searching forever.
    Limits::depth(12)
}

/// Parse the timing tokens of a `go` command into `GoLimits`.
fn parse_go_limits(line: &str) -> GoLimits {
    let mut it = line.split_whitespace().skip(1);
    let mut go = GoLimits::default();

    while let Some(tok) = it.next() {
        let mut val = || it.next().and_then(|s| s.parse::<u64>().ok());
        match tok {
            "depth" => go.depth = val().map(|v| v as u32),
            "movetime" => go.movetime = val(),
            "nodes" => go.nodes = val(),
            "wtime" => go.wtime = val(),
            "btime" => go.btime = val(),
            "winc" => go.winc = val().unwrap_or(0),
            "binc" => go.binc = val().unwrap_or(0),
            "movestogo" => go.movestogo = val().map(|v| v as u32),
            "infinite" => go.infinite = true,
            _ => {}
        }
    }
    go
}

/// UCI score: centipawns, or `mate N` when the score is in mate range.
fn format_score(score: i32) -> String {
    if score > MATE_THRESHOLD {
        format!("mate {}", (MATE_SCORE - score + 1) / 2)
    } else if score < -MATE_THRESHOLD {
        format!("mate -{}", (MATE_SCORE + score + 1) / 2)
    } else {
        format!("cp {}", score)
    }
}
