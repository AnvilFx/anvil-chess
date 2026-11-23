use crate::chess::{Board, Color, Move, MoveList, PieceKind, Square};
use crate::search::{Engine, IterInfo, Limits, MATE_SCORE, MATE_THRESHOLD, PARAM_SPECS, TT_DEFAULT_MB, TT_MAX_MB};
use crate::timeman::{self, GoLimits};
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const SEARCH_STACK_BYTES: usize = 32 * 1024 * 1024;

pub fn run() {
    let stdin = io::stdin();
    let mut board = Board::default();
    let mut engine = Engine::new();

    warm_up(&mut engine);
    let mut session = Session {
        stop: engine.stop_handle(),
        engine: Some(engine),
        search: None,
        move_overhead: timeman::MOVE_OVERHEAD_MS,
        normalize: true,
    };

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let received = Instant::now();
        let line = line.trim();
        let cmd = line.split_whitespace().next().unwrap_or("");

        match cmd {
            "uci" => {
                println!("id name Anvil {}", env!("CARGO_PKG_VERSION"));
                println!("id author AnvilFx");
                println!("option name Hash type spin default {TT_DEFAULT_MB} min 1 max {TT_MAX_MB}");
                println!("option name Move Overhead type spin default {} min 0 max 5000", timeman::MOVE_OVERHEAD_MS);
                println!("option name NormalizeScore type check default true");
                for s in PARAM_SPECS {
                    println!("option name {} type spin default {} min {} max {}", s.name, s.default, s.min, s.max);
                }
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "setoption" => session.set_option(line),
            "ucinewgame" => {
                board = Board::default();
                session.idle().reset();
            }
            "position" => {
                let history = parse_position(&mut board, line);
                session.idle().set_game_history(history);
            }
            "go" => session.go(&board, line, received),
            "stop" => session.stop(),
            // Static NNUE eval of the current position, side to move's view.
            "eval" => {
                let accs = crate::eval::Accumulator::recompute(&board);
                let bucket = crate::eval::output_bucket(board.occupancy);
                let raw = accs.evaluate(board.side_to_move, bucket);
                println!("info string eval {} raw {raw} bucket {bucket}", to_cp(raw, material(&board)));
            }
            "bench" => {
                session.idle();
                let depth = line.split_whitespace().nth(1).and_then(|d| d.parse().ok());
                crate::bench::run(depth.unwrap_or(crate::bench::BENCH_DEPTH));
            }
            "quit" => break,
            _ => {}
        }
        let _ = io::stdout().flush();
    }
    session.stop();
}

/// Owns the engine while idle; `go` moves it onto a search thread until the search ends.
struct Session {
    engine: Option<Engine>,
    search: Option<JoinHandle<Engine>>,
    stop: Arc<AtomicBool>,
    move_overhead: u64,
    /// Report win-rate normalized scores; datagen turns this off to get raw search scores.
    normalize: bool,
}

impl Session {
    /// Waits for a running search to finish, then returns the engine.
    fn idle(&mut self) -> &mut Engine {
        if let Some(handle) = self.search.take() {
            self.engine = Some(handle.join().expect("search thread panicked"));
        }
        self.engine.as_mut().expect("engine missing")
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.idle();
    }

    fn set_option(&mut self, line: &str) {
        let Some((name, value)) = parse_option(line) else { return };
        if name.eq_ignore_ascii_case("Move Overhead") {
            if let Ok(v) = value.parse::<u64>() {
                self.move_overhead = v.min(5000);
            }
        } else if name.eq_ignore_ascii_case("NormalizeScore") {
            self.normalize = !matches!(value.to_ascii_lowercase().as_str(), "false" | "0");
        } else {
            set_option(self.idle(), name, value);
        }
    }

    fn go(&mut self, board: &Board, line: &str, received: Instant) {
        self.idle();
        let mut engine = self.engine.take().expect("engine missing");
        let mut board = board.clone();
        let go = parse_go_limits(line);
        let mut limits = to_limits(&go, board.side_to_move, self.move_overhead);
        limits.start = Some(received);
        let scale = if self.normalize { Some(material(&board)) } else { None };
        let stop = self.stop.clone();
        // Reset before spawning so an early `stop` isn't lost.
        stop.store(false, Ordering::Relaxed);

        let handle = thread::Builder::new()
            .stack_size(SEARCH_STACK_BYTES)
            .spawn(move || {
                let (best, ..) = engine.search_with_info(&mut board, limits, &mut |info| print_info(info, scale));
                // UCI forbids sending bestmove during `go infinite` before `stop`.
                if go.infinite {
                    while !stop.load(Ordering::Relaxed) {
                        thread::sleep(Duration::from_millis(1));
                    }
                }
                match best {
                    Some(m) => println!("bestmove {}", m),
                    None => println!("bestmove 0000"),
                }
                let _ = io::stdout().flush();
                engine
            })
            .expect("failed to spawn search thread");
        self.search = Some(handle);
    }
}

/// `scale` is the root material for normalized scores, `None` for raw ones.
fn print_info(info: &IterInfo, scale: Option<i32>) {
    let nps = info.nodes * 1000 / info.elapsed_ms.max(1);
    let pv: Vec<String> = info.pv.iter().map(|m| m.to_string()).collect();
    println!(
        "info depth {} seldepth {} score {}{} nodes {} nps {} hashfull {} time {} pv {}",
        info.depth,
        info.seldepth,
        format_score(info.score, scale),
        if info.lower_bound { " lowerbound" } else { "" },
        info.nodes,
        nps,
        info.hashfull,
        info.elapsed_ms,
        pv.join(" ")
    );
}

/// Initialize lazy statics (NNUE, magic tables) before the first real search.
fn warm_up(engine: &mut Engine) {
    let mut b = Board::default();
    let _ = engine.search(&mut b, Limits::depth(6));
    engine.reset();
}

/// Returns the hashes of all positions before the current one, for repetition detection.
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
                    history.push(board.hash);
                    board.make_move(m);
                }
                None => break,
            }
        }
    }
    history
}

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

fn parse_option(line: &str) -> Option<(&str, &str)> {
    let after = &line[line.find(" name ")? + " name ".len()..];
    let i = after.find(" value ")?;
    Some((after[..i].trim(), after[i + " value ".len()..].trim()))
}

fn set_option(engine: &mut Engine, name: &str, value: &str) {
    if let Ok(v) = value.parse::<i32>() {
        if name.eq_ignore_ascii_case("Hash") {
            engine.set_hash_mb(v.clamp(1, TT_MAX_MB as i32) as usize);
        } else {
            engine.set_param(name, v);
        }
    }
}

fn to_limits(go: &GoLimits, stm: Color, move_overhead: u64) -> Limits {
    if let Some(d) = go.depth {
        return Limits::depth(d);
    }
    if let Some(n) = go.nodes {
        return Limits::nodes(n);
    }
    if go.infinite {
        return Limits::infinite();
    }
    if go.movetime.is_some() || go.wtime.is_some() || go.btime.is_some() {
        return Limits::budget(timeman::allocate(go, stm, move_overhead));
    }

    Limits::depth(12)
}

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

fn format_score(score: i32, scale: Option<i32>) -> String {
    if score > MATE_THRESHOLD {
        format!("mate {}", (MATE_SCORE - score + 1) / 2)
    } else if score < -MATE_THRESHOLD {
        format!("mate -{}", (MATE_SCORE + score + 1) / 2)
    } else {
        format!("cp {}", scale.map_or(score, |m| to_cp(score, m)))
    }
}

/// Material in pawns (N, B = 3, R = 5, Q = 9), both sides.
fn material(board: &Board) -> i32 {
    let count = |k: PieceKind| (board.piece_bb[0][k as usize] | board.piece_bb[1][k as usize]).count_ones() as i32;
    count(PieceKind::Pawn)
        + 3 * (count(PieceKind::Knight) + count(PieceKind::Bishop))
        + 5 * count(PieceKind::Rook)
        + 9 * count(PieceKind::Queen)
}

/// Internal score to UCI centipawns: +100 means the side to move wins half its games. The
/// 50% point (in thousandths, by material) is fitted on Anvil's own 50k-node self-play games.
fn to_cp(score: i32, material: i32) -> i32 {
    let m = material.clamp(8, 78) as i64;
    let half_win = 412_959 - 4_003 * m + 12 * m * m;
    let cp = score as i64 * 100_000;
    ((cp + cp.signum() * half_win / 2) / half_win) as i32
}
