use crate::chess::{
    bishop_attacks, lsb, rook_attacks, Board, Color, Move, MoveList, PieceKind, Square, KING_ATTACKS, KNIGHT_ATTACKS,
    PAWN_ATTACKS,
};
use crate::eval::AccStack;
use crate::timeman::{self, TimeBudget};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

mod context;
mod history;
mod negamax;
mod ordering;
mod params;
mod see;
#[cfg(test)]
mod tests;
mod tt;

use context::*;
use history::*;
use negamax::*;
use ordering::*;
pub use params::*;
use see::*;
pub use tt::*;

pub const INF: i32 = 32000;
pub const MATE_SCORE: i32 = 30000;
pub const MATE_THRESHOLD: i32 = MATE_SCORE - 1000;

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_depth: u32,
    pub soft_ms: u64,
    pub hard_ms: u64,
    /// No new iteration starts once this many nodes are searched.
    pub soft_nodes: u64,
    /// The search stops mid-iteration at this many nodes.
    pub max_nodes: u64,
    /// When the clock started (the UCI `go` arrival); `None` starts it with the search.
    pub start: Option<Instant>,
}

/// `go nodes N`: N is a soft limit checked between iterations; the hard cap is N times this.
pub const NODES_HARD_MUL: u64 = 8;

impl Limits {
    pub fn depth(d: u32) -> Self {
        Self { max_depth: d, soft_ms: u64::MAX, hard_ms: u64::MAX, soft_nodes: u64::MAX, max_nodes: u64::MAX, start: None }
    }
    pub fn time(ms: u64) -> Self {
        Self { max_depth: MAX_PLY as u32, soft_ms: ms, hard_ms: ms, soft_nodes: u64::MAX, max_nodes: u64::MAX, start: None }
    }
    /// Finishes the iteration that crosses `n` nodes, stopping mid-iteration only at `n * NODES_HARD_MUL`.
    pub fn nodes(n: u64) -> Self {
        Self {
            max_depth: MAX_PLY as u32,
            soft_ms: u64::MAX,
            hard_ms: u64::MAX,
            soft_nodes: n,
            max_nodes: n.saturating_mul(NODES_HARD_MUL),
            start: None,
        }
    }
    pub fn infinite() -> Self {
        Self::depth(MAX_PLY as u32)
    }
    pub fn budget(b: TimeBudget) -> Self {
        Self {
            max_depth: MAX_PLY as u32,
            soft_ms: b.soft_ms,
            hard_ms: b.hard_ms,
            soft_nodes: u64::MAX,
            max_nodes: u64::MAX,
            start: None,
        }
    }
}

/// Reported after each completed iteration.
pub struct IterInfo<'a> {
    pub depth: u32,
    pub seldepth: u32,
    pub score: i32,
    pub nodes: u64,
    pub elapsed_ms: u64,
    pub hashfull: u32,
    pub pv: &'a [Move],
    /// The iteration ended on a root fail high, so `score` is a lower bound.
    pub lower_bound: bool,
}

pub struct Engine {
    tt: Tt,
    /// Hashes of positions played before the root, for repetition detection.
    history: Vec<u64>,
    params: SearchParams,
    /// Kept across moves so move ordering starts warm; cleared on `reset`.
    ctx: SearchCtx,
}

impl Engine {
    pub fn new() -> Self {
        Self { tt: Tt::new(), history: Vec::new(), params: SearchParams::default(), ctx: SearchCtx::new() }
    }
    /// Returns (best move, score in cp, nodes searched, depth reached).
    pub fn search(&mut self, board: &mut Board, limits: Limits) -> (Option<Move>, i32, u64, u32) {
        run_search(board, limits, &mut self.tt, &mut self.ctx, &self.history, self.params, None)
    }
    pub fn search_with_info(
        &mut self,
        board: &mut Board,
        limits: Limits,
        on_iter: &mut dyn FnMut(&IterInfo),
    ) -> (Option<Move>, i32, u64, u32) {
        run_search(board, limits, &mut self.tt, &mut self.ctx, &self.history, self.params, Some(on_iter))
    }
    /// Setting this aborts the running search. It is not cleared by `search`.
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        self.ctx.abort.clone()
    }
    pub fn set_param(&mut self, name: &str, value: i32) -> bool {
        self.params.set(name, value)
    }
    pub fn set_hash_mb(&mut self, mb: usize) {
        // Free the old table first so both never coexist in memory.
        self.tt = Tt::with_mb(1);
        self.tt = Tt::with_mb(mb);
    }
    pub fn set_game_history(&mut self, history: Vec<u64>) {
        self.history = history;
    }
    pub fn reset(&mut self) {
        self.tt.clear();
        self.history.clear();
        self.ctx.clear_heuristics();
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_PLY: usize = 128;

/// `first` followed by the TT's best moves, for display only. Stops at a missing or
/// illegal move, a repeated position, or `max_len` moves.
fn pv_from_tt(board: &mut Board, first: Move, tt: &Tt, max_len: usize) -> Vec<Move> {
    let mut line = vec![first];
    let mut undos = vec![board.make_move(first)];
    let mut seen = vec![board.hash];
    while line.len() < max_len.max(1) {
        let Some(e) = tt.probe(board.hash) else { break };
        if e.move_packed == 0 {
            break;
        }
        let m = Move(e.move_packed);
        let mut legal = MoveList::new();
        board.gen_legal_into(&mut legal);
        if !legal.iter().any(|x| x == m) {
            break;
        }
        undos.push(board.make_move(m));
        line.push(m);
        if seen.contains(&board.hash) {
            break;
        }
        seen.push(board.hash);
    }
    for (m, u) in line.iter().zip(undos).rev() {
        board.unmake_move(*m, u);
    }
    line
}
/// Node-based time scaling starts once this many iterations have completed.
const NODE_TM_MIN_DEPTH: u32 = 7;

pub fn search(board: &mut Board, max_depth: u32) -> (Option<Move>, i32) {
    let mut tt = Tt::new();
    let (m, s, _, _) = search_with_limits(board, Limits::depth(max_depth), &mut tt, &[], SearchParams::default());
    (m, s)
}

/// Uses a fresh context; `Engine` keeps one across moves.
pub fn search_with_limits(
    board: &mut Board,
    limits: Limits,
    tt: &mut Tt,
    history: &[u64],
    params: SearchParams,
) -> (Option<Move>, i32, u64, u32) {
    let mut ctx = SearchCtx::new();
    run_search(board, limits, tt, &mut ctx, history, params, None)
}

fn run_search(
    board: &mut Board,
    limits: Limits,
    tt: &mut Tt,
    ctx: &mut SearchCtx,
    history: &[u64],
    params: SearchParams,
    mut on_iter: Option<&mut dyn FnMut(&IterInfo)>,
) -> (Option<Move>, i32, u64, u32) {
    let start = limits.start.unwrap_or_else(Instant::now);
    let hard_deadline =
        if limits.hard_ms == u64::MAX { None } else { Some(start + Duration::from_millis(limits.hard_ms)) };
    // Stability scaling and the mate early-out only apply to clock budgets.
    let fixed_time = limits.soft_ms != u64::MAX && limits.soft_ms == limits.hard_ms;

    ctx.begin_search(board, history, hard_deadline, limits.max_nodes, params);
    tt.new_search();

    let mut best_move: Option<Move> = None;
    let mut best_score = 0;
    let mut prev_score = 0;
    let mut bm_stability: u32 = 0;
    let mut completed_depth: u32 = 0;

    for d in 1..=limits.max_depth.min(MAX_PLY as u32) {
        // Depth 1 always runs so there is a move to return.
        if d > 1 && ctx.nodes >= limits.soft_nodes {
            break;
        }
        if limits.soft_ms != u64::MAX && d > 1 {
            let target = if fixed_time {
                limits.soft_ms
            } else {
                // Share of all nodes spent on the best move; unreliable at low depth.
                let best_share = best_move
                    .filter(|_| completed_depth >= NODE_TM_MIN_DEPTH && ctx.nodes > 0)
                    .map(|m| ctx.root_nodes[root_node_idx(m)] as f64 / ctx.nodes as f64);
                timeman::scaled_soft(limits.soft_ms, limits.hard_ms, bm_stability, best_score, prev_score, best_share, &params)
            };
            if start.elapsed().as_millis() as u64 >= target {
                break;
            }
        }

        let (m, s, lower_bound) = search_root_aspiration(board, d, best_move, best_score, ctx, tt);
        if ctx.stop && d > 1 {
            // A stopped iteration only returns a move that was fully searched and beat
            // the previous best at this depth, so it is the better choice.
            if m.is_some() {
                best_move = m;
            }
            break;
        }
        if let Some(mv) = m {
            bm_stability = if m == best_move { bm_stability.saturating_add(1) } else { 0 };
            prev_score = best_score;
            best_move = m;
            best_score = s;
            completed_depth = d;

            if let Some(report) = on_iter.as_mut() {
                let root_pv = ctx.root_pv();
                let tt_line;
                let pv = if root_pv.first() == m.as_ref() && root_pv.len() > 1 {
                    root_pv
                } else {
                    // A fail high records no line below the root move; the TT still has one.
                    tt_line = pv_from_tt(board, mv, tt, d as usize);
                    &tt_line[..]
                };
                report(&IterInfo {
                    depth: d,
                    seldepth: ctx.seldepth as u32,
                    score: s,
                    nodes: ctx.nodes,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    hashfull: tt.hashfull(),
                    pv,
                    lower_bound,
                });
            }
        }

        if !fixed_time && limits.soft_ms != u64::MAX && best_score.abs() >= MATE_THRESHOLD && bm_stability >= 2 {
            break;
        }
    }

    // Stopped before depth 1 finished: still return a legal move.
    if best_move.is_none() {
        let mut moves = MoveList::new();
        board.gen_legal_into(&mut moves);
        best_move = moves.iter().next();
    }
    (best_move, best_score, ctx.nodes, completed_depth)
}
