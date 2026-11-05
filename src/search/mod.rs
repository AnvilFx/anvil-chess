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
    pub max_nodes: u64,
}

impl Limits {
    pub fn depth(d: u32) -> Self {
        Self { max_depth: d, soft_ms: u64::MAX, hard_ms: u64::MAX, max_nodes: u64::MAX }
    }
    pub fn time(ms: u64) -> Self {
        Self { max_depth: MAX_PLY as u32, soft_ms: ms, hard_ms: ms, max_nodes: u64::MAX }
    }
    pub fn nodes(n: u64) -> Self {
        Self { max_depth: MAX_PLY as u32, soft_ms: u64::MAX, hard_ms: u64::MAX, max_nodes: n }
    }
    pub fn infinite() -> Self {
        Self::depth(MAX_PLY as u32)
    }
    pub fn budget(b: TimeBudget) -> Self {
        Self { max_depth: MAX_PLY as u32, soft_ms: b.soft_ms, hard_ms: b.hard_ms, max_nodes: u64::MAX }
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
const IIR_MIN_DEPTH: u32 = 4;
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
    let start = Instant::now();
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
        if limits.soft_ms != u64::MAX && d > 1 {
            let target = if fixed_time {
                limits.soft_ms
            } else {
                // Share of all nodes spent on the best move; unreliable at low depth.
                let best_share = best_move
                    .filter(|_| completed_depth >= NODE_TM_MIN_DEPTH && ctx.nodes > 0)
                    .map(|m| ctx.root_nodes[root_node_idx(m)] as f64 / ctx.nodes as f64);
                timeman::scaled_soft(limits.soft_ms, limits.hard_ms, bm_stability, best_score, prev_score, best_share)
            };
            if start.elapsed().as_millis() as u64 >= target {
                break;
            }
        }

        let (m, s) = search_root_aspiration(board, d, best_move, best_score, ctx, tt);
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
                let single = [mv];
                let pv = if root_pv.first() == m.as_ref() { root_pv } else { &single };
                report(&IterInfo {
                    depth: d,
                    seldepth: ctx.seldepth as u32,
                    score: s,
                    nodes: ctx.nodes,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    hashfull: tt.hashfull(),
                    pv,
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
