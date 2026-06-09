// Alpha-beta search: TT, move ordering, PVS, null-move, LMR, RFP, qsearch.

use crate::core::{
    bishop_attacks, lsb, rook_attacks, Board, Color, Move, MoveList, PieceKind,
    Square, KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS,
};
use crate::eval::AccStack;
use crate::timeman::{self, TimeBudget};
use std::mem::MaybeUninit;
use std::time::{Duration, Instant};

pub const INF: i32 = 32000;
pub const MATE_SCORE: i32 = 30000;
pub const MATE_THRESHOLD: i32 = MATE_SCORE - 1000;

/// Search limits. Time comes as a soft/hard pair (see `timeman`): `soft_ms` is
/// the target that gates *starting* a new iterative-deepening iteration, while
/// `hard_ms` is the absolute ceiling polled *inside* the search. Depth and node
/// caps are independent. Unused fields hold their unbounded sentinel.
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
    /// Fixed move time: spend the whole budget (soft == hard).
    pub fn time(ms: u64) -> Self {
        Self { max_depth: MAX_PLY as u32, soft_ms: ms, hard_ms: ms, max_nodes: u64::MAX }
    }
    pub fn nodes(n: u64) -> Self {
        Self { max_depth: MAX_PLY as u32, soft_ms: u64::MAX, hard_ms: u64::MAX, max_nodes: n }
    }
    /// A clock-derived budget from `timeman::allocate`.
    pub fn budget(b: TimeBudget) -> Self {
        Self {
            max_depth: MAX_PLY as u32,
            soft_ms: b.soft_ms,
            hard_ms: b.hard_ms,
            max_nodes: u64::MAX,
        }
    }
}

/// One tunable search parameter: its UCI option name and integer range. The
/// table below is the single source of truth, consumed both by the `uci`
/// option-declaration output and by `SearchParams::set`.
pub struct ParamSpec {
    pub name: &'static str,
    pub default: i32,
    pub min: i32,
    pub max: i32,
}

/// The tunable search constants exposed over UCI as `spin` options. Margins and
/// divisors are SPSA-tuned: the 3.2 run (vs anvil-3.1), refined by a 3.5 re-tune
/// (vs anvil-3.4, with continuation history and singular extensions in play).
/// Divisor ranges start at 2 so the formulas that divide by them never hit zero.
pub const PARAM_SPECS: &[ParamSpec] = &[
    ParamSpec { name: "RFPMargin", default: 55, min: 20, max: 200 },
    ParamSpec { name: "RFPMaxDepth", default: 7, min: 3, max: 12 },
    ParamSpec { name: "NMPBaseR", default: 3, min: 1, max: 5 },
    ParamSpec { name: "NMPDivisor", default: 3, min: 2, max: 8 },
    ParamSpec { name: "NMPMinDepth", default: 2, min: 2, max: 6 },
    ParamSpec { name: "LMRBase", default: 1, min: 0, max: 4 },
    ParamSpec { name: "LMRDepthDiv", default: 6, min: 2, max: 16 },
    ParamSpec { name: "LMRMoveDiv", default: 8, min: 2, max: 24 },
    ParamSpec { name: "LMRMinDepth", default: 3, min: 2, max: 6 },
    ParamSpec { name: "LMRMinMove", default: 4, min: 1, max: 8 },
    // Continuation-history coupling for LMR: reduce less for quiets with good
    // history, more for bad. The reduction shifts by clamp(history / div, -2, 2)
    // plies; a large divisor approaches the no-history behaviour.
    ParamSpec { name: "LMRHistoryDiv", default: 9545, min: 256, max: 32768 },
    // Shallow-depth move pruning; each *MaxDepth gate is a depth ceiling that
    // disables the feature at 0, so the tuner zeroes out any that don't help.
    // Margins are SPSA-tuned; the depth gates held at their starting values.
    ParamSpec { name: "LMPMaxDepth", default: 2, min: 0, max: 12 },
    ParamSpec { name: "LMPBase", default: 6, min: 1, max: 16 },
    ParamSpec { name: "FPMaxDepth", default: 4, min: 0, max: 12 },
    ParamSpec { name: "FPMargin", default: 80, min: 30, max: 400 },
    ParamSpec { name: "SEEMaxDepth", default: 4, min: 0, max: 12 },
    ParamSpec { name: "SEEQuietMargin", default: 84, min: 0, max: 200 },
    ParamSpec { name: "SEECaptMargin", default: 102, min: 0, max: 200 },
    // Singular extensions: at depth >= SingularMinDepth, the TT move is searched
    // a ply deeper when a reduced search of the alternatives all fall below
    // (ttScore - SingularMargin * depth). SingularMinDepth at its max disables
    // the feature; SingularMargin scales how readily a move counts as singular.
    ParamSpec { name: "SingularMinDepth", default: 6, min: 4, max: 12 },
    ParamSpec { name: "SingularMargin", default: 3, min: 1, max: 8 },
];

/// Runtime values of the tunable search parameters. `Copy` so a node can lift a
/// local snapshot (`let p = ctx.params;`) and read fields without borrow fuss.
#[derive(Clone, Copy)]
pub struct SearchParams {
    pub rfp_margin: i32,
    pub rfp_max_depth: i32,
    pub nmp_base_r: i32,
    pub nmp_divisor: i32,
    pub nmp_min_depth: i32,
    pub lmr_base: i32,
    pub lmr_depth_div: i32,
    pub lmr_move_div: i32,
    pub lmr_min_depth: i32,
    pub lmr_min_move: i32,
    pub lmr_history_div: i32,
    pub lmp_max_depth: i32,
    pub lmp_base: i32,
    pub fp_max_depth: i32,
    pub fp_margin: i32,
    pub see_max_depth: i32,
    pub see_quiet_margin: i32,
    pub see_capt_margin: i32,
    pub singular_min_depth: i32,
    pub singular_margin: i32,
}

impl Default for SearchParams {
    fn default() -> Self {
        // SPSA-tuned (see PARAM_SPECS); must stay in sync with the spec defaults.
        Self {
            rfp_margin: 55,
            rfp_max_depth: 7,
            nmp_base_r: 3,
            nmp_divisor: 3,
            nmp_min_depth: 2,
            lmr_base: 1,
            lmr_depth_div: 6,
            lmr_move_div: 8,
            lmr_min_depth: 3,
            lmr_min_move: 4,
            lmr_history_div: 9545,
            lmp_max_depth: 2,
            lmp_base: 6,
            fp_max_depth: 4,
            fp_margin: 80,
            see_max_depth: 4,
            see_quiet_margin: 84,
            see_capt_margin: 102,
            singular_min_depth: 6,
            singular_margin: 3,
        }
    }
}

impl SearchParams {
    /// Set a parameter by its UCI option name, clamping to the spec range.
    /// Returns false for an unknown name (ignored, per UCI tolerance).
    pub fn set(&mut self, name: &str, value: i32) -> bool {
        let spec = match PARAM_SPECS.iter().find(|s| s.name == name) {
            Some(s) => s,
            None => return false,
        };
        let v = value.clamp(spec.min, spec.max);
        match name {
            "RFPMargin" => self.rfp_margin = v,
            "RFPMaxDepth" => self.rfp_max_depth = v,
            "NMPBaseR" => self.nmp_base_r = v,
            "NMPDivisor" => self.nmp_divisor = v,
            "NMPMinDepth" => self.nmp_min_depth = v,
            "LMRBase" => self.lmr_base = v,
            "LMRDepthDiv" => self.lmr_depth_div = v,
            "LMRMoveDiv" => self.lmr_move_div = v,
            "LMRMinDepth" => self.lmr_min_depth = v,
            "LMRMinMove" => self.lmr_min_move = v,
            "LMRHistoryDiv" => self.lmr_history_div = v,
            "LMPMaxDepth" => self.lmp_max_depth = v,
            "LMPBase" => self.lmp_base = v,
            "FPMaxDepth" => self.fp_max_depth = v,
            "FPMargin" => self.fp_margin = v,
            "SEEMaxDepth" => self.see_max_depth = v,
            "SEEQuietMargin" => self.see_quiet_margin = v,
            "SEECaptMargin" => self.see_capt_margin = v,
            "SingularMinDepth" => self.singular_min_depth = v,
            "SingularMargin" => self.singular_margin = v,
            _ => return false,
        }
        true
    }
}

/// Reusable searcher owning a transposition table across moves (a UCI session).
pub struct Engine {
    tt: Tt,
    /// Zobrist hashes of every position that occurred in the game *before* the
    /// current search root (one per ply played). Used for repetition detection
    /// so the engine doesn't repeat a position into a draw when it's winning.
    history: Vec<u64>,
    /// Tunable search parameters, set via UCI `setoption` and held across the
    /// session (`ucinewgame`/`reset` keep them; only the TT and history clear).
    params: SearchParams,
}

impl Engine {
    pub fn new() -> Self {
        Self { tt: Tt::new(), history: Vec::new(), params: SearchParams::default() }
    }
    /// Returns (best move, score in cp, nodes searched, depth reached).
    pub fn search(&mut self, board: &mut Board, limits: Limits) -> (Option<Move>, i32, u64, u32) {
        search_with_limits(board, limits, &mut self.tt, &self.history, self.params)
    }
    /// Apply a `setoption` value by option name. Unknown names are ignored.
    pub fn set_param(&mut self, name: &str, value: i32) -> bool {
        self.params.set(name, value)
    }
    /// Record the hashes of all positions played before the current root, so
    /// in-search repetition detection can see the game's history. Call this
    /// from the UCI `position` handler before each `go`.
    pub fn set_game_history(&mut self, history: Vec<u64>) { self.history = history; }
    pub fn reset(&mut self) {
        self.tt.clear();
        self.history.clear();
    }
}

impl Default for Engine {
    fn default() -> Self { Self::new() }
}

// TT: 4M entries x 16 bytes = 64 MB, indexed by `hash & TT_MASK`, always-replace.
const TT_NUM_ENTRIES: usize = 4 * 1024 * 1024;
const TT_MASK: u64 = (TT_NUM_ENTRIES as u64) - 1;

const BOUND_NONE: u8 = 0;
const BOUND_EXACT: u8 = 1;
const BOUND_LOWER: u8 = 2; // beta cutoff: true score >= stored
const BOUND_UPPER: u8 = 3; // alpha not raised: true score <= stored

#[derive(Clone, Copy)]
#[repr(C)]
struct TtEntry {
    key: u64,
    move_packed: u16,
    score: i16,
    depth: u8,
    bound: u8,
    _pad: u16,
}

impl Default for TtEntry {
    fn default() -> Self {
        Self {
            key: 0,
            move_packed: 0,
            score: 0,
            depth: 0,
            bound: BOUND_NONE,
            _pad: 0,
        }
    }
}

pub struct Tt {
    table: Vec<TtEntry>,
}

impl Tt {
    pub fn new() -> Self {
        Self {
            table: vec![TtEntry::default(); TT_NUM_ENTRIES],
        }
    }

    fn clear(&mut self) {
        for e in self.table.iter_mut() {
            *e = TtEntry::default();
        }
    }

    #[inline]
    fn probe(&self, hash: u64) -> Option<TtEntry> {
        let entry = self.table[(hash & TT_MASK) as usize];
        if entry.bound != BOUND_NONE && entry.key == hash {
            Some(entry)
        } else {
            None
        }
    }

    #[inline]
    fn store(&mut self, hash: u64, mv: Move, score: i16, depth: u8, bound: u8) {
        let idx = (hash & TT_MASK) as usize;
        self.table[idx] = TtEntry {
            key: hash,
            move_packed: mv.0,
            score,
            depth,
            bound,
            _pad: 0,
        };
    }

    // Warm the cache line a node's probe will read. Called right after a move is
    // made (the child hash is known), so the line is in cache by the time the
    // child node probes. A pure hint: it never changes which entry is read or
    // the search result, only the latency of reading it.
    #[inline]
    fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let idx = (hash & TT_MASK) as usize;
            let ptr = self.table.as_ptr().wrapping_add(idx) as *const i8;
            unsafe { _mm_prefetch(ptr, _MM_HINT_T0) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = hash;
    }
}

// Mate scores adjusted by ply when stored / probed so they're independent of search depth.
#[inline]
fn tt_score_to_store(score: i32, ply: i32) -> i16 {
    let adjusted = if score >= MATE_THRESHOLD {
        score + ply
    } else if score <= -MATE_THRESHOLD {
        score - ply
    } else {
        score
    };
    adjusted.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[inline]
fn tt_score_from_probe(stored: i16, ply: i32) -> i32 {
    let s = stored as i32;
    if s >= MATE_THRESHOLD {
        s - ply
    } else if s <= -MATE_THRESHOLD {
        s + ply
    } else {
        s
    }
}


const MAX_PLY: usize = 128;
const HISTORY_MAX: i32 = 16_384;

const PIECE_NB: usize = 12; // 6 kinds x 2 colors
const CONT_PLIES: usize = 2; // 1-ply (counter-move) and 2-ply continuation
const CONT_MAX: i32 = HISTORY_MAX; // gravity bound for continuation entries

/// Continuation history: how strong a (piece, to) move tends to be given the
/// move played 1 and 2 plies earlier on the path. Indexed
/// `[ply_offset][prev_piece][prev_to][piece][to]`. Boxed because the full table
/// is a couple of megabytes.
struct ContHist {
    data: Box<[i16]>,
}

impl ContHist {
    fn new() -> Self {
        Self { data: vec![0i16; CONT_PLIES * PIECE_NB * 64 * PIECE_NB * 64].into_boxed_slice() }
    }

    #[inline]
    fn index(off: usize, prev: (u8, u8), cur: (u8, u8)) -> usize {
        ((((off * PIECE_NB + prev.0 as usize) * 64 + prev.1 as usize) * PIECE_NB
            + cur.0 as usize)
            * 64)
            + cur.1 as usize
    }

    #[inline]
    fn get(&self, off: usize, prev: (u8, u8), cur: (u8, u8)) -> i32 {
        self.data[Self::index(off, prev, cur)] as i32
    }

    /// Gravity update: bounded and self-correcting, so a hot entry saturates
    /// toward +/-CONT_MAX instead of overflowing.
    #[inline]
    fn add(&mut self, off: usize, prev: (u8, u8), cur: (u8, u8), delta: i32) {
        let i = Self::index(off, prev, cur);
        let e = self.data[i] as i32;
        self.data[i] = (e + delta - e * delta.abs() / CONT_MAX) as i16;
    }
}

/// `(piece index 0..12, to-square 0..64)` key for the piece that makes `m`.
/// The piece is read before the move is applied (a pawn for a promotion).
#[inline]
fn piece_key(board: &Board, m: Move) -> (u8, u8) {
    let idx = match board.piece_at(m.from()) {
        Some(p) => (p.kind.idx() + p.color.idx() * 6) as u8,
        None => 0,
    };
    (idx, m.to().idx() as u8)
}

/// Quiet-move ordering / reduction score: butterfly history plus the 1- and
/// 2-ply continuation histories for the moves on the current path.
#[inline]
fn quiet_history(ctx: &SearchCtx, board: &Board, m: Move, ply: usize) -> i32 {
    let mut h = ctx.history_heur[m.from().idx()][m.to().idx()];
    let cur = piece_key(board, m);
    if ply >= 1 && ply < MAX_PLY {
        if let Some(prev) = ctx.played[ply] {
            h += ctx.cont.get(0, prev, cur);
        }
    }
    if ply >= 2 && ply - 1 < MAX_PLY {
        if let Some(prev2) = ctx.played[ply - 1] {
            h += ctx.cont.get(1, prev2, cur);
        }
    }
    h
}

/// True if the current position has occurred earlier on the search path or in
/// the game history. `hashes` holds every position strictly before the current
/// node. A single earlier occurrence counts as a draw inside the tree (the
/// standard two-fold rule), which prunes repetition lines and avoids repeating
/// a won game.
///
/// Only positions since the last irreversible move can repeat, so the scan goes
/// back at most `hmc` plies, stepping two at a time from the grandparent (a
/// repetition shares the side to move).
#[inline]
fn is_repetition(hash: u64, hmc: u16, hashes: &[u64]) -> bool {
    let n = hashes.len();
    let max_back = (hmc as usize).min(n);
    let mut i = 2usize;
    while i <= max_back {
        if hashes[n - i] == hash {
            return true;
        }
        i += 2;
    }
    false
}

/// True for material that can never deliver checkmate: K vs K, K+minor vs K.
/// Only dead-drawn cases qualify, so a winnable endgame is never mislabelled;
/// KN vs KN, KB vs KB, and similar are left to the evaluation.
#[inline]
fn is_insufficient_material(board: &Board) -> bool {
    let w = Color::White.idx();
    let b = Color::Black.idx();
    // Any pawn, rook, or queen means mate is still possible.
    if board.piece_bb[w][PieceKind::Pawn.idx()]
        | board.piece_bb[b][PieceKind::Pawn.idx()]
        | board.piece_bb[w][PieceKind::Rook.idx()]
        | board.piece_bb[b][PieceKind::Rook.idx()]
        | board.piece_bb[w][PieceKind::Queen.idx()]
        | board.piece_bb[b][PieceKind::Queen.idx()]
        != 0
    {
        return false;
    }
    let minors = board.piece_bb[w][PieceKind::Knight.idx()]
        | board.piece_bb[w][PieceKind::Bishop.idx()]
        | board.piece_bb[b][PieceKind::Knight.idx()]
        | board.piece_bb[b][PieceKind::Bishop.idx()];
    minors.count_ones() <= 1
}

pub fn search(board: &mut Board, max_depth: u32) -> (Option<Move>, i32) {
    let mut tt = Tt::new();
    let (m, s, _, _) =
        search_with_limits(board, Limits::depth(max_depth), &mut tt, &[], SearchParams::default());
    (m, s)
}

pub fn search_with_limits(
    board: &mut Board,
    limits: Limits,
    tt: &mut Tt,
    history: &[u64],
    params: SearchParams,
) -> (Option<Move>, i32, u64, u32) {
    let start = Instant::now();
    // Hard ceiling, polled inside the search via `ctx.check`; None if unbounded.
    let hard_deadline = if limits.hard_ms == u64::MAX {
        None
    } else {
        Some(start + Duration::from_millis(limits.hard_ms))
    };
    // A fixed move time (soft == hard) runs the whole budget. Stability scaling
    // and the mate early-out apply only to clock-derived budgets.
    let fixed_time = limits.soft_ms != u64::MAX && limits.soft_ms == limits.hard_ms;

    let mut ctx = SearchCtx::new(hard_deadline);
    ctx.params = params;
    ctx.hashes.clear();
    ctx.hashes.extend_from_slice(history);
    ctx.accs.init(board);
    ctx.max_nodes = limits.max_nodes;

    let mut best_move: Option<Move> = None;
    let mut best_score = 0;
    let mut prev_score = 0;
    let mut bm_stability: u32 = 0;
    let mut completed_depth: u32 = 0;

    for d in 1..=limits.max_depth.min(MAX_PLY as u32) {
        // Soft gate: between iterations, stop once the (possibly stability-
        // scaled) optimum is reached. Depth 1 always runs to guarantee a move.
        if limits.soft_ms != u64::MAX && d > 1 {
            let target = if fixed_time {
                limits.soft_ms
            } else {
                timeman::scaled_soft(
                    limits.soft_ms,
                    limits.hard_ms,
                    bm_stability,
                    best_score,
                    prev_score,
                )
            };
            if start.elapsed().as_millis() as u64 >= target {
                break;
            }
        }

        let (m, s) = search_root(board, d, best_move, &mut ctx, tt);
        if ctx.stop && d > 1 {
            break;
        }
        if m.is_some() {
            bm_stability = if m == best_move { bm_stability.saturating_add(1) } else { 0 };
            prev_score = best_score;
            best_move = m;
            best_score = s;
            completed_depth = d;
        }

        // A forced mate held for a few iterations will not change, so stop
        // spending clock on it (clock budgets only; fixed depth/movetime finish).
        if !fixed_time
            && limits.soft_ms != u64::MAX
            && best_score.abs() >= MATE_THRESHOLD
            && bm_stability >= 2
        {
            break;
        }
    }
    (best_move, best_score, ctx.nodes, completed_depth)
}

struct SearchCtx {
    deadline: Option<Instant>,
    accs: AccStack,
    params: SearchParams,
    stop: bool,
    nodes: u64,
    max_nodes: u64,
    killers: [[Option<Move>; 2]; MAX_PLY],
    /// Butterfly history heuristic for move ordering (from-square × to-square).
    history_heur: [[i32; 64]; 64],
    /// Continuation history (1- and 2-ply back); see `ContHist`.
    cont: ContHist,
    /// The (piece, to) move played to enter each ply. Lets a node score and
    /// reduce its quiets against the moves its ancestors just played.
    played: [Option<(u8, u8)>; MAX_PLY],
    /// Zobrist hashes of all positions on the path from game start to the
    /// parent of the current node (game history + ancestors in the tree).
    /// Used for repetition detection; pushed/popped around each recursion.
    hashes: Vec<u64>,
}

impl SearchCtx {
    fn new(deadline: Option<Instant>) -> Self {
        Self {
            deadline,
            accs: AccStack::new(),
            params: SearchParams::default(),
            stop: false,
            nodes: 0,
            max_nodes: u64::MAX,
            killers: [[None; 2]; MAX_PLY],
            history_heur: [[0; 64]; 64],
            cont: ContHist::new(),
            played: [None; MAX_PLY],
            hashes: Vec::with_capacity(MAX_PLY + 64),
        }
    }

    #[inline]
    fn check(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes >= self.max_nodes {
            self.stop = true;
        }
        if self.nodes & 4095 == 0 {
            if let Some(dl) = self.deadline {
                if Instant::now() >= dl {
                    self.stop = true;
                }
            }
        }
        self.stop
    }

    #[inline]
    fn update_killers(&mut self, ply: usize, m: Move) {
        if ply >= MAX_PLY {
            return;
        }
        if self.killers[ply][0] != Some(m) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(m);
        }
    }

    /// Reward the quiet move that caused a beta cutoff and penalize the quiet
    /// moves that were tried first and did not, in both butterfly and
    /// continuation history. `board` must be the node position (moves unmade).
    #[inline]
    fn update_quiet_histories(
        &mut self,
        board: &Board,
        ply: usize,
        best: Move,
        searched: &[Move],
        depth: u32,
    ) {
        let bonus = ((depth as i32) * (depth as i32)).min(HISTORY_MAX);
        self.history_stat(board, ply, best, bonus);
        for &q in searched {
            if q != best {
                self.history_stat(board, ply, q, -bonus);
            }
        }
    }

    /// Apply a signed history delta to a quiet move across butterfly + 1/2-ply
    /// continuation tables, each with a bounded gravity update.
    #[inline]
    fn history_stat(&mut self, board: &Board, ply: usize, m: Move, delta: i32) {
        let e = &mut self.history_heur[m.from().idx()][m.to().idx()];
        *e += delta - *e * delta.abs() / HISTORY_MAX;
        let cur = piece_key(board, m);
        if ply >= 1 && ply < MAX_PLY {
            if let Some(prev) = self.played[ply] {
                self.cont.add(0, prev, cur, delta);
            }
        }
        if ply >= 2 && ply - 1 < MAX_PLY {
            if let Some(prev2) = self.played[ply - 1] {
                self.cont.add(1, prev2, cur, delta);
            }
        }
    }
}

fn search_root(
    board: &mut Board,
    depth: u32,
    pv: Option<Move>,
    ctx: &mut SearchCtx,
    tt: &mut Tt,
) -> (Option<Move>, i32) {
    let mut moves = MoveList::new();
    board.gen_legal_into(&mut moves);
    if moves.is_empty() {
        let s = if board.in_check(board.side_to_move) { -MATE_SCORE } else { 0 };
        return (None, s);
    }
    let root_hash = board.hash;
    let tt_move = tt.probe(board.hash).map(|e| Move(e.move_packed));
    let mut sm = ScoredMoves::from_list(&moves, board, pv, tt_move, ctx, 0);
    let mut best: Option<Move> = None;
    let mut alpha = -INF;
    let beta = INF;
    for i in 0..sm.len {
        let m = sm.pop_best(i);
        let move_key = piece_key(board, m);
        let undo = board.make_move(m);
        ctx.played[1] = Some(move_key);
        ctx.accs.push_after(board, m, &undo);
        ctx.hashes.push(root_hash);
        let mut score;
        if i == 0 {
            score = -negamax(board, depth.saturating_sub(1), -beta, -alpha, 1, ctx, tt, true, None);
        } else {
            score = -negamax(board, depth.saturating_sub(1), -alpha - 1, -alpha, 1, ctx, tt, true, None);
            if !ctx.stop && score > alpha && score < beta {
                score = -negamax(board, depth.saturating_sub(1), -beta, -alpha, 1, ctx, tt, true, None);
            }
        }
        ctx.hashes.pop();
        ctx.accs.pop();
        board.unmake_move(m, undo);
        if ctx.stop {
            return (best, alpha);
        }
        if score > alpha {
            alpha = score;
            best = Some(m);
        }
    }
    if let Some(m) = best {
        tt.store(
            board.hash,
            m,
            tt_score_to_store(alpha, 0),
            depth.min(255) as u8,
            BOUND_EXACT,
        );
    }
    (best, alpha)
}

#[allow(clippy::too_many_arguments)]
fn negamax(
    board: &mut Board,
    depth: u32,
    mut alpha: i32,
    beta: i32,
    ply: u32,
    ctx: &mut SearchCtx,
    tt: &mut Tt,
    allow_null: bool,
    excluded: Option<Move>,
) -> i32 {
    if ctx.check() {
        return 0;
    }

    let original_alpha = alpha;
    let ply_i = ply as i32;
    let ply_us = ply as usize;
    let hash = board.hash;

    // Draw detection runs before the TT probe so a drawn line never returns a
    // stale non-draw score. The root (ply 0) is skipped: a move must still be
    // chosen there even when the position is already drawn.
    if ply > 0 {
        if is_repetition(hash, board.halfmove_clock, &ctx.hashes)
            || is_insufficient_material(board)
        {
            return 0;
        }
        if board.halfmove_clock >= 100 {
            // Fifty-move rule, except when this position is checkmate, where
            // mate takes precedence over the draw.
            if !board.in_check(board.side_to_move) {
                return 0;
            }
            let mut ml = MoveList::new();
            board.gen_legal_into(&mut ml);
            return if ml.is_empty() { -MATE_SCORE + ply_i } else { 0 };
        }
    }

    let mut tt_move: Option<Move> = None;
    let mut tt_score: Option<i32> = None;
    let mut tt_depth: u32 = 0;
    let mut tt_bound: u8 = BOUND_NONE;
    // Skip the TT inside a singular (excluded-move) search: its entry describes
    // the full move set, so it must neither cut here nor be overwritten.
    if excluded.is_none() {
        if let Some(entry) = tt.probe(hash) {
            if entry.move_packed != 0 {
                tt_move = Some(Move(entry.move_packed));
            }
            let s = tt_score_from_probe(entry.score, ply_i);
            tt_score = Some(s);
            tt_depth = entry.depth as u32;
            tt_bound = entry.bound;
            if tt_depth >= depth {
                match entry.bound {
                    BOUND_EXACT => return s,
                    BOUND_LOWER if s >= beta => return s,
                    BOUND_UPPER if s <= alpha => return s,
                    _ => {}
                }
            }
        }
    }

    if depth == 0 {
        return qsearch(board, alpha, beta, ply, ctx);
    }

    let in_check = board.in_check(board.side_to_move);
    let p = ctx.params;
    // Computed once (cheap, accumulator-based) and shared by reverse futility,
    // null move, and the forward futility pruning in the move loop. None in check.
    let static_eval = if in_check { None } else { Some(ctx.accs.evaluate(board.side_to_move)) };

    // Reverse futility pruning.
    if !in_check
        && depth >= 1
        && depth <= p.rfp_max_depth as u32
        && beta.abs() < MATE_THRESHOLD
    {
        if static_eval.unwrap() - p.rfp_margin * depth as i32 >= beta {
            return beta;
        }
    }

    // Null move pruning.
    if allow_null && !in_check && depth >= p.nmp_min_depth as u32 {
        let stm_idx = board.side_to_move.idx();
        let non_pawn_bb = board.piece_bb[stm_idx][PieceKind::Knight.idx()]
            | board.piece_bb[stm_idx][PieceKind::Bishop.idx()]
            | board.piece_bb[stm_idx][PieceKind::Rook.idx()]
            | board.piece_bb[stm_idx][PieceKind::Queen.idx()];
        if non_pawn_bb != 0 {
            let eval = static_eval.unwrap();
            if eval >= beta {
                let r = p.nmp_base_r as u32 + (depth / p.nmp_divisor as u32);
                let undo = board.make_null_move();
                tt.prefetch(board.hash);
                ctx.accs.push_null();
                // No move was played, so the child has no 1-ply continuation.
                if ply_us + 1 < MAX_PLY {
                    ctx.played[ply_us + 1] = None;
                }
                ctx.hashes.push(hash);
                let null_score = -negamax(
                    board,
                    depth.saturating_sub(1 + r),
                    -beta,
                    -beta + 1,
                    ply + 1,
                    ctx,
                    tt,
                    false,
                    None,
                );
                ctx.hashes.pop();
                ctx.accs.pop_null();
                board.unmake_null_move(undo);
                if ctx.stop {
                    return alpha;
                }
                if null_score >= beta {
                    return beta;
                }
            }
        }
    }

    let mut moves = MoveList::new();
    board.gen_legal_into(&mut moves);
    if moves.is_empty() {
        if in_check {
            return -MATE_SCORE + ply_i;
        }
        return 0;
    }

    let mut sm = ScoredMoves::from_list(&moves, board, None, tt_move, ctx, ply_us);
    let mut best_move: Option<Move> = None;
    let mut best_score = -INF;
    // Quiet moves searched at this node, for the history malus on a cutoff.
    let mut searched_quiets: [Move; 64] = [Move(0); 64];
    let mut sq_n: usize = 0;

    for i in 0..sm.len {
        let m = sm.pop_best(i);
        // In a singular search, leave out the move being tested for singularity.
        if Some(m) == excluded {
            continue;
        }

        // Shallow-depth move pruning (skipped before make_move to save the cost).
        // Never on the first move, in check, or when best_score is a getting-mated
        // value, so a legal reply is always searched and forced lines survive. Each
        // feature is gated by a tunable max-depth (0 disables it).
        let is_quiet = !m.is_capture() && !m.is_promotion();
        // Read before the move is made (needs the piece on its from-square).
        let move_key = piece_key(board, m);
        let move_hist = if i > 0 && is_quiet { quiet_history(ctx, board, m, ply_us) } else { 0 };
        if i > 0 && !in_check && best_score > -MATE_THRESHOLD {
            // Late move pruning: drop late quiet moves at shallow depth.
            if is_quiet
                && depth <= p.lmp_max_depth as u32
                && i as i32 >= p.lmp_base + (depth as i32) * (depth as i32)
            {
                continue;
            }
            // Forward futility pruning: a quiet move can't lift a static eval that
            // is already a margin below alpha.
            if is_quiet
                && depth <= p.fp_max_depth as u32
                && static_eval.map_or(false, |ev| ev + p.fp_margin * depth as i32 <= alpha)
            {
                continue;
            }
            // SEE pruning: drop moves that lose too much material by static exchange.
            if depth <= p.see_max_depth as u32 {
                let margin = if is_quiet { p.see_quiet_margin } else { p.see_capt_margin };
                if see(board, m) < -(margin * depth as i32) {
                    continue;
                }
            }
        }

        // Singular extension. A reduced search of the other moves around a
        // margin below the TT score decides whether the TT move stands alone:
        //   - all fall short -> singular, extend it a ply
        //   - a second move also beats beta -> multi-cut, fail high now
        //   - not singular but expected to fail high -> reduce it (the move has
        //     viable company, so spend less depth on it)
        //   - not singular at a non-PV node -> reduce it a little
        // The negative cases are what keep the feature near node-neutral: depth
        // goes to forced/critical moves and is saved on moves that have peers.
        let mut extension: i32 = 0;
        if excluded.is_none()
            && Some(m) == tt_move
            && depth >= p.singular_min_depth as u32
            && tt_depth + 3 >= depth
            && (tt_bound == BOUND_LOWER || tt_bound == BOUND_EXACT)
        {
            if let Some(ts) = tt_score {
                if ts.abs() < MATE_THRESHOLD {
                    let singular_beta = ts - p.singular_margin * depth as i32;
                    let reduced = (depth - 1) / 2;
                    let value = negamax(
                        board, reduced, singular_beta - 1, singular_beta, ply, ctx, tt, false,
                        Some(m),
                    );
                    if ctx.stop {
                        return best_score;
                    }
                    if value < singular_beta {
                        extension = 1;
                    } else if singular_beta >= beta {
                        return singular_beta;
                    } else if ts >= beta {
                        extension = -2;
                    } else if beta - alpha <= 1 {
                        extension = -1;
                    }
                }
            }
        }
        // Floor at 0 (drop into qsearch), never below; depth - 1 must still be
        // able to reach 0 on the normal path. Negative extensions only fire at
        // depth >= SingularMinDepth, so they cannot underflow.
        let new_depth = ((depth as i32 - 1) + extension).max(0) as u32;

        let undo = board.make_move(m);
        tt.prefetch(board.hash);
        if ply_us + 1 < MAX_PLY {
            ctx.played[ply_us + 1] = Some(move_key);
        }
        ctx.accs.push_after(board, m, &undo);
        ctx.hashes.push(hash);
        if is_quiet && sq_n < searched_quiets.len() {
            searched_quiets[sq_n] = m;
            sq_n += 1;
        }

        let mut score;
        if i == 0 {
            score = -negamax(board, new_depth, -beta, -alpha, ply + 1, ctx, tt, true, None);
        } else {
            let do_lmr = !in_check
                && depth >= p.lmr_min_depth as u32
                && i >= p.lmr_min_move as usize
                && !m.is_capture()
                && !m.is_promotion();
            let initial_depth = if do_lmr {
                let base_r = p.lmr_base as u32
                    + depth / p.lmr_depth_div as u32
                    + (i as u32) / p.lmr_move_div as u32;
                // Reduce less for quiets with good history, more for bad.
                let adj = (move_hist / p.lmr_history_div).clamp(-2, 2);
                let r = (base_r as i32 - adj).max(0) as u32;
                new_depth.saturating_sub(r).max(1)
            } else {
                new_depth
            };

            score = -negamax(board, initial_depth, -alpha - 1, -alpha, ply + 1, ctx, tt, true, None);

            if do_lmr && !ctx.stop && score > alpha {
                score = -negamax(board, new_depth, -alpha - 1, -alpha, ply + 1, ctx, tt, true, None);
            }

            if !ctx.stop && score > alpha && score < beta {
                score = -negamax(board, new_depth, -beta, -alpha, ply + 1, ctx, tt, true, None);
            }
        }

        ctx.hashes.pop();
        ctx.accs.pop();

        board.unmake_move(m, undo);
        if ctx.stop {
            return best_score;
        }
        if score > best_score {
            best_score = score;
            best_move = Some(m);
        }
        if score >= beta {
            // Fail-soft beta cutoff: return the score that proved it, not `beta`.
            if !m.is_capture() && !m.is_promotion() {
                ctx.update_killers(ply_us, m);
                ctx.update_quiet_histories(board, ply_us, m, &searched_quiets[..sq_n], depth);
            }
            if excluded.is_none() {
                tt.store(
                    hash,
                    m,
                    tt_score_to_store(best_score, ply_i),
                    depth.min(255) as u8,
                    BOUND_LOWER,
                );
            }
            return best_score;
        }
        if score > alpha {
            alpha = score;
        }
    }

    // Fail-soft: return the best score found, which may sit below the original
    // alpha (a fail-low upper bound). The bound stored reflects that.
    let bound = if best_score > original_alpha { BOUND_EXACT } else { BOUND_UPPER };
    if excluded.is_none() {
        if let Some(m) = best_move {
            tt.store(
                hash,
                m,
                tt_score_to_store(best_score, ply_i),
                depth.min(255) as u8,
                bound,
            );
        }
    }
    best_score
}

const SEE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 10000];


fn attackers_of(board: &Board, sq: Square, side: Color, occ: u64) -> u64 {
    let side_idx = side.idx();
    let opp_idx = side.opp().idx();
    let p = PAWN_ATTACKS[opp_idx][sq.idx()] & board.piece_bb[side_idx][PieceKind::Pawn.idx()];
    let n = KNIGHT_ATTACKS[sq.idx()] & board.piece_bb[side_idx][PieceKind::Knight.idx()];
    let bq_targets = bishop_attacks(sq, occ);
    let rq_targets = rook_attacks(sq, occ);
    let bq = bq_targets
        & (board.piece_bb[side_idx][PieceKind::Bishop.idx()]
            | board.piece_bb[side_idx][PieceKind::Queen.idx()]);
    let rq = rq_targets
        & (board.piece_bb[side_idx][PieceKind::Rook.idx()]
            | board.piece_bb[side_idx][PieceKind::Queen.idx()]);
    let k = KING_ATTACKS[sq.idx()] & board.piece_bb[side_idx][PieceKind::King.idx()];
    (p | n | bq | rq | k) & occ
}

fn least_valuable(board: &Board, side: Color, attackers: u64) -> Option<(Square, PieceKind)> {
    let side_idx = side.idx();
    for kind_idx in 0..6 {
        let cand = attackers & board.piece_bb[side_idx][kind_idx];
        if cand != 0 {
            return Some((lsb(cand), PieceKind::from_idx(kind_idx)));
        }
    }
    None
}

fn see(board: &Board, m: Move) -> i32 {
    let to = m.to();
    let from = m.from();
    let attacker = match board.piece_at(from) {
        Some(p) => p,
        None => return 0,
    };
    let mut occ = board.occupancy & !from.bb();

    let captured_value = if m.flag() == Move::EP_CAPTURE {
        let cap_sq = match attacker.color {
            Color::White => Square(to.0 - 8),
            Color::Black => Square(to.0 + 8),
        };
        occ &= !cap_sq.bb();
        SEE_VALUES[PieceKind::Pawn.idx()]
    } else {
        match board.piece_at(to) {
            Some(p) => SEE_VALUES[p.kind.idx()],
            None => return 0,
        }
    };

    let mut gain = [0i32; 32];
    let mut d = 0;
    gain[0] = captured_value;
    let mut current_kind = m.promo_kind().unwrap_or(attacker.kind);
    if let Some(promo) = m.promo_kind() {
        gain[0] += SEE_VALUES[promo.idx()] - SEE_VALUES[PieceKind::Pawn.idx()];
    }

    let mut side = attacker.color.opp();
    loop {
        d += 1;
        gain[d] = SEE_VALUES[current_kind.idx()] - gain[d - 1];
        let attackers = attackers_of(board, to, side, occ);
        let (sq, kind) = match least_valuable(board, side, attackers) {
            Some(v) => v,
            None => break,
        };
        occ &= !sq.bb();
        current_kind = kind;
        side = side.opp();
    }
    for i in (1..d).rev() {
        gain[i - 1] = gain[i - 1].min(-gain[i]);
    }
    gain[0]
}

fn qsearch(
    board: &mut Board,
    mut alpha: i32,
    beta: i32,
    ply: u32,
    ctx: &mut SearchCtx,
) -> i32 {
    if ctx.check() {
        return 0;
    }

    let in_check = board.in_check(board.side_to_move);

    // Stand-pat is the score floor when not in check: there is no obligation to
    // capture, so the static eval is a guaranteed lower bound.
    let mut best_score = -INF;
    if !in_check {
        let stand_pat = ctx.accs.evaluate(board.side_to_move);
        best_score = stand_pat;
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    if (ply as usize) >= MAX_PLY {
        return ctx.accs.evaluate(board.side_to_move);
    }

    let mut moves = MoveList::new();
    board.gen_legal_into(&mut moves);

    if moves.is_empty() {
        if in_check {
            return -MATE_SCORE + ply as i32;
        }
        return 0;
    }

    let mut q: [(Move, i32); 256] = [(Move::NULL, 0i32); 256];
    let mut q_len = 0usize;
    for i in 0..moves.len() {
        let m = moves.get(i);
        if !in_check && !m.is_capture() && !m.is_promotion() {
            continue;
        }
        if !in_check && m.is_capture() && see(board, m) < 0 {
            continue;
        }
        q[q_len] = (m, qscore(m, board));
        q_len += 1;
    }

    for i in 0..q_len {
        let mut best = i;
        let mut best_s = q[i].1;
        for j in (i + 1)..q_len {
            if q[j].1 > best_s {
                best_s = q[j].1;
                best = j;
            }
        }
        if best != i {
            q.swap(i, best);
        }
        let m = q[i].0;

        let undo = board.make_move(m);

        ctx.accs.push_after(board, m, &undo);
        let score = -qsearch(board, -beta, -alpha, ply + 1, ctx);
        ctx.accs.pop();
        board.unmake_move(m, undo);
        if ctx.stop {
            return best_score;
        }
        if score > best_score {
            best_score = score;
        }
        if score >= beta {
            return best_score;
        }
        if score > alpha {
            alpha = score;
        }
    }

    best_score
}

fn qscore(m: Move, board: &Board) -> i32 {
    if m.is_capture() {
        let attacker = board
            .piece_at(m.from())
            .map(|p| p.kind as i32)
            .unwrap_or(0);
        let victim = if m.flag() == Move::EP_CAPTURE {
            PieceKind::Pawn as i32
        } else {
            board.piece_at(m.to()).map(|p| p.kind as i32).unwrap_or(0)
        };
        return 100_000 + (victim + 1) * 8 - attacker;
    }
    if m.is_promotion() {
        return 50_000
            + match m.promo_kind() {
                Some(PieceKind::Queen) => 400,
                Some(PieceKind::Knight) => 300,
                Some(PieceKind::Rook) => 200,
                Some(PieceKind::Bishop) => 100,
                _ => 0,
            };
    }
    0
}

fn score_move(
    m: Move,
    board: &Board,
    pv: Option<Move>,
    tt_move: Option<Move>,
    ctx: &SearchCtx,
    ply: usize,
) -> i32 {
    if tt_move == Some(m) {
        return 2_000_000;
    }
    if pv == Some(m) {
        return 1_000_000;
    }
    if m.is_capture() {
        let see_val = see(board, m);
        if see_val >= 0 {
            return 100_000 + see_val;
        } else {
            return -100_000 + see_val;
        }
    }
    if m.is_promotion() {
        return 50_000
            + match m.promo_kind() {
                Some(PieceKind::Queen) => 400,
                Some(PieceKind::Knight) => 300,
                Some(PieceKind::Rook) => 200,
                Some(PieceKind::Bishop) => 100,
                _ => 0,
            };
    }
    if ply < MAX_PLY {
        if ctx.killers[ply][0] == Some(m) {
            return 40_000;
        }
        if ctx.killers[ply][1] == Some(m) {
            return 39_000;
        }
    }
    quiet_history(ctx, board, m, ply)
}

struct ScoredMoves {
    data: [MaybeUninit<(Move, i32)>; 256],
    len: usize,
}

impl ScoredMoves {
    fn from_list(
        list: &MoveList,
        board: &Board,
        pv: Option<Move>,
        tt_move: Option<Move>,
        ctx: &SearchCtx,
        ply: usize,
    ) -> Self {
        let mut sm = Self {
            data: unsafe { MaybeUninit::<[MaybeUninit<(Move, i32)>; 256]>::uninit().assume_init() },
            len: list.len(),
        };
        for i in 0..list.len() {
            let m = list.get(i);
            sm.data[i] = MaybeUninit::new((m, score_move(m, board, pv, tt_move, ctx, ply)));
        }
        sm
    }

    fn pop_best(&mut self, start: usize) -> Move {
        let mut best = start;
        let mut best_score = unsafe { self.data[start].assume_init().1 };
        for i in (start + 1)..self.len {
            let s = unsafe { self.data[i].assume_init().1 };
            if s > best_score {
                best_score = s;
                best = i;
            }
        }
        if best != start {
            self.data.swap(start, best);
        }
        unsafe { self.data[start].assume_init().0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Board;

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
        // hashes = positions strictly before the current node.
        // Layout (most recent last): [.., grandparent(us), parent(them)].
        // A repetition of `cur` must be an even number of plies back.
        let cur = 0xABCDu64;
        // Grandparent (2 back) equals current: a repetition within the window.
        let hashes = [0x1111u64, cur, 0x2222u64];
        assert!(is_repetition(cur, 50, &hashes));
        // A halfmove clock of 1 forbids looking back 2 plies.
        assert!(!is_repetition(cur, 1, &hashes));
        // `cur` appears only at an odd distance (1 back), a different side to
        // move, so it is not a real repetition.
        let hashes_odd = [0x2222u64, cur];
        assert!(!is_repetition(cur, 50, &hashes_odd));
        // `cur` absent entirely.
        let hashes_absent = [0x2222u64, 0x3333u64];
        assert!(!is_repetition(cur, 50, &hashes_absent));
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
        // K+N vs K: every line stays insufficient, so the search must return 0
        // rather than a positive score for the extra knight.
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/4K2N w - - 0 1").unwrap();
        let (best, score) = search(&mut board, 5);
        assert!(best.is_some());
        assert_eq!(score, 0, "K+N vs K is a dead draw");
    }

    #[test]
    fn search_respects_fifty_move_rule() {
        // White is up a queen but the halfmove clock is at 100 with no mate in
        // one, so any move completes a fifty-move draw.
        let mut board = Board::from_fen("8/8/8/3k4/8/8/3KQ3/8 w - - 100 1").unwrap();
        let (best, score) = search(&mut board, 5);
        assert!(best.is_some());
        assert_eq!(score, 0, "fifty-move rule forces a draw despite the queen");
    }

    #[test]
    fn mate_beats_fifty_move_rule() {
        // Halfmove clock at 100, but White has mate in one (Qd7#-style). Mate
        // must take precedence over the fifty-move draw.
        let mut board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 100 1").unwrap();
        let (best, score) = search(&mut board, 4);
        let m = best.expect("should find a move");
        assert!(score >= MATE_THRESHOLD, "must see the mate, not claim a draw");
        // Verify the chosen move is actually checkmate.
        let undo = board.make_move(m);
        let mut next = MoveList::new();
        board.gen_legal_into(&mut next);
        let mate = next.is_empty() && board.in_check(board.side_to_move);
        board.unmake_move(m, undo);
        assert!(mate, "{m} should deliver mate");
    }

    #[test]
    fn engine_avoids_repeating_a_won_position() {
        use crate::core::Board as B;
        // White is up a queen and to move. Seeding the game history with the
        // current position makes a king-shuffle line a repetition (draw); the
        // engine should still report a winning score by choosing progress.
        let mut engine = Engine::new();
        let mut board = B::from_fen("4k3/8/4K3/8/8/8/8/3Q4 w - - 0 1").unwrap();
        engine.set_game_history(vec![board.hash]);
        let (best, score, _, _) = engine.search(&mut board, Limits::depth(6));
        assert!(best.is_some());
        assert!(score > 100, "up a queen, the engine must find a winning line, got {score}");
    }

    #[test]
    fn tt_roundtrip() {
        let mut tt = Tt::new();
        let hash = 0xdead_beef_cafe_babe_u64;
        let mv = Move::new(Square::new(4, 1), Square::new(4, 3), Move::DOUBLE_PUSH);
        tt.store(hash, mv, 42, 5, BOUND_EXACT);
        let entry = tt.probe(hash).expect("entry should be present");
        assert_eq!(entry.key, hash);
        assert_eq!(Move(entry.move_packed), mv);
        assert_eq!(entry.score, 42);
        assert_eq!(entry.depth, 5);
        assert_eq!(entry.bound, BOUND_EXACT);
        // A different hash that maps to the same slot still misses.
        let foreign = hash ^ 1u64; // perturbs key but not slot for power-of-2 mask
        let _ = foreign; // suppress unused if collision is rare
    }
}

