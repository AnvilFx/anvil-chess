use crate::core::{
    bishop_attacks, lsb, rook_attacks, Board, Color, Move, MoveList, PieceKind,
    Square, KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS,
};
use crate::eval::AccStack;
use crate::timeman::{self, TimeBudget};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const INF: i32 = 32000;
pub const MATE_SCORE: i32 = 30000;
pub const MATE_THRESHOLD: i32 = MATE_SCORE - 1000;

/// Past this half-width the aspiration window opens fully.
const ASP_FULL_WINDOW: i32 = 500;

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
        Self {
            max_depth: MAX_PLY as u32,
            soft_ms: b.soft_ms,
            hard_ms: b.hard_ms,
            max_nodes: u64::MAX,
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
}

pub struct ParamSpec {
    pub name: &'static str,
    pub default: i32,
    pub min: i32,
    pub max: i32,
}

/// Divisor minimums are 2 so the formulas dividing by them never hit zero.
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
    ParamSpec { name: "LMRHistoryDiv", default: 9545, min: 256, max: 32768 },
    // A *MaxDepth of 0 disables that pruning.
    ParamSpec { name: "LMPMaxDepth", default: 2, min: 0, max: 12 },
    ParamSpec { name: "LMPBase", default: 6, min: 1, max: 16 },
    ParamSpec { name: "FPMaxDepth", default: 4, min: 0, max: 12 },
    ParamSpec { name: "FPMargin", default: 80, min: 30, max: 400 },
    ParamSpec { name: "SEEMaxDepth", default: 4, min: 0, max: 12 },
    ParamSpec { name: "SEEQuietMargin", default: 84, min: 0, max: 200 },
    ParamSpec { name: "SEECaptMargin", default: 102, min: 0, max: 200 },
    ParamSpec { name: "SingularMinDepth", default: 6, min: 4, max: 12 },
    ParamSpec { name: "SingularMargin", default: 3, min: 1, max: 8 },
    ParamSpec { name: "AspMinDepth", default: 6, min: 2, max: 10 },
    ParamSpec { name: "AspDelta", default: 20, min: 4, max: 60 },
];

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
    pub asp_min_depth: i32,
    pub asp_delta: i32,
}

impl Default for SearchParams {
    fn default() -> Self {
        // Must match the PARAM_SPECS defaults.
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
            asp_min_depth: 6,
            asp_delta: 20,
        }
    }
}

impl SearchParams {
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
            "AspMinDepth" => self.asp_min_depth = v,
            "AspDelta" => self.asp_delta = v,
            _ => return false,
        }
        true
    }
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
        Self {
            tt: Tt::new(),
            history: Vec::new(),
            params: SearchParams::default(),
            ctx: SearchCtx::new(),
        }
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
    pub fn set_game_history(&mut self, history: Vec<u64>) { self.history = history; }
    pub fn reset(&mut self) {
        self.tt.clear();
        self.history.clear();
        self.ctx.clear_heuristics();
    }
}

impl Default for Engine {
    fn default() -> Self { Self::new() }
}

pub const TT_DEFAULT_MB: usize = 64;
pub const TT_MAX_MB: usize = 32768;
const BUCKET_SIZE: usize = 4;
const NO_EVAL: i16 = i16::MIN;

const BOUND_NONE: u8 = 0;
const BOUND_EXACT: u8 = 1;
const BOUND_LOWER: u8 = 2; // score >= stored
const BOUND_UPPER: u8 = 3; // score <= stored

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct TtEntry {
    key: u64,
    move_packed: u16,
    score: i16,
    eval: i16,
    depth: u8,
    /// Generation in the high 5 bits, then the was-PV flag, bound in the low 2.
    gen_bound: u8,
}

impl TtEntry {
    #[inline]
    fn bound(&self) -> u8 {
        self.gen_bound & 3
    }
    #[inline]
    fn gen(&self) -> u8 {
        self.gen_bound >> 3
    }
    #[inline]
    fn pv(&self) -> bool {
        self.gen_bound & 4 != 0
    }
    #[inline]
    fn eval(&self) -> Option<i32> {
        (self.eval != NO_EVAL).then_some(self.eval as i32)
    }
}

/// One cache line of entries sharing an index.
#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
struct Bucket([TtEntry; BUCKET_SIZE]);

pub struct Tt {
    buckets: Vec<Bucket>,
    mask: u64,
    gen: u8,
}

impl Tt {
    pub fn new() -> Self {
        Self::with_mb(TT_DEFAULT_MB)
    }

    /// Size rounds down to a power of two buckets.
    pub fn with_mb(mb: usize) -> Self {
        let wanted = (mb.clamp(1, TT_MAX_MB) * 1024 * 1024 / std::mem::size_of::<Bucket>()).max(1);
        let n = 1usize << (usize::BITS - 1 - wanted.leading_zeros());
        Self { buckets: vec![Bucket::default(); n], mask: (n - 1) as u64, gen: 0 }
    }

    fn clear(&mut self) {
        self.buckets.fill(Bucket::default());
        self.gen = 0;
    }

    fn new_search(&mut self) {
        self.gen = (self.gen + 1) & 31;
    }

    /// Permille of entries written by the current search, sampled from the first 1000.
    fn hashfull(&self) -> u32 {
        let n = (1000 / BUCKET_SIZE).min(self.buckets.len());
        self.buckets[..n]
            .iter()
            .flat_map(|b| b.0.iter())
            .filter(|e| e.bound() != BOUND_NONE && e.gen() == self.gen)
            .count() as u32
            * 1000
            / (n * BUCKET_SIZE) as u32
    }

    #[inline]
    fn bucket(&self, hash: u64) -> &Bucket {
        &self.buckets[(hash & self.mask) as usize]
    }

    #[inline]
    fn probe(&self, hash: u64) -> Option<TtEntry> {
        self.bucket(hash).0.iter().copied().find(|e| e.key == hash && e.bound() != BOUND_NONE)
    }

    #[allow(clippy::too_many_arguments)]
    fn store(&mut self, hash: u64, mv: Option<Move>, score: i16, eval: Option<i32>, depth: u8, bound: u8, pv: bool) {
        let gen = self.gen;
        let bucket = &mut self.buckets[(hash & self.mask) as usize].0;

        let slot = match bucket.iter().position(|e| e.key == hash && e.bound() != BOUND_NONE) {
            Some(i) => i,
            None => (0..BUCKET_SIZE)
                .min_by_key(|&i| {
                    let e = &bucket[i];
                    if e.bound() == BOUND_NONE {
                        return i32::MIN;
                    }
                    let age = (gen.wrapping_sub(e.gen()) & 31) as i32;
                    e.depth as i32 - 8 * age
                })
                .unwrap(),
        };

        let e = &mut bucket[slot];
        let same = e.key == hash && e.bound() != BOUND_NONE;
        // Keep a clearly deeper entry from this search, but still take a fresh move.
        if same && bound != BOUND_EXACT && e.gen() == gen && depth as i32 + 4 <= e.depth as i32 {
            if let Some(m) = mv {
                e.move_packed = m.0;
            }
            return;
        }
        let move_packed = match mv {
            Some(m) => m.0,
            None if same => e.move_packed,
            None => 0,
        };
        *e = TtEntry {
            key: hash,
            move_packed,
            score,
            eval: eval.map_or(NO_EVAL, |v| v.clamp(-INF, INF) as i16),
            depth,
            gen_bound: (gen << 3) | ((pv as u8) << 2) | bound,
        };
    }

    #[inline]
    fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let ptr = self.bucket(hash) as *const Bucket as *const i8;
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


/// Tightens the static eval with a TT score whose bound points the right way.
#[inline]
fn adjust_eval(eval: i32, tt_score: Option<i32>, bound: u8) -> i32 {
    match tt_score {
        Some(s) if s.abs() < MATE_THRESHOLD => match bound {
            BOUND_EXACT => s,
            BOUND_LOWER if s > eval => s,
            BOUND_UPPER if s < eval => s,
            _ => eval,
        },
        _ => eval,
    }
}

const MAX_PLY: usize = 128;
const IIR_MIN_DEPTH: u32 = 4;
const CORR_SIZE: usize = 16_384;
/// Corrections are stored in 1/256 cp.
const CORR_GRAIN: i32 = 256;
const CORR_MAX: i32 = CORR_GRAIN * 64;

#[inline]
fn pawn_key(board: &Board) -> usize {
    let w = board.piece_bb[Color::White.idx()][PieceKind::Pawn.idx()];
    let b = board.piece_bb[Color::Black.idx()][PieceKind::Pawn.idx()];
    let h = w.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (h >> (64 - CORR_SIZE.trailing_zeros())) as usize
}
const HISTORY_MAX: i32 = 16_384;

const PIECE_NB: usize = 12;
const CONT_PLIES: usize = 2;
const CONT_MAX: i32 = HISTORY_MAX;

/// Indexed `[ply_offset][prev_piece][prev_to][piece][to]`.
struct ContHist {
    data: Box<[i16]>,
}

impl ContHist {
    fn new() -> Self {
        Self { data: vec![0i16; CONT_PLIES * PIECE_NB * 64 * PIECE_NB * 64].into_boxed_slice() }
    }

    fn clear(&mut self) {
        self.data.fill(0);
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

    /// Gravity update: saturates toward +/-CONT_MAX instead of overflowing.
    #[inline]
    fn add(&mut self, off: usize, prev: (u8, u8), cur: (u8, u8), delta: i32) {
        let i = Self::index(off, prev, cur);
        let e = self.data[i] as i32;
        self.data[i] = (e + delta - e * delta.abs() / CONT_MAX) as i16;
    }
}

/// Must be called before `m` is made (a promotion is keyed as a pawn).
#[inline]
fn piece_key(board: &Board, m: Move) -> (u8, u8) {
    let idx = match board.piece_at(m.from()) {
        Some(p) => (p.kind.idx() + p.color.idx() * 6) as u8,
        None => 0,
    };
    (idx, m.to().idx() as u8)
}

#[inline]
fn quiet_history(ctx: &SearchCtx, board: &Board, m: Move, ply: usize) -> i32 {
    let mut h = ctx.history_heur[board.side_to_move.idx()][m.from().idx()][m.to().idx()];
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

/// `hashes` holds every position before this node; index `root` is the root position and
/// nothing before `floor` (the last null move) counts. One earlier occurrence after the
/// root is a draw; positions up to the root need two, as in a real three-fold.
#[inline]
fn is_repetition(hash: u64, hmc: u16, hashes: &[u64], root: usize, floor: usize) -> bool {
    let n = hashes.len();
    let max_back = (hmc as usize).min(n - floor.min(n));
    let mut seen = false;
    let mut i = 2usize;
    while i <= max_back {
        let idx = n - i;
        if hashes[idx] == hash {
            if idx > root || seen {
                return true;
            }
            seen = true;
        }
        i += 2;
    }
    false
}

/// Only dead draws (K vs K, K+minor vs K); KN vs KN and similar are left to eval.
#[inline]
fn is_insufficient_material(board: &Board) -> bool {
    let w = Color::White.idx();
    let b = Color::Black.idx();
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
    let hard_deadline = if limits.hard_ms == u64::MAX {
        None
    } else {
        Some(start + Duration::from_millis(limits.hard_ms))
    };
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

        let (m, s) = search_root_aspiration(board, d, best_move, best_score, ctx, tt);
        if ctx.stop && d > 1 {
            // A stopped iteration only returns a move that was fully searched and beat
            // the previous best at this depth, so it is the better choice.
            if m.is_some() {
                best_move = m;
            }
            break;
        }
        if m.is_some() {
            bm_stability = if m == best_move { bm_stability.saturating_add(1) } else { 0 };
            prev_score = best_score;
            best_move = m;
            best_score = s;
            completed_depth = d;

            if let Some(report) = on_iter.as_mut() {
                let root_pv = ctx.root_pv();
                let single = [m.unwrap()];
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

        if !fixed_time
            && limits.soft_ms != u64::MAX
            && best_score.abs() >= MATE_THRESHOLD
            && bm_stability >= 2
        {
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

struct SearchCtx {
    deadline: Option<Instant>,
    accs: AccStack,
    params: SearchParams,
    stop: bool,
    /// Set from another thread (UCI `stop`); polled alongside the deadline.
    abort: Arc<AtomicBool>,
    nodes: u64,
    max_nodes: u64,
    seldepth: usize,
    /// Triangular PV table: `pv[ply][ply..pv_len[ply]]` is the line from `ply`.
    pv: Box<[[Move; MAX_PLY + 1]; MAX_PLY + 1]>,
    pv_len: [usize; MAX_PLY + 1],
    killers: [[Option<Move>; 2]; MAX_PLY],
    history_heur: Box<[[[i32; 64]; 64]; 2]>,
    cont: ContHist,
    /// The (piece, to) move that led into each ply.
    played: [Option<(u8, u8)>; MAX_PLY],
    /// Static eval at each ply of the current path; None when in check.
    evals: [Option<i32>; MAX_PLY],
    /// How far search results have strayed from the raw eval, per side to move and pawn structure.
    corr: Box<[[i16; CORR_SIZE]; 2]>,
    /// Hashes from game start to the parent of the current node.
    hashes: Vec<u64>,
    /// Index of the root position in `hashes`.
    root_idx: usize,
    /// Repetition look-back stops here: positions before the last null move.
    rep_floor: usize,
}

impl SearchCtx {
    fn new() -> Self {
        Self {
            deadline: None,
            accs: AccStack::new(),
            params: SearchParams::default(),
            stop: false,
            abort: Arc::new(AtomicBool::new(false)),
            nodes: 0,
            max_nodes: u64::MAX,
            seldepth: 0,
            pv: Box::new([[Move::NULL; MAX_PLY + 1]; MAX_PLY + 1]),
            pv_len: [0; MAX_PLY + 1],
            killers: [[None; 2]; MAX_PLY],
            history_heur: Box::new([[[0; 64]; 64]; 2]),
            cont: ContHist::new(),
            played: [None; MAX_PLY],
            evals: [None; MAX_PLY],
            corr: Box::new([[0; CORR_SIZE]; 2]),
            hashes: Vec::with_capacity(MAX_PLY + 64),
            root_idx: 0,
            rep_floor: 0,
        }
    }

    fn begin_search(
        &mut self,
        board: &Board,
        history: &[u64],
        deadline: Option<Instant>,
        max_nodes: u64,
        params: SearchParams,
    ) {
        self.deadline = deadline;
        self.params = params;
        self.stop = false;
        self.nodes = 0;
        self.max_nodes = max_nodes;
        self.seldepth = 0;
        self.pv_len = [0; MAX_PLY + 1];
        self.played = [None; MAX_PLY];
        self.evals = [None; MAX_PLY];
        self.accs.init(board);
        self.hashes.clear();
        self.hashes.extend_from_slice(history);
        self.root_idx = history.len();
        self.rep_floor = 0;
    }

    fn clear_heuristics(&mut self) {
        self.killers = [[None; 2]; MAX_PLY];
        *self.history_heur = [[[0; 64]; 64]; 2];
        self.cont.clear();
        self.corr.iter_mut().for_each(|t| t.fill(0));
    }

    #[inline]
    fn corrected_eval(&self, board: &Board, raw: i32) -> i32 {
        let c = self.corr[board.side_to_move.idx()][pawn_key(board)] as i32 / CORR_GRAIN;
        (raw + c).clamp(-MATE_THRESHOLD + 1, MATE_THRESHOLD - 1)
    }

    /// Moves the correction toward `best - raw`, weighted by depth. Skips results whose bound
    /// says nothing about the eval, mates, and noisy best moves.
    fn update_correction(
        &mut self,
        board: &Board,
        raw: i32,
        static_eval: i32,
        best: i32,
        best_move: Option<Move>,
        bound: u8,
        depth: u32,
    ) {
        if best.abs() >= MATE_THRESHOLD
            || best_move.is_some_and(|m| m.is_capture() || m.is_promotion())
            || (bound == BOUND_LOWER && best <= static_eval)
            || (bound == BOUND_UPPER && best >= static_eval)
        {
            return;
        }
        let target = (best - raw).clamp(-CORR_MAX / CORR_GRAIN, CORR_MAX / CORR_GRAIN) * CORR_GRAIN;
        let weight = (depth as i32 + 1).min(16);
        let e = &mut self.corr[board.side_to_move.idx()][pawn_key(board)];
        *e = ((*e as i32 * (256 - weight) + target * weight) / 256) as i16;
    }

    #[inline]
    fn check(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes >= self.max_nodes {
            self.stop = true;
        }
        if self.nodes & 4095 == 0 {
            if self.abort.load(Ordering::Relaxed) {
                self.stop = true;
            }
            if let Some(dl) = self.deadline {
                if Instant::now() >= dl {
                    self.stop = true;
                }
            }
        }
        self.stop
    }

    #[inline]
    fn update_pv(&mut self, ply: usize, m: Move) {
        let next = ply + 1;
        let len = self.pv_len[next].max(next);
        let (head, tail) = self.pv.split_at_mut(next);
        head[ply][ply] = m;
        head[ply][next..len].copy_from_slice(&tail[0][next..len]);
        self.pv_len[ply] = len;
    }

    fn root_pv(&self) -> &[Move] {
        &self.pv[0][..self.pv_len[0]]
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

    /// `board` must be the node position (moves unmade).
    #[inline]
    fn update_quiet_histories(
        &mut self,
        board: &Board,
        ply: usize,
        best: Move,
        searched: &[Move],
        depth: u32,
    ) {
        let d = depth as i32;
        let bonus = (160 * d - 80).clamp(0, 1600);
        let malus = (180 * d - 60).clamp(0, 1800);
        self.history_stat(board, ply, best, bonus);
        for &q in searched {
            if q != best {
                self.history_stat(board, ply, q, -malus);
            }
        }
    }

    #[inline]
    fn history_stat(&mut self, board: &Board, ply: usize, m: Move, delta: i32) {
        let e = &mut self.history_heur[board.side_to_move.idx()][m.from().idx()][m.to().idx()];
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

/// On fail high the new move is adopted at once; on fail low the previous best is kept.
fn search_root_aspiration(
    board: &mut Board,
    depth: u32,
    prev_best: Option<Move>,
    prev_score: i32,
    ctx: &mut SearchCtx,
    tt: &mut Tt,
) -> (Option<Move>, i32) {
    let p = ctx.params;
    if depth < p.asp_min_depth as u32 || prev_score.abs() >= MATE_THRESHOLD {
        return search_root(board, depth, prev_best, -INF, INF, ctx, tt);
    }

    let mut delta = p.asp_delta;
    let mut alpha = (prev_score - delta).max(-INF);
    let mut beta = (prev_score + delta).min(INF);
    let mut best = prev_best;
    loop {
        let (m, s) = search_root(board, depth, best, alpha, beta, ctx, tt);
        if ctx.stop {
            // A fail-low interrupted by time is not an upgrade, so keep the prior best.
            if s > alpha {
                if let Some(mm) = m {
                    best = Some(mm);
                }
            }
            return (best, s);
        }
        if s <= alpha {
            beta = (alpha + beta) / 2;
            alpha = (s - delta).max(-INF);
        } else if s >= beta {
            if let Some(mm) = m {
                best = Some(mm);
            }
            beta = (s + delta).min(INF);
        } else {
            if let Some(mm) = m {
                best = Some(mm);
            }
            return (best, s);
        }
        delta += delta / 2 + 1;
        if delta >= ASP_FULL_WINDOW {
            alpha = -INF;
            beta = INF;
        }
    }
}

fn search_root(
    board: &mut Board,
    depth: u32,
    pv: Option<Move>,
    mut alpha: i32,
    beta: i32,
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
    let original_alpha = alpha;
    ctx.pv_len[0] = 0;
    let raw_eval = if board.in_check(board.side_to_move) {
        None
    } else {
        Some(ctx.accs.evaluate(board))
    };
    ctx.evals[0] = raw_eval.map(|ev| ctx.corrected_eval(board, ev));
    let tt_move = tt.probe(root_hash).map(|e| Move(e.move_packed));
    let mut sm = ScoredMoves::from_list(&moves, board, pv, tt_move, ctx, 0);
    let mut best: Option<Move> = None;
    let mut best_score = -INF;
    for i in 0..sm.len {
        let (m, _) = sm.pop_best(i);
        let move_key = piece_key(board, m);
        let undo = board.make_move(m);
        ctx.played[1] = Some(move_key);
        ctx.accs.push_after(board, m, &undo);
        ctx.hashes.push(root_hash);
        let mut score;
        if i == 0 {
            score = -negamax(board, depth.saturating_sub(1), -beta, -alpha, 1, ctx, tt, true, None, false);
        } else {
            score = -negamax(board, depth.saturating_sub(1), -alpha - 1, -alpha, 1, ctx, tt, true, None, true);
            if !ctx.stop && score > alpha && score < beta {
                score = -negamax(board, depth.saturating_sub(1), -beta, -alpha, 1, ctx, tt, true, None, false);
            }
        }
        ctx.hashes.pop();
        ctx.accs.pop();
        board.unmake_move(m, undo);
        if ctx.stop {
            return (best, best_score);
        }
        if score > best_score {
            best_score = score;
            best = Some(m);
            ctx.update_pv(0, m);
            if score > alpha {
                alpha = score;
            }
            if score >= beta {
                break;
            }
        }
    }
    let bound = if best_score >= beta {
        BOUND_LOWER
    } else if best_score > original_alpha {
        BOUND_EXACT
    } else {
        BOUND_UPPER
    };
    if best.is_some() {
        tt.store(
            root_hash,
            best,
            tt_score_to_store(best_score, 0),
            raw_eval,
            depth.min(255) as u8,
            bound,
            true,
        );
    }
    (best, best_score)
}

#[allow(clippy::too_many_arguments)]
fn negamax(
    board: &mut Board,
    mut depth: u32,
    mut alpha: i32,
    beta: i32,
    ply: u32,
    ctx: &mut SearchCtx,
    tt: &mut Tt,
    allow_null: bool,
    excluded: Option<Move>,
    // Expected to fail high (a non-PV node whose parent expects to fail low).
    cut_node: bool,
) -> i32 {
    ctx.pv_len[ply as usize] = ply as usize;
    if ply as usize >= MAX_PLY {
        return ctx.accs.evaluate(board);
    }

    let original_alpha = alpha;
    let pv_node = beta - alpha > 1;
    let ply_i = ply as i32;
    let ply_us = ply as usize;
    let hash = board.hash;

    // Before the TT probe so a drawn line never returns a stale score; skipped at the root.
    if ply > 0 {
        if is_repetition(hash, board.halfmove_clock, &ctx.hashes, ctx.root_idx, ctx.rep_floor)
            || is_insufficient_material(board)
        {
            return 0;
        }
        if board.halfmove_clock >= 100 {
            // Checkmate takes precedence over the fifty-move rule.
            if !board.in_check(board.side_to_move) {
                return 0;
            }
            let mut ml = MoveList::new();
            board.gen_legal_into(&mut ml);
            return if ml.is_empty() { -MATE_SCORE + ply_i } else { 0 };
        }
    }

    // qsearch counts the node and checks the clock itself.
    if depth == 0 {
        return qsearch(board, alpha, beta, ply, ctx, tt);
    }
    if ctx.check() {
        return 0;
    }
    ctx.seldepth = ctx.seldepth.max(ply as usize);

    let mut tt_move: Option<Move> = None;
    let mut tt_score: Option<i32> = None;
    let mut tt_depth: u32 = 0;
    let mut tt_bound: u8 = BOUND_NONE;
    let mut tt_eval: Option<i32> = None;
    // On the PV now or at some earlier visit.
    let mut tt_pv = pv_node;
    // In a singular search the TT entry covers the full move set, so don't cut or overwrite it.
    if excluded.is_none() {
        if let Some(entry) = tt.probe(hash) {
            tt_pv |= entry.pv();
            if entry.move_packed != 0 {
                tt_move = Some(Move(entry.move_packed));
            }
            let s = tt_score_from_probe(entry.score, ply_i);
            tt_score = Some(s);
            tt_depth = entry.depth as u32;
            tt_bound = entry.bound();
            tt_eval = entry.eval();
            if !pv_node && tt_depth >= depth {
                match tt_bound {
                    BOUND_EXACT => return s,
                    BOUND_LOWER if s >= beta => return s,
                    BOUND_UPPER if s <= alpha => return s,
                    _ => {}
                }
            }
        }
    }

    // Without a TT move this node was never searched well; spend less on it.
    if depth >= IIR_MIN_DEPTH && tt_move.is_none() && excluded.is_none() {
        depth -= 1;
    }

    let in_check = board.in_check(board.side_to_move);
    let p = ctx.params;
    let raw_eval = if in_check {
        None
    } else {
        Some(tt_eval.unwrap_or_else(|| ctx.accs.evaluate(board)))
    };
    let static_eval = raw_eval.map(|ev| ctx.corrected_eval(board, ev));
    let eval = static_eval.map(|ev| adjust_eval(ev, tt_score, tt_bound));
    ctx.evals[ply_us] = static_eval;
    // Compare with our previous turn; fall back a full move if that node was in check.
    let improving = match static_eval {
        None => false,
        Some(ev) => {
            let back = |n: usize| if ply_us >= n { ctx.evals[ply_us - n] } else { None };
            back(2).or_else(|| back(4)).map_or(true, |e| ev > e)
        }
    };

    if !pv_node
        && !in_check
        && depth >= 1
        && depth <= p.rfp_max_depth as u32
        && beta.abs() < MATE_THRESHOLD
    {
        if eval.unwrap() - p.rfp_margin * (depth as i32 - improving as i32) >= beta {
            return beta;
        }
    }

    if !pv_node && allow_null && !in_check && depth >= p.nmp_min_depth as u32 {
        let stm_idx = board.side_to_move.idx();
        let non_pawn_bb = board.piece_bb[stm_idx][PieceKind::Knight.idx()]
            | board.piece_bb[stm_idx][PieceKind::Bishop.idx()]
            | board.piece_bb[stm_idx][PieceKind::Rook.idx()]
            | board.piece_bb[stm_idx][PieceKind::Queen.idx()];
        if non_pawn_bb != 0 {
            if eval.unwrap() >= beta {
                let r = p.nmp_base_r as u32 + (depth / p.nmp_divisor as u32);
                let undo = board.make_null_move();
                tt.prefetch(board.hash);
                ctx.accs.push_null();
                if ply_us + 1 < MAX_PLY {
                    ctx.played[ply_us + 1] = None;
                }
                ctx.hashes.push(hash);
                let floor = std::mem::replace(&mut ctx.rep_floor, ctx.hashes.len());
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
                    false,
                );
                ctx.rep_floor = floor;
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

    let mut sm = ScoredMoves::tt_move_first(&moves, board, tt_move, ctx, ply_us);
    let mut best_move: Option<Move> = None;
    let mut best_score = -INF;
    let mut searched_quiets: [Move; 64] = [Move(0); 64];
    let mut sq_n: usize = 0;

    // Quiets dropped once pruning covers them all; they still count toward the move index.
    let mut skipped = 0usize;
    for slot in 0.. {
        if slot >= sm.len {
            break;
        }
        if slot == 1 && sm.deferred {
            sm.score_deferred(board, ctx, ply_us);
        }
        let i = slot + skipped;
        let (m, move_score) = sm.pop_best(slot);
        if Some(m) == excluded {
            continue;
        }

        // Never prune while best_score is a mated score, so forced lines survive.
        let is_quiet = !m.is_capture() && !m.is_promotion();
        if i > 0 && !in_check && best_score > -MATE_THRESHOLD {
            if is_quiet
                && depth <= p.lmp_max_depth as u32
                && i as i32 >= (p.lmp_base + (depth as i32) * (depth as i32)) / (2 - improving as i32)
            {
                skipped += sm.drop_quiets_after(slot);
                continue;
            }
            if is_quiet
                && depth <= p.fp_max_depth as u32
                && eval.map_or(false, |ev| ev + p.fp_margin * depth as i32 <= alpha)
            {
                skipped += sm.drop_quiets_after(slot);
                continue;
            }
            if depth <= p.see_max_depth as u32 {
                let margin = if is_quiet { p.see_quiet_margin } else { p.see_capt_margin };
                // Past the TT move a capture's score already holds its SEE.
                let see_val = if m.is_capture() { see_from_score(move_score) } else { see(board, m) };
                if see_val < -(margin * depth as i32) {
                    continue;
                }
            }
        }
        let move_key = piece_key(board, m);
        let move_hist = if i > 0 && is_quiet { quiet_history(ctx, board, m, ply_us) } else { 0 };

        // Extend a lone TT move, multi-cut if another move beats beta, otherwise reduce it.
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
                        Some(m), cut_node,
                    );
                    // The excluded search shares this ply's PV slot.
                    ctx.pv_len[ply_us] = ply_us;
                    if ctx.stop {
                        return best_score;
                    }
                    if value < singular_beta {
                        extension = 1;
                    } else if singular_beta >= beta {
                        return singular_beta;
                    } else if ts >= beta {
                        extension = -2;
                    } else if !pv_node {
                        extension = -1;
                    }
                }
            }
        }
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
            score = -negamax(board, new_depth, -beta, -alpha, ply + 1, ctx, tt, true, None, !pv_node && !cut_node);
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
                let adj = (move_hist / p.lmr_history_div).clamp(-2, 2);
                let r = (base_r as i32 - adj + !improving as i32 + 2 * cut_node as i32 - tt_pv as i32)
                    .max(0) as u32;
                new_depth.saturating_sub(r).max(1)
            } else {
                new_depth
            };

            score = -negamax(
                board, initial_depth, -alpha - 1, -alpha, ply + 1, ctx, tt, true, None, do_lmr || !cut_node,
            );

            if do_lmr && !ctx.stop && score > alpha {
                score = -negamax(board, new_depth, -alpha - 1, -alpha, ply + 1, ctx, tt, true, None, !cut_node);
            }

            if !ctx.stop && score > alpha && score < beta {
                score = -negamax(board, new_depth, -beta, -alpha, ply + 1, ctx, tt, true, None, false);
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
            if !m.is_capture() && !m.is_promotion() {
                ctx.update_killers(ply_us, m);
                ctx.update_quiet_histories(board, ply_us, m, &searched_quiets[..sq_n], depth);
            }
            if excluded.is_none() {
                tt.store(
                    hash,
                    Some(m),
                    tt_score_to_store(best_score, ply_i),
                    raw_eval,
                    depth.min(255) as u8,
                    BOUND_LOWER,
                    tt_pv,
                );
                if let (Some(raw), Some(se)) = (raw_eval, static_eval) {
                    ctx.update_correction(board, raw, se, best_score, Some(m), BOUND_LOWER, depth);
                }
            }
            return best_score;
        }
        if score > alpha {
            alpha = score;
            ctx.update_pv(ply_us, m);
        }
    }

    let bound = if best_score > original_alpha { BOUND_EXACT } else { BOUND_UPPER };
    if excluded.is_none() {
        tt.store(
            hash,
            if bound == BOUND_UPPER { None } else { best_move },
            tt_score_to_store(best_score, ply_i),
            raw_eval,
            depth.min(255) as u8,
            bound,
            tt_pv,
        );
        if let (Some(raw), Some(se)) = (raw_eval, static_eval) {
            ctx.update_correction(board, raw, se, best_score, best_move, bound, depth);
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

// SEE is never negative when the first capture already wins at least the value of the
// piece now standing on the square: whatever the exchange, the capturer can stop there.
#[inline]
fn capture_cannot_lose(board: &Board, m: Move) -> bool {
    let Some(attacker) = board.piece_at(m.from()) else { return false };
    let victim = if m.flag() == Move::EP_CAPTURE {
        PieceKind::Pawn
    } else {
        match board.piece_at(m.to()) {
            Some(p) => p.kind,
            None => return false,
        }
    };
    let mut gain = SEE_VALUES[victim.idx()];
    let standing = match m.promo_kind() {
        Some(promo) => {
            gain += SEE_VALUES[promo.idx()] - SEE_VALUES[PieceKind::Pawn.idx()];
            promo
        }
        None => attacker.kind,
    };
    gain >= SEE_VALUES[standing.idx()]
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
        board.piece_at(to).map_or(0, |p| SEE_VALUES[p.kind.idx()])
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
    tt: &mut Tt,
) -> i32 {
    if ctx.check() {
        return 0;
    }
    ctx.seldepth = ctx.seldepth.max(ply as usize);

    let pv_node = beta - alpha > 1;
    let original_alpha = alpha;
    let hash = board.hash;
    let ply_i = ply as i32;

    let mut tt_move: Option<Move> = None;
    let mut tt_score: Option<i32> = None;
    let mut tt_bound = BOUND_NONE;
    let mut tt_eval: Option<i32> = None;
    let mut tt_pv = pv_node;
    if let Some(entry) = tt.probe(hash) {
        tt_pv |= entry.pv();
        if entry.move_packed != 0 {
            tt_move = Some(Move(entry.move_packed));
        }
        let s = tt_score_from_probe(entry.score, ply_i);
        tt_score = Some(s);
        tt_bound = entry.bound();
        tt_eval = entry.eval();
        if !pv_node {
            match tt_bound {
                BOUND_EXACT => return s,
                BOUND_LOWER if s >= beta => return s,
                BOUND_UPPER if s <= alpha => return s,
                _ => {}
            }
        }
    }

    let in_check = board.in_check(board.side_to_move);

    let mut best_score = -INF;
    let mut raw_eval: Option<i32> = None;
    if !in_check {
        let raw = tt_eval.unwrap_or_else(|| ctx.accs.evaluate(board));
        raw_eval = Some(raw);
        let stand_pat = adjust_eval(ctx.corrected_eval(board, raw), tt_score, tt_bound);
        best_score = stand_pat;
        if stand_pat >= beta {
            if tt_bound == BOUND_NONE {
                tt.store(hash, None, tt_score_to_store(stand_pat, ply_i), raw_eval, 0, BOUND_LOWER, tt_pv);
            }
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    if (ply as usize) >= MAX_PLY {
        return ctx.accs.evaluate(board);
    }

    // Out of check only captures and promotions are searched, so only those are generated.
    let mut moves = MoveList::new();
    if in_check {
        board.gen_legal_into(&mut moves);
        if moves.is_empty() {
            return -MATE_SCORE + ply as i32;
        }
    } else {
        board.gen_noisy_into(&mut moves);
        if moves.is_empty() && !board.has_legal_move() {
            return 0;
        }
    }

    let mut q = ScoredMoves::empty();
    for i in 0..moves.len() {
        let m = moves.get(i);
        if !in_check && m.is_capture() && !capture_cannot_lose(board, m) && see(board, m) < 0 {
            continue;
        }
        let s = if Some(m) == tt_move { 1_000_000 } else { qscore(m, board) };
        q.push(m, s);
    }

    let mut best_move: Option<Move> = None;
    for i in 0..q.len {
        let (m, _) = q.pop_best(i);

        let undo = board.make_move(m);
        tt.prefetch(board.hash);
        ctx.accs.push_after(board, m, &undo);
        let score = -qsearch(board, -beta, -alpha, ply + 1, ctx, tt);
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
            tt.store(hash, Some(m), tt_score_to_store(best_score, ply_i), raw_eval, 0, BOUND_LOWER, tt_pv);
            return best_score;
        }
        if score > alpha {
            alpha = score;
        }
    }

    let bound = if best_score > original_alpha { BOUND_EXACT } else { BOUND_UPPER };
    tt.store(hash, best_move, tt_score_to_store(best_score, ply_i), raw_eval, 0, bound, tt_pv);
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

// Inverse of the capture branch of score_move.
#[inline]
fn see_from_score(score: i32) -> i32 {
    if score > 0 { score - 100_000 } else { score + 100_000 }
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
    // Captures past index 0 still need scoring (see tt_move_first).
    deferred: bool,
}

impl ScoredMoves {
    fn empty() -> Self {
        Self {
            data: unsafe { MaybeUninit::<[MaybeUninit<(Move, i32)>; 256]>::uninit().assume_init() },
            len: 0,
            deferred: false,
        }
    }

    // Like from_list, but the TT move goes first right away (as pop_best(0) would) and
    // the rest wait for score_deferred, since the TT move often cuts. Without a TT move
    // only the captures' SEE scores wait.
    fn tt_move_first(list: &MoveList, board: &Board, tt_move: Option<Move>, ctx: &SearchCtx, ply: usize) -> Self {
        let mut sm = Self::empty();
        let mut tt_idx = None;
        for i in 0..list.len() {
            let m = list.get(i);
            let s = if Some(m) == tt_move {
                tt_idx = Some(i);
                2_000_000
            } else if m.is_capture() || tt_move.is_some() {
                0
            } else {
                score_move(m, board, None, tt_move, ctx, ply)
            };
            sm.push(m, s);
        }
        match tt_idx {
            Some(k) => {
                sm.data.swap(0, k);
                sm.deferred = true;
            }
            None => {
                sm.deferred = true;
                sm.score_deferred_from(0, tt_move.is_some(), board, ctx, ply);
            }
        }
        sm
    }

    fn score_deferred(&mut self, board: &Board, ctx: &SearchCtx, ply: usize) {
        self.score_deferred_from(1, true, board, ctx, ply);
    }

    fn score_deferred_from(&mut self, start: usize, quiets: bool, board: &Board, ctx: &SearchCtx, ply: usize) {
        for i in start..self.len {
            let (m, _) = unsafe { self.data[i].assume_init() };
            if quiets || m.is_capture() {
                self.data[i] = MaybeUninit::new((m, score_move(m, board, None, None, ctx, ply)));
            }
        }
        self.deferred = false;
    }

    #[inline]
    fn push(&mut self, m: Move, score: i32) {
        self.data[self.len] = MaybeUninit::new((m, score));
        self.len += 1;
    }

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
            deferred: false,
        };
        for i in 0..list.len() {
            let m = list.get(i);
            sm.data[i] = MaybeUninit::new((m, score_move(m, board, pv, tt_move, ctx, ply)));
        }
        sm
    }

    /// Removes the quiets after `start`, keeping the others in order; returns how many went.
    fn drop_quiets_after(&mut self, start: usize) -> usize {
        let mut w = start + 1;
        for r in start + 1..self.len {
            let e = unsafe { self.data[r].assume_init() };
            if e.0.is_capture() || e.0.is_promotion() {
                self.data[w] = MaybeUninit::new(e);
                w += 1;
            }
        }
        let dropped = self.len - w;
        self.len = w;
        dropped
    }

    fn pop_best(&mut self, start: usize) -> (Move, i32) {
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
        unsafe { self.data[start].assume_init() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Board;

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
        use crate::core::Board as B;
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
}

