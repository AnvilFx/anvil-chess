// Per-search state: limits, PV, killers, histories and corrections.
use super::*;

/// Eval cache size in entries, log2. Must be at least 17 (see `eval_cache`).
const EVAL_CACHE_BITS: u32 = 18;
const _: () = assert!(EVAL_CACHE_BITS >= 17);
/// Material eval scaling: eval * (MAT_K + material) / (MAT_K + MAT_REF), material in pawn = 535 units, full board 25164.
const MAT_K: i32 = 77777;
const MAT_REF: i32 = 12000;

pub(super) struct SearchCtx {
    pub(super) deadline: Option<Instant>,
    pub(super) accs: AccStack,
    pub(super) params: SearchParams,
    /// Base LMR in 1/1024 ply by [depth][move number].
    pub(super) lmr: Box<[[i32; 64]; 64]>,
    pub(super) stop: bool,
    /// Set from another thread (UCI `stop`); polled alongside the deadline.
    pub(super) abort: Arc<AtomicBool>,
    pub(super) nodes: u64,
    pub(super) max_nodes: u64,
    /// Nodes spent under each root move this search, by from * 64 + to.
    pub(super) root_nodes: Box<[u64; 4096]>,
    pub(super) seldepth: usize,
    /// Triangular PV table: `pv[ply][ply..pv_len[ply]]` is the line from `ply`.
    pub(super) pv: Box<[[Move; MAX_PLY + 1]; MAX_PLY + 1]>,
    pub(super) pv_len: [usize; MAX_PLY + 1],
    pub(super) killers: [[Option<Move>; 2]; MAX_PLY],
    pub(super) history_heur: Box<ButterflyHist>,
    /// Squares the opponent attacks at each ply of the current path, for history indexing.
    pub(super) threats: [u64; MAX_PLY + 1],
    /// defence_maps at each ply of the current path.
    pub(super) defended: [(u64, u64); MAX_PLY + 1],
    pub(super) cont: ContHist,
    /// The (piece, to) move that led into each ply.
    pub(super) played: [Option<(u8, u8)>; MAX_PLY],
    /// Static eval at each ply of the current path; None when in check.
    pub(super) evals: [Option<i32>; MAX_PLY],
    /// Raw eval of a child the parent already evaluated for LMR, keyed by hash, so the child skips it.
    pub(super) child_eval: Option<(u64, i32)>,
    /// Whether that child is in check, keyed the same way.
    pub(super) child_check: Option<(u64, bool)>,
    /// Raw evals by hash, so positions that left no eval in the TT (an early return stores none)
    /// skip the network when they come up again. An entry is (hash >> 16) << 16 | eval as u16 at
    /// index hash's low EVAL_CACHE_BITS bits, so index and entry together match all 64 hash bits.
    /// Bit 16 of the hash is in both: empty entries get it opposite to their index, so they never match.
    pub(super) eval_cache: Box<[u64]>,
    /// How far search results have strayed from the raw eval, per side to move and pawn structure.
    pub(super) corr: Box<[[i16; CORR_SIZE]; 2]>,
    /// Hashes from game start to the parent of the current node.
    pub(super) hashes: Vec<u64>,
    /// Index of the root position in `hashes`.
    pub(super) root_idx: usize,
    /// Repetition look-back stops here: positions before the last null move.
    pub(super) rep_floor: usize,
    /// Eval offset by side to move, from the root side's recent score (positive when the root side is ahead).
    pub(super) optimism: [i32; 2],
    /// Optimism is on (UCI option `Optimism`); datagen turns it off so its labels stay unbiased.
    pub(super) optimism_on: bool,
}

impl SearchCtx {
    pub(super) fn new() -> Self {
        Self {
            deadline: None,
            accs: AccStack::new(),
            params: SearchParams::default(),
            lmr: Box::new([[0; 64]; 64]),
            stop: false,
            abort: Arc::new(AtomicBool::new(false)),
            nodes: 0,
            max_nodes: u64::MAX,
            root_nodes: Box::new([0; 4096]),
            seldepth: 0,
            pv: Box::new([[Move::NULL; MAX_PLY + 1]; MAX_PLY + 1]),
            pv_len: [0; MAX_PLY + 1],
            killers: [[None; 2]; MAX_PLY],
            history_heur: Box::new([[[[[0; 64]; 64]; 3]; 2]; 2]),
            threats: [0; MAX_PLY + 1],
            defended: [(0, 0); MAX_PLY + 1],
            cont: ContHist::new(),
            played: [None; MAX_PLY],
            evals: [None; MAX_PLY],
            child_eval: None,
            child_check: None,
            eval_cache: (0..1u64 << EVAL_CACHE_BITS).map(|i| (!i >> 16 & 1) << 16).collect(),
            corr: Box::new([[0; CORR_SIZE]; 2]),
            hashes: Vec::with_capacity(MAX_PLY + 64),
            root_idx: 0,
            rep_floor: 0,
            optimism: [0; 2],
            optimism_on: true,
        }
    }

    pub(super) fn begin_search(
        &mut self,
        board: &Board,
        history: &[u64],
        deadline: Option<Instant>,
        max_nodes: u64,
        params: SearchParams,
    ) {
        self.deadline = deadline;
        self.params = params;
        let (base, div) = (params.lmr_base as f64 / 100.0, params.lmr_div as f64 / 100.0);
        for d in 1..64 {
            for n in 1..64 {
                self.lmr[d][n] = ((base + (d as f64).ln() * (n as f64).ln() / div) * 1024.0) as i32;
            }
        }
        self.stop = false;
        self.nodes = 0;
        self.max_nodes = max_nodes;
        self.root_nodes.fill(0);
        self.seldepth = 0;
        self.pv_len = [0; MAX_PLY + 1];
        self.played = [None; MAX_PLY];
        self.evals = [None; MAX_PLY];
        self.child_eval = None;
        self.child_check = None;
        self.accs.init(board);
        self.hashes.clear();
        self.hashes.extend_from_slice(history);
        self.root_idx = history.len();
        self.rep_floor = 0;
        self.optimism = [0; 2];
    }

    pub(super) fn clear_heuristics(&mut self) {
        self.killers = [[None; 2]; MAX_PLY];
        *self.history_heur = [[[[[0; 64]; 64]; 3]; 2]; 2];
        self.cont.clear();
        self.corr.iter_mut().for_each(|t| t.fill(0));
    }

    /// The network's raw eval of `board`, from the eval cache when it holds this position.
    #[inline]
    pub(super) fn cached_raw_eval(&mut self, board: &Board) -> i32 {
        let hash = board.hash;
        let idx = hash as usize & ((1 << EVAL_CACHE_BITS) - 1);
        let e = self.eval_cache[idx];
        if e >> 16 == hash >> 16 {
            return e as u16 as i16 as i32;
        }
        let v = self.accs.evaluate(board);
        if let Ok(v16) = i16::try_from(v) {
            self.eval_cache[idx] = (hash >> 16) << 16 | v16 as u16 as u64;
        }
        v
    }

    #[inline]
    pub(super) fn prefetch_eval(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let p = &self.eval_cache[hash as usize & ((1 << EVAL_CACHE_BITS) - 1)] as *const u64 as *const i8;
            // SAFETY: a prefetch of a valid address has no side effects.
            unsafe { _mm_prefetch(p, _MM_HINT_T0) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = hash;
    }

    /// The raw eval scaled by the material on the board (larger with more material, smaller in
    /// endgames, 1.0 at MAT_REF), plus optimism weighted by the same material.
    #[inline]
    pub(super) fn base_eval(&self, board: &Board, raw: i32) -> i32 {
        let material = sf_material(board);
        let scaled = raw * (MAT_K + material) / (MAT_K + MAT_REF);
        let o = self.optimism[board.side_to_move.idx()];
        if o == 0 {
            return scaled;
        }
        scaled + o * (7777 + material) / 77777
    }

    #[inline]
    pub(super) fn corrected_eval(&self, board: &Board, raw: i32) -> i32 {
        let c = self.corr[board.side_to_move.idx()][pawn_key(board)] as i32 * self.params.corr_scale / (CORR_GRAIN * 256);
        (self.base_eval(board, raw) + c).clamp(-MATE_THRESHOLD + 1, MATE_THRESHOLD - 1)
    }

    /// Moves the correction toward `best - raw`, weighted by depth. Skips results whose bound
    /// says nothing about the eval, mates, and noisy best moves.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn update_correction(
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
        let raw = self.base_eval(board, raw);
        let target = (best - raw).clamp(-CORR_MAX / CORR_GRAIN, CORR_MAX / CORR_GRAIN) * CORR_GRAIN;
        let weight = (depth as i32 + 1).min(self.params.corr_weight_max);
        let e = &mut self.corr[board.side_to_move.idx()][pawn_key(board)];
        *e = ((*e as i32 * (256 - weight) + target * weight) / 256) as i16;
    }

    #[inline]
    pub(super) fn check(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes >= self.max_nodes {
            self.stop = true;
        }
        if self.nodes & 1023 == 0 {
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
    pub(super) fn update_pv(&mut self, ply: usize, m: Move) {
        let next = ply + 1;
        let len = self.pv_len[next].max(next);
        let (head, tail) = self.pv.split_at_mut(next);
        head[ply][ply] = m;
        head[ply][next..len].copy_from_slice(&tail[0][next..len]);
        self.pv_len[ply] = len;
    }

    pub(super) fn root_pv(&self) -> &[Move] {
        &self.pv[0][..self.pv_len[0]]
    }

    #[inline]
    pub(super) fn update_killers(&mut self, ply: usize, m: Move) {
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
    pub(super) fn update_quiet_histories(
        &mut self,
        board: &Board,
        ply: usize,
        best: Move,
        searched: &[Move],
        depth: u32,
    ) {
        let d = depth as i32;
        let p = &self.params;
        let bonus = (p.hist_bonus_mul * d - p.hist_bonus_off).clamp(0, p.hist_bonus_max);
        let malus = (p.hist_malus_mul * d - p.hist_malus_off).clamp(0, p.hist_malus_max);
        let rows = HistRows::new(self, board, ply);
        rows.add(self, board, best, bonus);
        for &q in searched {
            if q != best {
                rows.add(self, board, q, -malus);
            }
        }
    }
}

/// The eval offset that optimism `o` adds in `board`, weighted by the material on the board.
#[inline]
pub(super) fn optimism_offset(board: &Board, o: i32) -> i32 {
    o * (7777 + sf_material(board)) / 77777
}

/// Material on the board in Stockfish-style units (pawn 535), for eval scaling and optimism.
#[inline]
fn sf_material(board: &Board) -> i32 {
    let w = &board.piece_bb[0];
    let b = &board.piece_bb[1];
    let cnt = |k: usize| (w[k] | b[k]).count_ones() as i32;
    535 * cnt(0) + 781 * cnt(1) + 825 * cnt(2) + 1276 * cnt(3) + 2538 * cnt(4)
}
