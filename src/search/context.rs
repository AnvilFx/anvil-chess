// Per-search state: limits, PV, killers, histories and corrections.
use super::*;

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
    pub(super) cont: ContHist,
    /// The (piece, to) move that led into each ply.
    pub(super) played: [Option<(u8, u8)>; MAX_PLY],
    /// Static eval at each ply of the current path; None when in check.
    pub(super) evals: [Option<i32>; MAX_PLY],
    /// How far search results have strayed from the raw eval, per side to move and pawn structure.
    pub(super) corr: Box<[[i16; CORR_SIZE]; 2]>,
    /// Hashes from game start to the parent of the current node.
    pub(super) hashes: Vec<u64>,
    /// Index of the root position in `hashes`.
    pub(super) root_idx: usize,
    /// Repetition look-back stops here: positions before the last null move.
    pub(super) rep_floor: usize,
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
            history_heur: Box::new([[[[[0; 64]; 64]; 2]; 2]; 2]),
            threats: [0; MAX_PLY + 1],
            cont: ContHist::new(),
            played: [None; MAX_PLY],
            evals: [None; MAX_PLY],
            corr: Box::new([[0; CORR_SIZE]; 2]),
            hashes: Vec::with_capacity(MAX_PLY + 64),
            root_idx: 0,
            rep_floor: 0,
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
        self.accs.init(board);
        self.hashes.clear();
        self.hashes.extend_from_slice(history);
        self.root_idx = history.len();
        self.rep_floor = 0;
    }

    pub(super) fn clear_heuristics(&mut self) {
        self.killers = [[None; 2]; MAX_PLY];
        *self.history_heur = [[[[[0; 64]; 64]; 2]; 2]; 2];
        self.cont.clear();
        self.corr.iter_mut().for_each(|t| t.fill(0));
    }

    #[inline]
    pub(super) fn corrected_eval(&self, board: &Board, raw: i32) -> i32 {
        let c = self.corr[board.side_to_move.idx()][pawn_key(board)] as i32 * self.params.corr_scale / (CORR_GRAIN * 256);
        (raw + c).clamp(-MATE_THRESHOLD + 1, MATE_THRESHOLD - 1)
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
        self.history_stat(board, ply, best, bonus);
        for &q in searched {
            if q != best {
                self.history_stat(board, ply, q, -malus);
            }
        }
    }

    #[inline]
    pub(super) fn history_stat(&mut self, board: &Board, ply: usize, m: Move, delta: i32) {
        let (tf, tt) = butterfly_slot(self.threats[ply], m);
        let e = &mut self.history_heur[board.side_to_move.idx()][tf][tt][m.from().idx()][m.to().idx()];
        let v = *e as i32;
        *e = (v + delta - v * delta.abs() / HISTORY_MAX) as i16;
        let cur = piece_key(board, m);
        if (1..MAX_PLY).contains(&ply) {
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
