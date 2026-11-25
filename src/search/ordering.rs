// Move scoring and staged move picking.
use super::*;

pub(super) fn qscore(m: Move, board: &Board) -> i32 {
    if m.is_capture() {
        let attacker = board.piece_at(m.from()).map(|p| p.kind as i32).unwrap_or(0);
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
pub(super) fn see_from_score(score: i32) -> i32 {
    if score > 0 {
        score - 100_000
    } else {
        score + 100_000
    }
}

pub(super) fn score_move(
    m: Move,
    board: &Board,
    pv: Option<Move>,
    tt_move: Option<Move>,
    ctx: &SearchCtx,
    ply: usize,
    rows: &HistRows,
) -> i32 {
    if tt_move == Some(m) {
        return 2_000_000;
    }
    if pv == Some(m) {
        return 1_000_000;
    }
    score_plain(m, board, &killers_at(ctx, ply), ctx, rows)
}

#[inline]
fn killers_at(ctx: &SearchCtx, ply: usize) -> [Option<Move>; 2] {
    if ply < MAX_PLY {
        ctx.killers[ply]
    } else {
        [None; 2]
    }
}

/// score_plain for a capture: its SEE, split into winning and losing.
#[inline(never)]
fn score_capture(m: Move, board: &Board, threats: u64) -> i32 {
    let see_val = see_with_threats(board, m, threats);
    if see_val >= 0 {
        100_000 + see_val
    } else {
        -100_000 + see_val
    }
}

/// score_move for a move that is neither the TT nor the PV move. Inlined so the quiet path runs in
/// the caller's loop; the SEE path stays out of line.
#[inline(always)]
fn score_plain(m: Move, board: &Board, killers: &[Option<Move>; 2], ctx: &SearchCtx, rows: &HistRows) -> i32 {
    if m.is_capture() {
        return score_capture(m, board, rows.threats);
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
    if killers[0] == Some(m) {
        return 40_000;
    }
    if killers[1] == Some(m) {
        return 39_000;
    }
    rows.get(ctx, board, m) + rows.check_bonus(board, m)
}

/// Chunk width of the pop_best scan.
const PICK_CH: usize = 8;
const NO_HINT: usize = usize::MAX;

/// pop_best's sort key: its max is the highest score at the lowest index, as a first-wins scan picks.
/// Scores stay far inside +-2^22 (at most 2_000_000), so the key fits an i32.
#[inline(always)]
fn pick_key(i: usize, score: i32) -> i32 {
    debug_assert!(score > -(1 << 22) && score < (1 << 22) && i < 256);
    score * 256 + (255 - i as i32)
}

// Moves and scores in separate arrays so pop_best can scan the scores in vector chunks.
pub(super) struct ScoredMoves {
    moves: [MaybeUninit<Move>; 256],
    scores: [MaybeUninit<i32>; 256],
    pub(super) len: usize,
    // Captures past index 0 still need scoring (see tt_move_first).
    pub(super) deferred: bool,
    // While not NO_HINT, hint_key is the max pick_key over [hint_from, len): the first pop after a
    // fill needs no scan (a scan right after the stores would also stall on store forwarding).
    hint_from: usize,
    hint_key: i32,
}

impl ScoredMoves {
    pub(super) fn empty() -> Self {
        Self {
            moves: unsafe { MaybeUninit::<[MaybeUninit<Move>; 256]>::uninit().assume_init() },
            scores: unsafe { MaybeUninit::<[MaybeUninit<i32>; 256]>::uninit().assume_init() },
            len: 0,
            deferred: false,
            hint_from: 0,
            hint_key: i32::MIN,
        }
    }

    #[inline(always)]
    fn set(&mut self, i: usize, m: Move, score: i32) {
        self.moves[i] = MaybeUninit::new(m);
        self.scores[i] = MaybeUninit::new(score);
    }

    #[inline(always)]
    fn get(&self, i: usize) -> (Move, i32) {
        unsafe { (self.moves[i].assume_init(), self.scores[i].assume_init()) }
    }

    // Fills an empty list like from_list, but the TT move goes first right away (as pop_best(0)
    // would) and the rest wait for score_deferred, since the TT move often cuts. In place, since
    // returning the list copies 2 KB.
    pub(super) fn fill_tt_move_first(
        &mut self,
        list: &MoveList,
        board: &Board,
        tt_move: Option<Move>,
        ctx: &SearchCtx,
        ply: usize,
        rows: &HistRows,
    ) {
        let n = list.len();
        self.len = n;
        if let Some(k) = tt_move.and_then(|t| (0..n).position(|i| list.get(i) == t)) {
            // Only the moves: score_deferred gives slots 1.. their scores before any is read.
            for i in 0..n {
                self.moves[i] = MaybeUninit::new(list.get(i));
            }
            self.moves.swap(0, k);
            self.set(0, list.get(k), 2_000_000);
            self.hint_from = NO_HINT;
            self.deferred = true;
        } else {
            for i in 0..n {
                self.moves[i] = MaybeUninit::new(list.get(i));
            }
            self.hint_key = self.score_range(0, board, ctx, ply, rows);
            self.hint_from = 0;
        }
    }

    pub(super) fn score_deferred(&mut self, board: &Board, ctx: &SearchCtx, ply: usize, rows: &HistRows) {
        self.hint_key = self.score_range(1, board, ctx, ply, rows);
        self.hint_from = 1;
        self.deferred = false;
    }

    /// Gives moves[from..len) their score_plain scores; returns the max pick_key among them.
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    #[inline(always)]
    fn score_range(&mut self, from: usize, board: &Board, ctx: &SearchCtx, ply: usize, rows: &HistRows) -> i32 {
        let killers = killers_at(ctx, ply);
        let mut best = i32::MIN;
        for i in from..self.len {
            // SAFETY: moves[from..len) are initialised (after the TT branch their scores are not).
            let m = unsafe { self.moves[i].assume_init() };
            let s = score_plain(m, board, &killers, ctx, rows);
            self.set(i, m, s);
            best = best.max(pick_key(i, s));
        }
        best
    }

    /// Gives moves[from..len) their score_plain scores; returns the max pick_key among them.
    /// Eight moves at a time: every lane gets the quiet score (history gathers), then the killer and
    /// promotion scores by blend, and only the captures go through SEE one by one.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    fn score_range(&mut self, from: usize, board: &Board, ctx: &SearchCtx, ply: usize, rows: &HistRows) -> i32 {
        use std::arch::x86_64::*;
        let killers = killers_at(ctx, ply);
        let len = self.len;
        let mut best = i32::MIN;
        // SAFETY: AVX2 is enabled at compile time; moves[from..len) are initialised, the last partial
        // chunk is read from a padded copy, and only scores[from..len) are written.
        unsafe {
            let kv = |k: Option<Move>| _mm256_set1_epi32(k.map_or(-1, |m| m.0 as i32));
            let (k0, k1) = (kv(killers[0]), kv(killers[1]));
            let promo_tab = _mm256_setr_epi32(300, 100, 200, 400, 300, 100, 200, 400);
            let iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            let mut bestv = _mm256_set1_epi32(i32::MIN);
            let mut i = from;
            while i < len {
                let rem = (len - i).min(8);
                let mv = if rem == 8 {
                    _mm_loadu_si128(self.moves.as_ptr().add(i) as *const __m128i)
                } else {
                    let mut buf = [0u16; 8];
                    for (k, b) in buf.iter_mut().enumerate().take(rem) {
                        *b = self.moves[i + k].assume_init().0;
                    }
                    _mm_loadu_si128(buf.as_ptr() as *const __m128i)
                };
                let m = _mm256_cvtepu16_epi32(mv);
                let flag = _mm256_srli_epi32::<12>(m);
                let mut s = rows.quiet_scores8(ctx, board, m);
                s = _mm256_blendv_epi8(s, _mm256_set1_epi32(39_000), _mm256_cmpeq_epi32(m, k1));
                s = _mm256_blendv_epi8(s, _mm256_set1_epi32(40_000), _mm256_cmpeq_epi32(m, k0));
                let promo = _mm256_cmpeq_epi32(_mm256_and_si256(flag, _mm256_set1_epi32(8)), _mm256_set1_epi32(8));
                let promo_s = _mm256_add_epi32(
                    _mm256_set1_epi32(50_000),
                    _mm256_permutevar8x32_epi32(promo_tab, _mm256_and_si256(flag, _mm256_set1_epi32(3))),
                );
                s = _mm256_blendv_epi8(s, promo_s, promo);
                let cap = _mm256_or_si256(
                    _mm256_or_si256(
                        _mm256_cmpeq_epi32(flag, _mm256_set1_epi32(Move::CAPTURE as i32)),
                        _mm256_cmpeq_epi32(flag, _mm256_set1_epi32(Move::EP_CAPTURE as i32)),
                    ),
                    _mm256_cmpgt_epi32(flag, _mm256_set1_epi32(Move::PROMO_CAP_KNIGHT as i32 - 1)),
                );
                let valid = _mm256_cmpgt_epi32(_mm256_set1_epi32(rem as i32), iota);
                // pick_key = s * 256 + (255 - index)
                let idx = _mm256_add_epi32(_mm256_set1_epi32(i as i32), iota);
                let key = _mm256_sub_epi32(_mm256_add_epi32(_mm256_slli_epi32::<8>(s), _mm256_set1_epi32(255)), idx);
                let quiet = _mm256_andnot_si256(cap, valid);
                bestv = _mm256_max_epi32(bestv, _mm256_blendv_epi8(_mm256_set1_epi32(i32::MIN), key, quiet));
                let out = self.scores.as_mut_ptr().add(i) as *mut i32;
                if rem == 8 {
                    _mm256_storeu_si256(out as *mut __m256i, s);
                } else {
                    _mm256_maskstore_epi32(out, valid, s);
                }
                let mut cm = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_and_si256(cap, valid))) as u32;
                while cm != 0 {
                    let j = i + cm.trailing_zeros() as usize;
                    let (m, _) = self.get(j);
                    let sc = score_capture(m, board, rows.threats);
                    self.scores[j] = MaybeUninit::new(sc);
                    best = best.max(pick_key(j, sc));
                    cm &= cm - 1;
                }
                i += 8;
            }
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, bestv);
            lanes.into_iter().fold(best, i32::max)
        }
    }


    #[inline]
    pub(super) fn push(&mut self, m: Move, score: i32) {
        self.set(self.len, m, score);
        self.hint_key = self.hint_key.max(pick_key(self.len, score));
        self.len += 1;
    }

    pub(super) fn from_list(
        list: &MoveList,
        board: &Board,
        pv: Option<Move>,
        tt_move: Option<Move>,
        ctx: &SearchCtx,
        ply: usize,
    ) -> Self {
        let mut sm = Self::empty();
        let rows = HistRows::new(ctx, board, ply);
        for i in 0..list.len() {
            let m = list.get(i);
            sm.push(m, score_move(m, board, pv, tt_move, ctx, ply, &rows));
        }
        sm
    }

    /// Removes the quiets after `start`, keeping the others in order; returns how many went.
    pub(super) fn drop_quiets_after(&mut self, start: usize) -> usize {
        self.hint_from = NO_HINT;
        let mut w = start + 1;
        for r in start + 1..self.len {
            let (m, s) = self.get(r);
            if m.is_capture() || m.is_promotion() {
                self.set(w, m, s);
                w += 1;
            }
        }
        let dropped = self.len - w;
        self.len = w;
        dropped
    }

    /// The best-scored move from `start` on (the first one on ties), swapped into `start`.
    pub(super) fn pop_best(&mut self, start: usize) -> (Move, i32) {
        // Still deferred means the TT move sits at 0 and outscores everything.
        if self.deferred {
            return self.get(start);
        }
        let len = self.len;
        let key = if self.hint_from == start {
            self.hint_key
        } else {
            // Whole chunks in vectors (slots before `start` masked out), the tail one by one.
            let mut best = [i32::MIN; PICK_CH];
            let mut c = start & !(PICK_CH - 1);
            while c + PICK_CH <= len {
                let chunk: &[MaybeUninit<i32>; PICK_CH] = self.scores[c..c + PICK_CH].try_into().unwrap();
                for k in 0..PICK_CH {
                    // SAFETY: c + k < len, and [0, len) is initialised.
                    let key = pick_key(c + k, unsafe { chunk[k].assume_init() });
                    best[k] = best[k].max(if c + k >= start { key } else { i32::MIN });
                }
                c += PICK_CH;
            }
            let mut key = best.into_iter().max().unwrap();
            for i in c.max(start)..len {
                key = key.max(pick_key(i, unsafe { self.scores[i].assume_init() }));
            }
            key
        };
        self.hint_from = NO_HINT;
        let b = 255 - (key & 255) as usize;
        let picked = self.get(b);
        if b != start {
            // Slot `start` is never read again, so only `b` takes the displaced entry.
            let (m, s) = self.get(start);
            self.set(b, m, s);
        }
        picked
    }
}
