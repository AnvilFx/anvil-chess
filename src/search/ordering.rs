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

/// score_move for a move that is neither the TT nor the PV move.
#[inline]
fn score_plain(m: Move, board: &Board, killers: &[Option<Move>; 2], ctx: &SearchCtx, rows: &HistRows) -> i32 {
    if m.is_capture() {
        let see_val = see_with_threats(board, m, rows.threats);
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
    if killers[0] == Some(m) {
        return 40_000;
    }
    if killers[1] == Some(m) {
        return 39_000;
    }
    rows.get(ctx, board, m) + rows.check_bonus(board, m)
}

pub(super) struct ScoredMoves {
    pub(super) data: [MaybeUninit<(Move, i32)>; 256],
    pub(super) len: usize,
    // Captures past index 0 still need scoring (see tt_move_first).
    pub(super) deferred: bool,
}

impl ScoredMoves {
    pub(super) fn empty() -> Self {
        Self {
            data: unsafe { MaybeUninit::<[MaybeUninit<(Move, i32)>; 256]>::uninit().assume_init() },
            len: 0,
            deferred: false,
        }
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
        if let Some(k) = tt_move.and_then(|t| (0..n).position(|i| list.get(i) == t)) {
            for i in 0..n {
                self.push(list.get(i), 0);
            }
            self.data.swap(0, k);
            self.data[0] = MaybeUninit::new((list.get(k), 2_000_000));
            self.deferred = true;
        } else {
            let killers = killers_at(ctx, ply);
            for i in 0..n {
                let m = list.get(i);
                self.push(m, score_plain(m, board, &killers, ctx, rows));
            }
        }
    }

    pub(super) fn score_deferred(&mut self, board: &Board, ctx: &SearchCtx, ply: usize, rows: &HistRows) {
        let killers = killers_at(ctx, ply);
        for i in 1..self.len {
            let (m, _) = unsafe { self.data[i].assume_init() };
            self.data[i] = MaybeUninit::new((m, score_plain(m, board, &killers, ctx, rows)));
        }
        self.deferred = false;
    }


    #[inline]
    pub(super) fn push(&mut self, m: Move, score: i32) {
        self.data[self.len] = MaybeUninit::new((m, score));
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
        let mut sm = Self {
            data: unsafe { MaybeUninit::<[MaybeUninit<(Move, i32)>; 256]>::uninit().assume_init() },
            len: list.len(),
            deferred: false,
        };
        let rows = HistRows::new(ctx, board, ply);
        for i in 0..list.len() {
            let m = list.get(i);
            sm.data[i] = MaybeUninit::new((m, score_move(m, board, pv, tt_move, ctx, ply, &rows)));
        }
        sm
    }

    /// Removes the quiets after `start`, keeping the others in order; returns how many went.
    pub(super) fn drop_quiets_after(&mut self, start: usize) -> usize {
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

    pub(super) fn pop_best(&mut self, start: usize) -> (Move, i32) {
        // Still deferred means the TT move sits at 0 and outscores everything.
        if self.deferred {
            return unsafe { self.data[start].assume_init() };
        }
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
