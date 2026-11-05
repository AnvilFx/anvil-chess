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

    // Like from_list, but the TT move goes first right away (as pop_best(0) would) and
    // the rest wait for score_deferred, since the TT move often cuts. Without a TT move
    // only the captures' SEE scores wait.
    pub(super) fn tt_move_first(
        list: &MoveList,
        board: &Board,
        tt_move: Option<Move>,
        ctx: &SearchCtx,
        ply: usize,
    ) -> Self {
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

    pub(super) fn score_deferred(&mut self, board: &Board, ctx: &SearchCtx, ply: usize) {
        self.score_deferred_from(1, true, board, ctx, ply);
    }

    pub(super) fn score_deferred_from(
        &mut self,
        start: usize,
        quiets: bool,
        board: &Board,
        ctx: &SearchCtx,
        ply: usize,
    ) {
        for i in start..self.len {
            let (m, _) = unsafe { self.data[i].assume_init() };
            if quiets || m.is_capture() {
                self.data[i] = MaybeUninit::new((m, score_move(m, board, None, None, ctx, ply)));
            }
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
        for i in 0..list.len() {
            let m = list.get(i);
            sm.data[i] = MaybeUninit::new((m, score_move(m, board, pv, tt_move, ctx, ply)));
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
