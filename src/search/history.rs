// Move-ordering history tables and eval correction keys.
use super::*;

pub(super) const CORR_SIZE: usize = 16_384;
/// Corrections are stored in 1/256 cp.
pub(super) const CORR_GRAIN: i32 = 256;
pub(super) const CORR_MAX: i32 = CORR_GRAIN * 64;

#[inline]
pub(super) fn pawn_key(board: &Board) -> usize {
    let w = board.piece_bb[Color::White.idx()][PieceKind::Pawn.idx()];
    let b = board.piece_bb[Color::Black.idx()][PieceKind::Pawn.idx()];
    let h = w.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (h >> (64 - CORR_SIZE.trailing_zeros())) as usize
}
pub(super) const HISTORY_MAX: i32 = 16_384;

pub(super) const PIECE_NB: usize = 12;
pub(super) const CONT_PLIES: usize = 2;
pub(super) const CONT_MAX: i32 = HISTORY_MAX;

const CONT_ROW: usize = PIECE_NB * 64;
const CONT_LEN: usize = CONT_PLIES * PIECE_NB * 64 * CONT_ROW;

/// Indexed `[ply_offset][prev_piece][prev_to][piece][to]`, plus one trailing row that stays zero.
pub(super) struct ContHist {
    pub(super) data: Box<[i16]>,
}

impl ContHist {
    /// Row used when there is no previous move: never written, so it reads as 0.
    pub(super) const ZERO_ROW: usize = CONT_LEN;

    pub(super) fn new() -> Self {
        Self { data: vec![0i16; CONT_LEN + CONT_ROW].into_boxed_slice() }
    }

    pub(super) fn clear(&mut self) {
        self.data.fill(0);
    }

    #[inline]
    pub(super) fn index(off: usize, prev: (u8, u8), cur: (u8, u8)) -> usize {
        ((((off * PIECE_NB + prev.0 as usize) * 64 + prev.1 as usize) * PIECE_NB + cur.0 as usize) * 64)
            + cur.1 as usize
    }

    #[inline]
    fn add_at(&mut self, i: usize, delta: i32) {
        let e = &mut self.data[i];
        let v = *e as i32;
        *e = (v + delta - v * delta.abs() / CONT_MAX) as i16;
    }
}

/// Must be called before `m` is made (a promotion is keyed as a pawn).
#[inline]
pub(super) fn piece_key(board: &Board, m: Move) -> (u8, u8) {
    // The mailbox holds kind + color * 6 + 1, or 0 for empty (keyed as 0).
    (board.mailbox[m.from().idx()].saturating_sub(1), m.to().idx() as u8)
}

/// Indexed `[stm][from attacked][to: safe, attacked and undefended, attacked and defended][from][to]`.
pub(super) type ButterflyHist = [[[[[i16; 64]; 64]; 3]; 2]; 2];

#[inline]
pub(super) fn attacks_of_set(mut set: u64, f: impl Fn(usize) -> u64) -> u64 {
    let mut t = 0;
    while set != 0 {
        t |= f(set.trailing_zeros() as usize);
        set &= set - 1;
    }
    t
}

/// Our pieces attacked by a lower-valued enemy piece: material the side to move must spend its tempo saving.
pub(super) fn lesser_threats(board: &Board) -> u64 {
    let (us, them) = (board.side_to_move.idx(), board.side_to_move.opp().idx());
    let occ = board.occupancy;
    let ours = &board.piece_bb[us];
    let bb = &board.piece_bb[them];
    let rq = ours[PieceKind::Rook.idx()] | ours[PieceKind::Queen.idx()];
    let minors_up = ours[PieceKind::Knight.idx()] | ours[PieceKind::Bishop.idx()] | rq;
    let mut t = attacks_of_set(bb[PieceKind::Pawn.idx()], |s| PAWN_ATTACKS[them][s]) & minors_up;
    if rq != 0 {
        t |= (attacks_of_set(bb[PieceKind::Knight.idx()], |s| KNIGHT_ATTACKS[s])
            | attacks_of_set(bb[PieceKind::Bishop.idx()], |s| bishop_attacks(Square(s as u8), occ)))
            & rq;
        t |= attacks_of_set(bb[PieceKind::Rook.idx()], |s| rook_attacks(Square(s as u8), occ)) & ours[PieceKind::Queen.idx()];
    }
    t
}

/// Squares attacked by the side not to move.
pub(super) fn threat_map(board: &Board) -> u64 {
    let them = board.side_to_move.opp().idx();
    let occ = board.occupancy;
    let bb = &board.piece_bb[them];
    let queens = bb[PieceKind::Queen.idx()];
    attacks_of_set(bb[PieceKind::Pawn.idx()], |s| PAWN_ATTACKS[them][s])
        | attacks_of_set(bb[PieceKind::Knight.idx()], |s| KNIGHT_ATTACKS[s])
        | attacks_of_set(bb[PieceKind::Bishop.idx()] | queens, |s| bishop_attacks(Square(s as u8), occ))
        | attacks_of_set(bb[PieceKind::Rook.idx()] | queens, |s| rook_attacks(Square(s as u8), occ))
        | attacks_of_set(bb[PieceKind::King.idx()], |s| KING_ATTACKS[s])
}

/// Squares the side to move attacks with a pawn, and squares it attacks with at least two pieces.
pub(super) fn defence_maps(board: &Board) -> (u64, u64) {
    let us = board.side_to_move.idx();
    let occ = board.occupancy;
    let bb = &board.piece_bb[us];
    let pawns = attacks_of_set(bb[PieceKind::Pawn.idx()], |s| PAWN_ATTACKS[us][s]);
    let (mut once, mut twice) = (pawns, 0u64);
    let mut add = |a: u64| {
        twice |= once & a;
        once |= a;
    };
    let mut each = |mut set: u64, f: &dyn Fn(usize) -> u64| {
        while set != 0 {
            add(f(set.trailing_zeros() as usize));
            set &= set - 1;
        }
    };
    each(bb[PieceKind::Knight.idx()], &|s| KNIGHT_ATTACKS[s]);
    each(bb[PieceKind::Bishop.idx()] | bb[PieceKind::Queen.idx()], &|s| bishop_attacks(Square(s as u8), occ));
    each(bb[PieceKind::Rook.idx()] | bb[PieceKind::Queen.idx()], &|s| rook_attacks(Square(s as u8), occ));
    each(bb[PieceKind::King.idx()], &|s| KING_ATTACKS[s]);
    (once, twice)
}

#[inline]
pub(super) fn butterfly_slot(threats: u64, defended: u64, m: Move) -> (usize, usize) {
    let tt = (threats >> m.to().idx()) as usize & 1;
    ((threats >> m.from().idx()) as usize & 1, tt + (tt & (defended >> m.to().idx()) as usize))
}

/// A node's quiet-history context (side, threats, continuation rows), fixed over its move loop.
#[derive(Clone, Copy)]
pub(super) struct HistRows {
    stm: Color,
    pub(super) threats: u64,
    defended: (u64, u64),
    cont: [usize; CONT_PLIES],
    /// Squares from which each piece kind of ours would give direct check.
    checks: [u64; 6],
}

impl HistRows {
    #[inline]
    pub(super) fn new(ctx: &SearchCtx, board: &Board, ply: usize) -> Self {
        let row = |off: usize, prev: Option<(u8, u8)>| match prev {
            Some(p) => ContHist::index(off, p, (0, 0)),
            None => ContHist::ZERO_ROW,
        };
        let p0 = if (1..MAX_PLY).contains(&ply) { ctx.played[ply] } else { None };
        let p1 = if ply >= 2 && ply - 1 < MAX_PLY { ctx.played[ply - 1] } else { None };
        let stm = board.side_to_move;
        let ksq = lsb(board.piece_bb[stm.opp().idx()][PieceKind::King.idx()]);
        let (b, r) = (bishop_attacks(ksq, board.occupancy), rook_attacks(ksq, board.occupancy));
        let checks = [PAWN_ATTACKS[stm.opp().idx()][ksq.idx()], KNIGHT_ATTACKS[ksq.idx()], b, r, b | r, 0];
        Self { stm, threats: ctx.threats[ply], defended: ctx.defended[ply], cont: [row(0, p0), row(1, p1)], checks }
    }

    /// Ordering bonus for a quiet move that gives direct check from a square the opponent doesn't attack.
    #[inline]
    pub(super) fn check_bonus(&self, board: &Board, m: Move) -> i32 {
        let kind = (board.mailbox[m.from().idx()].saturating_sub(1) % 6) as usize;
        if self.checks[kind] & !self.threats & m.to().bb() != 0 {
            16_000
        } else {
            0
        }
    }

    /// Squares that stay defended once `m`'s piece has moved (a non-pawn mover attacks its own target).
    #[inline]
    fn defenders(&self, board: &Board, m: Move) -> u64 {
        if board.mailbox[m.from().idx()].saturating_sub(1) as usize % 6 == PieceKind::Pawn.idx() {
            self.defended.0
        } else {
            self.defended.1
        }
    }

    /// piece_key as an index within a continuation row (< CONT_ROW).
    #[inline]
    fn cont_key(board: &Board, m: Move) -> usize {
        board.mailbox[m.from().idx()].saturating_sub(1).min(PIECE_NB as u8 - 1) as usize * 64 + m.to().idx()
    }

    #[inline]
    pub(super) fn get(&self, ctx: &SearchCtx, board: &Board, m: Move) -> i32 {
        let (tf, tt) = butterfly_slot(self.threats, self.defenders(board, m), m);
        let h = ctx.history_heur[self.stm.idx()][tf][tt][m.from().idx()][m.to().idx()] as i32;
        let k = Self::cont_key(board, m);
        let d = &ctx.cont.data;
        h + d[self.cont[0] + k] as i32 + d[self.cont[1] + k] as i32
    }

    /// Gravity update of `m`'s butterfly and continuation entries; the zero row is skipped.
    #[inline]
    pub(super) fn add(&self, ctx: &mut SearchCtx, board: &Board, m: Move, delta: i32) {
        let (tf, tt) = butterfly_slot(self.threats, self.defenders(board, m), m);
        let e = &mut ctx.history_heur[self.stm.idx()][tf][tt][m.from().idx()][m.to().idx()];
        let v = *e as i32;
        *e = (v + delta - v * delta.abs() / HISTORY_MAX) as i16;
        let k = Self::cont_key(board, m);
        for &r in &self.cont {
            if r != ContHist::ZERO_ROW {
                ctx.cont.add_at(r + k, delta);
            }
        }
    }
}

