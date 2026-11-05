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

/// Indexed `[ply_offset][prev_piece][prev_to][piece][to]`.
pub(super) struct ContHist {
    pub(super) data: Box<[i16]>,
}

impl ContHist {
    pub(super) fn new() -> Self {
        Self { data: vec![0i16; CONT_PLIES * PIECE_NB * 64 * PIECE_NB * 64].into_boxed_slice() }
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
    pub(super) fn get(&self, off: usize, prev: (u8, u8), cur: (u8, u8)) -> i32 {
        self.data[Self::index(off, prev, cur)] as i32
    }

    /// Gravity update: saturates toward +/-CONT_MAX instead of overflowing.
    #[inline]
    pub(super) fn add(&mut self, off: usize, prev: (u8, u8), cur: (u8, u8), delta: i32) {
        let i = Self::index(off, prev, cur);
        let e = self.data[i] as i32;
        self.data[i] = (e + delta - e * delta.abs() / CONT_MAX) as i16;
    }
}

/// Must be called before `m` is made (a promotion is keyed as a pawn).
#[inline]
pub(super) fn piece_key(board: &Board, m: Move) -> (u8, u8) {
    let idx = match board.piece_at(m.from()) {
        Some(p) => (p.kind.idx() + p.color.idx() * 6) as u8,
        None => 0,
    };
    (idx, m.to().idx() as u8)
}

/// Indexed `[stm][from attacked][to attacked][from][to]`.
pub(super) type ButterflyHist = [[[[[i16; 64]; 64]; 2]; 2]; 2];

#[inline]
pub(super) fn attacks_of_set(mut set: u64, f: impl Fn(usize) -> u64) -> u64 {
    let mut t = 0;
    while set != 0 {
        t |= f(set.trailing_zeros() as usize);
        set &= set - 1;
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

#[inline]
pub(super) fn butterfly_slot(threats: u64, m: Move) -> (usize, usize) {
    ((threats >> m.from().idx()) as usize & 1, (threats >> m.to().idx()) as usize & 1)
}

#[inline]
pub(super) fn quiet_history(ctx: &SearchCtx, board: &Board, m: Move, ply: usize) -> i32 {
    let (tf, tt) = butterfly_slot(ctx.threats[ply], m);
    let mut h = ctx.history_heur[board.side_to_move.idx()][tf][tt][m.from().idx()][m.to().idx()] as i32;
    let cur = piece_key(board, m);
    if (1..MAX_PLY).contains(&ply) {
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
