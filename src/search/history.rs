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

const NOT_FILE_A: u64 = !0x0101_0101_0101_0101;
const NOT_FILE_H: u64 = !0x8080_8080_8080_8080;

/// Union of PAWN_ATTACKS[color] over `pawns`, by shifts.
#[inline]
fn pawn_attacks_of_set(pawns: u64, color: usize) -> u64 {
    let (west, east) = (pawns & NOT_FILE_A, pawns & NOT_FILE_H);
    if color == Color::White.idx() {
        (west << 7) | (east << 9)
    } else {
        (west >> 9) | (east >> 7)
    }
}

/// Union of KNIGHT_ATTACKS over `knights`, by shifts.
#[inline]
fn knight_attacks_of_set(knights: u64) -> u64 {
    let h1 = ((knights >> 1) & NOT_FILE_H) | ((knights << 1) & NOT_FILE_A);
    let h2 = ((knights >> 2) & !0xC0C0_C0C0_C0C0_C0C0) | ((knights << 2) & !0x0303_0303_0303_0303);
    (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
}

/// Our pieces attacked by a lower-valued enemy piece: material the side to move must spend its tempo saving.
pub(super) fn lesser_threats(board: &Board) -> u64 {
    let (us, them) = (board.side_to_move.idx(), board.side_to_move.opp().idx());
    let occ = board.occupancy;
    let ours = &board.piece_bb[us];
    let bb = &board.piece_bb[them];
    let rq = ours[PieceKind::Rook.idx()] | ours[PieceKind::Queen.idx()];
    let minors_up = ours[PieceKind::Knight.idx()] | ours[PieceKind::Bishop.idx()] | rq;
    let mut t = pawn_attacks_of_set(bb[PieceKind::Pawn.idx()], them) & minors_up;
    if rq != 0 {
        t |= (knight_attacks_of_set(bb[PieceKind::Knight.idx()])
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
    pawn_attacks_of_set(bb[PieceKind::Pawn.idx()], them)
        | knight_attacks_of_set(bb[PieceKind::Knight.idx()])
        | attacks_of_set(bb[PieceKind::Bishop.idx()] | queens, |s| bishop_attacks(Square(s as u8), occ))
        | attacks_of_set(bb[PieceKind::Rook.idx()] | queens, |s| rook_attacks(Square(s as u8), occ))
        | attacks_of_set(bb[PieceKind::King.idx()], |s| KING_ATTACKS[s])
}

/// Squares the side to move attacks with a pawn, and squares it attacks with at least two pieces.
pub(super) fn defence_maps(board: &Board) -> (u64, u64) {
    let us = board.side_to_move.idx();
    let occ = board.occupancy;
    let bb = &board.piece_bb[us];
    let pawns = pawn_attacks_of_set(bb[PieceKind::Pawn.idx()], us);
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


/// `get + check_bonus` for eight moves at once (AVX2 gathers), exactly as the scalar code computes them.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl HistRows {
    /// `m`: eight moves zero-extended into i32 lanes.
    #[inline(always)]
    pub(super) fn quiet_scores8(
        &self,
        ctx: &SearchCtx,
        board: &Board,
        m: std::arch::x86_64::__m256i,
    ) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        // Bit `sq` (0..64 per lane) of the u64 split into lo/hi lanes; variable shifts of 32+ give 0.
        #[inline(always)]
        unsafe fn bit(lo: __m256i, hi: __m256i, sq: __m256i) -> __m256i {
            let x = _mm256_or_si256(
                _mm256_srlv_epi32(lo, sq),
                _mm256_srlv_epi32(hi, _mm256_sub_epi32(sq, _mm256_set1_epi32(32))),
            );
            _mm256_and_si256(x, _mm256_set1_epi32(1))
        }
        // The i16 at each index (sign-extended), read as the aligned i32 pair holding it: callers
        // guarantee idx | 1 is still inside the table.
        #[inline(always)]
        unsafe fn gather16(base: *const i16, idx: __m256i) -> __m256i {
            let pair = _mm256_i32gather_epi32::<4>(base as *const i32, _mm256_srli_epi32::<1>(idx));
            let even = _mm256_andnot_si256(idx, _mm256_set1_epi32(1));
            _mm256_srai_epi32::<16>(_mm256_sllv_epi32(pair, _mm256_slli_epi32::<4>(even)))
        }
        let lo = |b: u64| b as u32 as i32;
        let hi = |b: u64| (b >> 32) as u32 as i32;
        // SAFETY: AVX2 is enabled at compile time; every gather stays inside its table (see each).
        unsafe {
            let c63 = _mm256_set1_epi32(63);
            let from = _mm256_and_si256(m, c63);
            let to = _mm256_and_si256(_mm256_srli_epi32::<6>(m), c63);
            // Mailbox byte of `from`, from the aligned 4 bytes holding it (inside the 64-byte array).
            let dw = _mm256_i32gather_epi32::<1>(
                board.mailbox.as_ptr() as *const i32,
                _mm256_andnot_si256(_mm256_set1_epi32(3), from),
            );
            let shift = _mm256_slli_epi32::<3>(_mm256_and_si256(from, _mm256_set1_epi32(3)));
            let code = _mm256_and_si256(_mm256_srlv_epi32(dw, shift), _mm256_set1_epi32(15));
            // mover_kind's table; the other bytes of each lane are 0 and look up KIND[0] = 0.
            let kind_tab = _mm256_setr_epi8(
                0, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2,
            );
            let kind = _mm256_shuffle_epi8(kind_tab, code);
            let pawn = _mm256_cmpeq_epi32(kind, _mm256_setzero_si256());

            let (t_lo, t_hi) = (_mm256_set1_epi32(lo(self.threats)), _mm256_set1_epi32(hi(self.threats)));
            let tf = bit(t_lo, t_hi, from);
            let ttt = bit(t_lo, t_hi, to);
            let (d0, d1) = self.defended;
            let d_lo = _mm256_blendv_epi8(_mm256_set1_epi32(lo(d1)), _mm256_set1_epi32(lo(d0)), pawn);
            let d_hi = _mm256_blendv_epi8(_mm256_set1_epi32(hi(d1)), _mm256_set1_epi32(hi(d0)), pawn);
            let tt = _mm256_add_epi32(ttt, _mm256_and_si256(ttt, bit(d_lo, d_hi, to)));
            let plane = _mm256_add_epi32(_mm256_add_epi32(tf, _mm256_slli_epi32::<1>(tf)), tt);
            let bidx = _mm256_add_epi32(_mm256_slli_epi32::<6>(_mm256_add_epi32(_mm256_slli_epi32::<6>(plane), from)), to);
            // One side's table is 2 * 3 * 64 * 64 entries (even); bidx is below that.
            let h = gather16(ctx.history_heur[self.stm.idx()].as_ptr() as *const i16, bidx);

            // cont_key: min(code - 1 saturating, 11) * 64 + to; rows + key stay inside the table (see get).
            let pc = _mm256_min_epi32(
                _mm256_max_epi32(_mm256_sub_epi32(code, _mm256_set1_epi32(1)), _mm256_setzero_si256()),
                _mm256_set1_epi32(PIECE_NB as i32 - 1),
            );
            let k = _mm256_add_epi32(_mm256_slli_epi32::<6>(pc), to);
            let d = ctx.cont.data.as_ptr();
            let c0 = gather16(d, _mm256_add_epi32(_mm256_set1_epi32(self.cont[0] as i32), k));
            let c1 = gather16(d, _mm256_add_epi32(_mm256_set1_epi32(self.cont[1] as i32), k));

            // check_bonus: checks[kind] & !threats has the `to` bit.
            let ck = |f: fn(u64) -> i32| {
                let c = |i: usize| f(self.checks[i] & !self.threats);
                _mm256_setr_epi32(c(0), c(1), c(2), c(3), c(4), c(5), 0, 0)
            };
            let (ck_lo, ck_hi) = (ck(|b| b as u32 as i32), ck(|b| (b >> 32) as u32 as i32));
            let cb = bit(_mm256_permutevar8x32_epi32(ck_lo, kind), _mm256_permutevar8x32_epi32(ck_hi, kind), to);
            let bonus = _mm256_and_si256(_mm256_sub_epi32(_mm256_setzero_si256(), cb), _mm256_set1_epi32(16_000));

            _mm256_add_epi32(_mm256_add_epi32(h, c0), _mm256_add_epi32(c1, bonus))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifted_attack_sets_match_tables() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..2000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let set = x & x.rotate_left(23);
            for c in 0..2 {
                assert_eq!(pawn_attacks_of_set(set, c), attacks_of_set(set, |s| PAWN_ATTACKS[c][s]));
            }
            assert_eq!(knight_attacks_of_set(set), attacks_of_set(set, |s| KNIGHT_ATTACKS[s]));
        }
        for s in 0..64 {
            for c in 0..2 {
                assert_eq!(pawn_attacks_of_set(1 << s, c), PAWN_ATTACKS[c][s]);
            }
            assert_eq!(knight_attacks_of_set(1 << s), KNIGHT_ATTACKS[s]);
        }
    }
}
