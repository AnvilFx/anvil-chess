// Built with target-cpu=native, so plain loops auto-vectorize; only the VNNI head uses intrinsics.

use super::params::*;
use crate::core::{Board, Color, Move, PieceKind, Square, UndoInfo};

// All lanes are summed at the end, so no lane permute is needed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn head_dot_vnni(
    us: &[i16; N_HIDDEN], them: &[i16; N_HIDDEN], head: &[i8; 2 * N_HIDDEN],
    qa: i16, head_bias: i32,
) -> i32 {
    use std::arch::x86_64::*;
    let mut act = [0u8; 2 * N_HIDDEN];
    for h in 0..N_HIDDEN {
        act[h] = us[h].clamp(0, qa) as u8;
        act[N_HIDDEN + h] = them[h].clamp(0, qa) as u8;
    }

    let total = 2 * N_HIDDEN;
    let chunks = total / 32;
    let mut acc = _mm256_setzero_si256();
    let ap = act.as_ptr();
    let wp = head.as_ptr();
    for c in 0..chunks {
        let a = _mm256_loadu_si256(ap.add(c * 32) as *const __m256i);
        let b = _mm256_loadu_si256(wp.add(c * 32) as *const __m256i);
        acc = _mm256_dpbusd_avx_epi32(acc, a, b);
    }

    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut sum: i32 = head_bias;
    for v in lanes {
        sum += v;
    }
    for i in (chunks * 32)..total {
        sum += act[i] as i32 * head[i] as i32;
    }
    sum
}

#[inline]
fn head_dot_scalar(
    us: &[i16; N_HIDDEN], them: &[i16; N_HIDDEN], head: &[i8; 2 * N_HIDDEN],
    qa: i16, head_bias: i32,
) -> i32 {
    let mut sum: i32 = head_bias;
    for h in 0..N_HIDDEN {
        sum += us[h].clamp(0, qa) as i32 * head[h] as i32;
        sum += them[h].clamp(0, qa) as i32 * head[N_HIDDEN + h] as i32;
    }
    sum
}

#[inline]
fn head_dot(
    us: &[i16; N_HIDDEN], them: &[i16; N_HIDDEN], head: &[i8; 2 * N_HIDDEN],
    qa: i16, head_bias: i32,
) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avxvnni") {
            return unsafe { head_dot_vnni(us, them, head, qa, head_bias) };
        }
    }
    head_dot_scalar(us, them, head, qa, head_bias)
}

// Returns (white-perspective, black-perspective) feature indices.
#[inline]
fn feature_indices(color: Color, kind: PieceKind, sq: usize) -> (usize, usize) {
    let kind_w = if color == Color::White { kind.idx() } else { kind.idx() + 6 };
    let feat_w = kind_w * 64 + sq;
    let sq_b = sq ^ 56;
    let kind_b = if color == Color::Black { kind.idx() } else { kind.idx() + 6 };
    let feat_b = kind_b * 64 + sq_b;
    (feat_w, feat_b)
}

// Real positions never overflow i16; wrapping ops just avoid debug-mode panics.
#[inline]
fn transition_add_sub(
    src_w: &[i16; N_HIDDEN], src_b: &[i16; N_HIDDEN],
    dst_w: &mut [i16; N_HIDDEN], dst_b: &mut [i16; N_HIDDEN],
    add: (usize, usize), sub: (usize, usize),
) {
    let ft = &WEIGHTS.ft_weights;
    let aw = &ft[add.0];
    let ab = &ft[add.1];
    let sw = &ft[sub.0];
    let sb = &ft[sub.1];
    for h in 0..N_HIDDEN {
        dst_w[h] = src_w[h].wrapping_add(aw[h]).wrapping_sub(sw[h]);
        dst_b[h] = src_b[h].wrapping_add(ab[h]).wrapping_sub(sb[h]);
    }
}

#[inline]
fn transition_add_sub2(
    src_w: &[i16; N_HIDDEN], src_b: &[i16; N_HIDDEN],
    dst_w: &mut [i16; N_HIDDEN], dst_b: &mut [i16; N_HIDDEN],
    add: (usize, usize), sub1: (usize, usize), sub2: (usize, usize),
) {
    let ft = &WEIGHTS.ft_weights;
    let aw = &ft[add.0];
    let ab = &ft[add.1];
    let s1w = &ft[sub1.0];
    let s1b = &ft[sub1.1];
    let s2w = &ft[sub2.0];
    let s2b = &ft[sub2.1];
    for h in 0..N_HIDDEN {
        dst_w[h] = src_w[h].wrapping_add(aw[h]).wrapping_sub(s1w[h]).wrapping_sub(s2w[h]);
        dst_b[h] = src_b[h].wrapping_add(ab[h]).wrapping_sub(s1b[h]).wrapping_sub(s2b[h]);
    }
}

#[inline]
fn transition_add2_sub2(
    src_w: &[i16; N_HIDDEN], src_b: &[i16; N_HIDDEN],
    dst_w: &mut [i16; N_HIDDEN], dst_b: &mut [i16; N_HIDDEN],
    add1: (usize, usize), add2: (usize, usize),
    sub1: (usize, usize), sub2: (usize, usize),
) {
    let ft = &WEIGHTS.ft_weights;
    let a1w = &ft[add1.0];
    let a1b = &ft[add1.1];
    let a2w = &ft[add2.0];
    let a2b = &ft[add2.1];
    let s1w = &ft[sub1.0];
    let s1b = &ft[sub1.1];
    let s2w = &ft[sub2.0];
    let s2b = &ft[sub2.1];
    for h in 0..N_HIDDEN {
        dst_w[h] = src_w[h]
            .wrapping_add(a1w[h]).wrapping_add(a2w[h])
            .wrapping_sub(s1w[h]).wrapping_sub(s2w[h]);
        dst_b[h] = src_b[h]
            .wrapping_add(a1b[h]).wrapping_add(a2b[h])
            .wrapping_sub(s1b[h]).wrapping_sub(s2b[h]);
    }
}

#[derive(Clone)]
pub struct Accumulator {
    pub white: [i16; N_HIDDEN],
    pub black: [i16; N_HIDDEN],
}

impl Accumulator {
    pub fn empty() -> Self {
        let bias = WEIGHTS.ft_bias;
        Self { white: bias, black: bias }
    }

    #[inline]
    fn add_piece(&mut self, color: Color, kind: PieceKind, sq: usize) {
        let (feat_w, feat_b) = feature_indices(color, kind, sq);
        let ft = &WEIGHTS.ft_weights;
        let row_w = &ft[feat_w];
        let row_b = &ft[feat_b];
        for h in 0..N_HIDDEN {
            self.white[h] = self.white[h].wrapping_add(row_w[h]);
            self.black[h] = self.black[h].wrapping_add(row_b[h]);
        }
    }

    pub fn recompute(board: &Board) -> Self {
        let mut a = Self::empty();
        for color_idx in 0..2 {
            let color = if color_idx == 0 { Color::White } else { Color::Black };
            for kind_idx in 0..6 {
                let kind = PieceKind::from_idx(kind_idx);
                let mut bb = board.piece_bb[color_idx][kind_idx];
                while bb != 0 {
                    let sq = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    a.add_piece(color, kind, sq);
                }
            }
        }
        a
    }

    pub fn evaluate(&self, stm: Color) -> i32 {
        let (us, them) = if stm == Color::White {
            (&self.white, &self.black)
        } else {
            (&self.black, &self.white)
        };
        let sum = head_dot(us, them, &WEIGHTS.head_weights, WEIGHTS.qa as i16, WEIGHTS.head_bias_q);
        ((sum as f32) / ((WEIGHTS.qa * WEIGHTS.qb) as f32)).round() as i32
    }
}

pub const MAX_ACC_PLY: usize = 256;

pub struct AccStack {
    stack: Box<[Accumulator]>,
    ply: usize,
}

impl AccStack {
    pub fn new() -> Self {
        let v: Vec<Accumulator> = (0..MAX_ACC_PLY).map(|_| Accumulator::empty()).collect();
        Self { stack: v.into_boxed_slice(), ply: 0 }
    }

    pub fn init(&mut self, board: &Board) {
        self.ply = 0;
        self.stack[0] = Accumulator::recompute(board);
    }

    #[inline]
    pub fn evaluate(&self, stm: Color) -> i32 {
        self.stack[self.ply].evaluate(stm)
    }

    pub fn push_after(&mut self, board_after: &Board, m: Move, undo: &UndoInfo) {
        debug_assert!(self.ply + 1 < MAX_ACC_PLY);
        let (cur, next) = self.stack.split_at_mut(self.ply + 1);
        let src = &cur[self.ply];
        let dst = &mut next[0];

        let from = m.from();
        let to = m.to();
        let flag = m.flag();
        let stm = board_after.side_to_move.opp();

        let arrived = board_after
            .piece_at(to)
            .expect("push_after: arrived square is empty");
        let original_kind = if m.is_promotion() { PieceKind::Pawn } else { arrived.kind };
        let sub_mover = feature_indices(stm, original_kind, from.0 as usize);
        let add_mover = feature_indices(stm, arrived.kind, to.0 as usize);

        if flag == Move::CASTLE_KING || flag == Move::CASTLE_QUEEN {
            let king_side = flag == Move::CASTLE_KING;
            let (rf, rt) = match (king_side, stm) {
                (true, Color::White) => (Square(7), Square(5)),
                (true, Color::Black) => (Square(63), Square(61)),
                (false, Color::White) => (Square(0), Square(3)),
                (false, Color::Black) => (Square(56), Square(59)),
            };
            let sub_rook = feature_indices(stm, PieceKind::Rook, rf.0 as usize);
            let add_rook = feature_indices(stm, PieceKind::Rook, rt.0 as usize);
            transition_add2_sub2(
                &src.white, &src.black, &mut dst.white, &mut dst.black,
                add_mover, add_rook, sub_mover, sub_rook,
            );
        } else if let Some(cap) = undo.captured {
            let cap_sq = if flag == Move::EP_CAPTURE {
                match stm {
                    Color::White => Square(to.0 - 8),
                    Color::Black => Square(to.0 + 8),
                }
            } else {
                to
            };
            let sub_cap = feature_indices(cap.color, cap.kind, cap_sq.0 as usize);
            transition_add_sub2(
                &src.white, &src.black, &mut dst.white, &mut dst.black,
                add_mover, sub_mover, sub_cap,
            );
        } else {
            transition_add_sub(
                &src.white, &src.black, &mut dst.white, &mut dst.black,
                add_mover, sub_mover,
            );
        }

        self.ply += 1;
    }

    // Piece placement is unchanged, so the parent's accumulator is reused as is.
    #[inline]
    pub fn push_null(&mut self) {}

    #[inline]
    pub fn pop(&mut self) {
        debug_assert!(self.ply > 0);
        self.ply -= 1;
    }

    #[inline]
    pub fn pop_null(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Board, MoveList};

    fn assert_incremental_matches_recompute(fen: &str) {
        let mut board = Board::from_fen(fen).unwrap();
        let mut list = MoveList::new();
        board.gen_legal_into(&mut list);
        for i in 0..list.len() {
            let m = list.get(i);
            let mut accs = AccStack::new();
            accs.init(&board);
            let undo = board.make_move(m);
            accs.push_after(&board, m, &undo);
            let fresh = Accumulator::recompute(&board);
            let inc = &accs.stack[accs.ply];
            assert!(
                inc.white == fresh.white && inc.black == fresh.black,
                "accumulator mismatch after {m} from {fen}",
            );
            board.unmake_move(m, undo);
        }
    }

    #[test]
    fn incremental_matches_recompute_all_move_shapes() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            // Kiwipete
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            // En passant
            "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1",
            // Promotions
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            // Capturing promotion
            "n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
        ];
        for fen in fens {
            assert_incremental_matches_recompute(fen);
        }
    }
}
