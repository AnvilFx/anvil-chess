// Built with target-cpu=native, so plain loops auto-vectorize; only the VNNI head uses intrinsics.

use super::params::*;
use crate::chess::{Board, Color, Move, PieceKind, Square, UndoInfo};

// All lanes are summed at the end, so no lane permute is needed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn head_dot_vnni(
    us: &[i16; N_HIDDEN],
    them: &[i16; N_HIDDEN],
    head: &[i8; 2 * N_HIDDEN],
    qa: i16,
    head_bias: i32,
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
    us: &[i16; N_HIDDEN],
    them: &[i16; N_HIDDEN],
    head: &[i8; 2 * N_HIDDEN],
    qa: i16,
    head_bias: i32,
) -> i32 {
    let mut sum: i32 = head_bias;
    for h in 0..N_HIDDEN {
        sum += us[h].clamp(0, qa) as i32 * head[h] as i32;
        sum += them[h].clamp(0, qa) as i32 * head[N_HIDDEN + h] as i32;
    }
    sum
}

#[inline]
fn head_dot(us: &[i16; N_HIDDEN], them: &[i16; N_HIDDEN], head: &[i8; 2 * N_HIDDEN], qa: i16, head_bias: i32) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avxvnni") {
            return unsafe { head_dot_vnni(us, them, head, qa, head_bias) };
        }
    }
    head_dot_scalar(us, them, head, qa, head_bias)
}

// sum of clamp(x)^2 * w at scale qa^2 * qb. clamp(x) * w fits i16 for |w| <= 128, and each
// i32 lane gathers N_HIDDEN / 4 terms (at most 255 * 32640 each), so lanes can't overflow up to
// N_HIDDEN 1024; the total needs i64.
const _: () = assert!(N_HIDDEN <= 1024, "screlu_dot_avx2 lanes would overflow");
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn screlu_dot_avx2(us: &[i16; N_HIDDEN], them: &[i16; N_HIDDEN], head: &[i16; 2 * N_HIDDEN], qa: i16) -> i64 {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let max = _mm256_set1_epi16(qa);
    let mut acc = zero;
    for (half, src) in [us, them].into_iter().enumerate() {
        let wp = head.as_ptr().add(half * N_HIDDEN);
        for c in 0..N_HIDDEN / 16 {
            let x = _mm256_loadu_si256(src.as_ptr().add(c * 16) as *const __m256i);
            let x = _mm256_min_epi16(_mm256_max_epi16(x, zero), max);
            let w = _mm256_loadu_si256(wp.add(c * 16) as *const __m256i);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, _mm256_mullo_epi16(x, w)));
        }
    }
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    lanes.iter().map(|&v| v as i64).sum()
}

#[inline]
fn screlu_dot_scalar(us: &[i16; N_HIDDEN], them: &[i16; N_HIDDEN], head: &[i16; 2 * N_HIDDEN], qa: i16) -> i64 {
    let mut sum = 0i64;
    for h in 0..N_HIDDEN {
        let (a, b) = (us[h].clamp(0, qa) as i64, them[h].clamp(0, qa) as i64);
        sum += a * a * head[h] as i64 + b * b * head[N_HIDDEN + h] as i64;
    }
    sum
}

#[inline]
fn screlu_dot(us: &[i16; N_HIDDEN], them: &[i16; N_HIDDEN], head: &[i16; 2 * N_HIDDEN], qa: i16) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if N_HIDDEN.is_multiple_of(16) && std::is_x86_feature_detected!("avx2") {
            return unsafe { screlu_dot_avx2(us, them, head, qa) };
        }
    }
    screlu_dot_scalar(us, them, head, qa)
}

const H: usize = N_HIDDEN / 2;
type L1Weights = [[i8; 4 * L1]; N_HIDDEN / 4];
const _: () = assert!(L1 == 16 && H.is_multiple_of(32), "dense SIMD paths assume L1 = 16 and 32-wide halves");

// (clamp(a) * clamp(b)) >> 9 of each side's accumulator halves, side to move first; at most 127,
// so maddubs pairs can't saturate.
fn pairwise_scalar(us: &[i16; N_HIDDEN], them: &[i16; N_HIDDEN], act: &mut [u8; N_HIDDEN]) {
    for (side, src) in [us, them].into_iter().enumerate() {
        for i in 0..H {
            let a = src[i].clamp(0, QA as i16) as i32;
            let b = src[H + i].clamp(0, QA as i16) as i32;
            act[side * H + i] = ((a * b) >> 9) as u8;
        }
    }
}

// mulhi(a << 7, b) is exactly (a * b) >> 9; packus interleaves 128-bit lanes, the permute undoes it.
// Also lists the 4-byte groups with any nonzero byte into `nz` (8 slack slots for the index
// stores) and returns their count.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pairwise_avx2(
    us: &[i16; N_HIDDEN],
    them: &[i16; N_HIDDEN],
    act: &mut [u8; N_HIDDEN],
    nz: &mut [u16; N_HIDDEN / 4 + 8],
) -> usize {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let max = _mm256_set1_epi16(QA as i16);
    let mut n = 0;
    for (side, src) in [us, them].into_iter().enumerate() {
        let p = src.as_ptr();
        let pair = |off: usize| {
            let a = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(p.add(off) as *const __m256i), zero), max);
            let b = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(p.add(H + off) as *const __m256i), zero), max);
            _mm256_mulhi_epi16(_mm256_slli_epi16(a, 7), b)
        };
        for c in 0..H / 32 {
            let packed = _mm256_packus_epi16(pair(c * 32), pair(c * 32 + 16));
            let v = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
            let at = side * H + c * 32;
            _mm256_storeu_si256(act.as_mut_ptr().add(at) as *mut __m256i, v);
            let m = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(v, zero))) as usize;
            let idx = _mm_loadu_si128(MASK_INDICES[m].as_ptr() as *const __m128i);
            _mm_storeu_si128(nz.as_mut_ptr().add(n) as *mut __m128i, _mm_add_epi16(idx, _mm_set1_epi16((at / 4) as i16)));
            n += m.count_ones() as usize;
        }
    }
    n
}

// Set-bit positions of every 8-bit mask, so a mask's indices are written in one store.
static MASK_INDICES: [[u16; 8]; 256] = {
    let mut t = [[0u16; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut bit, mut n) = (0, 0);
        while bit < 8 {
            if m & (1 << bit) != 0 {
                t[m][n] = bit as u16;
                n += 1;
            }
            bit += 1;
        }
        m += 1;
    }
    t
};

// Sparse L1 over the nonzero groups: each group's 4 bytes broadcast against its 16 x 4 weights.
macro_rules! l1_sparse {
    ($name:ident, $features:literal, $dot:expr) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $features)]
        unsafe fn $name(act: &[u8; N_HIDDEN], nz: &[u16], w: &L1Weights) -> [i32; L1] {
            use std::arch::x86_64::*;
            let n = nz.len();
            let groups = act.as_ptr() as *const i32;
            let z = _mm256_setzero_si256();
            // Two groups per step into separate accumulators, to halve the dependency chains.
            let (mut a0, mut a1, mut b0, mut b1) = (z, z, z, z);
            let mut i = 0;
            while i + 1 < n {
                let (g, h) = (nz[i] as usize, nz[i + 1] as usize);
                let x = _mm256_set1_epi32(groups.add(g).read_unaligned());
                let y = _mm256_set1_epi32(groups.add(h).read_unaligned());
                let wg = w[g].as_ptr() as *const __m256i;
                let wh = w[h].as_ptr() as *const __m256i;
                a0 = $dot(a0, x, _mm256_loadu_si256(wg));
                a1 = $dot(a1, x, _mm256_loadu_si256(wg.add(1)));
                b0 = $dot(b0, y, _mm256_loadu_si256(wh));
                b1 = $dot(b1, y, _mm256_loadu_si256(wh.add(1)));
                i += 2;
            }
            if i < n {
                let g = nz[i] as usize;
                let x = _mm256_set1_epi32(groups.add(g).read_unaligned());
                let wg = w[g].as_ptr() as *const __m256i;
                a0 = $dot(a0, x, _mm256_loadu_si256(wg));
                a1 = $dot(a1, x, _mm256_loadu_si256(wg.add(1)));
            }
            let (a0, a1) = (_mm256_add_epi32(a0, b0), _mm256_add_epi32(a1, b1));
            let mut out = [0i32; L1];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, a0);
            _mm256_storeu_si256(out.as_mut_ptr().add(8) as *mut __m256i, a1);
            out
        }
    };
}
l1_sparse!(l1_avxvnni, "avx2,avxvnni", |a, x, w| _mm256_dpbusd_avx_epi32(a, x, w));
l1_sparse!(l1_avx512vnni, "avx2,avx512vnni,avx512vl", |a, x, w| _mm256_dpbusd_epi32(a, x, w));
l1_sparse!(l1_avx2, "avx2", |a, x, w| _mm256_add_epi32(
    a,
    _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), _mm256_set1_epi16(1))
));

fn l1_scalar(act: &[u8; N_HIDDEN], w: &L1Weights) -> [i32; L1] {
    let mut out = [0i32; L1];
    for (g, gw) in w.iter().enumerate() {
        for (j, o) in out.iter_mut().enumerate() {
            for t in 0..4 {
                *o += act[g * 4 + t] as i32 * gw[j * 4 + t] as i32;
            }
        }
    }
    out
}

// Pairwise CReLU of each side's accumulator halves -> L1 -> SCReLU -> L2 -> SCReLU -> output.
fn dense_eval(us: &[i16; N_HIDDEN], them: &[i16; N_HIDDEN], bucket: usize) -> i32 {
    let d = &WEIGHTS.dense[0];
    let mut act = [0u8; N_HIDDEN];
    let l1w = &d.l1w[bucket];
    let s: [i32; L1];
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                let mut nz = [0u16; N_HIDDEN / 4 + 8];
                let n = pairwise_avx2(us, them, &mut act, &mut nz);
                let nz = &nz[..n];
                s = if std::is_x86_feature_detected!("avxvnni") {
                    l1_avxvnni(&act, nz, l1w)
                } else if std::is_x86_feature_detected!("avx512vnni") && std::is_x86_feature_detected!("avx512vl") {
                    l1_avx512vnni(&act, nz, l1w)
                } else {
                    l1_avx2(&act, nz, l1w)
                };
            }
        } else {
            pairwise_scalar(us, them, &mut act);
            s = l1_scalar(&act, l1w);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        pairwise_scalar(us, them, &mut act);
        s = l1_scalar(&act, l1w);
    }

    let mut h2 = d.l2b[bucket];
    for j in 0..L1 {
        let x = (s[j] as f32 * d.l1_dequant[bucket][j] + d.l1b[bucket][j]).clamp(0.0, 1.0);
        let h = x * x;
        for (o, &w) in h2.iter_mut().zip(&d.l2w[bucket][j]) {
            *o += h * w;
        }
    }
    let mut out = d.l3b[bucket];
    for (&x, &w) in h2.iter().zip(&d.l3w[bucket]) {
        let x = x.clamp(0.0, 1.0);
        out += w * x * x;
    }
    (out * OUT_SCALE as f32).round() as i32
}

// Where one side's features live: its king bucket's 768 rows, and a file flip
// when its king stands on files e-h of a mirrored net.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct KingView {
    base: usize,
    flip: usize,
}

impl KingView {
    // `ksq` is the king square from that side's view (rank 1 = its back rank).
    #[inline]
    fn new(ksq: usize) -> Self {
        let flip = if MIRRORED && ksq % 8 > 3 { 7 } else { 0 };
        Self { base: 768 * KING_BUCKET[ksq] as usize, flip }
    }

    #[inline]
    fn of(board: &Board) -> (Self, Self) {
        let king = PieceKind::King.idx();
        let wk = board.piece_bb[0][king].trailing_zeros() as usize;
        let bk = board.piece_bb[1][king].trailing_zeros() as usize;
        (Self::new(wk), Self::new(bk ^ 56))
    }

    // Slot in the refresh cache: one per bucket and mirror side.
    #[inline]
    fn slot(self) -> usize {
        self.base / 768 * 2 + (self.flip != 0) as usize
    }
}

// Returns (white-perspective, black-perspective) feature indices.
#[inline]
fn feature_indices(color: Color, kind: PieceKind, sq: usize, wv: KingView, bv: KingView) -> (usize, usize) {
    let kind_w = if color == Color::White { kind.idx() } else { kind.idx() + 6 };
    let feat_w = wv.base + kind_w * 64 + (sq ^ wv.flip);
    let kind_b = if color == Color::Black { kind.idx() } else { kind.idx() + 6 };
    let feat_b = bv.base + kind_b * 64 + (sq ^ 56 ^ bv.flip);
    (feat_w, feat_b)
}

// One side's accumulator from scratch.
fn refresh(board: &Board, view: KingView, white_view: bool) -> [i16; N_HIDDEN] {
    let mut acc = WEIGHTS.ft_bias;
    for color_idx in 0..2 {
        let color = if color_idx == 0 { Color::White } else { Color::Black };
        for kind_idx in 0..6 {
            let kind = PieceKind::from_idx(kind_idx);
            let mut bb = board.piece_bb[color_idx][kind_idx];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                let (feat_w, feat_b) = feature_indices(color, kind, sq, view, view);
                let row = &WEIGHTS.ft_weights[if white_view { feat_w } else { feat_b }];
                for h in 0..N_HIDDEN {
                    acc[h] = acc[h].wrapping_add(row[h]);
                }
            }
        }
    }
    acc
}

// The last accumulator built for one king view and the pieces it was built from, so a
// refresh only applies what changed since ("Finny table").
#[derive(Clone)]
struct CacheEntry {
    acc: [i16; N_HIDDEN],
    pieces: [[u64; 6]; 2],
}

fn refresh_cached(entry: &mut CacheEntry, board: &Board, view: KingView, white_view: bool) -> [i16; N_HIDDEN] {
    for color_idx in 0..2 {
        let color = if color_idx == 0 { Color::White } else { Color::Black };
        for kind_idx in 0..6 {
            let kind = PieceKind::from_idx(kind_idx);
            let now = board.piece_bb[color_idx][kind_idx];
            let old = entry.pieces[color_idx][kind_idx];
            let row = |sq: usize| {
                let (feat_w, feat_b) = feature_indices(color, kind, sq, view, view);
                &WEIGHTS.ft_weights[if white_view { feat_w } else { feat_b }]
            };
            let mut added = now & !old;
            while added != 0 {
                let r = row(added.trailing_zeros() as usize);
                added &= added - 1;
                for (a, &w) in entry.acc.iter_mut().zip(r) {
                    *a = a.wrapping_add(w);
                }
            }
            let mut removed = old & !now;
            while removed != 0 {
                let r = row(removed.trailing_zeros() as usize);
                removed &= removed - 1;
                for (a, &w) in entry.acc.iter_mut().zip(r) {
                    *a = a.wrapping_sub(w);
                }
            }
            entry.pieces[color_idx][kind_idx] = now;
        }
    }
    entry.acc
}

// Real positions never overflow i16; wrapping ops just avoid debug-mode panics.
#[inline]
fn transition_add_sub(
    src_w: &[i16; N_HIDDEN],
    src_b: &[i16; N_HIDDEN],
    dst_w: &mut [i16; N_HIDDEN],
    dst_b: &mut [i16; N_HIDDEN],
    add: (usize, usize),
    sub: (usize, usize),
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
    src_w: &[i16; N_HIDDEN],
    src_b: &[i16; N_HIDDEN],
    dst_w: &mut [i16; N_HIDDEN],
    dst_b: &mut [i16; N_HIDDEN],
    add: (usize, usize),
    sub1: (usize, usize),
    sub2: (usize, usize),
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
#[allow(clippy::too_many_arguments)]
fn transition_add2_sub2(
    src_w: &[i16; N_HIDDEN],
    src_b: &[i16; N_HIDDEN],
    dst_w: &mut [i16; N_HIDDEN],
    dst_b: &mut [i16; N_HIDDEN],
    add1: (usize, usize),
    add2: (usize, usize),
    sub1: (usize, usize),
    sub2: (usize, usize),
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
        dst_w[h] = src_w[h].wrapping_add(a1w[h]).wrapping_add(a2w[h]).wrapping_sub(s1w[h]).wrapping_sub(s2w[h]);
        dst_b[h] = src_b[h].wrapping_add(a1b[h]).wrapping_add(a2b[h]).wrapping_sub(s1b[h]).wrapping_sub(s2b[h]);
    }
}

#[derive(Clone)]
pub struct Accumulator {
    pub white: [i16; N_HIDDEN],
    pub black: [i16; N_HIDDEN],
    wv: KingView,
    bv: KingView,
}

impl Accumulator {
    pub fn empty() -> Self {
        let bias = WEIGHTS.ft_bias;
        Self { white: bias, black: bias, wv: KingView::default(), bv: KingView::default() }
    }

    pub fn recompute(board: &Board) -> Self {
        let (wv, bv) = KingView::of(board);
        Self { white: refresh(board, wv, true), black: refresh(board, bv, false), wv, bv }
    }

    pub fn evaluate(&self, stm: Color, bucket: usize) -> i32 {
        let (us, them) = if stm == Color::White { (&self.white, &self.black) } else { (&self.black, &self.white) };
        if DENSE {
            return dense_eval(us, them, bucket);
        }
        let w = &*WEIGHTS;
        let sum = if SCRELU {
            let sq = screlu_dot(us, them, &w.head_weights_i16[bucket], QA as i16);
            (sq / QA as i64) as i32 + w.head_bias_q[bucket]
        } else {
            head_dot(us, them, &w.head_weights[bucket], QA as i16, w.head_bias_q[bucket])
        };
        ((sum as f32) * OUT_SCALE as f32 / ((QA * QB) as f32)).round() as i32
    }
}

pub const MAX_ACC_PLY: usize = 256;

pub struct AccStack {
    stack: Box<[Accumulator]>,
    ply: usize,
    // Refresh cache per side (white, black), indexed by KingView::slot.
    cache: [Vec<CacheEntry>; 2],
}

impl Default for AccStack {
    fn default() -> Self {
        Self::new()
    }
}

impl AccStack {
    pub fn new() -> Self {
        let v: Vec<Accumulator> = (0..MAX_ACC_PLY).map(|_| Accumulator::empty()).collect();
        let empty = CacheEntry { acc: WEIGHTS.ft_bias, pieces: [[0; 6]; 2] };
        let slots = WEIGHTS.ft_weights.len() / 768 * 2;
        let cache = [vec![empty.clone(); slots], vec![empty; slots]];
        Self { stack: v.into_boxed_slice(), ply: 0, cache }
    }

    pub fn init(&mut self, board: &Board) {
        self.ply = 0;
        self.stack[0] = Accumulator::recompute(board);
    }

    #[inline]
    pub fn evaluate(&self, board: &Board) -> i32 {
        self.stack[self.ply].evaluate(board.side_to_move, output_bucket(board.occupancy))
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
        let (wv, bv) = KingView::of(board_after);

        let arrived = board_after.piece_at(to).expect("push_after: arrived square is empty");
        let original_kind = if m.is_promotion() { PieceKind::Pawn } else { arrived.kind };
        let sub_mover = feature_indices(stm, original_kind, from.0 as usize, wv, bv);
        let add_mover = feature_indices(stm, arrived.kind, to.0 as usize, wv, bv);

        if flag == Move::CASTLE_KING || flag == Move::CASTLE_QUEEN {
            let king_side = flag == Move::CASTLE_KING;
            let (rf, rt) = match (king_side, stm) {
                (true, Color::White) => (Square(7), Square(5)),
                (true, Color::Black) => (Square(63), Square(61)),
                (false, Color::White) => (Square(0), Square(3)),
                (false, Color::Black) => (Square(56), Square(59)),
            };
            let sub_rook = feature_indices(stm, PieceKind::Rook, rf.0 as usize, wv, bv);
            let add_rook = feature_indices(stm, PieceKind::Rook, rt.0 as usize, wv, bv);
            transition_add2_sub2(
                &src.white,
                &src.black,
                &mut dst.white,
                &mut dst.black,
                add_mover,
                add_rook,
                sub_mover,
                sub_rook,
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
            let sub_cap = feature_indices(cap.color, cap.kind, cap_sq.0 as usize, wv, bv);
            transition_add_sub2(&src.white, &src.black, &mut dst.white, &mut dst.black, add_mover, sub_mover, sub_cap);
        } else {
            transition_add_sub(&src.white, &src.black, &mut dst.white, &mut dst.black, add_mover, sub_mover);
        }

        // A king that changed bucket or crossed the mirror line invalidates its side's
        // incremental update (made with the new view), so that side is rebuilt.
        if wv != src.wv {
            dst.white = refresh_cached(&mut self.cache[0][wv.slot()], board_after, wv, true);
        }
        if bv != src.bv {
            dst.black = refresh_cached(&mut self.cache[1][bv.slot()], board_after, bv, false);
        }
        dst.wv = wv;
        dst.bv = bv;

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
    use crate::chess::{Board, MoveList};

    // Every SIMD path this CPU supports must match the scalar dense layers bit for bit.
    #[test]
    fn dense_simd_matches_scalar() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..50 {
            let us: [i16; N_HIDDEN] = std::array::from_fn(|_| (next() % 700) as i16 - 200);
            let them: [i16; N_HIDDEN] = std::array::from_fn(|_| (next() % 700) as i16 - 200);
            let w: Box<L1Weights> = Box::new(std::array::from_fn(|_| std::array::from_fn(|_| (next() % 255) as i8)));
            let mut want = [0u8; N_HIDDEN];
            pairwise_scalar(&us, &them, &mut want);
            let l1_want = l1_scalar(&want, &w);
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::is_x86_feature_detected as has;
                if !has!("avx2") {
                    continue;
                }
                let mut got = [0u8; N_HIDDEN];
                let mut nz = [0u16; N_HIDDEN / 4 + 8];
                let n = pairwise_avx2(&us, &them, &mut got, &mut nz);
                assert_eq!(got, want);
                let nz_want: Vec<u16> = (0..N_HIDDEN / 4).filter(|&g| want[g * 4..g * 4 + 4] != [0; 4]).map(|g| g as u16).collect();
                assert_eq!(&nz[..n], &nz_want[..]);
                let nz = &nz[..n];
                assert_eq!(l1_avx2(&want, nz, &w), l1_want);
                if has!("avxvnni") {
                    assert_eq!(l1_avxvnni(&want, nz, &w), l1_want);
                }
                if has!("avx512vnni") && has!("avx512vl") {
                    assert_eq!(l1_avx512vnni(&want, nz, &w), l1_want);
                }
            }
        }
    }

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
            assert!(inc.white == fresh.white && inc.black == fresh.black, "accumulator mismatch after {m} from {fen}",);
            board.unmake_move(m, undo);
        }
    }

    // Long games through one stack, so cached refreshes build on earlier ones.
    #[test]
    fn cached_refresh_matches_recompute_over_long_games() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 0 1",
            "r6r/1k6/8/8/8/8/5K2/7R w - - 0 1",
        ];
        for fen in fens {
            let mut board = Board::from_fen(fen).unwrap();
            let mut accs = AccStack::new();
            accs.init(&board);
            for ply in 0..200 {
                let mut list = MoveList::new();
                board.gen_legal_into(&mut list);
                if list.is_empty() {
                    break;
                }
                let m = list.get((ply * 7 + 3) % list.len());
                let undo = board.make_move(m);
                accs.push_after(&board, m, &undo);
                let fresh = Accumulator::recompute(&board);
                let inc = &accs.stack[accs.ply];
                assert!(
                    inc.white == fresh.white && inc.black == fresh.black,
                    "mismatch after {m} at ply {ply} from {fen}"
                );
                // Keep the stack shallow; the board walks on.
                accs.stack[0] = accs.stack[accs.ply].clone();
                accs.ply = 0;
            }
        }
    }

    #[test]
    fn screlu_simd_matches_scalar_at_extremes() {
        let us: [i16; N_HIDDEN] = std::array::from_fn(|h| (h as i16 * 37) % 400 - 100);
        let them = [i16::MAX; N_HIDDEN];
        let mixed: [i16; 2 * N_HIDDEN] = std::array::from_fn(|i| ((i * 53) % 255) as i16 - 127);
        for head in [mixed, [127; 2 * N_HIDDEN], [-127; 2 * N_HIDDEN]] {
            let want = screlu_dot_scalar(&us, &them, &head, 255);
            assert_eq!(screlu_dot(&us, &them, &head, 255), want);
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
