use super::types::Square;

/// `bb` must be non-zero.
#[inline]
pub const fn lsb(bb: u64) -> Square {
    Square(bb.trailing_zeros() as u8)
}

#[inline]
pub fn pop_lsb(bb: &mut u64) -> Option<Square> {
    if *bb == 0 {
        return None;
    }
    let sq = lsb(*bb);
    *bb &= *bb - 1;
    Some(sq)
}

const fn knight_attacks_from(sq: u8) -> u64 {
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;
    let mut bb = 0u64;
    let deltas: [(i8, i8); 8] = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)];
    let mut i = 0;
    while i < 8 {
        let (df, dr) = deltas[i];
        let nf = file + df;
        let nr = rank + dr;
        if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
            bb |= 1u64 << ((nr as u8) * 8 + (nf as u8));
        }
        i += 1;
    }
    bb
}

const fn king_attacks_from(sq: u8) -> u64 {
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;
    let mut bb = 0u64;
    let mut df: i8 = -1;
    while df <= 1 {
        let mut dr: i8 = -1;
        while dr <= 1 {
            if !(df == 0 && dr == 0) {
                let nf = file + df;
                let nr = rank + dr;
                if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                    bb |= 1u64 << ((nr as u8) * 8 + (nf as u8));
                }
            }
            dr += 1;
        }
        df += 1;
    }
    bb
}

/// `color`: 0 = white, 1 = black.
const fn pawn_attacks_from(color: u8, sq: u8) -> u64 {
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;
    let dr: i8 = if color == 0 { 1 } else { -1 };
    let mut bb = 0u64;
    let mut i = 0;
    let dfs: [i8; 2] = [-1, 1];
    while i < 2 {
        let nf = file + dfs[i];
        let nr = rank + dr;
        if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
            bb |= 1u64 << ((nr as u8) * 8 + (nf as u8));
        }
        i += 1;
    }
    bb
}

pub const KNIGHT_ATTACKS: [u64; 64] = {
    let mut arr = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        arr[i] = knight_attacks_from(i as u8);
        i += 1;
    }
    arr
};

pub const KING_ATTACKS: [u64; 64] = {
    let mut arr = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        arr[i] = king_attacks_from(i as u8);
        i += 1;
    }
    arr
};

pub const PAWN_ATTACKS: [[u64; 64]; 2] = {
    let mut arr = [[0u64; 64]; 2];
    let mut c = 0;
    while c < 2 {
        let mut i = 0;
        while i < 64 {
            arr[c][i] = pawn_attacks_from(c as u8, i as u8);
            i += 1;
        }
        c += 1;
    }
    arr
};

/// Squares attacked by all pawns in `pawns` of `color` (0 = white, 1 = black) at once.
#[inline(always)]
pub const fn pawn_attacks_set(pawns: u64, color: usize) -> u64 {
    const NOT_A: u64 = !0x0101_0101_0101_0101;
    const NOT_H: u64 = !0x8080_8080_8080_8080;
    if color == 0 {
        ((pawns & NOT_A) << 7) | ((pawns & NOT_H) << 9)
    } else {
        ((pawns & NOT_A) >> 9) | ((pawns & NOT_H) >> 7)
    }
}

const fn between_bb_calc(from: u8, to: u8) -> u64 {
    let af = (from % 8) as i8;
    let ar = (from / 8) as i8;
    let bf = (to % 8) as i8;
    let br = (to / 8) as i8;
    let df = bf - af;
    let dr = br - ar;
    if df == 0 && dr == 0 {
        return 0;
    }
    let (step_f, step_r) = if df == 0 {
        (0i8, if dr > 0 { 1i8 } else { -1i8 })
    } else if dr == 0 {
        (if df > 0 { 1i8 } else { -1i8 }, 0i8)
    } else if (if df > 0 { df } else { -df }) == (if dr > 0 { dr } else { -dr }) {
        (if df > 0 { 1i8 } else { -1i8 }, if dr > 0 { 1i8 } else { -1i8 })
    } else {
        return 0;
    };
    let mut bb = 0u64;
    let mut f = af + step_f;
    let mut r = ar + step_r;
    while f != bf || r != br {
        bb |= 1u64 << ((r as u8) * 8 + (f as u8));
        f += step_f;
        r += step_r;
    }
    bb
}

/// Squares strictly between `a` and `b` on a shared line; zero if not aligned.
pub(crate) static BETWEEN_BB: [[u64; 64]; 64] = {
    let mut arr = [[0u64; 64]; 64];
    let mut i = 0;
    while i < 64 {
        let mut j = 0;
        while j < 64 {
            arr[i][j] = between_bb_calc(i as u8, j as u8);
            j += 1;
        }
        i += 1;
    }
    arr
};

const fn line_bb_calc(a: u8, b: u8) -> u64 {
    if a == b {
        return 0;
    }
    let (af, ar, bf, br) = ((a % 8) as i8, (a / 8) as i8, (b % 8) as i8, (b / 8) as i8);
    let (df, dr) = (bf - af, br - ar);
    let aligned = df == 0 || dr == 0 || df == dr || df == -dr;
    if !aligned {
        return 0;
    }
    let (sf, sr) = (df.signum(), dr.signum());
    let mut bb = 1u64 << a;
    let mut dir = 0;
    while dir < 2 {
        let (stf, str) = if dir == 0 { (sf, sr) } else { (-sf, -sr) };
        let mut f = af + stf;
        let mut r = ar + str;
        while f >= 0 && f < 8 && r >= 0 && r < 8 {
            bb |= 1u64 << ((r as u8) * 8 + (f as u8));
            f += stf;
            r += str;
        }
        dir += 1;
    }
    bb
}

/// The full board line through `a` and `b`, edge to edge; zero if not aligned.
pub(crate) static LINE_BB: [[u64; 64]; 64] = {
    let mut arr = [[0u64; 64]; 64];
    let mut i = 0;
    while i < 64 {
        let mut j = 0;
        while j < 64 {
            arr[i][j] = line_bb_calc(i as u8, j as u8);
            j += 1;
        }
        i += 1;
    }
    arr
};

#[cfg(test)]
fn slider_attacks(sq: Square, occ: u64, directions: &[(i8, i8)]) -> u64 {
    let mut attacks = 0u64;
    let file0 = sq.file() as i8;
    let rank0 = sq.rank() as i8;
    for &(df, dr) in directions {
        let mut f = file0 + df;
        let mut r = rank0 + dr;
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let target = (r as u8) * 8 + (f as u8);
            let bit = 1u64 << target;
            attacks |= bit;
            if occ & bit != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    attacks
}

#[cfg(test)]
pub(crate) fn bishop_attacks_raycast(sq: Square, occ: u64) -> u64 {
    slider_attacks(sq, occ, &[(1, 1), (1, -1), (-1, 1), (-1, -1)])
}

#[cfg(test)]
pub(crate) fn rook_attacks_raycast(sq: Square, occ: u64) -> u64 {
    slider_attacks(sq, occ, &[(0, 1), (0, -1), (1, 0), (-1, 0)])
}

const fn ray_attacks(sq: u8, occ: u64, rook: bool) -> u64 {
    let f0 = (sq % 8) as i8;
    let r0 = (sq / 8) as i8;
    let dirs: [(i8, i8); 4] =
        if rook { [(0, 1), (0, -1), (1, 0), (-1, 0)] } else { [(1, 1), (1, -1), (-1, 1), (-1, -1)] };
    let mut attacks = 0u64;
    let mut d = 0;
    while d < 4 {
        let (df, dr) = dirs[d];
        let mut f = f0 + df;
        let mut r = r0 + dr;
        while f >= 0 && f < 8 && r >= 0 && r < 8 {
            let bit = 1u64 << ((r as u8) * 8 + f as u8);
            attacks |= bit;
            if occ & bit != 0 {
                break;
            }
            f += df;
            r += dr;
        }
        d += 1;
    }
    attacks
}

/// Relevant blockers: the empty-board rays without the board edge they run into.
const fn relevant_mask(sq: u8, rook: bool) -> u64 {
    let (f, r) = (sq % 8, sq / 8);
    let edges = if rook {
        (if f != 0 { 0x0101_0101_0101_0101 } else { 0 })
            | (if f != 7 { 0x8080_8080_8080_8080 } else { 0 })
            | (if r != 0 { 0xFF } else { 0 })
            | (if r != 7 { 0xFF00_0000_0000_0000 } else { 0 })
    } else {
        0xFF81_8181_8181_81FF
    };
    ray_attacks(sq, 0, rook) & !edges
}

const fn empty_rays(rook: bool) -> [u64; 64] {
    let mut t = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        t[sq] = ray_attacks(sq as u8, 0, rook);
        sq += 1;
    }
    t
}

/// Slider attacks on an empty board.
pub(crate) static BISHOP_RAYS: [u64; 64] = empty_rays(false);
pub(crate) static ROOK_RAYS: [u64; 64] = empty_rays(true);

// BMI2 PEXT indexes the attack tables directly; they are built at compile time.
#[cfg(all(target_feature = "bmi2", pext))]
mod pext {
    use super::{ray_attacks, relevant_mask, Square};

    #[derive(Clone, Copy)]
    struct PextEntry {
        mask: u64,
        offset: usize,
    }

    const fn pext_entries(rook: bool) -> [PextEntry; 64] {
        let mut e = [PextEntry { mask: 0, offset: 0 }; 64];
        let mut off = 0;
        let mut sq = 0;
        while sq < 64 {
            let mask = relevant_mask(sq as u8, rook);
            e[sq] = PextEntry { mask, offset: off };
            off += 1 << mask.count_ones();
            sq += 1;
        }
        e
    }

    const ROOK_PEXT_SIZE: usize = 102_400;
    const BISHOP_PEXT_SIZE: usize = 5_248;
    static ROOK_PEXT_ENTRIES: [PextEntry; 64] = pext_entries(true);
    static BISHOP_PEXT_ENTRIES: [PextEntry; 64] = pext_entries(false);

    // Carry-Rippler walks the subsets of a mask in increasing PEXT index order.
    const fn pext_table<const N: usize>(rook: bool) -> [u64; N] {
        let mut t = [0u64; N];
        let mut sq = 0;
        let mut off = 0;
        while sq < 64 {
            let mask = relevant_mask(sq as u8, rook);
            let mut sub = 0u64;
            loop {
                t[off] = ray_attacks(sq as u8, sub, rook);
                off += 1;
                sub = sub.wrapping_sub(mask) & mask;
                if sub == 0 {
                    break;
                }
            }
            sq += 1;
        }
        t
    }

    static ROOK_PEXT: [u64; ROOK_PEXT_SIZE] = pext_table(true);
    static BISHOP_PEXT: [u64; BISHOP_PEXT_SIZE] = pext_table(false);

    #[inline(always)]
    pub fn bishop_attacks(sq: Square, occ: u64) -> u64 {
        unsafe {
            let e = BISHOP_PEXT_ENTRIES.get_unchecked(sq.idx());
            *BISHOP_PEXT.get_unchecked(e.offset + std::arch::x86_64::_pext_u64(occ, e.mask) as usize)
        }
    }

    #[inline(always)]
    pub fn rook_attacks(sq: Square, occ: u64) -> u64 {
        unsafe {
            let e = ROOK_PEXT_ENTRIES.get_unchecked(sq.idx());
            *ROOK_PEXT.get_unchecked(e.offset + std::arch::x86_64::_pext_u64(occ, e.mask) as usize)
        }
    }
}

#[cfg(all(target_feature = "bmi2", pext))]
pub use pext::{bishop_attacks, rook_attacks};

// Magic bitboards for targets without fast PEXT. The magics were found offline (the xorshift
// search the engine used to run at startup) and the tables are built at compile time, so a
// lookup needs no lazy-init check and no pointer chase. A bad magic fails the build.
#[cfg(not(all(target_feature = "bmi2", pext)))]
mod magic {
    use super::{ray_attacks, relevant_mask, Square};

    const BISHOP_MAGICS: [u64; 64] = [
        0x1040904200882280, 0x0008500408404020, 0x1004410202042000, 0x8084040C80100084,
        0x8001104000080400, 0x00C2082308000002, 0x4011048874402340, 0x00A08044100108C0,
        0x2000C04222040100, 0x6110044404004A00, 0x0000040802004320, 0x0880080851010808,
        0x8C00020210300900, 0x04C40A0802280680, 0x4400010430020980, 0x40008904008404A4,
        0x0020004308021492, 0x00200202140420A0, 0x2010210800801010, 0x4004000241020000,
        0x4204108202020002, 0x0002020420842000, 0x8202003900A22000, 0x0000400084008800,
        0x42040400202084A0, 0x52029001A8900880, 0x9008080001060820, 0x00248080080200A2,
        0x00C1010050104000, 0x00009A0024880400, 0x01421A0000411010, 0x0944048002262108,
        0x080904C026A23800, 0x0044226800200100, 0x1000108210100400, 0x0042010041A40040,
        0x00400808214A0020, 0x4008020C20041011, 0x2048020084042080, 0x048203B204A100C0,
        0xA0010421200404A0, 0x0220480808184400, 0x4802010401088200, 0x001048420085A800,
        0x0010943092000400, 0x0120020060400200, 0x6C08080B014A4410, 0x4230023200210048,
        0x4025011090440001, 0x0014840401040000, 0x02600D8414090100, 0xA090E40042020000,
        0x4120001002022808, 0x00800910213200C0, 0x0020081001304082, 0x0002080855104001,
        0x04010423100A1000, 0x0002014046101100, 0x0014000100880400, 0x0000C00000C20880,
        0x2800000C04208200, 0x8020008460040104, 0x0080508519340400, 0x0120080108362041,
    ];
    const ROOK_MAGICS: [u64; 64] = [
        0x0180002950844000, 0x0280200040018030, 0x03001B0060004090, 0x1180080004100080,
        0x0200048802002090, 0x0100010002080400, 0x0400240081100822, 0x0100010000304882,
        0x4300800080400020, 0x0140401000200040, 0x0022002012008040, 0x2811800800100180,
        0x1408808004000800, 0x6082800201800400, 0x0001000100020004, 0x200500078A104100,
        0x0000828000400030, 0x0210004010402000, 0x2802020040201480, 0x0800808008001000,
        0x6A01010010060800, 0x0000808004000200, 0x0804010100040200, 0x2000220000409114,
        0x2020208080004010, 0x1202230300400080, 0x0010004101002000, 0x0008018180100008,
        0xC008000880800400, 0x0802040080020080, 0x4044814400021008, 0x0004801C80044100,
        0x0240008054800024, 0x0040201002400241, 0xD010120022004080, 0x2440100080800802,
        0x0200040080800801, 0x0600800400800200, 0x8020821004002118, 0x0000801060800100,
        0x0000208040008003, 0x0240002002838048, 0x0000200010008080, 0x0408100008008080,
        0x1180080004008080, 0x0052000510820008, 0x0020C23008040011, 0x00003102418A0024,
        0x0060800020410100, 0x2000200080400080, 0x4103412004150100, 0x0400080110008180,
        0x0302040208008080, 0x0009810200040080, 0x0212000401088200, 0x0000012444048200,
        0x000601A100144082, 0x0201042340008115, 0x8004102200800842, 0x0008081000042101,
        0x2401001004020801, 0x2001000400020847, 0x3908081210408104, 0x0001008024010842,
    ];

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MagicEntry {
        mask: u64,
        magic: u64,
        offset: usize,
        shift: u32,
    }

    const fn magic_entries(rook: bool) -> [MagicEntry; 64] {
        let mut e = [MagicEntry { mask: 0, magic: 0, offset: 0, shift: 0 }; 64];
        let mut off = 0;
        let mut sq = 0;
        while sq < 64 {
            let mask = relevant_mask(sq as u8, rook);
            let magic = if rook { ROOK_MAGICS[sq] } else { BISHOP_MAGICS[sq] };
            e[sq] = MagicEntry { mask, magic, offset: off, shift: 64 - mask.count_ones() };
            off += 1 << mask.count_ones();
            sq += 1;
        }
        e
    }

    const ROOK_SIZE: usize = 102_400;
    const BISHOP_SIZE: usize = 5_248;
    static ROOK_ENTRIES: [MagicEntry; 64] = magic_entries(true);
    static BISHOP_ENTRIES: [MagicEntry; 64] = magic_entries(false);

    // Every subset of every mask goes to its magic index; two subsets with different attacks on one
    // index would be a bad magic, which stops compilation. Attack sets are never empty, so 0 = unused.
    const fn magic_table<const N: usize>(rook: bool) -> [u64; N] {
        let entries = magic_entries(rook);
        let mut t = [0u64; N];
        let mut sq = 0;
        while sq < 64 {
            let e = entries[sq];
            let mut sub = 0u64;
            loop {
                let a = ray_attacks(sq as u8, sub, rook);
                let i = e.offset + (sub.wrapping_mul(e.magic) >> e.shift) as usize;
                assert!(t[i] == 0 || t[i] == a, "bad magic");
                t[i] = a;
                sub = sub.wrapping_sub(e.mask) & e.mask;
                if sub == 0 {
                    break;
                }
            }
            sq += 1;
        }
        t
    }

    static ROOK_TABLE: [u64; ROOK_SIZE] = magic_table(true);
    static BISHOP_TABLE: [u64; BISHOP_SIZE] = magic_table(false);

    #[inline(always)]
    pub fn bishop_attacks(sq: Square, occ: u64) -> u64 {
        // SAFETY: sq < 64, and offset + (masked occ * magic >> shift) < offset + 2^bits, inside the table.
        unsafe {
            let e = BISHOP_ENTRIES.get_unchecked(sq.idx());
            *BISHOP_TABLE.get_unchecked(e.offset + ((occ & e.mask).wrapping_mul(e.magic) >> e.shift) as usize)
        }
    }

    #[inline(always)]
    pub fn rook_attacks(sq: Square, occ: u64) -> u64 {
        // SAFETY: as in bishop_attacks.
        unsafe {
            let e = ROOK_ENTRIES.get_unchecked(sq.idx());
            *ROOK_TABLE.get_unchecked(e.offset + ((occ & e.mask).wrapping_mul(e.magic) >> e.shift) as usize)
        }
    }
}

#[cfg(not(all(target_feature = "bmi2", pext)))]
pub use magic::{bishop_attacks, rook_attacks};

#[inline]
pub fn queen_attacks(sq: Square, occ: u64) -> u64 {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}

#[cfg(test)]
pub(crate) struct XorShift64 {
    state: u64,
}

#[cfg(test)]
impl XorShift64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }
    pub(crate) fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}
