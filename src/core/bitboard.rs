// Bitboard helpers: population utilities, precomputed attack tables, and
// magic-bitboard slider attacks.

use super::types::Square;
use std::sync::LazyLock;

// =============================================================================
// BITBOARD HELPERS
// =============================================================================

/// Square of the least-significant set bit. Caller must ensure `bb != 0`.
#[inline]
pub const fn lsb(bb: u64) -> Square {
    Square(bb.trailing_zeros() as u8)
}

/// Pop the LSB from `bb`, returning its square. Returns None if `bb` is
/// already empty. Used to iterate the set squares of a bitboard:
///
/// ```ignore
/// let mut bb = some_bitboard;
/// while let Some(sq) = pop_lsb(&mut bb) {
///     // ... do something with sq ...
/// }
/// ```
#[inline]
pub fn pop_lsb(bb: &mut u64) -> Option<Square> {
    if *bb == 0 {
        return None;
    }
    let sq = lsb(*bb);
    *bb &= *bb - 1; // Clear the lowest set bit.
    Some(sq)
}

// =============================================================================
// ATTACK TABLES (const, evaluated at compile time)
// =============================================================================
//
// Knight, king, and pawn attacks depend only on the source square (and pawn
// color), not on board state - precomputed once at compile time and indexed
// at runtime. Slider attacks (bishop, rook, queen) depend on blocker
// positions and use magic bitboards (`bishop_attacks` / `rook_attacks`);
// `*_raycast` variants are kept as a correctness reference.

const fn knight_attacks_from(sq: u8) -> u64 {
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;
    let mut bb = 0u64;
    // 8 L-shape offsets (df, dr).
    let deltas: [(i8, i8); 8] = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2),  (1, 2),  (2, -1),  (2, 1),
    ];
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

/// Pawn attack squares from `sq` if a pawn of `color` (0 = white, 1 = black)
/// stood there. White pawns attack NE/NW; black pawns attack SE/SW.
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

/// PAWN_ATTACKS[color][sq] = squares a pawn of that color on sq would attack.
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

// =============================================================================
// BETWEEN-SQUARE BITBOARDS (for pin / check-mask computation)
// =============================================================================
// BETWEEN_BB[a][b] = bitboard of squares strictly between `a` and `b` on
// their shared rank, file, or diagonal. Zero if they don't share a line.

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
        (
            if df > 0 { 1i8 } else { -1i8 },
            if dr > 0 { 1i8 } else { -1i8 },
        )
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

pub(crate) const BETWEEN_BB: [[u64; 64]; 64] = {
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

// =============================================================================
// SLIDER ATTACKS (raycast)
// =============================================================================

/// Walk outward from `sq` in each of the supplied directions, stopping at
/// the first occupied square in that direction (inclusive: the blocker
/// square itself is part of the attack set so captures land correctly).
fn slider_attacks(sq: Square, occ: u64, directions: &[(i8, i8)]) -> u64 {
    let mut attacks = 0u64;
    let file0 = sq.file() as i8;
    let rank0 = sq.rank() as i8;
    for &(df, dr) in directions {
        let mut f = file0 + df;
        let mut r = rank0 + dr;
        while f >= 0 && f < 8 && r >= 0 && r < 8 {
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

pub(crate) fn bishop_attacks_raycast(sq: Square, occ: u64) -> u64 {
    slider_attacks(sq, occ, &[(1, 1), (1, -1), (-1, 1), (-1, -1)])
}

pub(crate) fn rook_attacks_raycast(sq: Square, occ: u64) -> u64 {
    slider_attacks(sq, occ, &[(0, 1), (0, -1), (1, 0), (-1, 0)])
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: u64) -> u64 {
    let mi = &*BISHOP_MAGIC;
    let idx = sq.idx();
    unsafe {
        let entry = mi.entries.get_unchecked(idx);
        let blockers = occ & entry.mask;
        let table_idx = (blockers.wrapping_mul(entry.magic) >> entry.shift) as usize;
        *mi.table.get_unchecked(entry.offset + table_idx)
    }
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occ: u64) -> u64 {
    let mi = &*ROOK_MAGIC;
    let idx = sq.idx();
    unsafe {
        let entry = mi.entries.get_unchecked(idx);
        let blockers = occ & entry.mask;
        let table_idx = (blockers.wrapping_mul(entry.magic) >> entry.shift) as usize;
        *mi.table.get_unchecked(entry.offset + table_idx)
    }
}

#[inline]
pub fn queen_attacks(sq: Square, occ: u64) -> u64 {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}

// -----------------------------------------------------------------------------
// Magic bitboards: O(1) slider attack lookup.
//
// Per square, precomputed:
//   - mask: bitboard of "relevant" occupancy squares (the squares whose
//     occupancy actually changes the attack pattern - inner squares of the
//     attack rays, excluding board edges).
//   - magic: a u64 chosen so that `(occ & mask) * magic >> shift` produces
//     a unique table index for every distinct attack pattern.
//   - shift: 64 - popcount(mask)
//   - offset: where this square's table starts in the shared attack table.
//
// Magic numbers are found via random sparse-u64 search at first use; the
// LazyLock ensures init runs once and is then a plain pointer load.
// -----------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct MagicEntry {
    mask: u64,
    magic: u64,
    offset: usize,
    shift: u32,
}

struct MagicTable {
    entries: [MagicEntry; 64],
    table: Box<[u64]>,
}

static BISHOP_MAGIC: LazyLock<MagicTable> = LazyLock::new(|| build_magic_table(true));
static ROOK_MAGIC: LazyLock<MagicTable> = LazyLock::new(|| build_magic_table(false));

fn bishop_mask(sq: u8) -> u64 {
    let file = (sq % 8) as i8;
    let rank = (sq / 8) as i8;
    let mut mask = 0u64;
    let dirs: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    for (df, dr) in dirs {
        let mut f = file + df;
        let mut r = rank + dr;
        while f > 0 && f < 7 && r > 0 && r < 7 {
            mask |= 1u64 << ((r as u8) * 8 + (f as u8));
            f += df;
            r += dr;
        }
    }
    mask
}

fn rook_mask(sq: u8) -> u64 {
    let file = (sq % 8) as i8;
    let rank = (sq / 8) as i8;
    let mut mask = 0u64;
    let mut r = rank + 1;
    while r < 7 {
        mask |= 1u64 << ((r as u8) * 8 + (file as u8));
        r += 1;
    }
    let mut r = rank - 1;
    while r > 0 {
        mask |= 1u64 << ((r as u8) * 8 + (file as u8));
        r -= 1;
    }
    let mut f = file + 1;
    while f < 7 {
        mask |= 1u64 << ((rank as u8) * 8 + (f as u8));
        f += 1;
    }
    let mut f = file - 1;
    while f > 0 {
        mask |= 1u64 << ((rank as u8) * 8 + (f as u8));
        f -= 1;
    }
    mask
}

pub(crate) struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
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

fn find_magic_for_sq(sq: u8, mask: u64, is_bishop: bool, rng: &mut XorShift64) -> (u64, Vec<u64>) {
    let bits = mask.count_ones();
    let n_entries = 1usize << bits;
    let shift = 64 - bits;

    let mut subsets: Vec<(u64, u64)> = Vec::with_capacity(n_entries);
    let mut sub = 0u64;
    loop {
        let attack = if is_bishop {
            bishop_attacks_raycast(Square(sq), sub)
        } else {
            rook_attacks_raycast(Square(sq), sub)
        };
        subsets.push((sub, attack));
        sub = sub.wrapping_sub(mask) & mask;
        if sub == 0 {
            break;
        }
    }

    loop {
        let magic = rng.next() & rng.next() & rng.next();
        if (mask.wrapping_mul(magic) >> 56).count_ones() < 6 {
            continue;
        }

        let mut table = vec![0u64; n_entries];
        let mut used = vec![false; n_entries];
        let mut conflict = false;

        for &(s, attack) in &subsets {
            let idx = ((s.wrapping_mul(magic)) >> shift) as usize;
            if used[idx] {
                if table[idx] != attack {
                    conflict = true;
                    break;
                }
            } else {
                table[idx] = attack;
                used[idx] = true;
            }
        }

        if !conflict {
            return (magic, table);
        }
    }
}

fn build_magic_table(is_bishop: bool) -> MagicTable {
    let mut entries = [MagicEntry { mask: 0, magic: 0, offset: 0, shift: 0 }; 64];
    let mut table: Vec<u64> = Vec::new();
    let mut rng = XorShift64::new(0x6F61D1735F2B3D1E);
    for sq in 0..64u8 {
        let mask = if is_bishop { bishop_mask(sq) } else { rook_mask(sq) };
        let (magic, sub_table) = find_magic_for_sq(sq, mask, is_bishop, &mut rng);
        entries[sq as usize] = MagicEntry {
            mask,
            magic,
            offset: table.len(),
            shift: 64 - mask.count_ones(),
        };
        table.extend(sub_table);
    }
    MagicTable {
        entries,
        table: table.into_boxed_slice(),
    }
}

