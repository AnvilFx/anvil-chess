use super::types::{CastleRights, Color, PieceKind, Square};

struct ZobristTables {
    pieces: [[[u64; 64]; 6]; 2], // [color][kind][sq]
    side: u64,
    // XOR of the keys for each castling-rights mask (bits WK, WQ, BK, BQ).
    castle: [u64; 16],
    ep_file: [u64; 8],
}

const fn xorshift(mut s: u64) -> u64 {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    s
}

// Built at compile time from one xorshift stream: pieces, side, castling rights, ep files.
static ZOBRIST: ZobristTables = {
    let mut state: u64 = 0x1571_E0DC_AFE1_7572;
    let mut pieces = [[[0u64; 64]; 6]; 2];
    let mut c = 0;
    while c < 2 {
        let mut k = 0;
        while k < 6 {
            let mut s = 0;
            while s < 64 {
                state = xorshift(state);
                pieces[c][k][s] = state;
                s += 1;
            }
            k += 1;
        }
        c += 1;
    }
    state = xorshift(state);
    let side = state;
    let mut rights = [0u64; 4];
    let mut i = 0;
    while i < 4 {
        state = xorshift(state);
        rights[i] = state;
        i += 1;
    }
    let mut castle = [0u64; 16];
    let mut mask = 0;
    while mask < 16 {
        let mut b = 0;
        while b < 4 {
            if mask & (1 << b) != 0 {
                castle[mask] ^= rights[b];
            }
            b += 1;
        }
        mask += 1;
    }
    let mut ep_file = [0u64; 8];
    let mut e = 0;
    while e < 8 {
        state = xorshift(state);
        ep_file[e] = state;
        e += 1;
    }
    ZobristTables { pieces, side, castle, ep_file }
};

#[inline]
pub(crate) fn zob_piece(color: Color, kind: PieceKind, sq: Square) -> u64 {
    ZOBRIST.pieces[color.idx()][kind.idx()][sq.idx()]
}

#[inline]
pub(crate) fn zob_meta(side: Color, castle: CastleRights, ep: Option<Square>) -> u64 {
    let mut h = ZOBRIST.castle[(castle.0 & 15) as usize];
    if side == Color::Black {
        h ^= ZOBRIST.side;
    }
    if let Some(sq) = ep {
        h ^= ZOBRIST.ep_file[sq.file() as usize];
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    // Keys must equal those of the original runtime generator.
    #[test]
    fn keys_match_xorshift_stream() {
        let mut state: u64 = 0x1571_E0DC_AFE1_7572;
        let mut next = || {
            state = xorshift(state);
            state
        };
        for c in 0..2 {
            for k in 0..6 {
                for s in 0..64 {
                    assert_eq!(ZOBRIST.pieces[c][k][s], next());
                }
            }
        }
        assert_eq!(ZOBRIST.side, next());
        let rights: Vec<u64> = (0..4).map(|_| next()).collect();
        for e in 0..8 {
            assert_eq!(ZOBRIST.ep_file[e], next());
        }
        for mask in 0..16usize {
            let want = (0..4).filter(|b| mask & (1 << b) != 0).fold(0, |h, b| h ^ rights[b]);
            assert_eq!(ZOBRIST.castle[mask], want);
        }
    }
}
