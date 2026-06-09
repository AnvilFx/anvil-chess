// Zobrist hashing keys and incremental-hash helpers.

use super::types::{CastleRights, Color, PieceKind, Square};
use std::sync::LazyLock;

// =============================================================================
// ZOBRIST HASHING
// =============================================================================
// Each piece-square, the side-to-move toggle, each castle right, and each
// possible en-passant file gets a random u64 key. The position hash is the
// XOR of every applicable key. XOR is its own inverse, so make/unmake can
// update incrementally instead of rescanning the board.
//
// Keys are generated once at startup from a fixed-seed xorshift, so hashes
// are reproducible across runs (matters for tests and TT comparisons).

struct ZobristTables {
    pieces: [[[u64; 64]; 6]; 2], // [color][kind][sq]
    side: u64,
    castle: [u64; 4], // WK, WQ, BK, BQ
    ep_file: [u64; 8],
}

static ZOBRIST: LazyLock<ZobristTables> = LazyLock::new(|| {
    let mut state: u64 = 0x1571_E0DC_AFE1_7572;
    let mut next = || -> u64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut pieces = [[[0u64; 64]; 6]; 2];
    for c in 0..2 {
        for k in 0..6 {
            for s in 0..64 {
                pieces[c][k][s] = next();
            }
        }
    }
    let side = next();
    let mut castle = [0u64; 4];
    for c in castle.iter_mut() {
        *c = next();
    }
    let mut ep_file = [0u64; 8];
    for e in ep_file.iter_mut() {
        *e = next();
    }
    ZobristTables { pieces, side, castle, ep_file }
});

#[inline]
pub(crate) fn zob_piece(color: Color, kind: PieceKind, sq: Square) -> u64 {
    ZOBRIST.pieces[color.idx()][kind.idx()][sq.idx()]
}

// Hash bits for side-to-move + castle rights + en-passant file. Pure
// function of the meta state; piece bits live in the per-square XORs.
pub(crate) fn zob_meta(side: Color, castle: CastleRights, ep: Option<Square>) -> u64 {
    let mut h = 0u64;
    if side == Color::Black {
        h ^= ZOBRIST.side;
    }
    if castle.has(CastleRights::WHITE_KING) {
        h ^= ZOBRIST.castle[0];
    }
    if castle.has(CastleRights::WHITE_QUEEN) {
        h ^= ZOBRIST.castle[1];
    }
    if castle.has(CastleRights::BLACK_KING) {
        h ^= ZOBRIST.castle[2];
    }
    if castle.has(CastleRights::BLACK_QUEEN) {
        h ^= ZOBRIST.castle[3];
    }
    if let Some(sq) = ep {
        h ^= ZOBRIST.ep_file[sq.file() as usize];
    }
    h
}
