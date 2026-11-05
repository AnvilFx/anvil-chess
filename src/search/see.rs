// Static exchange evaluation.
use super::*;

pub(super) const SEE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 10000];

pub(super) fn attackers_of(board: &Board, sq: Square, side: Color, occ: u64) -> u64 {
    let side_idx = side.idx();
    let opp_idx = side.opp().idx();
    let p = PAWN_ATTACKS[opp_idx][sq.idx()] & board.piece_bb[side_idx][PieceKind::Pawn.idx()];
    let n = KNIGHT_ATTACKS[sq.idx()] & board.piece_bb[side_idx][PieceKind::Knight.idx()];
    let bq_targets = bishop_attacks(sq, occ);
    let rq_targets = rook_attacks(sq, occ);
    let bq = bq_targets
        & (board.piece_bb[side_idx][PieceKind::Bishop.idx()] | board.piece_bb[side_idx][PieceKind::Queen.idx()]);
    let rq = rq_targets
        & (board.piece_bb[side_idx][PieceKind::Rook.idx()] | board.piece_bb[side_idx][PieceKind::Queen.idx()]);
    let k = KING_ATTACKS[sq.idx()] & board.piece_bb[side_idx][PieceKind::King.idx()];
    (p | n | bq | rq | k) & occ
}

pub(super) fn least_valuable(board: &Board, side: Color, attackers: u64) -> Option<(Square, PieceKind)> {
    let side_idx = side.idx();
    for kind_idx in 0..6 {
        let cand = attackers & board.piece_bb[side_idx][kind_idx];
        if cand != 0 {
            return Some((lsb(cand), PieceKind::from_idx(kind_idx)));
        }
    }
    None
}

// SEE is never negative when the first capture already wins at least the value of the
// piece now standing on the square: whatever the exchange, the capturer can stop there.
#[inline]
pub(super) fn capture_cannot_lose(board: &Board, m: Move) -> bool {
    let Some(attacker) = board.piece_at(m.from()) else { return false };
    let victim = if m.flag() == Move::EP_CAPTURE {
        PieceKind::Pawn
    } else {
        match board.piece_at(m.to()) {
            Some(p) => p.kind,
            None => return false,
        }
    };
    let mut gain = SEE_VALUES[victim.idx()];
    let standing = match m.promo_kind() {
        Some(promo) => {
            gain += SEE_VALUES[promo.idx()] - SEE_VALUES[PieceKind::Pawn.idx()];
            promo
        }
        None => attacker.kind,
    };
    gain >= SEE_VALUES[standing.idx()]
}

pub(super) fn see(board: &Board, m: Move) -> i32 {
    let to = m.to();
    let from = m.from();
    let attacker = match board.piece_at(from) {
        Some(p) => p,
        None => return 0,
    };
    let mut occ = board.occupancy & !from.bb();

    let captured_value = if m.flag() == Move::EP_CAPTURE {
        let cap_sq = match attacker.color {
            Color::White => Square(to.0 - 8),
            Color::Black => Square(to.0 + 8),
        };
        occ &= !cap_sq.bb();
        SEE_VALUES[PieceKind::Pawn.idx()]
    } else {
        board.piece_at(to).map_or(0, |p| SEE_VALUES[p.kind.idx()])
    };

    let mut gain = [0i32; 32];
    let mut d = 0;
    gain[0] = captured_value;
    let mut current_kind = m.promo_kind().unwrap_or(attacker.kind);
    if let Some(promo) = m.promo_kind() {
        gain[0] += SEE_VALUES[promo.idx()] - SEE_VALUES[PieceKind::Pawn.idx()];
    }

    let mut side = attacker.color.opp();
    loop {
        d += 1;
        gain[d] = SEE_VALUES[current_kind.idx()] - gain[d - 1];
        let attackers = attackers_of(board, to, side, occ);
        let (sq, kind) = match least_valuable(board, side, attackers) {
            Some(v) => v,
            None => break,
        };
        occ &= !sq.bb();
        current_kind = kind;
        side = side.opp();
    }
    for i in (1..d).rev() {
        gain[i - 1] = gain[i - 1].min(-gain[i]);
    }
    gain[0]
}
