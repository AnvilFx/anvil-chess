// Static exchange evaluation.
use super::*;

pub(super) const SEE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 10000];

// Empty-board slider lines from each square, to skip magic lookups when no slider stands on them.
const fn empty_board_rays(diagonal: bool) -> [u64; 64] {
    let mut rays = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let (f, r) = ((sq % 8) as i32, (sq / 8) as i32);
        let mut d = 0;
        while d < 4 {
            let (df, dr) = if diagonal { ([1, 1, -1, -1][d], [1, -1, 1, -1][d]) } else { ([0, 0, 1, -1][d], [1, -1, 0, 0][d]) };
            let (mut nf, mut nr) = (f + df, r + dr);
            while nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                rays[sq] |= 1u64 << (nr * 8 + nf);
                nf += df;
                nr += dr;
            }
            d += 1;
        }
        sq += 1;
    }
    rays
}

static BISHOP_RAYS: [u64; 64] = empty_board_rays(true);
static ROOK_RAYS: [u64; 64] = empty_board_rays(false);

#[inline(always)]
fn diagonal_sliders(board: &Board) -> u64 {
    let w = &board.piece_bb[Color::White.idx()];
    let b = &board.piece_bb[Color::Black.idx()];
    w[PieceKind::Bishop.idx()] | b[PieceKind::Bishop.idx()] | w[PieceKind::Queen.idx()] | b[PieceKind::Queen.idx()]
}

#[inline(always)]
fn orthogonal_sliders(board: &Board) -> u64 {
    let w = &board.piece_bb[Color::White.idx()];
    let b = &board.piece_bb[Color::Black.idx()];
    w[PieceKind::Rook.idx()] | b[PieceKind::Rook.idx()] | w[PieceKind::Queen.idx()] | b[PieceKind::Queen.idx()]
}

/// Pieces of both colors attacking `sq` through `occ`.
#[inline(always)]
fn all_attackers(board: &Board, sq: Square, occ: u64) -> u64 {
    let w = &board.piece_bb[Color::White.idx()];
    let b = &board.piece_bb[Color::Black.idx()];
    let s = sq.idx();
    let mut att = (PAWN_ATTACKS[Color::Black.idx()][s] & w[PieceKind::Pawn.idx()])
        | (PAWN_ATTACKS[Color::White.idx()][s] & b[PieceKind::Pawn.idx()])
        | (KNIGHT_ATTACKS[s] & (w[PieceKind::Knight.idx()] | b[PieceKind::Knight.idx()]))
        | (KING_ATTACKS[s] & (w[PieceKind::King.idx()] | b[PieceKind::King.idx()]));
    let diag = diagonal_sliders(board) & occ;
    if BISHOP_RAYS[s] & diag != 0 {
        att |= bishop_attacks(sq, occ) & diag;
    }
    let orth = orthogonal_sliders(board) & occ;
    if ROOK_RAYS[s] & orth != 0 {
        att |= rook_attacks(sq, occ) & orth;
    }
    att & occ
}

/// `attackers` once an attacker of `kind` has left `occ`: only lines through its square can open.
#[inline(always)]
fn remove_attacker(board: &Board, sq: Square, kind: PieceKind, occ: u64, attackers: u64) -> u64 {
    let mut att = attackers & occ;
    let s = sq.idx();
    if matches!(kind, PieceKind::Pawn | PieceKind::Bishop | PieceKind::Queen | PieceKind::King) {
        let hidden = diagonal_sliders(board) & occ & !att;
        if BISHOP_RAYS[s] & hidden != 0 {
            att |= bishop_attacks(sq, occ) & hidden;
        }
    }
    if matches!(kind, PieceKind::Rook | PieceKind::Queen | PieceKind::King) {
        let hidden = orthogonal_sliders(board) & occ & !att;
        if ROOK_RAYS[s] & hidden != 0 {
            att |= rook_attacks(sq, occ) & hidden;
        }
    }
    att
}

#[inline(always)]
fn least_valuable(board: &Board, side: Color, attackers: u64) -> Option<(Square, PieceKind)> {
    let own = attackers & board.color_bb[side.idx()];
    if own == 0 {
        return None;
    }
    let bbs = &board.piece_bb[side.idx()];
    for kind_idx in 0..6 {
        let cand = own & bbs[kind_idx];
        if cand != 0 {
            return Some((lsb(cand), PieceKind::from_idx(kind_idx)));
        }
    }
    None
}

// SEE is never negative when the first capture already wins at least the value of the
// piece now standing on the square: whatever the exchange, the capturer can stop there.
#[cfg(test)]
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

/// First capture's gain, the piece left on the target, the mover and the occupancy after the move.
#[inline(always)]
fn see_start(board: &Board, m: Move) -> Option<(i32, PieceKind, Color, u64)> {
    let to = m.to();
    let from = m.from();
    let attacker = board.piece_at(from)?;
    let mut occ = board.occupancy & !from.bb();
    let mut gain = if m.flag() == Move::EP_CAPTURE {
        let cap_sq = match attacker.color {
            Color::White => Square(to.0 - 8),
            Color::Black => Square(to.0 + 8),
        };
        occ &= !cap_sq.bb();
        SEE_VALUES[PieceKind::Pawn.idx()]
    } else {
        board.piece_at(to).map_or(0, |p| SEE_VALUES[p.kind.idx()])
    };
    let kind = match m.promo_kind() {
        Some(promo) => {
            gain += SEE_VALUES[promo.idx()] - SEE_VALUES[PieceKind::Pawn.idx()];
            promo
        }
        None => attacker.kind,
    };
    Some((gain, kind, attacker.color, occ))
}

pub(super) fn see(board: &Board, m: Move) -> i32 {
    let Some((gain0, mut current_kind, mover, mut occ)) = see_start(board, m) else { return 0 };
    let to = m.to();
    let mut attackers = all_attackers(board, to, occ);
    let mut gain = [0i32; 32];
    let mut d = 0;
    gain[0] = gain0;
    let mut side = mover.opp();
    loop {
        d += 1;
        gain[d] = SEE_VALUES[current_kind.idx()] - gain[d - 1];
        let (sq, kind) = match least_valuable(board, side, attackers) {
            Some(v) => v,
            None => break,
        };
        occ &= !sq.bb();
        attackers = remove_attacker(board, to, kind, occ, attackers);
        current_kind = kind;
        side = side.opp();
    }
    for i in (1..d).rev() {
        gain[i - 1] = gain[i - 1].min(-gain[i]);
    }
    gain[0]
}

/// `see` given `threats`, the squares the opponent attacks: with neither square attacked there is no recapture.
#[inline]
pub(super) fn see_with_threats(board: &Board, m: Move, threats: u64) -> i32 {
    if threats & (m.from().bb() | m.to().bb()) == 0 && m.flag() != Move::EP_CAPTURE {
        let captured = board.piece_at(m.to()).map_or(0, |p| SEE_VALUES[p.kind.idx()]);
        return captured + m.promo_kind().map_or(0, |k| SEE_VALUES[k.idx()] - SEE_VALUES[PieceKind::Pawn.idx()]);
    }
    see(board, m)
}

/// Same as `see(board, m) >= threshold`, but stops as soon as the answer is settled.
pub(super) fn see_ge(board: &Board, m: Move, threshold: i32) -> bool {
    let Some((mut score, mut current_kind, mover, mut occ)) = see_start(board, m) else { return threshold <= 0 };
    // `score` is the mover's result if the exchange stops now; each side may stop before capturing.
    if score < threshold {
        return false;
    }
    if score - SEE_VALUES[current_kind.idx()] >= threshold {
        return true;
    }
    let to = m.to();
    let mut attackers = all_attackers(board, to, occ);
    let mut side = mover.opp();
    loop {
        let them = side != mover;
        let next = if them { score - SEE_VALUES[current_kind.idx()] } else { score + SEE_VALUES[current_kind.idx()] };
        if them && next >= threshold {
            return true;
        }
        if !them && next < threshold {
            return false;
        }
        let (sq, kind) = match least_valuable(board, side, attackers) {
            Some(v) => v,
            None => return them,
        };
        score = next;
        occ &= !sq.bb();
        attackers = remove_attacker(board, to, kind, occ, attackers);
        current_kind = kind;
        side = side.opp();
    }
}
