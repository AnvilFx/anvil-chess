// Legal and noisy move generation, and perft.
use super::bitboard::*;
use super::board::Board;
use super::types::*;

struct CheckInfo {
    king_sq: Square,
    checkers: u64,
    /// Squares non-king moves must land on: all if not in check, none if double check.
    check_mask: u64,
    /// `!0` if not pinned, otherwise the line through king and pinner.
    pin_ray: [u64; 64],
    /// Enemy attacks with our king removed, so it can't retreat along a checking ray.
    danger: u64,
}

impl Board {
    pub fn gen_legal_into(&self, list: &mut MoveList) {
        let stm = self.side_to_move;
        let king_bb = self.piece_bb[stm.idx()][PieceKind::King.idx()];
        if king_bb == 0 {
            return;
        }
        let info = self.compute_check_info(stm);

        if info.checkers.count_ones() < 2 {
            self.gen_pawn_legal(&info, stm, list);
            self.gen_knight_legal(&info, stm, list);
            self.gen_bishop_legal(&info, stm, list);
            self.gen_rook_legal(&info, stm, list);
            self.gen_queen_legal(&info, stm, list);
        }
        self.gen_king_legal(&info, stm, list);
        if info.checkers == 0 {
            self.gen_castling_legal(&info, stm, list);
        }
    }

    /// Captures and promotions only, in the order `gen_legal_into` emits them.
    /// The side to move must not be in check.
    pub fn gen_noisy_into(&self, list: &mut MoveList) {
        let stm = self.side_to_move;
        let king_bb = self.piece_bb[stm.idx()][PieceKind::King.idx()];
        if king_bb == 0 {
            return;
        }
        let info = self.compute_pin_info(stm);
        debug_assert_eq!(self.attackers_of_square(info.king_sq, stm.opp()), 0);
        let opp_bb = self.color_bb[stm.opp().idx()];

        self.gen_pawn_noisy(&info, stm, list);
        let mut knights = self.piece_bb[stm.idx()][PieceKind::Knight.idx()];
        while let Some(from) = pop_lsb(&mut knights) {
            if info.pin_ray[from.idx()] == !0u64 {
                emit_captures(from, KNIGHT_ATTACKS[from.idx()] & opp_bb, list);
            }
        }
        let mut bishops = self.piece_bb[stm.idx()][PieceKind::Bishop.idx()];
        while let Some(from) = pop_lsb(&mut bishops) {
            let targets = bishop_attacks(from, self.occupancy) & opp_bb & info.pin_ray[from.idx()];
            emit_captures(from, targets, list);
        }
        let mut rooks = self.piece_bb[stm.idx()][PieceKind::Rook.idx()];
        while let Some(from) = pop_lsb(&mut rooks) {
            let targets = rook_attacks(from, self.occupancy) & opp_bb & info.pin_ray[from.idx()];
            emit_captures(from, targets, list);
        }
        let mut queens = self.piece_bb[stm.idx()][PieceKind::Queen.idx()];
        while let Some(from) = pop_lsb(&mut queens) {
            let targets = queen_attacks(from, self.occupancy) & opp_bb & info.pin_ray[from.idx()];
            emit_captures(from, targets, list);
        }
        // King captures, checked one target at a time instead of building the danger map.
        let occ_no_king = self.occupancy & !king_bb;
        let mut caps = KING_ATTACKS[info.king_sq.idx()] & opp_bb;
        while let Some(to) = pop_lsb(&mut caps) {
            if !self.attacked_with_occ(to, stm.opp(), occ_no_king) {
                list.push(Move::new(info.king_sq, to, Move::CAPTURE));
            }
        }
    }

    /// Whether the side to move, not in check, has any legal move. Stops at the first
    /// group of pieces that has one; the king, which needs the danger map, goes last.
    pub fn has_legal_move(&self) -> bool {
        let stm = self.side_to_move;
        if self.piece_bb[stm.idx()][PieceKind::King.idx()] == 0 {
            return false;
        }
        let info = self.compute_pin_info(stm);
        let mut list = MoveList::new();
        self.gen_pawn_legal(&info, stm, &mut list);
        self.gen_knight_legal(&info, stm, &mut list);
        if !list.is_empty() {
            return true;
        }
        self.gen_bishop_legal(&info, stm, &mut list);
        self.gen_rook_legal(&info, stm, &mut list);
        self.gen_queen_legal(&info, stm, &mut list);
        if !list.is_empty() {
            return true;
        }
        // Castling needs a safe empty square next to the king, which is also a king move.
        let full = self.compute_check_info(stm);
        self.gen_king_legal(&full, stm, &mut list);
        !list.is_empty()
    }

    fn attacked_with_occ(&self, sq: Square, by: Color, occ: u64) -> bool {
        let pb = &self.piece_bb[by.idx()];
        PAWN_ATTACKS[by.opp().idx()][sq.idx()] & pb[PieceKind::Pawn.idx()] != 0
            || KNIGHT_ATTACKS[sq.idx()] & pb[PieceKind::Knight.idx()] != 0
            || KING_ATTACKS[sq.idx()] & pb[PieceKind::King.idx()] != 0
            || bishop_attacks(sq, occ) & (pb[PieceKind::Bishop.idx()] | pb[PieceKind::Queen.idx()]) != 0
            || rook_attacks(sq, occ) & (pb[PieceKind::Rook.idx()] | pb[PieceKind::Queen.idx()]) != 0
    }

    pub fn generate_legal_moves(&mut self) -> Vec<Move> {
        let mut list = MoveList::new();
        self.gen_legal_into(&mut list);
        list.iter().collect()
    }

    fn compute_check_info(&self, stm: Color) -> CheckInfo {
        let stm_idx = stm.idx();
        let opp = stm.opp();
        let opp_idx = opp.idx();
        let king_bb = self.piece_bb[stm_idx][PieceKind::King.idx()];
        let king_sq = lsb(king_bb);

        let occ_no_king = self.occupancy & !king_bb;
        let mut danger = 0u64;
        let mut p = self.piece_bb[opp_idx][PieceKind::Pawn.idx()];
        while let Some(sq) = pop_lsb(&mut p) {
            danger |= PAWN_ATTACKS[opp_idx][sq.idx()];
        }
        let mut n = self.piece_bb[opp_idx][PieceKind::Knight.idx()];
        while let Some(sq) = pop_lsb(&mut n) {
            danger |= KNIGHT_ATTACKS[sq.idx()];
        }
        let mut bq = self.piece_bb[opp_idx][PieceKind::Bishop.idx()] | self.piece_bb[opp_idx][PieceKind::Queen.idx()];
        while let Some(sq) = pop_lsb(&mut bq) {
            danger |= bishop_attacks(sq, occ_no_king);
        }
        let mut rq = self.piece_bb[opp_idx][PieceKind::Rook.idx()] | self.piece_bb[opp_idx][PieceKind::Queen.idx()];
        while let Some(sq) = pop_lsb(&mut rq) {
            danger |= rook_attacks(sq, occ_no_king);
        }
        let opp_king_bb = self.piece_bb[opp_idx][PieceKind::King.idx()];
        if opp_king_bb != 0 {
            danger |= KING_ATTACKS[lsb(opp_king_bb).idx()];
        }

        let checkers = self.attackers_of_square(king_sq, opp);

        let check_mask = match checkers.count_ones() {
            0 => !0u64,
            1 => {
                let checker_sq = lsb(checkers);
                let sliders = self.piece_bb[opp_idx][PieceKind::Bishop.idx()]
                    | self.piece_bb[opp_idx][PieceKind::Rook.idx()]
                    | self.piece_bb[opp_idx][PieceKind::Queen.idx()];
                if sliders & checkers != 0 {
                    BETWEEN_BB[king_sq.idx()][checker_sq.idx()] | checkers
                } else {
                    checkers
                }
            }
            _ => 0u64,
        };

        CheckInfo { king_sq, checkers, check_mask, pin_ray: self.pin_rays(stm, king_sq), danger }
    }

    /// Check info for a side known not to be in check, without the king danger map.
    fn compute_pin_info(&self, stm: Color) -> CheckInfo {
        let king_sq = lsb(self.piece_bb[stm.idx()][PieceKind::King.idx()]);
        CheckInfo { king_sq, checkers: 0, check_mask: !0u64, pin_ray: self.pin_rays(stm, king_sq), danger: 0 }
    }

    fn pin_rays(&self, stm: Color, king_sq: Square) -> [u64; 64] {
        let opp_idx = stm.opp().idx();
        let own_bb = self.color_bb[stm.idx()];
        let mut pin_ray = [!0u64; 64];
        let bishop_q = self.piece_bb[opp_idx][PieceKind::Bishop.idx()] | self.piece_bb[opp_idx][PieceKind::Queen.idx()];
        let rook_q = self.piece_bb[opp_idx][PieceKind::Rook.idx()] | self.piece_bb[opp_idx][PieceKind::Queen.idx()];
        let mut snipers = (bishop_q & bishop_attacks(king_sq, 0)) | (rook_q & rook_attacks(king_sq, 0));
        while let Some(sniper_sq) = pop_lsb(&mut snipers) {
            let between = BETWEEN_BB[king_sq.idx()][sniper_sq.idx()] & self.occupancy;
            if between.count_ones() == 1 {
                let pinned_sq = lsb(between);
                if pinned_sq.bb() & own_bb != 0 {
                    pin_ray[pinned_sq.idx()] = BETWEEN_BB[king_sq.idx()][sniper_sq.idx()] | sniper_sq.bb();
                }
            }
        }
        pin_ray
    }

    fn gen_knight_legal(&self, info: &CheckInfo, stm: Color, out: &mut MoveList) {
        let mut knights = self.piece_bb[stm.idx()][PieceKind::Knight.idx()];
        let own = self.color_bb[stm.idx()];
        let opp_bb = self.color_bb[stm.opp().idx()];
        while let Some(from) = pop_lsb(&mut knights) {
            if info.pin_ray[from.idx()] != !0u64 {
                continue;
            }
            let targets = KNIGHT_ATTACKS[from.idx()] & !own & info.check_mask;
            emit_targets(from, targets, opp_bb, out);
        }
    }

    fn gen_bishop_legal(&self, info: &CheckInfo, stm: Color, out: &mut MoveList) {
        let mut bishops = self.piece_bb[stm.idx()][PieceKind::Bishop.idx()];
        let own = self.color_bb[stm.idx()];
        let opp_bb = self.color_bb[stm.opp().idx()];
        while let Some(from) = pop_lsb(&mut bishops) {
            let targets = bishop_attacks(from, self.occupancy) & !own & info.check_mask & info.pin_ray[from.idx()];
            emit_targets(from, targets, opp_bb, out);
        }
    }

    fn gen_rook_legal(&self, info: &CheckInfo, stm: Color, out: &mut MoveList) {
        let mut rooks = self.piece_bb[stm.idx()][PieceKind::Rook.idx()];
        let own = self.color_bb[stm.idx()];
        let opp_bb = self.color_bb[stm.opp().idx()];
        while let Some(from) = pop_lsb(&mut rooks) {
            let targets = rook_attacks(from, self.occupancy) & !own & info.check_mask & info.pin_ray[from.idx()];
            emit_targets(from, targets, opp_bb, out);
        }
    }

    fn gen_queen_legal(&self, info: &CheckInfo, stm: Color, out: &mut MoveList) {
        let mut queens = self.piece_bb[stm.idx()][PieceKind::Queen.idx()];
        let own = self.color_bb[stm.idx()];
        let opp_bb = self.color_bb[stm.opp().idx()];
        while let Some(from) = pop_lsb(&mut queens) {
            let targets = queen_attacks(from, self.occupancy) & !own & info.check_mask & info.pin_ray[from.idx()];
            emit_targets(from, targets, opp_bb, out);
        }
    }

    fn gen_king_legal(&self, info: &CheckInfo, stm: Color, out: &mut MoveList) {
        let from = info.king_sq;
        let own = self.color_bb[stm.idx()];
        let opp_bb = self.color_bb[stm.opp().idx()];
        let targets = KING_ATTACKS[from.idx()] & !own & !info.danger;
        emit_targets(from, targets, opp_bb, out);
    }

    fn gen_castling_legal(&self, info: &CheckInfo, stm: Color, out: &mut MoveList) {
        let occ = self.occupancy;
        match stm {
            Color::White => {
                if self.castle.has(CastleRights::WHITE_KING) {
                    let between = (1u64 << 5) | (1u64 << 6);
                    let must_safe = (1u64 << 4) | (1u64 << 5) | (1u64 << 6);
                    if occ & between == 0 && info.danger & must_safe == 0 {
                        out.push(Move::new(Square(4), Square(6), Move::CASTLE_KING));
                    }
                }
                if self.castle.has(CastleRights::WHITE_QUEEN) {
                    let between = (1u64 << 1) | (1u64 << 2) | (1u64 << 3);
                    let must_safe = (1u64 << 2) | (1u64 << 3) | (1u64 << 4);
                    if occ & between == 0 && info.danger & must_safe == 0 {
                        out.push(Move::new(Square(4), Square(2), Move::CASTLE_QUEEN));
                    }
                }
            }
            Color::Black => {
                if self.castle.has(CastleRights::BLACK_KING) {
                    let between = (1u64 << 61) | (1u64 << 62);
                    let must_safe = (1u64 << 60) | (1u64 << 61) | (1u64 << 62);
                    if occ & between == 0 && info.danger & must_safe == 0 {
                        out.push(Move::new(Square(60), Square(62), Move::CASTLE_KING));
                    }
                }
                if self.castle.has(CastleRights::BLACK_QUEEN) {
                    let between = (1u64 << 57) | (1u64 << 58) | (1u64 << 59);
                    let must_safe = (1u64 << 58) | (1u64 << 59) | (1u64 << 60);
                    if occ & between == 0 && info.danger & must_safe == 0 {
                        out.push(Move::new(Square(60), Square(58), Move::CASTLE_QUEEN));
                    }
                }
            }
        }
    }

    fn gen_pawn_legal(&self, info: &CheckInfo, stm: Color, out: &mut MoveList) {
        let pawns = self.piece_bb[stm.idx()][PieceKind::Pawn.idx()];
        if pawns == 0 {
            return;
        }
        let opp_bb = self.color_bb[stm.opp().idx()];
        let empty = !self.occupancy;
        let step: i8 = if stm == Color::White { 8 } else { -8 };
        let start_rank: u8 = if stm == Color::White { 1 } else { 6 };
        let promo_rank: u8 = if stm == Color::White { 7 } else { 0 };

        let mut p = pawns;
        while let Some(from) = pop_lsb(&mut p) {
            let pin = info.pin_ray[from.idx()];

            let to_idx = from.0 as i8 + step;
            if (0..64).contains(&to_idx) {
                let to = Square(to_idx as u8);
                let to_bb = to.bb();
                if empty & to_bb != 0 && pin & to_bb != 0 {
                    if info.check_mask & to_bb != 0 {
                        if to.rank() == promo_rank {
                            out.push(Move::new(from, to, Move::PROMO_QUEEN));
                            out.push(Move::new(from, to, Move::PROMO_ROOK));
                            out.push(Move::new(from, to, Move::PROMO_BISHOP));
                            out.push(Move::new(from, to, Move::PROMO_KNIGHT));
                        } else {
                            out.push(Move::new(from, to, Move::QUIET));
                        }
                    }
                    if from.rank() == start_rank {
                        let dbl_idx = from.0 as i8 + 2 * step;
                        if (0..64).contains(&dbl_idx) {
                            let dbl = Square(dbl_idx as u8);
                            let dbl_bb = dbl.bb();
                            if empty & dbl_bb != 0 && pin & dbl_bb != 0 && info.check_mask & dbl_bb != 0 {
                                out.push(Move::new(from, dbl, Move::DOUBLE_PUSH));
                            }
                        }
                    }
                }
            }

            let attacks = PAWN_ATTACKS[stm.idx()][from.idx()];
            let mut caps = attacks & opp_bb & pin & info.check_mask;
            while let Some(to) = pop_lsb(&mut caps) {
                if to.rank() == promo_rank {
                    out.push(Move::new(from, to, Move::PROMO_CAP_QUEEN));
                    out.push(Move::new(from, to, Move::PROMO_CAP_ROOK));
                    out.push(Move::new(from, to, Move::PROMO_CAP_BISHOP));
                    out.push(Move::new(from, to, Move::PROMO_CAP_KNIGHT));
                } else {
                    out.push(Move::new(from, to, Move::CAPTURE));
                }
            }

            if let Some(ep_sq) = self.en_passant {
                if attacks & ep_sq.bb() & pin != 0 && self.is_ep_legal(from, ep_sq, stm, info) {
                    out.push(Move::new(from, ep_sq, Move::EP_CAPTURE));
                }
            }
        }
    }

    /// The noisy part of `gen_pawn_legal`, in the same order: push-promotions, captures, en passant.
    fn gen_pawn_noisy(&self, info: &CheckInfo, stm: Color, out: &mut MoveList) {
        let pawns = self.piece_bb[stm.idx()][PieceKind::Pawn.idx()];
        if pawns == 0 {
            return;
        }
        let opp_bb = self.color_bb[stm.opp().idx()];
        let empty = !self.occupancy;
        let step: i8 = if stm == Color::White { 8 } else { -8 };
        let promo_rank: u8 = if stm == Color::White { 7 } else { 0 };

        let mut p = pawns;
        while let Some(from) = pop_lsb(&mut p) {
            let pin = info.pin_ray[from.idx()];
            let to_idx = from.0 as i8 + step;
            if (0..64).contains(&to_idx) {
                let to = Square(to_idx as u8);
                let to_bb = to.bb();
                if to.rank() == promo_rank && empty & to_bb != 0 && pin & to_bb != 0 {
                    out.push(Move::new(from, to, Move::PROMO_QUEEN));
                    out.push(Move::new(from, to, Move::PROMO_ROOK));
                    out.push(Move::new(from, to, Move::PROMO_BISHOP));
                    out.push(Move::new(from, to, Move::PROMO_KNIGHT));
                }
            }

            let attacks = PAWN_ATTACKS[stm.idx()][from.idx()];
            let mut caps = attacks & opp_bb & pin;
            while let Some(to) = pop_lsb(&mut caps) {
                if to.rank() == promo_rank {
                    out.push(Move::new(from, to, Move::PROMO_CAP_QUEEN));
                    out.push(Move::new(from, to, Move::PROMO_CAP_ROOK));
                    out.push(Move::new(from, to, Move::PROMO_CAP_BISHOP));
                    out.push(Move::new(from, to, Move::PROMO_CAP_KNIGHT));
                } else {
                    out.push(Move::new(from, to, Move::CAPTURE));
                }
            }

            if let Some(ep_sq) = self.en_passant {
                if attacks & ep_sq.bb() & pin != 0 && self.is_ep_legal(from, ep_sq, stm, info) {
                    out.push(Move::new(from, ep_sq, Move::EP_CAPTURE));
                }
            }
        }
    }

    /// EP removes two pawns from one rank, which can expose the king to a slider the pin rays miss.
    fn is_ep_legal(&self, from: Square, to: Square, stm: Color, info: &CheckInfo) -> bool {
        let captured_sq = match stm {
            Color::White => Square(to.0 - 8),
            Color::Black => Square(to.0 + 8),
        };
        if info.checkers != 0 && info.check_mask & to.bb() == 0 && info.check_mask & captured_sq.bb() == 0 {
            return false;
        }
        let mut test = self.clone();
        let m = Move::new(from, to, Move::EP_CAPTURE);
        let _ = test.make_move(m);
        !test.in_check(stm)
    }

    pub fn perft(&mut self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let mut list = MoveList::new();
        self.gen_legal_into(&mut list);
        if depth == 1 {
            return list.len() as u64;
        }
        let mut nodes = 0u64;
        for i in 0..list.len() {
            let m = list.get(i);
            let undo = self.make_move(m);
            nodes += self.perft(depth - 1);
            self.unmake_move(m, undo);
        }
        nodes
    }
}

fn emit_captures(from: Square, mut caps: u64, out: &mut MoveList) {
    while let Some(to) = pop_lsb(&mut caps) {
        out.push(Move::new(from, to, Move::CAPTURE));
    }
}

fn emit_targets(from: Square, targets: u64, opp_bb: u64, out: &mut MoveList) {
    let mut quiet = targets & !opp_bb;
    while let Some(to) = pop_lsb(&mut quiet) {
        out.push(Move::new(from, to, Move::QUIET));
    }
    let mut caps = targets & opp_bb;
    while let Some(to) = pop_lsb(&mut caps) {
        out.push(Move::new(from, to, Move::CAPTURE));
    }
}
