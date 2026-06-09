// Board state, mailbox encoding, move generation, make/unmake, FEN.

use super::bitboard::*;
use super::types::*;
use super::zobrist::*;

// =============================================================================
// MAILBOX ENCODING
// =============================================================================
// One byte per square:
//   0      = empty
//   1..=6  = white {P, N, B, R, Q, K}  (in PieceKind order)
//   7..=12 = black {P, N, B, R, Q, K}
//
// Total: 64 bytes per board, exactly one CPU cache line.
//
// Kept in sync with the bitboards by every code path that mutates the
// board (currently only `set_square`; later also `make_move`/`unmake_move`).

const MB_EMPTY: u8 = 0;

const fn mb_encode(piece: Piece) -> u8 {
    1 + piece.kind as u8 + (piece.color as u8) * 6
}

#[inline]
const fn mb_encode_ck(color: Color, kind: PieceKind) -> u8 {
    1 + kind as u8 + (color as u8) * 6
}

fn mb_decode(code: u8) -> Option<Piece> {
    if code == MB_EMPTY {
        return None;
    }
    let idx = (code - 1) as usize;
    let color = if idx < 6 { Color::White } else { Color::Black };
    let kind = PieceKind::from_idx(idx % 6);
    Some(Piece { color, kind })
}

// =============================================================================
// BOARD
// =============================================================================

#[derive(Clone)]
pub struct Board {
    /// `piece_bb[color][kind]` is the bitboard of that color's pieces of
    /// that kind. Index with `Color::idx()` and `PieceKind::idx()`.
    pub piece_bb: [[u64; 6]; 2],

    /// `color_bb[color]` is the union of `piece_bb[color][..]`. Cached so
    /// occupancy checks for one side don't have to OR six bitboards.
    pub color_bb: [u64; 2],

    /// All occupied squares (`color_bb[0] | color_bb[1]`).
    pub occupancy: u64,

    /// Quick "what's on square X" lookup. See MAILBOX ENCODING above for
    /// the byte values.
    pub mailbox: [u8; 64],

    pub side_to_move: Color,
    pub castle: CastleRights,
    pub en_passant: Option<Square>,
    pub halfmove_clock: u16,
    pub fullmove_number: u16,

    /// Zobrist hash of the full position (pieces + side + castle + ep).
    /// Maintained incrementally by `make_move`/`unmake_move`.
    pub hash: u64,
}

impl Default for Board {
    fn default() -> Self {
        Self::from_fen(STARTING_FEN).expect("STARTING_FEN must parse")
    }
}

/// Per-position info needed for already-legal movegen. Computed once at the
/// start of `gen_legal_into`; reused across all gen_*_legal calls.
struct CheckInfo {
    king_sq: Square,
    /// Enemy pieces attacking the king (0, 1, or 2+).
    checkers: u64,
    /// Squares non-king pieces must land on. Full board if not in check;
    /// between-king-and-checker | checker.bb() if singly checked;
    /// 0 if double-checked (only king moves allowed).
    check_mask: u64,
    /// Per-square pin restriction. `!0` if not pinned; otherwise the squares
    /// the pinned piece can move to (the line through king and pinner).
    pin_ray: [u64; 64],
    /// Squares attacked by enemy with the king removed from occupancy.
    /// Used to filter king moves (king can't move into attacked square,
    /// and slider attacks pass through the king square).
    danger: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct UndoInfo {
    pub captured: Option<Piece>,
    pub castle: CastleRights,
    pub en_passant: Option<Square>,
    pub halfmove_clock: u16,
    pub hash: u64,
}

/// Undo state for `make_null_move` (engine pass-the-turn move). Castle
/// rights don't change and no piece moves, so only restore
/// en-passant + halfmove + hash.
#[derive(Clone, Copy, Debug)]
pub struct NullUndo {
    pub en_passant: Option<Square>,
    pub halfmove_clock: u16,
    pub hash: u64,
}

impl Board {
    /// An all-empty board, white to move, no castle rights, no en passant.
    /// Used by `from_fen` as the base before placing pieces, and useful
    /// for tests / position editor scenarios.
    pub fn empty() -> Self {
        Self {
            piece_bb: [[0; 6]; 2],
            color_bb: [0; 2],
            occupancy: 0,
            mailbox: [MB_EMPTY; 64],
            side_to_move: Color::White,
            castle: CastleRights::NONE,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
        }
    }

    /// Decode the piece at this square. O(1) - one mailbox load plus a
    /// small decode.
    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        mb_decode(self.mailbox[sq.idx()])
    }

    /// Place a piece on a square, or clear it if `piece` is None.
    ///
    /// Used by FEN parsing and the UI's drag-and-drop (position editor).
    /// The engine will eventually call this via `make_move`/`unmake_move`
    /// instead - those paths can be faster because they know more about
    /// the previous state (e.g. the from-square is always empty after a
    /// move, no need to bother decoding what was there).
    #[inline]
    fn add_piece(&mut self, sq: Square, color: Color, kind: PieceKind) {
        let bit = sq.bb();
        let ci = color.idx();
        let ki = kind.idx();
        self.piece_bb[ci][ki] |= bit;
        self.color_bb[ci] |= bit;
        self.occupancy |= bit;
        self.mailbox[sq.idx()] = mb_encode_ck(color, kind);
        self.hash ^= zob_piece(color, kind, sq);
    }

    #[inline]
    fn remove_piece(&mut self, sq: Square, color: Color, kind: PieceKind) {
        let mask = !sq.bb();
        let ci = color.idx();
        let ki = kind.idx();
        self.piece_bb[ci][ki] &= mask;
        self.color_bb[ci] &= mask;
        self.occupancy &= mask;
        self.mailbox[sq.idx()] = MB_EMPTY;
        self.hash ^= zob_piece(color, kind, sq);
    }

    #[inline]
    fn move_piece(&mut self, from: Square, to: Square, color: Color, kind: PieceKind) {
        let xor = from.bb() | to.bb();
        let ci = color.idx();
        let ki = kind.idx();
        self.piece_bb[ci][ki] ^= xor;
        self.color_bb[ci] ^= xor;
        self.occupancy ^= xor;
        self.mailbox[from.idx()] = MB_EMPTY;
        self.mailbox[to.idx()] = mb_encode_ck(color, kind);
        self.hash ^= zob_piece(color, kind, from) ^ zob_piece(color, kind, to);
    }

    pub fn set_square(&mut self, sq: Square, piece: Option<Piece>) {
        let bit = sq.bb();
        let inv_bit = !bit;

        // Clear any existing piece on this square.
        let old_code = self.mailbox[sq.idx()];
        if let Some(old) = mb_decode(old_code) {
            self.piece_bb[old.color.idx()][old.kind.idx()] &= inv_bit;
            self.color_bb[old.color.idx()] &= inv_bit;
            self.occupancy &= inv_bit;
            self.hash ^= zob_piece(old.color, old.kind, sq);
        }

        // Install the new piece (or just leave the square empty).
        match piece {
            None => self.mailbox[sq.idx()] = MB_EMPTY,
            Some(p) => {
                self.mailbox[sq.idx()] = mb_encode(p);
                self.piece_bb[p.color.idx()][p.kind.idx()] |= bit;
                self.color_bb[p.color.idx()] |= bit;
                self.occupancy |= bit;
                self.hash ^= zob_piece(p.color, p.kind, sq);
            }
        }
    }

    // -------------------------------------------------------------------------
    // FEN
    // -------------------------------------------------------------------------

    /// Parse a full 6-field FEN string. Returns a human-readable error
    /// describing where parsing failed.
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let mut parts = fen.split_whitespace();
        let board_part = parts
            .next()
            .ok_or_else(|| "FEN missing piece-placement field".to_string())?;

        let mut board = Self::empty();

        // FEN's piece-placement section is rank-by-rank starting with the
        // 8th rank. In the bitboard layout, the 8th rank is rank
        // index 7. Reading characters descends rank index from 7
        // down to 0, advancing file from 0 toward 7 within each rank.
        let mut rank: i8 = 7;
        let mut file: u8 = 0;

        for ch in board_part.chars() {
            if ch == '/' {
                if file != 8 {
                    return Err(format!(
                        "rank {} ended with {} files (expected 8)",
                        rank + 1,
                        file
                    ));
                }
                rank -= 1;
                file = 0;
                if rank < 0 {
                    return Err("too many ranks in FEN".into());
                }
                continue;
            }
            if let Some(n) = ch.to_digit(10) {
                if n == 0 || n > 8 {
                    return Err(format!("invalid empty-square count '{}'", ch));
                }
                file += n as u8;
                if file > 8 {
                    return Err(format!("rank {} overflowed past file h", rank + 1));
                }
                continue;
            }
            if file >= 8 {
                return Err(format!(
                    "piece '{}' on rank {} past file h",
                    ch,
                    rank + 1
                ));
            }
            let (color, kind) = match ch {
                'P' => (Color::White, PieceKind::Pawn),
                'N' => (Color::White, PieceKind::Knight),
                'B' => (Color::White, PieceKind::Bishop),
                'R' => (Color::White, PieceKind::Rook),
                'Q' => (Color::White, PieceKind::Queen),
                'K' => (Color::White, PieceKind::King),
                'p' => (Color::Black, PieceKind::Pawn),
                'n' => (Color::Black, PieceKind::Knight),
                'b' => (Color::Black, PieceKind::Bishop),
                'r' => (Color::Black, PieceKind::Rook),
                'q' => (Color::Black, PieceKind::Queen),
                'k' => (Color::Black, PieceKind::King),
                _ => return Err(format!("unrecognized piece character '{}'", ch)),
            };
            board.set_square(
                Square::new(file, rank as u8),
                Some(Piece { color, kind }),
            );
            file += 1;
        }
        if rank != 0 || file != 8 {
            return Err(format!(
                "incomplete board (last rank index = {}, files filled = {})",
                rank + 1,
                file
            ));
        }

        // Side to move.
        match parts.next() {
            Some("w") => board.side_to_move = Color::White,
            Some("b") => board.side_to_move = Color::Black,
            Some(s) => return Err(format!("bad side-to-move field '{}'", s)),
            None => return Err("FEN missing side-to-move field".into()),
        }

        // Castle rights.
        match parts.next() {
            Some("-") => board.castle = CastleRights::NONE,
            Some(s) => {
                let mut rights = CastleRights::NONE;
                for ch in s.chars() {
                    match ch {
                        'K' => rights.add(CastleRights::WHITE_KING),
                        'Q' => rights.add(CastleRights::WHITE_QUEEN),
                        'k' => rights.add(CastleRights::BLACK_KING),
                        'q' => rights.add(CastleRights::BLACK_QUEEN),
                        _ => {
                            return Err(format!(
                                "bad castle-rights character '{}'",
                                ch
                            ))
                        }
                    }
                }
                board.castle = rights;
            }
            None => return Err("FEN missing castle-rights field".into()),
        }

        // En passant target.
        match parts.next() {
            Some("-") => board.en_passant = None,
            Some(s) => {
                board.en_passant = Some(
                    Square::from_algebraic(s)
                        .ok_or_else(|| format!("bad en-passant square '{}'", s))?,
                );
            }
            None => return Err("FEN missing en-passant field".into()),
        }

        // Halfmove clock (50-move counter).
        match parts.next() {
            Some(s) => {
                board.halfmove_clock = s
                    .parse()
                    .map_err(|_| format!("bad halfmove clock '{}'", s))?;
            }
            None => return Err("FEN missing halfmove-clock field".into()),
        }

        // Fullmove number.
        match parts.next() {
            Some(s) => {
                board.fullmove_number = s
                    .parse()
                    .map_err(|_| format!("bad fullmove number '{}'", s))?;
            }
            None => return Err("FEN missing fullmove-number field".into()),
        }

        // Mix the meta state into the hash. Piece bits were already folded
        // in by the `set_square` calls during piece placement above.
        board.hash ^= zob_meta(board.side_to_move, board.castle, board.en_passant);

        Ok(board)
    }

    /// Serialize the board back to a 6-field FEN string. Round-trips
    /// cleanly with `from_fen` for valid positions.
    pub fn to_fen(&self) -> String {
        let mut out = String::with_capacity(80);

        // Piece placement: rank 8 (index 7) first, descending to rank 1.
        for rank in (0..8u8).rev() {
            let mut empty_run = 0u8;
            for file in 0..8u8 {
                let sq = Square::new(file, rank);
                match self.piece_at(sq) {
                    None => empty_run += 1,
                    Some(p) => {
                        if empty_run > 0 {
                            out.push((b'0' + empty_run) as char);
                            empty_run = 0;
                        }
                        out.push(piece_to_fen_char(p));
                    }
                }
            }
            if empty_run > 0 {
                out.push((b'0' + empty_run) as char);
            }
            if rank > 0 {
                out.push('/');
            }
        }

        out.push(' ');
        out.push(match self.side_to_move {
            Color::White => 'w',
            Color::Black => 'b',
        });

        out.push(' ');
        if self.castle.0 == 0 {
            out.push('-');
        } else {
            if self.castle.has(CastleRights::WHITE_KING) {
                out.push('K');
            }
            if self.castle.has(CastleRights::WHITE_QUEEN) {
                out.push('Q');
            }
            if self.castle.has(CastleRights::BLACK_KING) {
                out.push('k');
            }
            if self.castle.has(CastleRights::BLACK_QUEEN) {
                out.push('q');
            }
        }

        out.push(' ');
        match self.en_passant {
            None => out.push('-'),
            Some(sq) => {
                use std::fmt::Write;
                let _ = write!(out, "{}", sq);
            }
        }

        out.push(' ');
        out.push_str(&self.halfmove_clock.to_string());
        out.push(' ');
        out.push_str(&self.fullmove_number.to_string());

        out
    }

    // -------------------------------------------------------------------------
    // ATTACK QUERIES
    // -------------------------------------------------------------------------

    /// Bitboard of pieces of `by_color` that attack `sq` in the current
    /// position. Useful for SEE, check detection, castle-path safety,
    /// and as a movegen primitive.
    ///
    /// Pawn term: a pawn of color C attacks `sq` iff an imaginary pawn of the
    /// opposite color on `sq` would attack C's pawn, hence the opposite color's
    /// pawn-attack table indexed by `sq`.
    pub fn attackers_of_square(&self, sq: Square, by_color: Color) -> u64 {
        let c = by_color.idx();
        let opp = by_color.opp().idx();
        let mut atk = 0u64;

        // Pawns
        atk |= PAWN_ATTACKS[opp][sq.idx()] & self.piece_bb[c][PieceKind::Pawn.idx()];
        // Knights
        atk |= KNIGHT_ATTACKS[sq.idx()] & self.piece_bb[c][PieceKind::Knight.idx()];
        // Bishop-like sliders (bishops + queens on diagonals)
        let diag = bishop_attacks(sq, self.occupancy);
        atk |= diag
            & (self.piece_bb[c][PieceKind::Bishop.idx()]
                | self.piece_bb[c][PieceKind::Queen.idx()]);
        // Rook-like sliders (rooks + queens on files/ranks)
        let orth = rook_attacks(sq, self.occupancy);
        atk |= orth
            & (self.piece_bb[c][PieceKind::Rook.idx()]
                | self.piece_bb[c][PieceKind::Queen.idx()]);
        // King
        atk |= KING_ATTACKS[sq.idx()] & self.piece_bb[c][PieceKind::King.idx()];

        atk
    }

    #[inline]
    pub fn is_square_attacked(&self, sq: Square, by_color: Color) -> bool {
        self.attackers_of_square(sq, by_color) != 0
    }

    /// True if `side`'s king is in check. False if `side` has no king on the
    /// board, so partial positions in tests do not panic.
    pub fn in_check(&self, side: Color) -> bool {
        let king_bb = self.piece_bb[side.idx()][PieceKind::King.idx()];
        if king_bb == 0 {
            return false;
        }
        let king_sq = lsb(king_bb);
        self.is_square_attacked(king_sq, side.opp())
    }

    // -------------------------------------------------------------------------
    // MAKE / UNMAKE
    // -------------------------------------------------------------------------

    /// Apply `m` to the board, returning info needed to undo it. Assumes
    /// the move is at least pseudo-legal (the from-square has a piece of
    /// the side to move, the to-square is a valid target, the flag
    /// matches the move's semantics). The caller guarantees this.
    pub fn make_move(&mut self, m: Move) -> UndoInfo {
        let from = m.from();
        let to = m.to();
        let flag = m.flag();
        let stm = self.side_to_move;

        let moving_kind = self
            .piece_at(from)
            .expect("make_move: from-square is empty")
            .kind;

        let ep_capture = flag == Move::EP_CAPTURE;
        let captured_sq = if ep_capture {
            match stm {
                Color::White => Square(to.0 - 8),
                Color::Black => Square(to.0 + 8),
            }
        } else {
            to
        };
        let captured = if m.is_capture() {
            self.piece_at(captured_sq)
        } else {
            None
        };

        let undo = UndoInfo {
            captured,
            castle: self.castle,
            en_passant: self.en_passant,
            halfmove_clock: self.halfmove_clock,
            hash: self.hash,
        };

        // Strip the old meta contribution from the hash; the new meta state
        // is folded back in at the end of this function. Piece movements
        // hash themselves inside add_piece/remove_piece/move_piece.
        self.hash ^= zob_meta(self.side_to_move, self.castle, self.en_passant);

        self.en_passant = None;

        if moving_kind == PieceKind::Pawn || captured.is_some() {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock = self.halfmove_clock.saturating_add(1);
        }

        if let Some(cap) = captured {
            self.remove_piece(captured_sq, cap.color, cap.kind);
        }

        if let Some(promo) = m.promo_kind() {
            self.remove_piece(from, stm, PieceKind::Pawn);
            self.add_piece(to, stm, promo);
        } else {
            self.move_piece(from, to, stm, moving_kind);
        }

        if flag == Move::DOUBLE_PUSH {
            let ep_sq = match stm {
                Color::White => Square(from.0 + 8),
                Color::Black => Square(from.0 - 8),
            };
            self.en_passant = Some(ep_sq);
        }

        if flag == Move::CASTLE_KING {
            let (rf, rt) = match stm {
                Color::White => (Square(7), Square(5)),
                Color::Black => (Square(63), Square(61)),
            };
            self.move_piece(rf, rt, stm, PieceKind::Rook);
        } else if flag == Move::CASTLE_QUEEN {
            let (rf, rt) = match stm {
                Color::White => (Square(0), Square(3)),
                Color::Black => (Square(56), Square(59)),
            };
            self.move_piece(rf, rt, stm, PieceKind::Rook);
        }

        if moving_kind == PieceKind::King {
            match stm {
                Color::White => {
                    self.castle.remove(CastleRights::WHITE_KING);
                    self.castle.remove(CastleRights::WHITE_QUEEN);
                }
                Color::Black => {
                    self.castle.remove(CastleRights::BLACK_KING);
                    self.castle.remove(CastleRights::BLACK_QUEEN);
                }
            }
        }
        if moving_kind == PieceKind::Rook {
            match from.0 {
                0 => self.castle.remove(CastleRights::WHITE_QUEEN),
                7 => self.castle.remove(CastleRights::WHITE_KING),
                56 => self.castle.remove(CastleRights::BLACK_QUEEN),
                63 => self.castle.remove(CastleRights::BLACK_KING),
                _ => {}
            }
        }
        if let Some(cap) = captured {
            if cap.kind == PieceKind::Rook {
                match captured_sq.0 {
                    0 => self.castle.remove(CastleRights::WHITE_QUEEN),
                    7 => self.castle.remove(CastleRights::WHITE_KING),
                    56 => self.castle.remove(CastleRights::BLACK_QUEEN),
                    63 => self.castle.remove(CastleRights::BLACK_KING),
                    _ => {}
                }
            }
        }

        if stm == Color::Black {
            self.fullmove_number = self.fullmove_number.saturating_add(1);
        }
        self.side_to_move = stm.opp();

        // Fold the new meta state back into the hash.
        self.hash ^= zob_meta(self.side_to_move, self.castle, self.en_passant);

        undo
    }

    pub fn unmake_move(&mut self, m: Move, undo: UndoInfo) {
        let from = m.from();
        let to = m.to();
        let flag = m.flag();

        self.side_to_move = self.side_to_move.opp();
        let stm = self.side_to_move;

        if stm == Color::Black {
            self.fullmove_number = self.fullmove_number.saturating_sub(1);
        }

        self.castle = undo.castle;
        self.en_passant = undo.en_passant;
        self.halfmove_clock = undo.halfmove_clock;

        if flag == Move::CASTLE_KING {
            let (rf, rt) = match stm {
                Color::White => (Square(7), Square(5)),
                Color::Black => (Square(63), Square(61)),
            };
            self.move_piece(rt, rf, stm, PieceKind::Rook);
        } else if flag == Move::CASTLE_QUEEN {
            let (rf, rt) = match stm {
                Color::White => (Square(0), Square(3)),
                Color::Black => (Square(56), Square(59)),
            };
            self.move_piece(rt, rf, stm, PieceKind::Rook);
        }

        if let Some(promo) = m.promo_kind() {
            self.remove_piece(to, stm, promo);
            self.add_piece(from, stm, PieceKind::Pawn);
        } else {
            let moving_kind = self
                .piece_at(to)
                .expect("unmake: to-square is empty")
                .kind;
            self.move_piece(to, from, stm, moving_kind);
        }

        if let Some(cap) = undo.captured {
            let cap_sq = if flag == Move::EP_CAPTURE {
                match stm {
                    Color::White => Square(to.0 - 8),
                    Color::Black => Square(to.0 + 8),
                }
            } else {
                to
            };
            self.add_piece(cap_sq, cap.color, cap.kind);
        }

        // Piece-XORs inside the reverse moves above leave hash equal to
        // the make-time pre-meta value (XOR is self-inverse on the pieces),
        // but the meta-state bits never got unwound. Restoring from the
        // saved snapshot is cheaper than retracing make's meta XORs.
        self.hash = undo.hash;
    }

    /// Pass-the-turn move. The side-to-move and en-passant are updated,
    /// halfmove clock advances, hash is XOR'd accordingly. Caller MUST
    /// ensure the side to move is not in check (a null move would be
    /// illegal). Used by null-move pruning in search.
    pub fn make_null_move(&mut self) -> NullUndo {
        let undo = NullUndo {
            en_passant: self.en_passant,
            halfmove_clock: self.halfmove_clock,
            hash: self.hash,
        };
        self.hash ^= zob_meta(self.side_to_move, self.castle, self.en_passant);
        self.en_passant = None;
        self.halfmove_clock = self.halfmove_clock.saturating_add(1);
        self.side_to_move = self.side_to_move.opp();
        self.hash ^= zob_meta(self.side_to_move, self.castle, self.en_passant);
        undo
    }

    pub fn unmake_null_move(&mut self, undo: NullUndo) {
        self.side_to_move = self.side_to_move.opp();
        self.en_passant = undo.en_passant;
        self.halfmove_clock = undo.halfmove_clock;
        self.hash = undo.hash;
    }

    // -------------------------------------------------------------------------
    // MOVE GENERATION
    // -------------------------------------------------------------------------

    /// Generate all pseudo-legal moves for the side to move. "Pseudo-legal"
    /// means rule-compliant per piece (pawns push forward, knights leap,
    /// etc.) but moves that leave the side-to-move king in check are NOT filtered
    /// out. Use `generate_legal_moves` for the filtered list.
    ///
    /// Castles are filtered for check and transit safety here. Their legality
    /// rule is more than "does not leave the king in check": it also requires
    /// the king not currently in check and not passing through an attacked
    /// square, which is simplest to handle at generation time.
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
        let own_bb = self.color_bb[stm_idx];

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
        let mut bq = self.piece_bb[opp_idx][PieceKind::Bishop.idx()]
            | self.piece_bb[opp_idx][PieceKind::Queen.idx()];
        while let Some(sq) = pop_lsb(&mut bq) {
            danger |= bishop_attacks(sq, occ_no_king);
        }
        let mut rq = self.piece_bb[opp_idx][PieceKind::Rook.idx()]
            | self.piece_bb[opp_idx][PieceKind::Queen.idx()];
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

        let mut pin_ray = [!0u64; 64];
        let bishop_q = self.piece_bb[opp_idx][PieceKind::Bishop.idx()]
            | self.piece_bb[opp_idx][PieceKind::Queen.idx()];
        let rook_q = self.piece_bb[opp_idx][PieceKind::Rook.idx()]
            | self.piece_bb[opp_idx][PieceKind::Queen.idx()];
        let mut snipers = (bishop_q & bishop_attacks(king_sq, 0))
            | (rook_q & rook_attacks(king_sq, 0));
        while let Some(sniper_sq) = pop_lsb(&mut snipers) {
            let between = BETWEEN_BB[king_sq.idx()][sniper_sq.idx()] & self.occupancy;
            if between.count_ones() == 1 {
                let pinned_sq = lsb(between);
                if pinned_sq.bb() & own_bb != 0 {
                    pin_ray[pinned_sq.idx()] =
                        BETWEEN_BB[king_sq.idx()][sniper_sq.idx()] | sniper_sq.bb();
                }
            }
        }

        CheckInfo {
            king_sq,
            checkers,
            check_mask,
            pin_ray,
            danger,
        }
    }

    fn gen_knight_legal(&self, info: &CheckInfo, stm: Color, out: &mut MoveList) {
        let mut knights = self.piece_bb[stm.idx()][PieceKind::Knight.idx()];
        let own = self.color_bb[stm.idx()];
        let opp_bb = self.color_bb[stm.opp().idx()];
        while let Some(from) = pop_lsb(&mut knights) {
            // Knights can never move while pinned (no knight target lies on a line).
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
            let targets = bishop_attacks(from, self.occupancy)
                & !own
                & info.check_mask
                & info.pin_ray[from.idx()];
            emit_targets(from, targets, opp_bb, out);
        }
    }

    fn gen_rook_legal(&self, info: &CheckInfo, stm: Color, out: &mut MoveList) {
        let mut rooks = self.piece_bb[stm.idx()][PieceKind::Rook.idx()];
        let own = self.color_bb[stm.idx()];
        let opp_bb = self.color_bb[stm.opp().idx()];
        while let Some(from) = pop_lsb(&mut rooks) {
            let targets = rook_attacks(from, self.occupancy)
                & !own
                & info.check_mask
                & info.pin_ray[from.idx()];
            emit_targets(from, targets, opp_bb, out);
        }
    }

    fn gen_queen_legal(&self, info: &CheckInfo, stm: Color, out: &mut MoveList) {
        let mut queens = self.piece_bb[stm.idx()][PieceKind::Queen.idx()];
        let own = self.color_bb[stm.idx()];
        let opp_bb = self.color_bb[stm.opp().idx()];
        while let Some(from) = pop_lsb(&mut queens) {
            let targets = queen_attacks(from, self.occupancy)
                & !own
                & info.check_mask
                & info.pin_ray[from.idx()];
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
                            if empty & dbl_bb != 0
                                && pin & dbl_bb != 0
                                && info.check_mask & dbl_bb != 0
                            {
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

    /// EP capture has a rare special case: removing both the moving pawn AND
    /// the captured pawn from the rank can expose the king to a horizontal
    /// slider attack. The pin_ray machinery doesn't catch this because the
    /// pin requires exactly one piece between, and here there are two. Verify
    /// via clone+make+check.
    fn is_ep_legal(&self, from: Square, to: Square, stm: Color, info: &CheckInfo) -> bool {
        // First check that check resolution is satisfied. EP captures either:
        // (1) put the moving pawn on a check_mask square, or
        // (2) capture the checker (if checker is the just-pushed pawn).
        let captured_sq = match stm {
            Color::White => Square(to.0 - 8),
            Color::Black => Square(to.0 + 8),
        };
        if info.checkers != 0
            && info.check_mask & to.bb() == 0
            && info.check_mask & captured_sq.bb() == 0
        {
            return false;
        }
        // Now verify no horizontal-pin discovered check.
        let mut test = self.clone();
        let m = Move::new(from, to, Move::EP_CAPTURE);
        let _ = test.make_move(m);
        !test.in_check(stm)
    }

    // -------------------------------------------------------------------------
    // PERFT
    // -------------------------------------------------------------------------

    /// Count legal-move leaf positions at the given depth. Used to verify
    /// movegen correctness against chessprogramming.org's published
    /// values. `perft(0) = 1` by convention.
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

/// Helper used by piece-by-piece movegen: take a target bitboard `targets`
/// and split it into quiet moves and captures (based on enemy occupancy
/// `opp_bb`), emitting each as a separate `Move` from `from`.
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

fn piece_to_fen_char(p: Piece) -> char {
    match (p.color, p.kind) {
        (Color::White, PieceKind::Pawn) => 'P',
        (Color::White, PieceKind::Knight) => 'N',
        (Color::White, PieceKind::Bishop) => 'B',
        (Color::White, PieceKind::Rook) => 'R',
        (Color::White, PieceKind::Queen) => 'Q',
        (Color::White, PieceKind::King) => 'K',
        (Color::Black, PieceKind::Pawn) => 'p',
        (Color::Black, PieceKind::Knight) => 'n',
        (Color::Black, PieceKind::Bishop) => 'b',
        (Color::Black, PieceKind::Rook) => 'r',
        (Color::Black, PieceKind::Queen) => 'q',
        (Color::Black, PieceKind::King) => 'k',
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starting_fen_roundtrip() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(board.to_fen(), STARTING_FEN);
    }

    #[test]
    fn empty_board_has_no_pieces() {
        let board = Board::empty();
        assert_eq!(board.occupancy, 0);
        for sq_idx in 0..64u8 {
            assert_eq!(board.piece_at(Square(sq_idx)), None);
        }
    }

    #[test]
    fn set_square_updates_bitboards_and_mailbox_in_sync() {
        let mut board = Board::empty();
        let e1 = Square::new(4, 0);
        let king = Piece {
            color: Color::White,
            kind: PieceKind::King,
        };

        board.set_square(e1, Some(king));
        assert_eq!(board.piece_at(e1), Some(king));
        assert_eq!(board.occupancy, e1.bb());
        assert_eq!(board.color_bb[Color::White.idx()], e1.bb());
        assert_eq!(board.color_bb[Color::Black.idx()], 0);
        assert_eq!(
            board.piece_bb[Color::White.idx()][PieceKind::King.idx()],
            e1.bb()
        );

        // Overwrite with a different piece - the old one should be cleanly
        // removed from all bitboards.
        let bk = Piece {
            color: Color::Black,
            kind: PieceKind::Knight,
        };
        board.set_square(e1, Some(bk));
        assert_eq!(board.piece_at(e1), Some(bk));
        assert_eq!(
            board.piece_bb[Color::White.idx()][PieceKind::King.idx()],
            0
        );
        assert_eq!(board.color_bb[Color::White.idx()], 0);
        assert_eq!(board.color_bb[Color::Black.idx()], e1.bb());

        // And clear.
        board.set_square(e1, None);
        assert_eq!(board.piece_at(e1), None);
        assert_eq!(board.occupancy, 0);
    }

    #[test]
    fn starting_position_piece_counts() {
        let board = Board::default();
        assert_eq!(board.occupancy.count_ones(), 32);
        assert_eq!(board.color_bb[Color::White.idx()].count_ones(), 16);
        assert_eq!(board.color_bb[Color::Black.idx()].count_ones(), 16);
        // 8 pawns per side.
        assert_eq!(
            board.piece_bb[Color::White.idx()][PieceKind::Pawn.idx()]
                .count_ones(),
            8
        );
        assert_eq!(
            board.piece_bb[Color::Black.idx()][PieceKind::Pawn.idx()]
                .count_ones(),
            8
        );
        // White king on e1.
        let wk = board.piece_bb[Color::White.idx()][PieceKind::King.idx()];
        assert_eq!(wk, Square::new(4, 0).bb());
    }

    #[test]
    fn kiwipete_roundtrip() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn en_passant_parses_and_roundtrips() {
        // Position after 1. e4 - black to move, e3 is the EP target square.
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.en_passant, Some(Square::new(4, 2)));
        assert_eq!(board.side_to_move, Color::Black);
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn no_castle_rights_renders_as_dash() {
        let fen = "8/8/8/8/8/8/8/4K2k w - - 0 1";
        let board = Board::from_fen(fen).unwrap();
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn fen_rejects_bad_input() {
        assert!(Board::from_fen("").is_err());
        assert!(Board::from_fen("garbage").is_err());
        // Missing the trailing fields.
        assert!(Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
        )
        .is_err());
        // Bad piece character.
        assert!(Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX w KQkq - 0 1"
        )
        .is_err());
    }

    #[test]
    fn square_algebraic_roundtrip() {
        for sq_idx in 0..64u8 {
            let sq = Square(sq_idx);
            let s = sq.to_string();
            assert_eq!(Square::from_algebraic(&s), Some(sq));
        }
    }

    // -------------------------------------------------------------------------
    // Move encoding
    // -------------------------------------------------------------------------

    #[test]
    fn move_pack_unpack() {
        let m = Move::new(Square::new(4, 1), Square::new(4, 3), Move::DOUBLE_PUSH);
        assert_eq!(m.from(), Square::new(4, 1));
        assert_eq!(m.to(), Square::new(4, 3));
        assert_eq!(m.flag(), Move::DOUBLE_PUSH);
        assert!(!m.is_capture());
        assert!(!m.is_promotion());
        assert!(!m.is_castle());
    }

    #[test]
    fn move_capture_promotion_flags() {
        let m = Move::new(Square::new(0, 6), Square::new(0, 7), Move::PROMO_QUEEN);
        assert!(m.is_promotion());
        assert!(!m.is_capture());
        assert_eq!(m.promo_kind(), Some(PieceKind::Queen));

        let m = Move::new(Square::new(1, 6), Square::new(0, 7), Move::PROMO_CAP_KNIGHT);
        assert!(m.is_promotion());
        assert!(m.is_capture());
        assert_eq!(m.promo_kind(), Some(PieceKind::Knight));

        let m = Move::new(Square::new(0, 0), Square::new(7, 0), Move::CASTLE_KING);
        assert!(m.is_castle());
        assert!(!m.is_capture());
    }

    #[test]
    fn move_uci_format() {
        let m = Move::new(Square::new(4, 1), Square::new(4, 3), Move::DOUBLE_PUSH);
        assert_eq!(m.to_string(), "e2e4");
        let m = Move::new(Square::new(0, 6), Square::new(0, 7), Move::PROMO_QUEEN);
        assert_eq!(m.to_string(), "a7a8q");
    }

    // -------------------------------------------------------------------------
    // Bitboard helpers
    // -------------------------------------------------------------------------

    #[test]
    fn pop_lsb_iterates_set_bits() {
        let mut bb = Square::new(0, 0).bb() | Square::new(3, 4).bb() | Square::new(7, 7).bb();
        let mut squares = Vec::new();
        while let Some(sq) = pop_lsb(&mut bb) {
            squares.push(sq);
        }
        assert_eq!(squares, vec![Square::new(0, 0), Square::new(3, 4), Square::new(7, 7)]);
        assert_eq!(bb, 0);
    }

    // -------------------------------------------------------------------------
    // Attack tables
    // -------------------------------------------------------------------------

    #[test]
    fn knight_attacks_corner_and_center() {
        // Knight on a1 attacks exactly b3 and c2.
        let a1 = KNIGHT_ATTACKS[Square::new(0, 0).idx()];
        let expected = Square::new(1, 2).bb() | Square::new(2, 1).bb();
        assert_eq!(a1, expected);

        // Knight on d4 attacks 8 squares.
        let d4 = KNIGHT_ATTACKS[Square::new(3, 3).idx()];
        assert_eq!(d4.count_ones(), 8);
    }

    #[test]
    fn king_attacks_corner_and_center() {
        // King on a1 attacks a2, b1, b2 (3 squares).
        let a1 = KING_ATTACKS[Square::new(0, 0).idx()];
        let expected = Square::new(0, 1).bb()
            | Square::new(1, 0).bb()
            | Square::new(1, 1).bb();
        assert_eq!(a1, expected);
        assert_eq!(a1.count_ones(), 3);

        // King on e4 attacks 8 squares.
        let e4 = KING_ATTACKS[Square::new(4, 3).idx()];
        assert_eq!(e4.count_ones(), 8);
    }

    #[test]
    fn pawn_attacks_directional() {
        // White pawn on e4 attacks d5 and f5.
        let wp = PAWN_ATTACKS[Color::White.idx()][Square::new(4, 3).idx()];
        let expected = Square::new(3, 4).bb() | Square::new(5, 4).bb();
        assert_eq!(wp, expected);

        // Black pawn on e4 attacks d3 and f3.
        let bp = PAWN_ATTACKS[Color::Black.idx()][Square::new(4, 3).idx()];
        let expected = Square::new(3, 2).bb() | Square::new(5, 2).bb();
        assert_eq!(bp, expected);

        // Pawn on a-file only has one attack square (no wraparound).
        let wa = PAWN_ATTACKS[Color::White.idx()][Square::new(0, 3).idx()];
        assert_eq!(wa, Square::new(1, 4).bb());
        assert_eq!(wa.count_ones(), 1);
    }

    #[test]
    fn bishop_attacks_empty_board() {
        // Bishop on d4 with no blockers: 13 squares on the two diagonals.
        let attacks = bishop_attacks(Square::new(3, 3), 0);
        assert_eq!(attacks.count_ones(), 13);
    }

    #[test]
    fn rook_attacks_empty_board() {
        // Rook on d4 with no blockers: 14 squares on its rank and file.
        let attacks = rook_attacks(Square::new(3, 3), 0);
        assert_eq!(attacks.count_ones(), 14);
    }

    #[test]
    #[ignore]
    fn bench_slider_attacks() {
        // Warm up the magic tables
        let _ = bishop_attacks(Square(0), 0);
        let _ = rook_attacks(Square(0), 0);

        const ITERS: u64 = 1_000_000;
        let mut sum = 0u64;
        let mut rng = XorShift64::new(0x12345);
        let occs: [u64; 16] = std::array::from_fn(|_| rng.next() & rng.next());

        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            for &occ in &occs {
                for sq in 0..64u8 {
                    sum ^= bishop_attacks(Square(sq), occ);
                    sum ^= rook_attacks(Square(sq), occ);
                }
            }
        }
        let magic_t = start.elapsed().as_secs_f64();
        let total_ops = (ITERS * 16 * 64 * 2) as f64;

        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            for &occ in &occs {
                for sq in 0..64u8 {
                    sum ^= bishop_attacks_raycast(Square(sq), occ);
                    sum ^= rook_attacks_raycast(Square(sq), occ);
                }
            }
        }
        let ray_t = start.elapsed().as_secs_f64();

        eprintln!("magic:   {:.3}s  ({:.0} M ops/sec)", magic_t, total_ops / magic_t / 1e6);
        eprintln!("raycast: {:.3}s  ({:.0} M ops/sec)", ray_t, total_ops / ray_t / 1e6);
        eprintln!("speedup: {:.2}x", ray_t / magic_t);
        eprintln!("sum (don't optimize out): {}", sum);
    }

    #[test]
    fn magic_matches_raycast() {
        let mut rng = XorShift64::new(0xDEADBEEFCAFEBABE);
        for _ in 0..200 {
            let occ = rng.next() & rng.next();
            for sq_idx in 0..64u8 {
                let sq = Square(sq_idx);
                assert_eq!(
                    bishop_attacks(sq, occ),
                    bishop_attacks_raycast(sq, occ),
                    "bishop mismatch at sq={} occ={:x}", sq_idx, occ
                );
                assert_eq!(
                    rook_attacks(sq, occ),
                    rook_attacks_raycast(sq, occ),
                    "rook mismatch at sq={} occ={:x}", sq_idx, occ
                );
            }
        }
    }

    #[test]
    fn bishop_attacks_with_blocker() {
        // Bishop on a1, blocker on c3. Attacks should include c3 (capture
        // square) but stop there - d4, e5, f6, g7, h8 not reachable.
        let occ = Square::new(2, 2).bb();
        let attacks = bishop_attacks(Square::new(0, 0), occ);
        assert!(attacks & Square::new(1, 1).bb() != 0);  // b2 reached
        assert!(attacks & Square::new(2, 2).bb() != 0);  // c3 reached (blocker)
        assert!(attacks & Square::new(3, 3).bb() == 0);  // d4 not reached
        assert!(attacks & Square::new(7, 7).bb() == 0);  // h8 not reached
    }

    // -------------------------------------------------------------------------
    // Attack queries on real positions
    // -------------------------------------------------------------------------

    #[test]
    fn starting_position_not_in_check() {
        let board = Board::default();
        assert!(!board.in_check(Color::White));
        assert!(!board.in_check(Color::Black));
    }

    #[test]
    fn fools_mate_position_is_check() {
        // 1. f3 e5 2. g4 Qh4# - mate position with white king in check from
        // the queen on h4 (long diagonal e1-h4 isn't blocked because f3 went
        // to f3 not f2).
        let fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3";
        let board = Board::from_fen(fen).unwrap();
        assert!(board.in_check(Color::White));
        assert!(!board.in_check(Color::Black));
    }

    // -------------------------------------------------------------------------
    // Move generation & make/unmake
    // -------------------------------------------------------------------------

    #[test]
    fn starting_position_has_20_moves() {
        let mut board = Board::default();
        let moves = board.generate_legal_moves();
        assert_eq!(moves.len(), 20);
    }

    #[test]
    fn make_unmake_starting_position_round_trip() {
        // Apply every legal move from the start, immediately undo it, and
        // assert the board state is byte-identical to the original.
        let mut board = Board::default();
        let original_fen = board.to_fen();
        let moves = board.generate_legal_moves();
        for m in moves {
            let undo = board.make_move(m);
            board.unmake_move(m, undo);
            assert_eq!(
                board.to_fen(),
                original_fen,
                "make/unmake didn't restore state for move {}",
                m
            );
        }
    }

    #[test]
    fn pawn_double_push_sets_en_passant() {
        let mut board = Board::default();
        // 1. e4 (e2 = sq 12, e4 = sq 28)
        let m = Move::new(Square::new(4, 1), Square::new(4, 3), Move::DOUBLE_PUSH);
        let _ = board.make_move(m);
        assert_eq!(board.en_passant, Some(Square::new(4, 2))); // e3
        assert_eq!(board.side_to_move, Color::Black);
    }

    #[test]
    fn en_passant_capture_removes_the_passed_pawn() {
        // White pushed e2-e4; black has a pawn on d4 that can capture en passant.
        let fen = "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3";
        let mut board = Board::from_fen(fen).unwrap();
        let from = Square::new(3, 3); // d4
        let to = Square::new(4, 2); // e3
        let m = Move::new(from, to, Move::EP_CAPTURE);
        let undo = board.make_move(m);
        // White pawn on e4 should now be gone.
        assert_eq!(board.piece_at(Square::new(4, 3)), None);
        // Captured pawn recorded in undo.
        assert_eq!(
            undo.captured,
            Some(Piece { color: Color::White, kind: PieceKind::Pawn })
        );
        // Unmake restores everything.
        board.unmake_move(m, undo);
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn castle_kingside_moves_rook() {
        // Position with white able to castle kingside.
        let fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1";
        let mut board = Board::from_fen(fen).unwrap();
        let m = Move::new(Square(4), Square(6), Move::CASTLE_KING); // e1g1
        let undo = board.make_move(m);
        assert_eq!(
            board.piece_at(Square(6)),
            Some(Piece { color: Color::White, kind: PieceKind::King })
        );
        assert_eq!(
            board.piece_at(Square(5)),
            Some(Piece { color: Color::White, kind: PieceKind::Rook })
        );
        // King and h-rook moved out of their starting squares.
        assert_eq!(board.piece_at(Square(4)), None);
        assert_eq!(board.piece_at(Square(7)), None);
        // Castle rights for white are gone after castling.
        assert!(!board.castle.has(CastleRights::WHITE_KING));
        assert!(!board.castle.has(CastleRights::WHITE_QUEEN));
        // Unmake restores.
        board.unmake_move(m, undo);
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn promotion_changes_piece_kind() {
        // White pawn on a7 about to promote.
        let fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1";
        let mut board = Board::from_fen(fen).unwrap();
        let m = Move::new(Square::new(0, 6), Square::new(0, 7), Move::PROMO_QUEEN);
        let undo = board.make_move(m);
        assert_eq!(
            board.piece_at(Square::new(0, 7)),
            Some(Piece { color: Color::White, kind: PieceKind::Queen })
        );
        // Unmake restores the pawn.
        board.unmake_move(m, undo);
        assert_eq!(
            board.piece_at(Square::new(0, 6)),
            Some(Piece { color: Color::White, kind: PieceKind::Pawn })
        );
        assert_eq!(board.piece_at(Square::new(0, 7)), None);
    }

    #[test]
    fn cannot_castle_through_check() {
        // Black rook on f8 attacks f1; white kingside castling has to pass
        // the king through f1, so it must be rejected.
        let fen = "5rk1/8/8/8/8/8/8/4K2R w K - 0 1";
        let mut board = Board::from_fen(fen).unwrap();
        let moves = board.generate_legal_moves();
        assert!(
            !moves.iter().any(|m| m.is_castle()),
            "kingside castling should be blocked (king passes through f1)"
        );
    }

    #[test]
    fn legal_filter_rejects_pinned_piece_moving_away() {
        // White king on e1, white rook on e2, black rook on e8. The e2
        // rook is absolutely pinned; moving it sideways is illegal.
        let fen = "4r3/8/8/8/8/8/4R3/4K3 w - - 0 1";
        let mut board = Board::from_fen(fen).unwrap();
        let moves = board.generate_legal_moves();
        // The e2 rook can move along the e-file (still blocks the pin) but
        // not sideways (would expose the king).
        let e2 = Square::new(4, 1);
        let rook_moves: Vec<Move> = moves.iter().copied().filter(|m| m.from() == e2).collect();
        for m in &rook_moves {
            assert_eq!(m.to().file(), 4, "pinned rook move went off the e-file: {}", m);
        }
    }

    // -------------------------------------------------------------------------
    // Perft (chessprogramming.org test values)
    // -------------------------------------------------------------------------

    #[test]
    fn perft_starting_position() {
        let mut board = Board::default();
        assert_eq!(board.perft(1), 20);
        assert_eq!(board.perft(2), 400);
        assert_eq!(board.perft(3), 8902);
    }

    #[test]
    fn perft_kiwipete() {
        // Kiwipete - exercises captures, castles, promotions, EP. Standard
        // test position from Peter McKenzie / Marcel van Kervinck.
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let mut board = Board::from_fen(fen).unwrap();
        assert_eq!(board.perft(1), 48);
        assert_eq!(board.perft(2), 2039);
    }

    #[test]
    fn perft_position_3() {
        // Endgame-ish position - heavy slider movement, narrow line.
        // chessprogramming.org "Position 3".
        let fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
        let mut board = Board::from_fen(fen).unwrap();
        assert_eq!(board.perft(1), 14);
        assert_eq!(board.perft(2), 191);
        assert_eq!(board.perft(3), 2812);
    }

    #[test]
    fn perft_position_4() {
        // chessprogramming.org "Position 4". Note: g2 = white pawn (P),
        // not black - several AI-generated transcripts of this FEN flip
        // the case and produce a different position with the same depth-1
        // count but wrong deeper counts.
        let fen = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
        let mut board = Board::from_fen(fen).unwrap();
        assert_eq!(board.perft(1), 6);
        assert_eq!(board.perft(2), 264);
    }

    #[test]
    fn attackers_of_square_in_kiwipete() {
        // Kiwipete e5: white knight sits there but nothing else on either
        // side hits the square - the g7 bishop's a1-h8 diagonal is blocked
        // by the black knight on f6, the d5 pawn attacks c6/e6, and the
        // black queen on e7 is blocked by the e6 pawn on the e-file.
        let kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let board = Board::from_fen(kiwipete).unwrap();
        let e5 = Square::new(4, 4);
        assert_eq!(board.attackers_of_square(e5, Color::White), 0);
        assert_eq!(board.attackers_of_square(e5, Color::Black), 0);
    }

    // -------------------------------------------------------------------------
    // Zobrist hashing
    // -------------------------------------------------------------------------

    #[test]
    fn zobrist_starting_position_nonzero() {
        let board = Board::default();
        assert_ne!(board.hash, 0, "starting position hash should not be 0");
    }

    #[test]
    fn zobrist_make_unmake_roundtrip() {
        // Walk every legal move from kiwipete, make/unmake each, and confirm
        // the hash returns to the original value.
        let kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let mut board = Board::from_fen(kiwipete).unwrap();
        let before = board.hash;
        let mut moves = MoveList::new();
        board.gen_legal_into(&mut moves);
        for i in 0..moves.len() {
            let m = moves.get(i);
            let undo = board.make_move(m);
            assert_ne!(board.hash, before, "hash should change after {}", m);
            board.unmake_move(m, undo);
            assert_eq!(board.hash, before, "hash should restore after unmake of {}", m);
        }
    }

    #[test]
    fn zobrist_matches_from_scratch() {
        // After making a move, the incremental hash should equal a fresh
        // from_fen on the resulting position.
        let mut board = Board::default();
        let mut moves = MoveList::new();
        board.gen_legal_into(&mut moves);
        // Play e2-e4.
        let e4 = (0..moves.len())
            .map(|i| moves.get(i))
            .find(|m| m.to_string() == "e2e4")
            .expect("e2e4 should be legal from start");
        board.make_move(e4);
        let fresh = Board::from_fen(&board.to_fen()).unwrap();
        assert_eq!(board.hash, fresh.hash, "incremental hash != from_fen hash");
    }

    #[test]
    fn zobrist_transposition_equality() {
        // Reach the same position via two move orders. Knights only so EP
        // target stays None for both sequences (a pawn double push would
        // leave EP set after the last move, breaking the equality).
        let play = |ucis: &[&str]| {
            let mut b = Board::default();
            for uci in ucis {
                let mut ml = MoveList::new();
                b.gen_legal_into(&mut ml);
                let m = (0..ml.len())
                    .map(|i| ml.get(i))
                    .find(|m| m.to_string() == *uci)
                    .unwrap();
                b.make_move(m);
            }
            b
        };
        let a = play(&["b1c3", "b8c6", "g1f3", "g8f6"]);
        let b = play(&["g1f3", "g8f6", "b1c3", "b8c6"]);
        assert_eq!(a.hash, b.hash);
    }

    #[test]
    fn null_move_roundtrip() {
        // make_null_move followed by unmake_null_move must restore the
        // position exactly (hash, side, ep, halfmove).
        let kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let mut board = Board::from_fen(kiwipete).unwrap();
        let before_hash = board.hash;
        let before_side = board.side_to_move;
        let before_ep = board.en_passant;
        let before_hm = board.halfmove_clock;
        let undo = board.make_null_move();
        assert_ne!(board.hash, before_hash, "null move should change hash");
        assert_ne!(board.side_to_move, before_side, "side to move must toggle");
        board.unmake_null_move(undo);
        assert_eq!(board.hash, before_hash);
        assert_eq!(board.side_to_move, before_side);
        assert_eq!(board.en_passant, before_ep);
        assert_eq!(board.halfmove_clock, before_hm);
    }

    #[test]
    fn null_move_clears_en_passant() {
        // Position with ep target set (right after e2e4). A null move must
        // null out the ep target (en passant rights expire after one ply).
        let mut board = Board::default();
        let mut ml = MoveList::new();
        board.gen_legal_into(&mut ml);
        let e2e4 = (0..ml.len())
            .map(|i| ml.get(i))
            .find(|m| m.to_string() == "e2e4")
            .unwrap();
        board.make_move(e2e4);
        assert!(board.en_passant.is_some(), "e2e4 should set ep target");
        let undo = board.make_null_move();
        assert!(board.en_passant.is_none(), "null move clears ep");
        board.unmake_null_move(undo);
        assert!(board.en_passant.is_some(), "unmake restores ep");
    }
}

