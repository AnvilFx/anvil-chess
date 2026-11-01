use std::fmt;
use std::mem::MaybeUninit;

pub const STARTING_FEN: &str =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline]
    pub const fn opp(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline]
    pub const fn idx(self) -> usize {
        self as usize
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PieceKind {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceKind {
    #[inline]
    pub const fn idx(self) -> usize {
        self as usize
    }

    pub const fn from_idx(idx: usize) -> PieceKind {
        match idx {
            0 => PieceKind::Pawn,
            1 => PieceKind::Knight,
            2 => PieceKind::Bishop,
            3 => PieceKind::Rook,
            4 => PieceKind::Queen,
            5 => PieceKind::King,
            _ => panic!("PieceKind::from_idx: index out of range"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Square(pub u8);

impl Square {
    #[inline]
    pub const fn new(file: u8, rank: u8) -> Square {
        Square(rank * 8 + file)
    }

    #[inline]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    #[inline]
    pub const fn bb(self) -> u64 {
        1u64 << self.0
    }

    pub fn from_algebraic(s: &str) -> Option<Square> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = bytes[0].checked_sub(b'a')?;
        let rank = bytes[1].checked_sub(b'1')?;
        if file < 8 && rank < 8 {
            Some(Square::new(file, rank))
        } else {
            None
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}{}",
            (b'a' + self.file()) as char,
            (b'1' + self.rank()) as char
        )
    }
}

/// Bits 0-5 from, 6-11 to, 12-15 flag (chessprogramming.org "Encoding Moves").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Move(pub u16);

impl Move {
    pub const NULL: Move = Move(0);

    pub const QUIET: u16 = 0;
    pub const DOUBLE_PUSH: u16 = 1;
    pub const CASTLE_KING: u16 = 2;
    pub const CASTLE_QUEEN: u16 = 3;
    pub const CAPTURE: u16 = 4;
    pub const EP_CAPTURE: u16 = 5;
    pub const PROMO_KNIGHT: u16 = 8;
    pub const PROMO_BISHOP: u16 = 9;
    pub const PROMO_ROOK: u16 = 10;
    pub const PROMO_QUEEN: u16 = 11;
    pub const PROMO_CAP_KNIGHT: u16 = 12;
    pub const PROMO_CAP_BISHOP: u16 = 13;
    pub const PROMO_CAP_ROOK: u16 = 14;
    pub const PROMO_CAP_QUEEN: u16 = 15;

    #[inline]
    pub const fn new(from: Square, to: Square, flag: u16) -> Move {
        Move((from.0 as u16) | ((to.0 as u16) << 6) | (flag << 12))
    }

    #[inline]
    pub const fn from(self) -> Square {
        Square((self.0 & 0x3F) as u8)
    }

    #[inline]
    pub const fn to(self) -> Square {
        Square(((self.0 >> 6) & 0x3F) as u8)
    }

    #[inline]
    pub const fn flag(self) -> u16 {
        (self.0 >> 12) & 0xF
    }

    #[inline]
    pub const fn is_promotion(self) -> bool {
        self.flag() & 0b1000 != 0
    }

    #[inline]
    pub const fn is_capture(self) -> bool {
        let f = self.flag();
        f == Self::CAPTURE
            || f == Self::EP_CAPTURE
            || f >= Self::PROMO_CAP_KNIGHT
    }

    #[inline]
    pub const fn is_castle(self) -> bool {
        let f = self.flag();
        f == Self::CASTLE_KING || f == Self::CASTLE_QUEEN
    }

    pub fn promo_kind(self) -> Option<PieceKind> {
        if !self.is_promotion() {
            return None;
        }
        Some(match self.flag() & 0b11 {
            0 => PieceKind::Knight,
            1 => PieceKind::Bishop,
            2 => PieceKind::Rook,
            _ => PieceKind::Queen,
        })
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.from(), self.to())?;
        if let Some(k) = self.promo_kind() {
            let ch = match k {
                PieceKind::Knight => 'n',
                PieceKind::Bishop => 'b',
                PieceKind::Rook => 'r',
                PieceKind::Queen => 'q',
                _ => unreachable!(),
            };
            write!(f, "{}", ch)?;
        }
        Ok(())
    }
}

pub const MAX_MOVES: usize = 256;

pub struct MoveList {
    moves: [MaybeUninit<Move>; MAX_MOVES],
    len: usize,
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl MoveList {
    #[inline]
    pub fn new() -> Self {
        Self {
            moves: unsafe {
                MaybeUninit::<[MaybeUninit<Move>; MAX_MOVES]>::uninit().assume_init()
            },
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES);
        self.moves[self.len] = MaybeUninit::new(m);
        self.len += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn get(&self, i: usize) -> Move {
        debug_assert!(i < self.len);
        unsafe { self.moves[i].assume_init() }
    }

    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        (0..self.len).map(move |i| self.get(i))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CastleRights(pub u8);

impl CastleRights {
    pub const NONE: CastleRights = CastleRights(0);
    pub const ALL: CastleRights = CastleRights(0b1111);

    pub const WHITE_KING: u8 = 0b0001;
    pub const WHITE_QUEEN: u8 = 0b0010;
    pub const BLACK_KING: u8 = 0b0100;
    pub const BLACK_QUEEN: u8 = 0b1000;

    #[inline]
    pub fn has(self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    #[inline]
    pub fn add(&mut self, flag: u8) {
        self.0 |= flag;
    }

    #[inline]
    pub fn remove(&mut self, flag: u8) {
        self.0 &= !flag;
    }
}
