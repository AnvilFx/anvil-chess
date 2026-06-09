//! Anvil chess engine: bitboard core, NNUE evaluation, and alpha-beta search.
//! The binary (`src/main.rs`) wraps this in a UCI front-end.

pub mod core;
pub mod eval;
pub mod search;
pub mod timeman;
pub mod uci;

pub use search::{Engine, Limits};
pub use timeman::{GoLimits, TimeBudget};
