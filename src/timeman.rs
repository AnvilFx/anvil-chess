// Time management: turn a UCI `go` command into a per-move time budget, then
// shape that budget during iterative deepening from how settled the search is.
// Policy (how much time to spend) stays separate from mechanism (when the clock
// is polled, in `search.rs`).
//
// The budget is a pair:
//   soft (optimum): the target, checked between deepening iterations. Once it is
//     reached no new iteration starts; `scaled_soft` adjusts it from PV
//     stability and score swings.
//   hard (maximum): an absolute ceiling, checked inside the search every few
//     thousand nodes. Crossing it aborts at once. Always leaves a clock reserve
//     so the engine cannot flag.

use crate::core::Color;

/// Communication and scheduling slack subtracted from the clock so a move is
/// reported before the flag falls. Covers GUI, pipe, and OS-scheduling latency.
pub const MOVE_OVERHEAD_MS: u64 = 15;

/// Assumed remaining moves when `movestogo` is absent. Dividing the clock by a
/// fixed horizon paces a sudden-death or increment game and self-corrects: a
/// shrinking clock yields a shrinking slice.
const MOVES_HORIZON: u64 = 40;

/// Raw timing fields from a UCI `go` command. All optional; no clock field set
/// means no wall-clock limit (depth, nodes, or infinite).
#[derive(Clone, Copy, Debug, Default)]
pub struct GoLimits {
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub movetime: Option<u64>,
    pub infinite: bool,
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: u64,
    pub binc: u64,
    pub movestogo: Option<u32>,
}

/// A move's time allotment in milliseconds. `u64::MAX` means unbounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeBudget {
    pub soft_ms: u64,
    pub hard_ms: u64,
}

impl TimeBudget {
    /// No wall-clock limit (fixed depth, nodes, or `go infinite`).
    pub const UNLIMITED: TimeBudget = TimeBudget {
        soft_ms: u64::MAX,
        hard_ms: u64::MAX,
    };

    #[inline]
    pub fn is_unlimited(&self) -> bool {
        self.soft_ms == u64::MAX
    }
}

/// Time budget for the side to move, derived from a `go` command.
///
/// An explicit `movetime` takes priority and is used in full. Otherwise a clock
/// (`wtime`/`btime`) yields a soft/hard pair. With neither, the budget is
/// unbounded and the caller falls back to a depth limit.
pub fn allocate(go: &GoLimits, stm: Color) -> TimeBudget {
    // Fixed move time is an explicit per-move instruction, so it is used in full
    // (soft == hard). The overhead reserve guards a cumulative clock from
    // flagging and does not apply here; trimming movetime would hand the engine
    // less time than instructed.
    if let Some(mt) = go.movetime {
        let t = mt.max(1);
        return TimeBudget { soft_ms: t, hard_ms: t };
    }

    let (clock, inc) = match stm {
        Color::White => (go.wtime, go.winc),
        Color::Black => (go.btime, go.binc),
    };
    let remaining = match clock {
        Some(t) => t,
        None => return TimeBudget::UNLIMITED,
    };

    // Reserve comms overhead before any clock arithmetic.
    let usable = remaining.saturating_sub(MOVE_OVERHEAD_MS).max(1);

    // Moves until the next increment. Fall back to the horizon for sudden-death
    // or increment games where the GUI gives no `movestogo`.
    let mtg = go
        .movestogo
        .map(|m| (m as u64).max(1))
        .unwrap_or(MOVES_HORIZON);

    // Ideal slice for this move: an even share of the clock plus most of the
    // increment, which is earned back next move.
    let base = usable / mtg + inc.saturating_mul(3) / 4;

    // Soft target, capped so a large increment cannot push spending past a safe
    // share of the clock.
    let cap = usable.saturating_mul(4) / 5; // keeps a ~20% reserve
    let soft = base.min(cap).max(1);

    // Hard ceiling: allows extra time on a critical or unstable move while the
    // cap preserves a reserve, so the flag cannot fall.
    let hard = base.saturating_mul(4).min(cap).max(soft);

    TimeBudget { soft_ms: soft, hard_ms: hard }
}

/// Reshape the soft target for the next iteration from how settled the search
/// is. Both signals spend more when the result is uncertain:
///
///   `bm_stability`: consecutive completed iterations with an unchanged root
///     best move. A long-stable PV is confident, so the target shrinks.
///   score drop since the previous iteration: a sign of trouble, so the target
///     grows.
///
/// The result is clamped to `hard_ms`. An unbounded `soft_ms` (`u64::MAX`)
/// passes through unchanged.
pub fn scaled_soft(soft_ms: u64, hard_ms: u64, bm_stability: u32, score: i32, prev_score: i32) -> u64 {
    if soft_ms == u64::MAX {
        return u64::MAX;
    }

    // No recent change keeps the full target; each stable iteration shaves a
    // little, down to ~0.68x after 8 stable plies.
    let stab = bm_stability.min(8) as f64;
    let stability_scale = 1.0 - 0.04 * stab;

    // A falling score extends the target up to +50% as the drop nears a pawn.
    let drop = (prev_score - score).max(0);
    let fall_scale = 1.0 + (drop as f64 / 100.0).min(1.0) * 0.5;

    let scaled = (soft_ms as f64) * stability_scale * fall_scale;
    (scaled.round() as u64).clamp(1, hard_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn go_clock(wtime: u64, winc: u64) -> GoLimits {
        GoLimits { wtime: Some(wtime), winc, ..Default::default() }
    }

    #[test]
    fn movetime_is_used_in_full() {
        // An explicit movetime is honored exactly, with no overhead shaved off.
        let go = GoLimits { movetime: Some(1000), ..Default::default() };
        let b = allocate(&go, Color::White);
        assert_eq!(b.soft_ms, 1000);
        assert_eq!(b.hard_ms, 1000);
    }

    #[test]
    fn no_clock_is_unlimited() {
        assert!(allocate(&GoLimits::default(), Color::White).is_unlimited());
    }

    #[test]
    fn hard_never_risks_flagging() {
        // Across a range of clocks and increments, the hard cap keeps a reserve.
        for &(t, inc) in &[(1000u64, 0u64), (60_000, 1000), (300_000, 0), (200, 100)] {
            let b = allocate(&go_clock(t, inc), Color::White);
            let usable = t - MOVE_OVERHEAD_MS;
            assert!(b.hard_ms <= usable, "hard {} must stay under usable {}", b.hard_ms, usable);
            assert!(b.soft_ms <= b.hard_ms, "soft {} <= hard {}", b.soft_ms, b.hard_ms);
            assert!(b.soft_ms >= 1);
        }
    }

    #[test]
    fn more_time_with_more_on_the_clock() {
        let small = allocate(&go_clock(10_000, 0), Color::White);
        let big = allocate(&go_clock(120_000, 0), Color::White);
        assert!(big.soft_ms > small.soft_ms);
    }

    #[test]
    fn movestogo_one_spends_a_big_share() {
        // Last move before the next time control spends most of the clock.
        let b = allocate(&GoLimits { wtime: Some(10_000), movestogo: Some(1), ..Default::default() }, Color::White);
        assert!(b.soft_ms >= 7_000, "movestogo=1 should spend a big share, got {}", b.soft_ms);
    }

    #[test]
    fn picks_side_to_move_clock() {
        let go = GoLimits { wtime: Some(100_000), btime: Some(1_000), ..Default::default() };
        let w = allocate(&go, Color::White);
        let b = allocate(&go, Color::Black);
        assert!(w.soft_ms > b.soft_ms, "white has far more time than black");
    }

    #[test]
    fn stability_shortens_and_drops_lengthen() {
        let soft = 1000;
        let hard = 4000;
        let steady = scaled_soft(soft, hard, 8, 50, 50);
        let fresh = scaled_soft(soft, hard, 0, 50, 50);
        assert!(steady < fresh, "a stable PV should shorten the target");

        let falling = scaled_soft(soft, hard, 0, 10, 90); // dropped 80cp
        assert!(falling > fresh, "a falling score should lengthen the target");
        assert!(falling <= hard, "never exceed the hard ceiling");
    }

    #[test]
    fn unlimited_soft_passes_through() {
        assert_eq!(scaled_soft(u64::MAX, u64::MAX, 0, 0, 0), u64::MAX);
    }
}
