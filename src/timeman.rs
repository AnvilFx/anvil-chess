// Soft limit: no new iteration starts past it. Hard limit: the search aborts mid-iteration.

use crate::chess::Color;

pub const MOVE_OVERHEAD_MS: u64 = 15;

/// Assumed moves remaining when `movestogo` is absent.
const MOVES_HORIZON: u64 = 40;

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

/// Milliseconds; `u64::MAX` means unbounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeBudget {
    pub soft_ms: u64,
    pub hard_ms: u64,
}

impl TimeBudget {
    pub const UNLIMITED: TimeBudget = TimeBudget { soft_ms: u64::MAX, hard_ms: u64::MAX };

    #[inline]
    pub fn is_unlimited(&self) -> bool {
        self.soft_ms == u64::MAX
    }
}

pub fn allocate(go: &GoLimits, stm: Color) -> TimeBudget {
    // Movetime is used in full; the overhead reserve only matters for a running clock.
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

    let usable = remaining.saturating_sub(MOVE_OVERHEAD_MS).max(1);

    let mtg = go.movestogo.map(|m| (m as u64).max(1)).unwrap_or(MOVES_HORIZON);

    let base = usable / mtg + inc.saturating_mul(3) / 4;

    let cap = usable.saturating_mul(4) / 5;
    let soft = base.min(cap).max(1);

    let hard = base.saturating_mul(4).min(cap).max(soft);

    TimeBudget { soft_ms: soft, hard_ms: hard }
}

/// Shrink the soft target while the best move is stable or takes most of the nodes;
/// grow it when the score drops or the nodes are spread over several moves.
pub fn scaled_soft(
    soft_ms: u64,
    hard_ms: u64,
    bm_stability: u32,
    score: i32,
    prev_score: i32,
    best_node_share: Option<f64>,
) -> u64 {
    if soft_ms == u64::MAX {
        return u64::MAX;
    }

    let stab = bm_stability.min(8) as f64;
    let stability_scale = 1.0 - 0.04 * stab;

    let drop = (prev_score - score).max(0);
    let fall_scale = 1.0 + (drop as f64 / 100.0).min(1.0) * 0.5;

    let node_scale = best_node_share.map_or(1.0, |f| (1.5 - f.clamp(0.0, 1.0)) * 1.35);

    let scaled = (soft_ms as f64) * stability_scale * fall_scale * node_scale;
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
        let steady = scaled_soft(soft, hard, 8, 50, 50, None);
        let fresh = scaled_soft(soft, hard, 0, 50, 50, None);
        assert!(steady < fresh, "a stable PV should shorten the target");

        let falling = scaled_soft(soft, hard, 0, 10, 90, None);
        assert!(falling > fresh, "a falling score should lengthen the target");
        assert!(falling <= hard, "never exceed the hard ceiling");
    }

    #[test]
    fn unlimited_soft_passes_through() {
        assert_eq!(scaled_soft(u64::MAX, u64::MAX, 0, 0, 0, None), u64::MAX);
    }

    #[test]
    fn node_share_scales_the_target() {
        let (soft, hard) = (1000, 4000);
        let plain = scaled_soft(soft, hard, 0, 0, 0, None);
        let clear = scaled_soft(soft, hard, 0, 0, 0, Some(0.95));
        let split = scaled_soft(soft, hard, 0, 0, 0, Some(0.3));
        assert!(clear < plain, "one move taking the nodes should stop sooner");
        assert!(split > plain, "nodes spread over several moves should think longer");
        assert!(scaled_soft(soft, 1200, 0, 0, 0, Some(0.0)) <= 1200);
    }
}
