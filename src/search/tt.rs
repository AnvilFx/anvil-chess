// Transposition table.
use super::*;

pub const TT_DEFAULT_MB: usize = 64;
pub const TT_MAX_MB: usize = 32768;
pub(super) const BUCKET_SIZE: usize = 4;
pub(super) const NO_EVAL: i16 = i16::MIN;

pub(super) const BOUND_NONE: u8 = 0;
pub(super) const BOUND_EXACT: u8 = 1;
pub(super) const BOUND_LOWER: u8 = 2; // score >= stored
pub(super) const BOUND_UPPER: u8 = 3; // score <= stored

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(super) struct TtEntry {
    pub(super) key: u64,
    pub(super) move_packed: u16,
    pub(super) score: i16,
    pub(super) eval: i16,
    pub(super) depth: u8,
    /// Generation in the high 5 bits, then the was-PV flag, bound in the low 2.
    pub(super) gen_bound: u8,
}

impl TtEntry {
    #[inline]
    pub(super) fn bound(&self) -> u8 {
        self.gen_bound & 3
    }
    #[inline]
    pub(super) fn gen(&self) -> u8 {
        self.gen_bound >> 3
    }
    #[inline]
    pub(super) fn pv(&self) -> bool {
        self.gen_bound & 4 != 0
    }
    #[inline]
    pub(super) fn eval(&self) -> Option<i32> {
        (self.eval != NO_EVAL).then_some(self.eval as i32)
    }
}

/// One cache line of entries sharing an index.
#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
pub(super) struct Bucket([TtEntry; BUCKET_SIZE]);

pub struct Tt {
    pub(super) buckets: crate::mem::HugeBuf<Bucket>,
    pub(super) mask: u64,
    pub(super) gen: u8,
}

impl Default for Tt {
    fn default() -> Self {
        Self::new()
    }
}

impl Tt {
    pub fn new() -> Self {
        Self::with_mb(TT_DEFAULT_MB)
    }

    /// Size rounds down to a power of two buckets.
    pub fn with_mb(mb: usize) -> Self {
        let wanted = (mb.clamp(1, TT_MAX_MB) * 1024 * 1024 / std::mem::size_of::<Bucket>()).max(1);
        let n = 1usize << (usize::BITS - 1 - wanted.leading_zeros());
        // All-zero bytes are an empty entry.
        Self { buckets: unsafe { crate::mem::HugeBuf::zeroed(n) }, mask: (n - 1) as u64, gen: 0 }
    }

    pub(super) fn clear(&mut self) {
        self.buckets.fill(Bucket::default());
        self.gen = 0;
    }

    pub(super) fn new_search(&mut self) {
        self.gen = (self.gen + 1) & 31;
    }

    /// Permille of entries written by the current search, sampled from the first 1000.
    pub(super) fn hashfull(&self) -> u32 {
        let n = (1000 / BUCKET_SIZE).min(self.buckets.len());
        self.buckets[..n]
            .iter()
            .flat_map(|b| b.0.iter())
            .filter(|e| e.bound() != BOUND_NONE && e.gen() == self.gen)
            .count() as u32
            * 1000
            / (n * BUCKET_SIZE) as u32
    }

    #[inline]
    pub(super) fn bucket(&self, hash: u64) -> &Bucket {
        &self.buckets[(hash & self.mask) as usize]
    }

    #[inline]
    pub(super) fn probe(&self, hash: u64) -> Option<TtEntry> {
        self.bucket(hash).0.iter().copied().find(|e| e.key == hash && e.bound() != BOUND_NONE)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn store(
        &mut self,
        hash: u64,
        mv: Option<Move>,
        score: i16,
        eval: Option<i32>,
        depth: u8,
        bound: u8,
        pv: bool,
    ) {
        let gen = self.gen;
        let bucket = &mut self.buckets[(hash & self.mask) as usize].0;

        let slot = match bucket.iter().position(|e| e.key == hash && e.bound() != BOUND_NONE) {
            Some(i) => i,
            None => (0..BUCKET_SIZE)
                .min_by_key(|&i| {
                    let e = &bucket[i];
                    if e.bound() == BOUND_NONE {
                        return i32::MIN;
                    }
                    let age = (gen.wrapping_sub(e.gen()) & 31) as i32;
                    e.depth as i32 - 8 * age
                })
                .unwrap(),
        };

        let e = &mut bucket[slot];
        let same = e.key == hash && e.bound() != BOUND_NONE;
        // Keep a clearly deeper entry from this search, but still take a fresh move.
        if same && bound != BOUND_EXACT && e.gen() == gen && depth as i32 + 4 <= e.depth as i32 {
            if let Some(m) = mv {
                e.move_packed = m.0;
            }
            return;
        }
        let move_packed = match mv {
            Some(m) => m.0,
            None if same => e.move_packed,
            None => 0,
        };
        *e = TtEntry {
            key: hash,
            move_packed,
            score,
            eval: eval.map_or(NO_EVAL, |v| v.clamp(-INF, INF) as i16),
            depth,
            gen_bound: (gen << 3) | ((pv as u8) << 2) | bound,
        };
    }

    #[inline]
    pub(super) fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let ptr = self.bucket(hash) as *const Bucket as *const i8;
            unsafe { _mm_prefetch(ptr, _MM_HINT_T0) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = hash;
    }
}

// Mate scores adjusted by ply when stored / probed so they're independent of search depth.
#[inline]
pub(super) fn tt_score_to_store(score: i32, ply: i32) -> i16 {
    let adjusted = if score >= MATE_THRESHOLD {
        score + ply
    } else if score <= -MATE_THRESHOLD {
        score - ply
    } else {
        score
    };
    adjusted.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[inline]
pub(super) fn tt_score_from_probe(stored: i16, ply: i32) -> i32 {
    let s = stored as i32;
    if s >= MATE_THRESHOLD {
        s - ply
    } else if s <= -MATE_THRESHOLD {
        s + ply
    } else {
        s
    }
}

/// Tightens the static eval with a TT score whose bound points the right way.
#[inline]
pub(super) fn adjust_eval(eval: i32, tt_score: Option<i32>, bound: u8) -> i32 {
    match tt_score {
        Some(s) if s.abs() < MATE_THRESHOLD => match bound {
            BOUND_EXACT => s,
            BOUND_LOWER if s > eval => s,
            BOUND_UPPER if s < eval => s,
            _ => eval,
        },
        _ => eval,
    }
}
