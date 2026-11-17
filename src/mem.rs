// Large zeroed buffers backed by transparent huge pages where the OS allows it, to cut TLB misses
// on the randomly accessed TT and feature weights.

use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

pub struct HugeBuf<T> {
    ptr: NonNull<T>,
    len: usize,
    #[cfg(target_os = "linux")]
    map_len: usize,
    #[cfg(not(target_os = "linux"))]
    layout: std::alloc::Layout,
}

unsafe impl<T: Send> Send for HugeBuf<T> {}
unsafe impl<T: Sync> Sync for HugeBuf<T> {}

#[cfg(target_os = "linux")]
mod sys {
    extern "C" {
        pub fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
        pub fn munmap(addr: *mut u8, len: usize) -> i32;
        pub fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    }
    pub const PROT_READ_WRITE: i32 = 0x1 | 0x2;
    pub const MAP_PRIVATE_ANON: i32 = 0x02 | 0x20;
    pub const MADV_HUGEPAGE: i32 = 14;
    pub const HUGE: usize = 2 << 20;

    /// `len` (a multiple of HUGE) zeroed bytes aligned to HUGE and advised as huge pages; null on failure.
    /// Over-maps by one huge page to align the start, as madvise only takes effect on aligned 2 MB
    /// ranges; it is just a hint, so its result is ignored.
    pub unsafe fn map_huge(len: usize) -> *mut u8 {
        let base = mmap(std::ptr::null_mut(), len + HUGE, PROT_READ_WRITE, MAP_PRIVATE_ANON, -1, 0);
        if base as isize == -1 {
            return std::ptr::null_mut();
        }
        let head = base.align_offset(HUGE);
        if head > 0 {
            munmap(base, head);
        }
        let start = base.add(head);
        munmap(start.add(len), HUGE - head);
        madvise(start, len, MADV_HUGEPAGE);
        start
    }
}

/// Global allocator that puts every block of 4 KB or more on transparent huge pages, so the
/// search's tables (histories, accumulators, magic attack tables) cost no 4 KB TLB misses.
/// Blocks up to 1 MB come in power-of-two classes from shared 2 MB chunks and are recycled per class;
/// bigger ones get their own mapping. Everything smaller goes to the system allocator.
#[cfg(target_os = "linux")]
mod huge_alloc {
    use super::sys::{map_huge, munmap, HUGE};
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicBool, Ordering};

    const MIN_SHIFT: u32 = 12;
    const MAX_SHIFT: u32 = 20;
    const CLASSES: usize = (MAX_SHIFT - MIN_SHIFT + 1) as usize;

    struct State {
        free: [*mut u8; CLASSES],
        bump: *mut u8,
        end: *mut u8,
    }

    pub struct HugeAlloc {
        lock: AtomicBool,
        state: std::cell::UnsafeCell<State>,
    }

    unsafe impl Sync for HugeAlloc {}

    #[inline]
    fn is_small(l: &Layout) -> bool {
        l.size() < 1 << MIN_SHIFT || l.align() > 1 << MIN_SHIFT
    }

    impl HugeAlloc {
        pub const fn new() -> Self {
            let s = State { free: [std::ptr::null_mut(); CLASSES], bump: std::ptr::null_mut(), end: std::ptr::null_mut() };
            Self { lock: AtomicBool::new(false), state: std::cell::UnsafeCell::new(s) }
        }

        fn with<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
            while self.lock.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
                std::hint::spin_loop();
            }
            let r = f(unsafe { &mut *self.state.get() });
            self.lock.store(false, Ordering::Release);
            r
        }

        unsafe fn alloc_big(&self, l: Layout) -> *mut u8 {
            let shift = l.size().next_power_of_two().trailing_zeros().max(MIN_SHIFT);
            if shift > MAX_SHIFT {
                return map_huge(l.size().next_multiple_of(HUGE));
            }
            let class = (shift - MIN_SHIFT) as usize;
            let size = 1usize << shift;
            self.with(|s| {
                let p = s.free[class];
                if !p.is_null() {
                    s.free[class] = *(p as *mut *mut u8);
                    return p;
                }
                // Blocks are carved in order of size class alignment from a 2 MB aligned chunk.
                let mut start = if s.bump.is_null() { s.end } else { s.bump.add(s.bump.align_offset(size)) };
                if s.bump.is_null() || start.add(size) > s.end {
                    let chunk = map_huge(HUGE);
                    if chunk.is_null() {
                        return chunk;
                    }
                    s.end = chunk.add(HUGE);
                    start = chunk;
                }
                s.bump = start.add(size);
                start
            })
        }
    }

    unsafe impl GlobalAlloc for HugeAlloc {
        #[inline]
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            if is_small(&l) {
                System.alloc(l)
            } else {
                self.alloc_big(l)
            }
        }

        #[inline]
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            if is_small(&l) {
                return System.alloc_zeroed(l);
            }
            let p = self.alloc_big(l);
            // Fresh mappings are zero, recycled class blocks are not.
            if !p.is_null() && l.size() <= 1 << MAX_SHIFT {
                std::ptr::write_bytes(p, 0, l.size());
            }
            p
        }

        #[inline]
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            if is_small(&l) {
                return System.dealloc(p, l);
            }
            let shift = l.size().next_power_of_two().trailing_zeros().max(MIN_SHIFT);
            if shift > MAX_SHIFT {
                munmap(p, l.size().next_multiple_of(HUGE));
                return;
            }
            let class = (shift - MIN_SHIFT) as usize;
            self.with(|s| {
                *(p as *mut *mut u8) = s.free[class];
                s.free[class] = p;
            });
        }

        #[inline]
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
            let nl = Layout::from_size_align_unchecked(new_size, l.align());
            if is_small(&l) && is_small(&nl) {
                return System.realloc(p, l, new_size);
            }
            let q = self.alloc(nl);
            if !q.is_null() {
                std::ptr::copy_nonoverlapping(p, q, l.size().min(new_size));
                self.dealloc(p, l);
            }
            q
        }
    }

    #[global_allocator]
    static GLOBAL: HugeAlloc = HugeAlloc::new();
}

impl<T: Copy> HugeBuf<T> {
    /// `len` elements of all-zero bytes.
    ///
    /// # Safety
    /// All-zero bytes must be a valid `T`.
    pub unsafe fn zeroed(len: usize) -> Self {
        let bytes = (len * std::mem::size_of::<T>()).max(1);
        #[cfg(target_os = "linux")]
        {
            let map_len = bytes.next_multiple_of(sys::HUGE);
            let start = sys::map_huge(map_len);
            assert!(!start.is_null(), "mmap of {map_len} bytes failed");
            Self { ptr: NonNull::new_unchecked(start as *mut T), len, map_len }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let layout = std::alloc::Layout::from_size_align(bytes, std::mem::align_of::<T>().max(64)).unwrap();
            let p = std::alloc::alloc_zeroed(layout);
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Self { ptr: NonNull::new_unchecked(p as *mut T), len, layout }
        }
    }
}

impl<T> Drop for HugeBuf<T> {
    fn drop(&mut self) {
        unsafe {
            #[cfg(target_os = "linux")]
            sys::munmap(self.ptr.as_ptr() as *mut u8, self.map_len);
            #[cfg(not(target_os = "linux"))]
            std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

impl<T> Deref for HugeBuf<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for HugeBuf<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_buffer_reads_and_writes() {
        let mut b = unsafe { HugeBuf::<[u64; 8]>::zeroed(100_000) };
        assert_eq!(b.len(), 100_000);
        assert!(b.iter().all(|r| r.iter().all(|&x| x == 0)));
        b[99_999][7] = 42;
        b.fill([1; 8]);
        assert_eq!(b[12_345][3], 1);
    }

    #[test]
    fn large_allocations_are_zeroed_and_kept_apart() {
        for round in 0..3 {
            let sizes = [4096usize, 5000, 65_536, 1 << 20, (1 << 20) + 1, 3 << 20];
            let mut bufs: Vec<Vec<u8>> = sizes.iter().map(|&n| vec![0u8; n]).collect();
            for (i, b) in bufs.iter_mut().enumerate() {
                assert!(b.iter().all(|&x| x == 0), "round {round} size {}", b.len());
                b.fill(i as u8 + 1);
            }
            for (i, b) in bufs.iter().enumerate() {
                assert!(b.iter().all(|&x| x == i as u8 + 1));
            }
            let mut v: Vec<u64> = Vec::new();
            for k in 0..300_000u64 {
                v.push(k);
            }
            assert!(v.iter().enumerate().all(|(k, &x)| x == k as u64));
        }
    }
}
