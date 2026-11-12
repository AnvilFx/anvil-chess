// Large zeroed buffers backed by transparent huge pages where the OS allows it, to cut TLB misses
// on the randomly accessed TT and feature weights.

use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

pub struct HugeBuf<T> {
    ptr: NonNull<T>,
    len: usize,
    #[cfg(target_os = "linux")]
    map: (*mut u8, usize),
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
            use sys::*;
            // Over-map by one huge page so the start can be aligned to it; madvise only takes effect
            // on aligned 2 MB ranges, and is just a hint, so its result is ignored.
            let map_len = bytes + HUGE;
            let base = mmap(std::ptr::null_mut(), map_len, PROT_READ_WRITE, MAP_PRIVATE_ANON, -1, 0);
            assert!(base as isize != -1, "mmap of {map_len} bytes failed");
            let start = base.add(base.align_offset(HUGE));
            madvise(start, bytes.next_multiple_of(HUGE), MADV_HUGEPAGE);
            Self { ptr: NonNull::new_unchecked(start as *mut T), len, map: (base, map_len) }
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
            sys::munmap(self.map.0, self.map.1);
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
}
