#![doc = include_str!("../README.md")]

use std::fmt::Debug;

use deepsize::DeepSizeOf;
use serde::{Deserialize, Serialize};

/// A densely-packed vector of integers with fixed bit-length
#[derive(DeepSizeOf, Serialize, Deserialize)]
pub struct BitpackVec {
    data: Vec<u64>,
    len: usize,
    width: u8,
}

impl BitpackVec {
    /// Constructs a new `BitpackVec` with the given bit-width.
    ///
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let bv = BitpackVec::new(5);
    /// assert_eq!(bv.width(), 5);
    /// ```
    pub fn new(width: usize) -> BitpackVec {
        assert!(width > 0, "bitpack width must be greater than 0");
        assert!(
            width <= 64,
            "bitpack width must be less than or equal to 64"
        );

        BitpackVec {
            data: Vec::with_capacity(1),
            len: 0,
            width: width as u8,
        }
    }

    /// Construct a `BitpackVec` from a vector of `u64`s, interpreting the
    /// vector with the given bitwidth.
    ///
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    ///
    /// let v = vec![6];
    /// let bv = BitpackVec::from_raw_vec(v, 5, 12);
    /// assert_eq!(bv.at(0), 6);
    /// assert_eq!(bv.at(1), 0);
    /// ```
    pub fn from_raw_vec(data: Vec<u64>, width: usize, len: usize) -> BitpackVec {
        assert!(
            data.len() * 64 > width * len,
            "data is not long enough to be valid"
        );
        BitpackVec {
            data,
            width: width as u8,
            len,
        }
    }

    /// Returns the internal vector representing the bitpacked data
    ///
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    ///
    /// let bv = BitpackVec::from_slice(&[10, 15]);
    /// let v = bv.into_raw_vec();
    /// assert_eq!(v.len(), 1);
    /// assert_eq!(v[0], (15 << 4) | 10);
    /// ```
    pub fn into_raw_vec(self) -> Vec<u64> {
        self.data
    }

    /// Reference to the internal vector, see [into_raw_vec](Self::into_raw_vec)
    pub fn as_raw(&self) -> &[u64] {
        &self.data
    }

    /// Construct a `BitpackVec` from a slice of `u64`s. The smallest
    /// correct bitwidth will be computed.
    ///
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let bv = BitpackVec::from_slice(&[5, 12, 13]);
    /// assert_eq!(bv.width(), 4);
    /// assert_eq!(bv.at(2), 13);
    /// ```
    pub fn from_slice(x: &[u64]) -> BitpackVec {
        assert!(
            !x.is_empty(),
            "Cannot make bitpacked vector from empty slice"
        );

        // scan the data to figure out the fewest bits needed
        let max = x.iter().max().unwrap();
        let bits = 64 - max.leading_zeros();

        let mut bv = BitpackVec::new(bits as usize);

        for i in x {
            bv.push(*i);
        }

        bv
    }

    /// Returns the number of items in the bitpacked vector.
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let bv = BitpackVec::from_slice(&[5, 12, 13]);
    /// assert_eq!(bv.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the packing width of the vector (the size in bits of each
    /// element).
    ///
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let bv = BitpackVec::from_slice(&[5, 12, 13]);
    /// assert_eq!(bv.width(), 4);
    /// ```
    pub fn width(&self) -> usize {
        self.width as usize
    }

    /// Returns the number of items that can be inserted into the
    /// `BitpackVec` before the vector will grow.
    pub fn capacity(&self) -> usize {
        (self.data.capacity() * 64) / self.width()
    }

    /// Determines if `x` can fit inside this vector.
    ///
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    ///
    /// let bv = BitpackVec::new(5);
    ///
    /// assert!(bv.fits(31));
    /// assert!(!bv.fits(32));
    /// ```
    pub fn fits(&self, x: u64) -> bool {
        x < 2_u64.pow(self.width as u32)
    }

    /// Appends an item to the back of the `BitpackVec`. Panics if the item
    /// is too large for the vector's bitwidth.
    ///
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let mut bv = BitpackVec::new(5);
    ///
    /// bv.push(4);
    /// bv.push(22);
    ///
    /// assert_eq!(bv.at(0), 4);
    /// assert_eq!(bv.at(1), 22);
    /// ```
    ///
    /// Adding items that are too large will cause a panic:
    /// ```should_panic
    /// use bitpack_vec::BitpackVec;
    /// let mut bv = BitpackVec::new(5);
    ///
    /// bv.push(90);  // panics
    /// ```
    pub fn push(&mut self, x: u64) {
        assert!(
            self.fits(x),
            "value {} too large to bitpack to width {}",
            x,
            self.width
        );

        // calculate position info
        let start_bit = self.len * self.width();
        let stop_bit = start_bit + self.width() - 1;

        let start_u64 = start_bit / 64;
        let stop_u64 = stop_bit / 64;

        // ensure we have enough u64s to store the number
        while self.data.len() <= stop_u64 {
            self.data.push(0);
        }

        let local_start_bit = start_bit % 64;
        if start_u64 == stop_u64 {
            // pack the whole value into the same u64
            self.data[start_u64] |= x << local_start_bit;
        } else {
            // we have to pack part of the number into one u64, and the
            // rest into the next
            let bits_in_first_cell = 64 - local_start_bit;
            self.data[start_u64] |= x << local_start_bit;
            self.data[stop_u64] |= x >> bits_in_first_cell;
        }

        self.len += 1;
    }

    /// Sets an existing element to a particular value. Panics if `idx` is
    /// out of range or if `x` does not fit within the bitwidth.
    ///
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let mut bv = BitpackVec::new(5);
    ///
    /// bv.push(5);
    ///
    /// assert_eq!(bv.at(0), 5);
    /// bv.set(0, 9);
    /// assert_eq!(bv.at(0), 9);
    /// ```
    pub fn set(&mut self, idx: usize, x: u64) {
        assert!(
            idx < self.len,
            "index {} out of bounds for len {}",
            idx,
            self.len
        );
        assert!(
            self.fits(x),
            "value {} too large to bitpack to width {}",
            x,
            self.width
        );

        let start_bit = idx * self.width();
        let stop_bit = start_bit + self.width() - 1;

        let start_u64 = start_bit / 64;
        let stop_u64 = stop_bit / 64;

        if start_u64 == stop_u64 {
            // all in the same u64
            let local_start_bit = start_bit % 64;
            let local_stop_bit = local_start_bit + self.width();
            let v = self.data[start_u64];

            // zero out all the data at index `idx`
            let mut mask = ((!0_u64) >> local_start_bit) << local_start_bit;
            mask = (mask << (64 - local_stop_bit)) >> (64 - local_stop_bit);
            mask = !mask;

            let v = v & mask;

            // now or the value into it
            let x = x << local_start_bit;
            let v = v | x;

            self.data[start_u64] = v;
        } else {
            // bits are split between two cells
            let local_start_bit = start_bit % 64;

            // clear the bits currently in the cell
            let bit_count_in_first = 64 - local_start_bit;
            let v = self.data[start_u64];
            let v = (v << bit_count_in_first) >> bit_count_in_first;
            let prefix_bits = x << (64 - bit_count_in_first);
            let v = v | prefix_bits;
            self.data[start_u64] = v;

            // clear the bits at the front of the next cell
            let remaining_bit_count = self.width() - bit_count_in_first;
            let v = self.data[stop_u64];
            let v = (v >> remaining_bit_count) << remaining_bit_count;
            let x = x >> bit_count_in_first;
            let v = v | x;

            self.data[stop_u64] = v;
        }
    }

    /// Returns the value at the specified index. Panics if `idx` is out of
    /// range.
    ///
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let mut bv = BitpackVec::new(5);
    ///
    /// bv.push(5);
    ///
    /// assert_eq!(bv.at(0), 5);
    /// ```
    pub fn at(&self, idx: usize) -> u64 {
        assert!(
            idx < self.len,
            "index {} out of bounds for len {}",
            idx,
            self.len
        );
        let start_bit = idx * self.width();
        let stop_bit = start_bit + self.width() - 1;

        let start_u64 = start_bit / 64;
        let stop_u64 = stop_bit / 64;

        if start_u64 == stop_u64 {
            // all in the same u64
            let local_start_bit = start_bit % 64;
            let local_stop_bit = local_start_bit + self.width();
            let v = self.data[start_u64];

            let v = v << (64 - local_stop_bit);
            v >> (64 - self.width)
        } else {
            // bits are split between two cells
            let mut x = 0;
            let local_start_bit = start_bit % 64;
            x |= self.data[start_u64] >> local_start_bit;
            x |= self.data[stop_u64] << (64 - local_start_bit);
            x = (x << (64 - self.width)) >> (64 - self.width);
            x
        }
    }

    /// Determines if the vector's length is 0 (empty)
    ///
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let mut bv = BitpackVec::new(5);
    ///
    /// assert!(bv.is_empty());
    /// bv.push(5);
    /// assert!(!bv.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Removes and returns the last element in the vector. Returns `None`
    /// is the vector is empty.
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let mut bv = BitpackVec::new(5);
    ///
    /// bv.push(5);
    /// bv.push(6);
    ///
    /// assert_eq!(bv.pop(), Some(6));
    /// assert_eq!(bv.pop(), Some(5));
    /// assert_eq!(bv.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<u64> {
        if self.is_empty() {
            return None;
        }

        let idx = self.len() - 1;
        let last = self.at(idx);

        // zero out the value so our xor works later
        let start_bit = idx * self.width();
        let stop_bit = start_bit + self.width() - 1;

        let start_u64 = start_bit / 64;
        let stop_u64 = stop_bit / 64;

        let local_start_bit = start_bit % 64;

        if local_start_bit == 0 {
            // this is the only value in the cell
            self.data.pop();
        } else {
            // clear the data in the cell containing the start of the number
            self.data[start_u64] =
                (self.data[start_u64] << (64 - local_start_bit)) >> (64 - local_start_bit);

            if start_u64 != stop_u64 {
                // the last cell is now clear
                self.data.pop();
            }
        }

        self.len -= 1;

        Some(last)
    }

    /// Truncates the vector to the given length.
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let mut bv = BitpackVec::new(5);
    ///
    /// bv.push(5);
    /// bv.push(6);
    /// bv.push(7);
    ///
    /// assert_eq!(bv.len(), 3);
    /// bv.truncate(1);
    /// assert_eq!(bv.len(), 1);
    /// assert_eq!(bv.at(0), 5);
    /// ```
    pub fn truncate(&mut self, len: usize) {
        // TODO this should compute which entire cells can be dropped, and
        // do that first
        while self.len() > len {
            self.pop();
        }
    }

    /// Split the vector into two parts, so that `self` will contain all
    /// elements from 0 to `idx` (exclusive), and the returned value will
    /// contain all elements from `idx` to the end of the vector.
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let mut bv = BitpackVec::new(5);
    ///
    /// bv.push(5);
    /// bv.push(6);
    /// bv.push(7);
    /// bv.push(8);
    ///
    /// let bv_rest = bv.split_off(2);
    ///
    /// assert_eq!(bv.to_vec(), &[5, 6]);
    /// assert_eq!(bv_rest.to_vec(), &[7, 8]);
    /// ```
    pub fn split_off(&mut self, idx: usize) -> BitpackVec {
        assert!(
            idx <= self.len(),
            "split index ({}) should be <= len ({})",
            idx,
            self.len()
        );
        let mut rest = BitpackVec::new(self.width());

        for i in idx..self.len() {
            rest.push(self.at(i));
        }

        self.truncate(idx);
        rest
    }

    /// Copies the bitpacked vector into a standard, non-packed vector of
    /// `u64`s.
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let mut bv = BitpackVec::new(5);
    ///
    /// bv.push(5);
    /// bv.push(6);
    /// bv.push(7);
    /// bv.push(8);
    ///
    /// let v = bv.to_vec();
    ///
    /// assert_eq!(v, vec![5, 6, 7, 8]);
    /// ```    
    pub fn to_vec(&self) -> Vec<u64> {
        let mut r = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            r.push(self.at(i))
        }

        r
    }

    /// Allows iteration over the values in the bit vector.
    /// ```rust
    /// use bitpack_vec::BitpackVec;
    /// let mut bv = BitpackVec::new(5);
    ///
    /// bv.push(5);
    /// bv.push(6);
    /// bv.push(7);
    /// bv.push(8);
    ///
    /// let v: Vec<u64> = bv.iter().filter(|x| x % 2 == 0).collect();
    /// assert_eq!(v, vec![6, 8]);
    /// ```    
    pub fn iter(&self) -> BitpackIter {
        BitpackIter { bv: self, curr: 0 }
    }
}

/// Iterator over a `BitpackVec`
pub struct BitpackIter<'a> {
    bv: &'a BitpackVec,
    curr: usize,
}

impl<'a> Iterator for BitpackIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr >= self.bv.len() {
            return None;
        }

        let v = self.bv.at(self.curr);
        self.curr += 1;

        Some(v)
    }
}

impl PartialEq for BitpackVec {
    fn eq(&self, other: &Self) -> bool {
        if self.width != other.width {
            return false;
        }

        if self.len != other.len {
            return false;
        }

        // TODO could use a faster, bitwise check here, but we can't just
        // compare the underlying vecs because an element may have been
        // removed from one, but the data left and unzero'd
        for i in 0..self.len() {
            if self.at(i) != other.at(i) {
                return false;
            }
        }

        true
    }
}

impl Debug for BitpackVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("BitpackVec (width: {}) [ ", self.width()))?;

        for i in self.iter() {
            f.write_str(&format!("{} ", i))?;
        }

        f.write_str("]\n")
    }
}

impl Extend<u64> for BitpackVec {
    fn extend<T: IntoIterator<Item = u64>>(&mut self, iter: T) {
        for i in iter {
            self.push(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_simple_6bit() {
        let mut v = BitpackVec::new(6);

        for i in 0..64 {
            v.push(i)
        }

        for i in 0..64 {
            assert_eq!(v.at(i as usize), i);
        }

        assert_eq!(v.len(), 64);
    }

    #[test]
    pub fn test_simple_5bit() {
        let mut v = BitpackVec::new(5);

        for i in 0..32 {
            v.push(i)
        }

        for i in 0..32 {
            assert_eq!(v.at(i as usize), i);
        }

        assert_eq!(v.len(), 32);
    }

    #[test]
    pub fn test_perfect_pack_8bit() {
        let mut v = BitpackVec::new(8);

        v.push(10);
        v.push(11);
        v.push(12);
        v.push(13);
        v.push(14);
        v.push(15);
        v.push(16);
        v.push(17);

        assert_eq!(v.at(0), 10);
        assert_eq!(v.at(1), 11);
        assert_eq!(v.at(2), 12);
        assert_eq!(v.at(3), 13);
        assert_eq!(v.at(4), 14);
        assert_eq!(v.at(5), 15);
        assert_eq!(v.at(6), 16);
        assert_eq!(v.at(7), 17);

        assert_eq!(v.capacity(), 8);
    }

    #[test]
    pub fn test_immut_n_bit() {
        for bits in 1..13 {
            let mut v = BitpackVec::new(bits);

            for x in 0..2_u64.pow(bits as u32) {
                v.push(x);
            }

            for i in 0..2_u64.pow(bits as u32) {
                assert_eq!(v.at(i as usize), i);
            }

            assert_eq!(v.len(), 2_u64.pow(bits as u32) as usize);
        }
    }

    #[test]
    pub fn test_set() {
        let mut v = BitpackVec::new(5);

        for i in 0..16 {
            v.push(i);
        }

        assert_eq!(v.at(5), 5);
        v.set(5, 20);
        assert_eq!(v.at(5), 20);
        v.set(5, 5);
        assert_eq!(v.at(5), 5);

        v.set(12, 20);
        assert_eq!(v.at(12), 20);

        for i in 0..16 {
            v.set(i as usize, i + 5);
        }

        for i in 0..16 {
            assert_eq!(v.at(i as usize), i + 5);
        }
    }

    #[test]
    pub fn test_pop() {
        let mut v = BitpackVec::new(11);
        let mut rv = Vec::new();

        for i in 0..2048 {
            v.push(i);
            rv.push(i);
        }

        while let Some(i) = rv.pop() {
            assert_eq!(i, v.pop().unwrap());
        }

        assert!(v.pop().is_none());
    }

    #[test]
    pub fn test_push_pop() {
        let mut v = BitpackVec::new(11);
        let mut rv = Vec::new();

        for i in 0..2048 {
            v.push(i);
            rv.push(i);
        }

        for _ in 0..100 {
            assert_eq!(v.pop(), rv.pop());
        }

        for i in 0..10_000 {
            v.push(i % 2048);
            rv.push(i % 2048);
        }

        while let Some(i) = rv.pop() {
            assert_eq!(i, v.pop().unwrap());
        }

        assert!(v.pop().is_none());
    }

    #[test]
    pub fn test_split_off() {
        let mut v = Vec::new();
        let mut bv = BitpackVec::new(7);

        for i in 0..10_000 {
            v.push(i % 128);
            bv.push(i % 128);
        }

        assert_eq!(v, bv.to_vec());

        let rv = v.split_off(500);
        let rbv = bv.split_off(500);

        assert_eq!(v, bv.to_vec());
        assert_eq!(rv, rbv.to_vec());
    }

    #[test]
    pub fn test_raw() {
        let mut v = BitpackVec::new(7);

        for i in 0..10_000 {
            v.push(i % 128);
        }

        let data = v.into_raw_vec();

        let nv = BitpackVec::from_raw_vec(data, 7, 10_000);

        for i in 0..10_000 {
            assert_eq!(nv.at(i), i as u64 % 128);
        }
    }

    #[test]
    pub fn test_iter() {
        let mut v = BitpackVec::new(7);

        for i in 0..10_000 {
            v.push(i % 128);
        }

        let from_iter: Vec<u64> = v.iter().collect();
        assert_eq!(v.to_vec(), from_iter);
    }

    #[test]
    pub fn test_eq() {
        let mut v = BitpackVec::new(7);
        let mut v2 = BitpackVec::new(7);

        for i in 0..10_000 {
            v.push(i % 128);
            v2.push(i % 128);
        }

        assert_eq!(v, v2);

        v.push(7);
        assert_ne!(v, v2);

        v.pop();
        assert_eq!(v, v2);

        v.push(8);
        v2.push(8);
        assert_eq!(v, v2);
    }

    #[test]
    pub fn test_from_slice() {
        let v = vec![4, 19, 184, 18314, 62];
        let bv = BitpackVec::from_slice(&v);

        assert_eq!(bv.width(), 15);
        assert_eq!(bv.to_vec(), v);
    }

    #[test]
    #[should_panic]
    fn test_does_not_fit_push() {
        let mut bv = BitpackVec::new(5);
        bv.push(32);
    }

    #[test]
    #[should_panic]
    fn test_does_not_fit_set() {
        let mut bv = BitpackVec::new(5);
        bv.push(18);
        bv.set(0, 32);
    }
}
