use std::fmt::Debug;

use deepsize::DeepSizeOf;
use serde::{Deserialize, Serialize};

#[derive(DeepSizeOf, Serialize, Deserialize)]
pub struct BitpackVec {
    data: Vec<u64>,
    width: usize,
    len: usize,
}

impl BitpackVec {
    pub fn new(width: usize) -> BitpackVec {
        assert!(width > 0, "bitpack width must be greater than 0");
        assert!(
            width <= 64,
            "bitpack width must be less than or equal to 64"
        );

        BitpackVec {
            data: Vec::new(),
            len: 0,
            width,
        }
    }

    pub fn from_raw_vec(data: Vec<u64>, width: usize, len: usize) -> BitpackVec {
        assert!(
            data.len() * 64 > width * len,
            "data is not long enough to be valid"
        );
        BitpackVec { data, width, len }
    }

    pub fn into_raw_vec(self) -> Vec<u64> {
        self.data
    }

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

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn capacity(&self) -> usize {
        (self.data.len() * 64) / self.width()
    }

    pub fn fits(&self, x: u64) -> bool {
        x < 2_u64.pow(self.width as u32)
    }

    pub fn push(&mut self, x: u64) {
        assert!(
            self.fits(x),
            "value {} too large to bitpack to width {}",
            x,
            self.width
        );

        // calculate position info
        let start_bit = self.len * self.width;
        let stop_bit = start_bit + self.width - 1;

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

    pub fn set(&mut self, idx: usize, x: u64) {
        assert!(
            idx < self.len,
            "index {} out of bounds for len {}",
            idx,
            self.len
        );
        let start_bit = idx * self.width;
        let stop_bit = start_bit + self.width - 1;

        let start_u64 = start_bit / 64;
        let stop_u64 = stop_bit / 64;

        if start_u64 == stop_u64 {
            // all in the same u64
            let local_start_bit = start_bit % 64;
            let local_stop_bit = local_start_bit + self.width;
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
            let remaining_bit_count = self.width - bit_count_in_first;
            let v = self.data[stop_u64];
            let v = (v >> remaining_bit_count) << remaining_bit_count;
            let x = x >> bit_count_in_first;
            let v = v | x;

            self.data[stop_u64] = v;
        }
    }

    pub fn at(&self, idx: usize) -> u64 {
        assert!(
            idx < self.len,
            "index {} out of bounds for len {}",
            idx,
            self.len
        );
        let start_bit = idx * self.width;
        let stop_bit = start_bit + self.width - 1;

        let start_u64 = start_bit / 64;
        let stop_u64 = stop_bit / 64;

        if start_u64 == stop_u64 {
            // all in the same u64
            let local_start_bit = start_bit % 64;
            let local_stop_bit = local_start_bit + self.width;
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

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn pop(&mut self) -> Option<u64> {
        if self.is_empty() {
            return None;
        }

        let idx = self.len() - 1;
        let last = self.at(idx);

        // zero out the value so our xor works later
        let start_bit = idx * self.width;
        let stop_bit = start_bit + self.width - 1;

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

    pub fn truncate(&mut self, len: usize) {
        // TODO this should compute which entire cells can be dropped, and
        // do that first
        while self.len() > len {
            self.pop();
        }
    }

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

    pub fn to_vec(&self) -> Vec<u64> {
        let mut r = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            r.push(self.at(i))
        }

        r
    }

    pub fn iter(&self) -> BitpackIter {
        BitpackIter { bv: self, curr: 0 }
    }
}

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
}
