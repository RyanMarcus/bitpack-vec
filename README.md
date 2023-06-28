# bitpack-vec

[![Rust](https://github.com/RyanMarcus/bitpack-vec/actions/workflows/rust.yml/badge.svg)](https://github.com/RyanMarcus/bitpack-vec/actions/workflows/rust.yml) [![docs.rs](https://img.shields.io/docsrs/bitpack-vec/latest)](https://docs.rs/bitpack-vec/)

A dense bitpacked vector type for unnsigned integers.

```rust
use bitpack_vec::BitpackVec;
let mut bv = BitpackVec::new(5);  // 5-bit integers

for i in 0..12 {
    bv.push(i);
}

assert_eq!(bv.at(6), 6);
assert_eq!(bv.at(9), 9);

use deepsize::DeepSizeOf;
assert_eq!(bv.as_raw().len(), 1);  // underlying vector length is just 1 (60 bits)

// total in-memory size (not strictly specified by Rust):
assert_eq!(
    bv.deep_size_of(), 
    std::mem::size_of::<Vec<u64>>()  // size of the vector structure
    + std::mem::size_of::<usize>() // the length counter (separate from the Vec's)
    + std::mem::size_of::<u8>() // the bitwidth of the structure
    + 15 // padding
);
```

* `O(1)` random access to single elements
* `O(1)` pop
* `O(1)` set
* Amortized `O(1)` push (same as Rust `Vec`)
* Any bitlength from 1 to 63
* Serde serializable

This package does an "as you'd expect" bitpacking of integers, with no fancy SIMD or additional compression. Values are stored in a `Vec<u64>`, so no more than 63 bits should be wasted. Values can overlap `u64` values.

Compared to other bitpacking packages for Rust:

* [`bitpacking`](https://docs.rs/bitpacking/latest/bitpacking/) uses SIMD compression to pack values into blocks, but entire blocks must be decompressed in order to access values. If you don't care about random access, `bitpacking` is probably what you want.
* [`vorbis_bitpack`](https://docs.rs/vorbis_bitpack/latest/vorbis_bitpack/) allows for streaming compression and decompression of packed integers, using the Vorbis format. No random access, but has streaming readers / writers.
* [`parquet`](https://docs.rs/parquet) implements a number of Apache-backed formats (feather, arrow, parquet), many of which support bitpacking and other types of compression.

## License

This code is available under the terms of the GPL-3.0 (or later) license.
