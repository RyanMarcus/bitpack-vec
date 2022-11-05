# dense-bitpack

A dense bitpacked vector type for unnsigned integers.

* `O(1)` random access to single elements
* `O(1)` pop
* Any bitlength from 1 to 63
* Serde serializable

This packages does an "as you'd expect" bitpacking of integers, with no fancy SIMD or additional compression. Values are stored in a `Vec<u64>`, so no more than 63 bits should be wasted. Values can overlap `u64` values.

Compared to other bitpacking packages for Rust:

* [`bitpacking`](https://docs.rs/bitpacking/latest/bitpacking/) uses SIMD compression to pack values into blocks, but entire blocks must be decompressed in order to access values. If you don't care about random access, `bitpacking` is probably what you want.
* [`vorbis_bitpack`](https://docs.rs/vorbis_bitpack/latest/vorbis_bitpack/) allows for streaming compression and decompression of packed integers, using the Vorbis format. No random access, but has streaming readers / writers.
* [`parquet`](https://docs.rs/parquet) implements a number of Apache-backed formats (feather, arrow, parquet), many of which support bitpacking and other types of compression.

## License

This code is available under the terms of the GPL-3.0 (or later) license.