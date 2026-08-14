# Built-in normative assets

`avs3a_hyper_model.bin` is the 79,930-byte `g_model` array from the adjacent
UWA AVS3-P3 C reference implementation, with its byte-wise XOR `0x55`
obfuscation removed. It remains covered by the repository's UWA Code Sharing
Policy license.

The plaintext form is intentional: it is embedded read-only with
`include_bytes!`, removes an unnecessary startup copy/XOR pass, and still goes
through the checked little-endian Rust model parser.

To reproduce it from a reference checkout:

```bash
rustc tools/extract_model.rs -O -o /tmp/avs3a-extract-model
/tmp/avs3a-extract-model ../avs3a/include/model.h assets/avs3a_hyper_model.bin
```

Expected SHA-256 is recorded in the root README; the Rust test suite checks
the byte length, FNV fingerprint, complete parsed topology and CNN outputs.

`avs3a_fd_tables.bin` contains 10,992 normative `f32` values used by HBR/LBR
LSF dequantization and inverse FD shaping. Values are concatenated in the
named order checked by `tools/extract_fd_tables.rs` and serialized explicitly
as little-endian bytes. This avoids thousands of source literals while
remaining read-only, portable, and allocation-free at decode time.

To reproduce the FD table asset from the reference checkout:

```bash
rustc tools/extract_fd_tables.rs -O -o /tmp/avs3a-extract-fd-tables
/tmp/avs3a-extract-fd-tables ../avs3a/src/avs3_rom_com.c assets/avs3a_fd_tables.bin
```

Expected size is 43,968 bytes. SHA-256 is
`6b8e25a332edf722c81c494c85ab57d90f145d1524fd808e01333e1c9a6d39d5`;
the Rust suite also checks FNV-1a `9ce264f019b75cc4`.

`avs3a_mcr_rotations.bin` contains the normative MCR VQ codebooks converted to
little-endian `(cos(theta), sin(theta))` `f32` pairs. The long 9-bit codebook
comes first, followed by the short-window 8-bit codebook; this lets the decode
hot path avoid per-frame trigonometry and allocation.

To reproduce it from a reference checkout:

```bash
rustc tools/extract_mcr_rotations.rs -O -o /tmp/avs3a-extract-mcr-rotations
/tmp/avs3a-extract-mcr-rotations ../avs3a/src/avs3_rom_com.c assets/avs3a_mcr_rotations.bin
```

Expected size is 18,432 bytes. SHA-256 is
`9fe0ece1f78509f66847b9b31c60efed6a2185b5ee533e0b17e66cf3b5df61bc`; the Rust
suite also checks FNV-1a `5b62aa9a6b23145a` and representative long/short
rotation values.

`avs3a_hoa_spatial_tables.bin` contains the 1343 pairs from
`avs3_hoa_fixed_angle_basis_matrix`, followed by all 257 values from
`avs3_hoa_sin_table`. Angle entries are signed 16-bit integers and sine entries
are `f32`; both use explicit little-endian serialization. The decoder derives
the 16 third-order spherical-harmonic coefficients from these normative tables
without target-libm trigonometry.

The C source explicitly initializes only 1340 of its declared 1343 angle rows;
the extractor writes the three implicit all-zero rows explicitly.

To reproduce it from a reference checkout:

```bash
rustc tools/extract_hoa_spatial_tables.rs -O -o /tmp/avs3a-extract-hoa-tables
/tmp/avs3a-extract-hoa-tables ../avs3a/src/avs3_rom_com.c assets/avs3a_hoa_spatial_tables.bin
```

Expected size is 6,400 bytes. SHA-256 is
`641e93f65c86376815560119d6704064d33528ecddd331bab133f189164aec50`;
the Rust suite also checks FNV-1a `91a0296fd4def1af`.
