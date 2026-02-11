#!/usr/bin/env python3
"""
Convert Bullet quantised.bin → KiyEngine KYNN format.

Supports two modes:
  --buckets 1  (v2 legacy): single output bucket
  --buckets 8  (v3 default): 8 output buckets by MaterialCount

Bullet's quantised.bin layout WITH output buckets (transposed l1w):
  l0w: i16[768 * HIDDEN]              (feature weights, feature-first)
  l0b: i16[HIDDEN]                     (feature biases)
  l1w: i16[NUM_BUCKETS * 2 * HIDDEN]  (output weights, transposed = row-major per bucket)
  l1b: i16[NUM_BUCKETS]               (output biases, in QA*QB units)

KiyEngine KYNN v3 format:
  Magic: "KYNN" (4 bytes)
  Version: u32 LE (3)
  hidden_size: u32 LE
  num_buckets: u32 LE
  ft_weights: i16[768 * hidden_size] LE
  ft_biases: i16[hidden_size] LE
  psqt_weights: i16[768 * num_buckets] LE  (zeros if no PSQT trained)
  output_weights: i16[num_buckets * 2 * hidden_size] LE
  output_biases: i16[num_buckets] LE
"""

import struct
import sys
import os

NNUE_MAGIC = b"KYNN"
NUM_FEATURES = 768


def convert(input_path: str, output_path: str, hidden_size: int = 512, num_buckets: int = 8):
    data = open(input_path, "rb").read()

    # Expected sizes
    ft_w_count = NUM_FEATURES * hidden_size
    ft_b_count = hidden_size
    out_w_count = num_buckets * 2 * hidden_size
    out_b_count = num_buckets
    expected_bytes = (ft_w_count + ft_b_count + out_w_count + out_b_count) * 2

    if len(data) < expected_bytes:
        print(f"ERROR: File too small. Expected >= {expected_bytes} bytes for hidden={hidden_size}, buckets={num_buckets}, got {len(data)}")
        print(f"  Try different --hidden-size or --buckets values.")
        sys.exit(1)

    if len(data) > expected_bytes:
        extra = len(data) - expected_bytes
        print(f"  Note: Stripping {extra} bytes of Bullet padding from end of file")
        data = data[:expected_bytes]

    # Parse all i16 values
    total_values = len(data) // 2
    values = struct.unpack(f"<{total_values}h", data)

    offset = 0
    ft_weights = values[offset:offset + ft_w_count]
    offset += ft_w_count
    ft_biases = values[offset:offset + ft_b_count]
    offset += ft_b_count
    out_weights = values[offset:offset + out_w_count]
    offset += out_w_count
    out_biases = values[offset:offset + out_b_count]

    psqt_count = NUM_FEATURES * num_buckets

    if num_buckets == 1:
        # Write v2 format (legacy single bucket)
        version = 2
        with open(output_path, "wb") as f:
            f.write(NNUE_MAGIC)
            f.write(struct.pack("<I", version))
            f.write(struct.pack("<I", hidden_size))
            f.write(struct.pack(f"<{ft_w_count}h", *ft_weights))
            f.write(struct.pack(f"<{ft_b_count}h", *ft_biases))
            f.write(struct.pack(f"<{out_w_count}h", *out_weights))
            f.write(struct.pack("<h", out_biases[0]))
    else:
        # Write v3 format (output buckets + PSQT placeholder)
        version = 3
        with open(output_path, "wb") as f:
            f.write(NNUE_MAGIC)
            f.write(struct.pack("<I", version))
            f.write(struct.pack("<I", hidden_size))
            f.write(struct.pack("<I", num_buckets))
            f.write(struct.pack(f"<{ft_w_count}h", *ft_weights))
            f.write(struct.pack(f"<{ft_b_count}h", *ft_biases))
            # PSQT weights (zeros — not trained yet)
            f.write(struct.pack(f"<{psqt_count}h", *([0] * psqt_count)))
            f.write(struct.pack(f"<{out_w_count}h", *out_weights))
            f.write(struct.pack(f"<{out_b_count}h", *out_biases))

    out_size = os.path.getsize(output_path)
    print(f"Converted: {input_path} -> {output_path}")
    print(f"  Version:     v{version}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Buckets:     {num_buckets}")
    print(f"  FT weights:  {ft_w_count} i16 values")
    print(f"  FT biases:   {ft_b_count} i16 values")
    print(f"  Out weights: {out_w_count} i16 values")
    print(f"  Out biases:  {list(out_biases)}")
    print(f"  PSQT:        {psqt_count} i16 values (zeros)" if num_buckets > 1 else "  PSQT:        N/A (v2)")
    print(f"  Output size: {out_size} bytes ({out_size / 1024:.1f} KB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Bullet quantised.bin to KiyEngine KYNN format")
    parser.add_argument("input", help="Path to Bullet quantised.bin")
    parser.add_argument("output", help="Output path for .nnue file")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden layer size (default: 512)")
    parser.add_argument("--buckets", type=int, default=8, help="Number of output buckets (default: 8, use 1 for legacy v2)")
    args = parser.parse_args()

    convert(args.input, args.output, args.hidden_size, args.buckets)
