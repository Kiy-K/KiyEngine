#!/usr/bin/env python3
"""
Convert Bullet quantised.bin → KiyEngine KYNN format.

Supports two output formats:
  --format v5 (default): Deep network (FT → SCReLU → L1 → CReLU → L2)
  --format v4:           Shallow network (FT → SCReLU → output)

Bullet's quantised.bin layout for v5 deep network:
  l0w: i16[NUM_INPUT_BUCKETS * 768 * HIDDEN]  (FT weights, column-major from Bullet)
  l0b: i16[HIDDEN]                              (FT biases)
  l1w: i8[L1_SIZE * 2 * HIDDEN]               (L1 weights, transposed = row-major)
  l1b: i32[L1_SIZE]                             (L1 biases, in QA*Q1 scale)
  l2w: i16[NUM_BUCKETS * L1_SIZE]              (L2 weights, transposed = row-major)
  l2b: i16[NUM_BUCKETS]                         (L2 biases, in QA*Q2 scale)

KiyEngine KYNN v5 format:
  Magic: "KYNN" (4 bytes)
  Version: u32 LE (5)
  hidden_size: u32 LE
  num_output_buckets: u32 LE
  num_input_buckets: u32 LE
  l1_size: u32 LE
  ft_weights: i16[...] LE (row-major)
  ft_biases: i16[hidden_size] LE
  l1_weights: i8[l1_size * 2 * hidden_size]
  l1_biases: i32[l1_size] LE
  l2_weights: i16[num_output_buckets * l1_size] LE
  l2_biases: i16[num_output_buckets] LE

Note on weight layout:
  Bullet stores weights column-major; the engine expects row-major.
  l0w needs transpose; l1w/l2w are explicitly transposed in save_format.
"""

import struct
import sys
import os
import numpy as np

NNUE_MAGIC = b"KYNN"
NUM_FEATURES = 768


def convert_v5(input_path: str, output_path: str, hidden_size: int = 512,
               l1_size: int = 16, num_output_buckets: int = 8, num_input_buckets: int = 10):
    """Convert Bullet quantised.bin to KYNN v5 (deep network) format."""
    data = open(input_path, "rb").read()

    # Validate file size before parsing
    expected = (num_input_buckets * NUM_FEATURES * hidden_size * 2  # l0w i16
                + hidden_size * 2                                    # l0b i16
                + l1_size * 2 * hidden_size                          # l1w i8
                + l1_size * 4                                        # l1b i32
                + num_output_buckets * l1_size * 2                   # l2w i16
                + num_output_buckets * 2)                             # l2b i16
    if len(data) < expected:
        print(f"ERROR: quantised.bin too small for v5 format.")
        print(f"  Got {len(data)} bytes, need >= {expected} bytes.")
        print(f"  Config: hidden={hidden_size}, l1={l1_size}, in_buckets={num_input_buckets}, out_buckets={num_output_buckets}")
        print(f"  Did you train with the deep architecture (L1→CReLU→L2)?")
        sys.exit(1)

    offset = 0

    # FT weights: i16
    ft_w_count = num_input_buckets * NUM_FEATURES * hidden_size
    ft_w_bytes = ft_w_count * 2
    ft_weights = np.frombuffer(data[offset:offset + ft_w_bytes], dtype=np.int16).copy()
    offset += ft_w_bytes

    # FT biases: i16
    ft_b_count = hidden_size
    ft_b_bytes = ft_b_count * 2
    ft_biases = np.frombuffer(data[offset:offset + ft_b_bytes], dtype=np.int16).copy()
    offset += ft_b_bytes

    # L1 weights: i8 (already transposed in save_format)
    l1_w_count = l1_size * 2 * hidden_size
    l1_w_bytes = l1_w_count  # i8 = 1 byte each
    l1_weights = np.frombuffer(data[offset:offset + l1_w_bytes], dtype=np.int8).copy()
    offset += l1_w_bytes

    # L1 biases: i32
    l1_b_count = l1_size
    l1_b_bytes = l1_b_count * 4
    l1_biases = np.frombuffer(data[offset:offset + l1_b_bytes], dtype=np.int32).copy()
    offset += l1_b_bytes

    # L2 weights: i16 (already transposed in save_format)
    l2_w_count = num_output_buckets * l1_size
    l2_w_bytes = l2_w_count * 2
    l2_weights = np.frombuffer(data[offset:offset + l2_w_bytes], dtype=np.int16).copy()
    offset += l2_w_bytes

    # L2 biases: i16
    l2_b_count = num_output_buckets
    l2_b_bytes = l2_b_count * 2
    l2_biases = np.frombuffer(data[offset:offset + l2_b_bytes], dtype=np.int16).copy()
    offset += l2_b_bytes

    # Transpose FT weights: Bullet column-major → engine row-major
    num_inputs = num_input_buckets * NUM_FEATURES
    ft_matrix = ft_weights.reshape(hidden_size, num_inputs).T
    ft_weights_rm = ft_matrix.flatten()

    # Write KYNN v5
    version = 5
    with open(output_path, "wb") as f:
        f.write(NNUE_MAGIC)
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<I", hidden_size))
        f.write(struct.pack("<I", num_output_buckets))
        f.write(struct.pack("<I", num_input_buckets))
        f.write(struct.pack("<I", l1_size))
        f.write(ft_weights_rm.tobytes())
        f.write(ft_biases.tobytes())
        f.write(l1_weights.tobytes())
        f.write(l1_biases.tobytes())
        f.write(l2_weights.tobytes())
        f.write(l2_biases.tobytes())

    out_size = os.path.getsize(output_path)
    print(f"Converted: {input_path} -> {output_path}")
    print(f"  Version:        v{version} (deep: FT→SCReLU→L1→CReLU→L2)")
    print(f"  Hidden size:    {hidden_size}")
    print(f"  L1 size:        {l1_size}")
    print(f"  Input buckets:  {num_input_buckets}")
    print(f"  Output buckets: {num_output_buckets}")
    print(f"  FT weights:     {ft_w_count} i16 ({ft_w_count * 2 / 1024:.1f} KB)")
    print(f"  L1 weights:     {l1_w_count} i8  ({l1_w_count / 1024:.1f} KB)")
    print(f"  L2 weights:     {l2_w_count} i16 ({l2_w_count * 2 / 1024:.1f} KB)")
    print(f"  Output size:    {out_size} bytes ({out_size / 1024:.1f} KB)")


def convert_v4(input_path: str, output_path: str, hidden_size: int = 512,
               num_output_buckets: int = 8, num_input_buckets: int = 10):
    """Convert Bullet quantised.bin to KYNN v4 (shallow network) format."""
    data = open(input_path, "rb").read()

    ft_w_count = num_input_buckets * NUM_FEATURES * hidden_size
    ft_b_count = hidden_size
    out_w_count = num_output_buckets * 2 * hidden_size
    out_b_count = num_output_buckets
    expected_bytes = (ft_w_count + ft_b_count + out_w_count + out_b_count) * 2

    if len(data) < expected_bytes:
        print(f"ERROR: File too small. Expected >= {expected_bytes} bytes, got {len(data)}")
        sys.exit(1)

    values = struct.unpack(f"<{len(data) // 2}h", data[:expected_bytes])

    offset = 0
    ft_weights = values[offset:offset + ft_w_count]; offset += ft_w_count
    ft_biases = values[offset:offset + ft_b_count]; offset += ft_b_count
    out_weights = values[offset:offset + out_w_count]; offset += out_w_count
    out_biases = values[offset:offset + out_b_count]

    # Transpose FT weights
    num_inputs = num_input_buckets * NUM_FEATURES
    ft_arr = np.array(ft_weights, dtype=np.int16)
    ft_matrix = ft_arr.reshape(hidden_size, num_inputs).T
    ft_weights_rm = ft_matrix.flatten().tolist()

    version = 4
    with open(output_path, "wb") as f:
        f.write(NNUE_MAGIC)
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<I", hidden_size))
        f.write(struct.pack("<I", num_output_buckets))
        f.write(struct.pack("<I", num_input_buckets))
        f.write(struct.pack(f"<{ft_w_count}h", *ft_weights_rm))
        f.write(struct.pack(f"<{ft_b_count}h", *ft_biases))
        f.write(struct.pack(f"<{out_w_count}h", *out_weights))
        f.write(struct.pack(f"<{out_b_count}h", *out_biases))

    out_size = os.path.getsize(output_path)
    print(f"Converted: {input_path} -> {output_path}")
    print(f"  Version:        v{version} (shallow: FT→SCReLU→output)")
    print(f"  Hidden size:    {hidden_size}")
    print(f"  Input buckets:  {num_input_buckets}")
    print(f"  Output buckets: {num_output_buckets}")
    print(f"  Output size:    {out_size} bytes ({out_size / 1024:.1f} KB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Bullet quantised.bin to KiyEngine KYNN format")
    parser.add_argument("input", help="Path to Bullet quantised.bin")
    parser.add_argument("output", help="Output path for .nnue file", nargs="?", default="kiyengine.nnue")
    parser.add_argument("--format", choices=["v4", "v5"], default="v5", help="Output format (default: v5 deep)")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden layer size (default: 512)")
    parser.add_argument("--l1-size", type=int, default=16, help="L1 hidden size for v5 (default: 16)")
    parser.add_argument("--input-buckets", type=int, default=10, help="Number of king input buckets (default: 10)")
    parser.add_argument("--output-buckets", type=int, default=8, help="Number of output buckets (default: 8)")
    args = parser.parse_args()

    if args.format == "v5":
        convert_v5(args.input, args.output, args.hidden_size, args.l1_size, args.output_buckets, args.input_buckets)
    else:
        convert_v4(args.input, args.output, args.hidden_size, args.output_buckets, args.input_buckets)
