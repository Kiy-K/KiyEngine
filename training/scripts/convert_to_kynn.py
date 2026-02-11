#!/usr/bin/env python3
"""
Convert Bullet quantised.bin → KiyEngine KYNN format.

Supports three modes:
  --input-buckets 10  (v4 default): Stockfish-style king-bucketed HalfKAv2_hm
  --input-buckets 1   (v3 fallback): legacy Chess768, no king buckets

Bullet's quantised.bin layout for king-bucketed network (with merged factoriser):
  l0w: i16[NUM_INPUT_BUCKETS * 768 * HIDDEN]  (feature weights, column-major from Bullet)
  l0b: i16[HIDDEN]                              (feature biases)
  l1w: i16[NUM_BUCKETS * 2 * HIDDEN]           (output weights, transposed = row-major)
  l1b: i16[NUM_BUCKETS]                         (output biases, in QA*QB units)

KiyEngine KYNN v4 format:
  Magic: "KYNN" (4 bytes)
  Version: u32 LE (4)
  hidden_size: u32 LE
  num_output_buckets: u32 LE
  num_input_buckets: u32 LE
  ft_weights: i16[num_input_buckets * 768 * hidden_size] LE (row-major)
  ft_biases: i16[hidden_size] LE
  output_weights: i16[num_output_buckets * 2 * hidden_size] LE
  output_biases: i16[num_output_buckets] LE

Note on weight layout:
  Bullet stores weights in column-major order by default.
  The engine expects row-major (ft_weights[feature * hidden + neuron]).
  Bullet's column-major for (rows=inputs, cols=hidden) means:
    data[col * rows + row] = weight[row][col]
  We need to transpose: weight[row][col] = data[col * rows + row]
  But since we read as flat i16 and the feature transformer in Bullet
  is already in the layout we expect (feature-major), no transpose needed
  for l0w. l1w is explicitly transposed in the save_format.
"""

import struct
import sys
import os
import numpy as np

NNUE_MAGIC = b"KYNN"
NUM_FEATURES = 768


def convert(input_path: str, output_path: str, hidden_size: int = 512,
            num_output_buckets: int = 8, num_input_buckets: int = 10):
    data = open(input_path, "rb").read()

    # Expected sizes
    ft_w_count = num_input_buckets * NUM_FEATURES * hidden_size
    ft_b_count = hidden_size
    out_w_count = num_output_buckets * 2 * hidden_size
    out_b_count = num_output_buckets
    expected_bytes = (ft_w_count + ft_b_count + out_w_count + out_b_count) * 2

    if len(data) < expected_bytes:
        print(f"ERROR: File too small. Expected >= {expected_bytes} bytes, got {len(data)}")
        print(f"  Config: hidden={hidden_size}, input_buckets={num_input_buckets}, output_buckets={num_output_buckets}")
        print(f"  Try different --hidden-size, --input-buckets, or --output-buckets values.")
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

    if num_input_buckets > 1:
        # ═══ v4: King-bucketed HalfKAv2_hm ═══
        # Bullet stores l0w in column-major: shape (num_inputs, hidden) stored as
        # [col0_all_rows, col1_all_rows, ...]. We need row-major for the engine.
        num_inputs = num_input_buckets * NUM_FEATURES
        ft_arr = np.array(ft_weights, dtype=np.int16)

        # Bullet column-major: reshape as (hidden, num_inputs) then transpose → (num_inputs, hidden)
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
    else:
        # ═══ v3: Legacy Chess768 (no king buckets) ═══
        # Same column-major → row-major transpose for ft_weights
        ft_arr = np.array(ft_weights, dtype=np.int16)
        ft_matrix = ft_arr.reshape(hidden_size, NUM_FEATURES).T
        ft_weights_rm = ft_matrix.flatten().tolist()

        psqt_count = NUM_FEATURES * num_output_buckets
        version = 3
        with open(output_path, "wb") as f:
            f.write(NNUE_MAGIC)
            f.write(struct.pack("<I", version))
            f.write(struct.pack("<I", hidden_size))
            f.write(struct.pack("<I", num_output_buckets))
            f.write(struct.pack(f"<{ft_w_count}h", *ft_weights_rm))
            f.write(struct.pack(f"<{ft_b_count}h", *ft_biases))
            f.write(struct.pack(f"<{psqt_count}h", *([0] * psqt_count)))
            f.write(struct.pack(f"<{out_w_count}h", *out_weights))
            f.write(struct.pack(f"<{out_b_count}h", *out_biases))

    out_size = os.path.getsize(output_path)
    print(f"Converted: {input_path} -> {output_path}")
    print(f"  Version:        v{version}")
    print(f"  Hidden size:    {hidden_size}")
    print(f"  Input buckets:  {num_input_buckets} ({'HalfKAv2_hm' if num_input_buckets > 1 else 'Chess768'})")
    print(f"  Output buckets: {num_output_buckets}")
    print(f"  FT weights:     {ft_w_count} i16 values ({ft_w_count * 2 / 1024:.1f} KB)")
    print(f"  FT biases:      {ft_b_count} i16 values")
    print(f"  Out weights:    {out_w_count} i16 values")
    print(f"  Out biases:     {list(out_biases)}")
    print(f"  Output size:    {out_size} bytes ({out_size / 1024:.1f} KB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Bullet quantised.bin to KiyEngine KYNN format")
    parser.add_argument("input", help="Path to Bullet quantised.bin")
    parser.add_argument("output", help="Output path for .nnue file", nargs="?", default="kiyengine.nnue")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden layer size (default: 512)")
    parser.add_argument("--input-buckets", type=int, default=10, help="Number of king input buckets (default: 10)")
    parser.add_argument("--output-buckets", type=int, default=8, help="Number of output buckets (default: 8)")
    args = parser.parse_args()

    convert(args.input, args.output, args.hidden_size, args.output_buckets, args.input_buckets)
