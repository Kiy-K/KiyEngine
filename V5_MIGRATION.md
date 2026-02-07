# KiyEngine V5 Migration Guide

## Overview

KiyEngine V5 introduces **GGUF (GGML Universal File)** format support, replacing the legacy `.safetensors` format. This change brings:

- **Faster loading** via memory mapping (mmap)
- **Better quantization** support for BitNet 1.58-bit models
- **Single-file format** with embedded metadata
- **Standard format** compatible with llama.cpp ecosystem

## Architecture Unchanged

The neural network architecture remains identical to V4:
- **Vocab Size**: 4608
- **Context Length**: 32
- **D-Model**: 1024
- **Num Layers**: 4
- **Quantization**: BitNet 1.58-bit (ternary {-1, 0, +1} weights)

## Migrating from V4 (safetensors) to V5 (GGUF)

### 1. Convert Your Model

Use the provided conversion script:

```bash
# Install dependencies if needed
pip install safetensors numpy

# Convert your model
python scripts/convert_to_gguf.py \
    --input kiyengine.safetensors \
    --output kiyengine.gguf
```

### 2. Update Your Code

The `Engine::new()` method automatically detects the format:

```rust
// Automatically tries kiyengine.gguf first, falls back to kiyengine.safetensors
let engine = Engine::new()?;
```

Or explicitly load a specific format:

```rust
// Load GGUF (V5)
let engine = Engine::load_from_gguf("kiyengine.gguf", Device::Cpu)?;

// Or load legacy safetensors (v4)
let engine = Engine::load_from_safetensors("kiyengine.safetensors", Device::Cpu)?;
```

### 3. Using ChessModel API

```rust
use kiy_engine_v4_omega::engine::{ChessModel, ModelConfig};
use candle_core::Device;

// Load from GGUF
let model = ChessModel::from_gguf("kiyengine.gguf", Device::Cpu)?;

// Or from safetensors (backward compatible)
let model = ChessModel::from_safetensors("kiyengine.safetensors", Device::Cpu)?;
```

## GGUF Format Details

### File Structure

```
GGUF File Layout:
├── Header (24 bytes)
│   ├── Magic: "GGUF" (4 bytes)
│   ├── Version: u32 (4 bytes)
│   ├── Tensor Count: u64 (8 bytes)
│   └── Metadata KV Count: u64 (8 bytes)
├── Metadata KV Section
│   └── Architecture-specific metadata
├── Tensor Info Section
│   └── Tensor names, shapes, offsets, types
└── Tensor Data Section (32-byte aligned)
    └── Raw tensor bytes
```

### Metadata Keys

KiyEngine V5 uses the following GGUF metadata keys:

```
general.architecture = "kiyengine"
general.name = "KiyEngine V5"
general.version = "5.0.0"
general.file_type = 26  (I8 - BitNet 1.58-bit)

kiyengine.vocab_size = 4608
kiyengine.context_length = 32
kiyengine.embedding_length = 1024
kiyengine.block_count = 4
kiyengine.feed_forward_length = 1024
kiyengine.attention.head_count = 16
kiyengine.attention.layer_norm_rms_epsilon = 1e-5

# Per-layer scale factors (BitNet dequantization)
kiyengine.layers.0.linear.scale = 0.0234
kiyengine.layers.1.linear.scale = 0.0256
...
```

### Quantization Types

- **I8** (type 26): BitNet 1.58-bit ternary weights {-1, 0, +1}
- **F16**: 16-bit floating point (activations, embeddings)
- **F32**: 32-bit floating point (fallback)

## Tensor Naming Convention

Tensors in GGUF follow the same naming as the original model:

```
embed.weight          # Token embedding [4608, 1024]
pos_embed             # Positional embedding [1, 32, 1024]

layers.0.linear.weight       # Layer 0 linear weights
layers.0.linear.norm.weight  # Layer 0 RMS norm weights

policy_head.0.weight   # Policy RMS norm
policy_head.1.weight # Policy linear [4608, 1024]

value_head.0.weight  # Value RMS norm
value_head.1.weight  # Value hidden [512, 1024]
value_head.3.weight  # Value output [1, 512]
```

## Inspecting GGUF Files

### Using the inspect binary

```bash
# Build the inspection tool
cargo build --release --bin inspect_gguf

# Inspect a GGUF file
./target/release/inspect_gguf kiyengine.gguf
```

### Using Python

```python
from gguf import GGUFReader

reader = GGUFReader("kiyengine.gguf")
print(f"Architecture: {reader.get_field('general.architecture')}")
print(f"Tensors: {len(reader.tensors)}")
for tensor in reader.tensors:
    print(f"  {tensor.name}: {tensor.shape} ({tensor.dtype})")
```

## File Size Comparison

| Format | Size | Compression |
|--------|------|-------------|
| safetensors (v4) | ~24 MB | None (F32) |
| GGUF with I8 (V5) | ~6 MB | **4x smaller** |
| GGUF with F16 | ~12 MB | 2x smaller |

## Backward Compatibility

V5 maintains full backward compatibility:
- Old `.safetensors` files can still be loaded
- `Engine::new()` auto-detects format
- Both formats use the same inference code path

## Performance Benefits

### Loading Speed
- **GGUF**: Memory-mapped loading (~10ms for 6MB)
- **safetensors**: Buffer copy (~50ms for 24MB)

### Memory Usage
- **GGUF**: Lazy loading via mmap (pages loaded on demand)
- **safetensors**: Entire file loaded into RAM

### Inference Speed
- Same performance (both use identical BitNet kernels)
- GGUF I8 weights may be slightly faster due to smaller memory bandwidth

## Troubleshooting

### "Invalid GGUF magic" Error
The file is not a valid GGUF file. Check:
- File was properly converted using `convert_to_gguf.py`
- File wasn't corrupted during transfer

### Missing Scale Factors
If layer scales aren't found, the model uses default scale=1.0:
```
[WARN] Scale factor not found for layers.0.linear, using default 1.0
```
This is safe but may affect model accuracy slightly.

### Tensor Shape Mismatch
Ensure your GGUF file uses the standard KiyEngine architecture:
- Vocab size: 4608
- D-model: 1024
- Context length: 32

## Migration Checklist

- [ ] Run `python scripts/convert_to_gguf.py` on your model
- [ ] Verify `kiyengine.gguf` is created successfully
- [ ] Test with `cargo run --release`
- [ ] Verify inference produces same results as v4
- [ ] Delete old `kiyengine.safetensors` (optional)

## References

- [GGUF Specification](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/README.md)
- [llama.cpp GGUF Format](https://github.com/ggml-org/llama.cpp/wiki/GGUF-format)
- [BitNet 1.58-bit Paper](https://arxiv.org/abs/2402.12354)
