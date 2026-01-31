# KiyEngine V3 Surgical Audit & V4 Pre-flight Checklist

## 🛠 Completed V3 Optimizations

### 1. Concurrency & Transposition Table
- **Lock-free TT**: Refactored `src/tt.rs` to use `AtomicU64` for both key and data. Implemented the **XOR trick** (`stored_key = key ^ data`) to achieve 128-bit atomicity without `AtomicU128`, ensuring no data corruption during multi-threaded probes.
- **Lazy SMP**: Search now scales across multiple CPU cores. Launched via `std::thread::scope`, threads communicate through the shared TT. Only thread 0 reports UCI info to maintain protocol compliance.
- **Table Aging**: Added an aging mechanism to the TT to improve replacement strategies across multiple search sessions.

### 2. Inference Performance (The "Mamba way")
- **Stateful Inference**: Implemented `MambaState` (`conv_state` and `ssm_state`) in `src/mamba.rs`. The engine now performs **O(1) incremental inference** per move instead of re-processing the entire move history ($O(L^2)$).
- **CPU Roundtrip Elimination**: Optimized MoE routing in `src/moe.rs`. Large tensors remain on the device (GPU/CPU) during the search. Host-side data transfer is limited to tiny 8-expert routing scores.
- **Native Operations**: Replaced manual depthwise convolution loops with optimized `candle` operations.

### 3. Search & Evaluation Logic
- **Neural Integration**: The Mamba-MoE brain is now called at every search node with `depth >= 2`.
- **Hybrid Evaluation**: Implemented formula: `Score = 0.7 * AI_Score + 0.3 * Static_Tutor_Score`.
- **Policy-Driven Ordering**: Neural policy logits are now used to order quiet moves, significantly improving the pruning efficiency.
- **Tight Time Controls**: Increased time-check frequency (every 64 nodes) to ensure millisecond responsiveness in bullet/blitz games.

---

## 🚀 V4 Integration Hooks

The current "perfect skeleton" is ready for V4 (RWKV-6 + BitNet) integration:

### 🦅 RWKV-6 Eagle (Move Ordering)
- **Location**: `src/engine.rs` -> `Engine::forward_incremental`
- **Hook**: Replace the current Mamba block logic with RWKV-6 Eagle blocks. The `EngineState` struct is already designed to hold recurrent states.
- **Integration**: The 4096-dim latent space of RWKV-6 will replace the `policy_head`. Use the Eagle's state-update logic in place of the current SSM update.

### 💎 BitNet (Binary Evaluation)
- **Location**: `src/engine.rs` -> `ValueHead` and `src/search.rs` -> `evaluate_hybrid`
- **Hook**: BitNet-based evaluation will replace the `ValueHead`.
- **Integration**: Replace the floating-point `Linear` layers in the value head with BitNet 1.58-bit quantized layers. The `evaluate_hybrid` function in search will then call this binary evaluator.

---

## 📋 Pre-flight Checklist for V4
- [ ] **Quantization**: Implement custom `candle` kernels for BitNet's -1, 0, 1 weights to realize the 10x speedup potential.
- [ ] **State Compaction**: Further optimize `EngineState` cloning if NPS drops below 1000 at higher thread counts.
- [ ] **History Heuristics**: Combine neural policy with traditional history heuristics for even better move ordering.
- [ ] **Memory Mapping**: Ensure `model.safetensors` remains memory-mapped to minimize RAM footprint when using large RWKV-6 weights.
