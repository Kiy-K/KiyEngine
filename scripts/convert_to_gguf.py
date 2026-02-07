import argparse
import gguf
import numpy as np
from safetensors.torch import load_file
import os
import sys

def convert_to_gguf(input_path, output_path):
    print(f"🔄 Đang convert: {input_path} -> {output_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Không tìm thấy file input: {input_path}")
        sys.exit(1)

    # 1. Load Model (Raw weights từ quá trình train)
    print("📦 Đang load tensors...")
    tensors = load_file(input_path)
    
    # 2. Khởi tạo GGUF Writer
    # Lưu ý: "kiyengine" là tên kiến trúc custom để Rust nhận diện
    gguf_writer = gguf.GGUFWriter(output_path, "kiyengine")
    
    # 3. Metadata (Config cho Rust)
    print("⚙️ Đang ghi Metadata...")
    gguf_writer.add_name("KiyEngine-V5-AutoSandwich")
    gguf_writer.add_block_count(4)
    gguf_writer.add_embedding_length(1024)
    gguf_writer.add_context_length(32)
    # ⚠️ QUAN TRỌNG: Đánh dấu file là F16 để Rust không báo lỗi Type 48
    gguf_writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16) 

    print("\n🔨 BẮT ĐẦU XỬ LÝ & ĐÓNG GÓI (AUTO-QUANTIZATION):")
    
    for k, w in tensors.items():
        # Dọn dẹp tên (Fix vụ _orig_mod từ torch.compile)
        clean_k = k.replace("_orig_mod.", "")
        
        # Chuyển sang Numpy để xử lý toán học
        data_np = w.cpu().float().numpy() # Dùng float32 để tính toán cho chuẩn
        
        # --- LOGIC TỰ ĐỘNG SANDWICH ---
        # Lớp ẩn: Nằm trong 'layers', là 'linear', KHÔNG phải 'norm'
        is_hidden_layer = ("layers." in clean_k) and \
                          ("linear.weight" in clean_k) and \
                          ("norm" not in clean_k)
        
        final_data = None
        
        if is_hidden_layer:
            # === NHÂN THỊT (BITNET TERNARY) ===
            # Tự động tính toán scale và ép về {-1, 0, 1}
            scale = np.max(np.abs(data_np))
            
            if scale > 1e-6:
                # Công thức BitNet: round(w / scale).clamp(-1, 1)
                ternary = np.clip(np.round(data_np / scale), -1, 1)
            else:
                ternary = data_np # Zero tensor
            
            # Lưu ý: Rust cần đọc Float16, nên ta cast về F16
            final_data = ternary.astype(np.float16)
            
            gguf_writer.add_tensor(clean_k, final_data)
            print(f"  🥩 {clean_k:40s} | Auto-Quantized to {-1, 0, 1} (F16 Container)")
            
        else:
            # === VỎ BÁNH (FP16 HIGH PRECISION) ===
            # Embed, Heads, Norms -> Giữ nguyên giá trị thực
            final_data = data_np.astype(np.float16)
            
            gguf_writer.add_tensor(clean_k, final_data)
            print(f"  🛡️ {clean_k:40s} | Giữ nguyên High Precision (F16)")

    # 4. Ghi file
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    
    print(f"\n✅ XONG! File GGUF chuẩn F16 đã ra lò: {output_path}")
    print("👉 File này chứa Sandwich Architecture (BitNet giữa, FP16 đầu đuôi).")
    print("👉 Rust sẽ đọc file này 'một phát ăn ngay' không lỗi Type 48.")

if __name__ == "__main__":
    # Giữ đúng interface CLI như script gốc ông yêu cầu
    parser = argparse.ArgumentParser(description="Convert KiyEngine safetensors to GGUF")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input .safetensors file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output .gguf file")
    
    args = parser.parse_args()
    
    convert_to_gguf(args.input, args.output)