// src/safetensor_loader.rs
//
// A robust safetensors loader that handles large metadata headers (>10MB)
// and provides type-safe tensor retrieval with shape validation.

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Represents tensor metadata from the safetensors header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// A memory-mapped safetensors file with parsed metadata.
pub struct SafeTensorFile {
    mmap: Mmap,
    tensors: HashMap<String, TensorInfo>,
    data_start: usize,
}

impl SafeTensorFile {
    /// Opens a safetensors file and parses its header.
    /// Handles large headers by reading the size prefix first.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened, is too small, or has an invalid header.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open {}", path.as_ref().display()))?;

        // Safety: We're only reading from the file, never modifying it
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            bail!("Safetensors file too small: {} bytes", mmap.len());
        }

        // Read 8-byte little-endian header size
        #[allow(clippy::cast_possible_truncation)]
        let header_size = u64::from_le_bytes(mmap[0..8].try_into()?) as usize;

        if header_size > mmap.len() - 8 {
            bail!(
                "Invalid header size: {header_size} (file size: {})",
                mmap.len()
            );
        }

        // Parse JSON header
        let header_bytes = &mmap[8..8 + header_size];
        let header_str = std::str::from_utf8(header_bytes).context("Header is not valid UTF-8")?;

        let header: serde_json::Value =
            serde_json::from_str(header_str).context("Failed to parse header JSON")?;

        let header_obj = header
            .as_object()
            .ok_or_else(|| anyhow!("Header is not a JSON object"))?;

        let mut tensors = HashMap::new();

        for (key, value) in header_obj {
            // Skip metadata
            if key == "__metadata__" {
                continue;
            }

            let tensor_info = value
                .as_object()
                .ok_or_else(|| anyhow!("Tensor {key} info is not an object"))?;

            let dtype = tensor_info
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow!("Missing dtype for tensor {key}"))?
                .to_string();

            #[allow(clippy::cast_possible_truncation)]
            let shape: Vec<usize> = tensor_info
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| anyhow!("Missing shape for tensor {key}"))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();

            let offsets = tensor_info
                .get("data_offsets")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| anyhow!("Missing data_offsets for tensor {key}"))?;

            if offsets.len() != 2 {
                bail!("Invalid data_offsets for tensor {key}");
            }

            #[allow(clippy::cast_possible_truncation)]
            let start = offsets[0].as_u64().unwrap_or(0) as usize;
            #[allow(clippy::cast_possible_truncation)]
            let end = offsets[1].as_u64().unwrap_or(0) as usize;

            tensors.insert(
                key.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    data_offsets: (start, end),
                },
            );
        }

        let data_start = 8 + header_size;

        Ok(Self {
            mmap,
            tensors,
            data_start,
        })
    }

    /// Lists all tensor names in the file.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Gets information about a specific tensor.
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Loads a tensor by name, converting to the specified dtype.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not found, has an unsupported dtype, or if size mismatch occurs.
    ///
    /// # Panics
    ///
    /// Panics if the bytes cannot be converted to the target type (unreachable if data is valid).
    pub fn load_tensor(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{name}' not found in safetensors file"))?;

        let (start, end) = info.data_offsets;
        let data = &self.mmap[self.data_start + start..self.data_start + end];

        // Determine source dtype
        let src_dtype = match info.dtype.as_str() {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F64" => DType::F64,
            "I64" => bail!("I64 tensors not supported for neural network weights"),
            "I32" => bail!("I32 tensors not supported for neural network weights"),
            other => bail!("Unsupported dtype: {other}"),
        };

        // Calculate expected size
        let elem_size = match src_dtype {
            DType::F32 | DType::I64 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F64 => 8,
            _ => bail!("Unsupported dtype"),
        };

        let num_elements: usize = info.shape.iter().product();
        let expected_size = num_elements * elem_size;

        if data.len() != expected_size {
            bail!(
                "Size mismatch for tensor '{}': expected {} bytes, got {}",
                name,
                expected_size,
                data.len()
            );
        }

        // Create tensor from raw bytes
        let tensor = match src_dtype {
            DType::F32 => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Tensor::from_vec(floats, info.shape.as_slice(), device)?
            }
            DType::F16 => {
                let halfs: Vec<half::f16> = data
                    .chunks_exact(2)
                    .map(|chunk| half::f16::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Tensor::from_vec(halfs, info.shape.as_slice(), device)?
            }
            DType::BF16 => {
                let bfloats: Vec<half::bf16> = data
                    .chunks_exact(2)
                    .map(|chunk| half::bf16::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Tensor::from_vec(bfloats, info.shape.as_slice(), device)?
            }
            _ => bail!("Unsupported source dtype: {src_dtype:?}"),
        };

        // Convert to target dtype if needed
        if src_dtype == dtype {
            Ok(tensor)
        } else {
            tensor
                .to_dtype(dtype)
                .with_context(|| format!("Failed to convert tensor '{name}' to {dtype:?}"))
        }
    }

    /// Loads a tensor with shape validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not found, has a shape mismatch, or if loading fails.
    pub fn load_tensor_with_shape(
        &self,
        name: &str,
        expected_shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{name}' not found"))?;

        if info.shape != expected_shape {
            bail!(
                "Shape mismatch for tensor '{}': expected {:?}, got {:?}",
                name,
                expected_shape,
                info.shape
            );
        }

        self.load_tensor(name, dtype, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_safetensors() {
        // This test requires the actual model file
        if std::path::Path::new("model.safetensors").exists() {
            let st = SafeTensorFile::open("model.safetensors").expect("Failed to open");
            assert!(!st.tensor_names().is_empty());
            println!("Found {} tensors", st.tensor_names().len());
        }
    }
}
