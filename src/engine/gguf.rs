// src/engine/gguf.rs
//! GGUF (GGML Universal File) format parser and loader for KiyEngine V5
//!
//! Supports fast memory-mapped loading of BitNet 1.58-bit quantized models.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;

/// GGUF Magic number: "GGUF"
pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
/// Current GGUF version supported
pub const GGUF_VERSION: u32 = 3;
/// Tensor data alignment (32 bytes)
pub const GGUF_ALIGNMENT: u64 = 32;

/// GGUF Value Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    /// Get type from raw value
    pub fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// GGML Quantization Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlQuantizationType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 26, // Custom: BitNet 1.58-bit ternary weights
    I16 = 27,
    I32 = 28,
}

impl GgmlQuantizationType {
    /// Get type from raw value
    pub fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            26 => Some(Self::I8),
            27 => Some(Self::I16),
            28 => Some(Self::I32),
            _ => None,
        }
    }

    /// Get size in bytes per element (for non-quantized types)
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            _ => 1, // Quantized types have block-based sizes
        }
    }
}

/// GGUF Metadata Value
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// Tensor information
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub quantization: GgmlQuantizationType,
    pub offset: u64,
    pub n_elements: u64,
}

/// Parsed GGUF file
#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufMetadataValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub tensor_data_offset: u64,
    pub mmap: Mmap,
}

impl GgufFile {
    /// Load GGUF file from path with memory mapping
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    /// Parse GGUF from memory-mapped buffer
    pub fn from_mmap(mmap: Mmap) -> anyhow::Result<Self> {
        if mmap.len() < 24 {
            anyhow::bail!("GGUF file too small (minimum 24 bytes)");
        }

        // Verify magic
        if &mmap[0..4] != GGUF_MAGIC {
            anyhow::bail!("Invalid GGUF magic (expected 'GGUF')");
        }

        // Parse header
        let version = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]);
        if version != GGUF_VERSION {
            anyhow::bail!(
                "Unsupported GGUF version: {} (expected {})",
                version,
                GGUF_VERSION
            );
        }

        let n_tensors = u64::from_le_bytes([
            mmap[8], mmap[9], mmap[10], mmap[11], mmap[12], mmap[13], mmap[14], mmap[15],
        ]);
        let n_kv = u64::from_le_bytes([
            mmap[16], mmap[17], mmap[18], mmap[19], mmap[20], mmap[21], mmap[22], mmap[23],
        ]);

        let mut offset: usize = 24;

        // Parse metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let (key, value, consumed) = Self::parse_kv(&mmap[offset..])?;
            metadata.insert(key, value);
            offset += consumed;
        }

        // Parse tensor info
        let mut tensors = Vec::with_capacity(n_tensors as usize);

        for _ in 0..n_tensors {
            let (info, consumed) = Self::parse_tensor_info(&mmap[offset..])?;
            tensors.push(info);
            offset += consumed;
        }

        // Data section starts at next alignment boundary after all headers
        let tensor_data_offset =
            ((offset as u64 + GGUF_ALIGNMENT - 1) / GGUF_ALIGNMENT) * GGUF_ALIGNMENT;

        Ok(Self {
            version,
            metadata,
            tensors,
            tensor_data_offset,
            mmap,
        })
    }

    /// Parse a key-value pair from metadata section
    fn parse_kv(data: &[u8]) -> anyhow::Result<(String, GgufMetadataValue, usize)> {
        let mut offset = 0;

        // Read key string
        let key_len = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]) as usize;
        offset += 8;

        let key = String::from_utf8(data[offset..offset + key_len].to_vec())?;
        offset += key_len;

        // Read value type
        let vtype_raw = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        let vtype = GgufValueType::from_raw(vtype_raw)
            .ok_or_else(|| anyhow::anyhow!("Unknown value type: {}", vtype_raw))?;
        offset += 4;

        // Read value
        let (value, consumed) = Self::parse_value(&data[offset..], vtype)?;
        offset += consumed;

        Ok((key, value, offset))
    }

    /// Parse a typed value
    fn parse_value(
        data: &[u8],
        vtype: GgufValueType,
    ) -> anyhow::Result<(GgufMetadataValue, usize)> {
        let mut offset;

        let value = match vtype {
            GgufValueType::Uint8 => {
                offset = 1;
                GgufMetadataValue::Uint8(data[0])
            }
            GgufValueType::Int8 => {
                offset = 1;
                GgufMetadataValue::Int8(data[0] as i8)
            }
            GgufValueType::Uint16 => {
                offset = 2;
                GgufMetadataValue::Uint16(u16::from_le_bytes([data[0], data[1]]))
            }
            GgufValueType::Int16 => {
                offset = 2;
                GgufMetadataValue::Int16(i16::from_le_bytes([data[0], data[1]]))
            }
            GgufValueType::Uint32 => {
                offset = 4;
                GgufMetadataValue::Uint32(u32::from_le_bytes([data[0], data[1], data[2], data[3]]))
            }
            GgufValueType::Int32 => {
                offset = 4;
                GgufMetadataValue::Int32(i32::from_le_bytes([data[0], data[1], data[2], data[3]]))
            }
            GgufValueType::Float32 => {
                offset = 4;
                GgufMetadataValue::Float32(f32::from_le_bytes([data[0], data[1], data[2], data[3]]))
            }
            GgufValueType::Bool => {
                offset = 1;
                GgufMetadataValue::Bool(data[0] != 0)
            }
            GgufValueType::String => {
                let len = u64::from_le_bytes([
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                ]) as usize;
                offset = 8 + len;
                let s = String::from_utf8(data[8..8 + len].to_vec())?;
                GgufMetadataValue::String(s)
            }
            GgufValueType::Uint64 => {
                offset = 8;
                GgufMetadataValue::Uint64(u64::from_le_bytes([
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                ]))
            }
            GgufValueType::Int64 => {
                offset = 8;
                GgufMetadataValue::Int64(i64::from_le_bytes([
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                ]))
            }
            GgufValueType::Float64 => {
                offset = 8;
                GgufMetadataValue::Float64(f64::from_le_bytes([
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                ]))
            }
            GgufValueType::Array => {
                let elem_type_raw = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let elem_type = GgufValueType::from_raw(elem_type_raw).ok_or_else(|| {
                    anyhow::anyhow!("Unknown array element type: {}", elem_type_raw)
                })?;
                let count = u64::from_le_bytes([
                    data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11],
                ]) as usize;

                offset = 12;
                let mut elements = Vec::with_capacity(count);
                for _ in 0..count {
                    let (elem, consumed) = Self::parse_value(&data[offset..], elem_type)?;
                    elements.push(elem);
                    offset += consumed;
                }
                GgufMetadataValue::Array(elements)
            }
        };

        Ok((value, offset))
    }

    /// Parse tensor info
    fn parse_tensor_info(data: &[u8]) -> anyhow::Result<(GgufTensorInfo, usize)> {
        let mut offset = 0;

        // Read name
        let name_len = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]) as usize;
        offset += 8;

        let name = String::from_utf8(data[offset..offset + name_len].to_vec())?;
        offset += name_len;

        // Read dimensions count FIRST (GGUF format)
        let n_dims = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        // Read dimensions
        let mut dimensions = Vec::with_capacity(n_dims);
        for i in 0..n_dims {
            let dim = u64::from_le_bytes([
                data[offset + i * 8],
                data[offset + i * 8 + 1],
                data[offset + i * 8 + 2],
                data[offset + i * 8 + 3],
                data[offset + i * 8 + 4],
                data[offset + i * 8 + 5],
                data[offset + i * 8 + 6],
                data[offset + i * 8 + 7],
            ]);
            dimensions.push(dim);
        }
        dimensions.reverse(); // Convert GGML to standard order
        offset += n_dims * 8;

        // Read quantization type (AFTER dimensions in GGUF)
        let qtype_raw = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        let quantization = GgmlQuantizationType::from_raw(qtype_raw)
            .ok_or_else(|| anyhow::anyhow!("Unknown quantization type: {}", qtype_raw))?;
        offset += 4;

        // Read offset
        let tensor_offset = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        // Calculate element count
        let n_elements: u64 = dimensions.iter().product();

        Ok((
            GgufTensorInfo {
                name,
                dimensions,
                quantization,
                offset: tensor_offset,
                n_elements,
            },
            offset,
        ))
    }

    /// Get tensor raw data as byte slice
    /// Tensor offsets in GGUF are relative to the data section start
    pub fn get_tensor_data(&self, info: &GgufTensorInfo) -> &[u8] {
        let type_size = info.quantization.type_size();
        let data_size = info.n_elements as usize * type_size;
        let abs_offset = self.tensor_data_offset as usize + info.offset as usize;
        &self.mmap[abs_offset..abs_offset + data_size]
    }

    /// Get metadata value as f32
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| match v {
            GgufMetadataValue::Float32(f) => Some(*f),
            GgufMetadataValue::Float64(f) => Some(*f as f32),
            GgufMetadataValue::Int32(i) => Some(*i as f32),
            GgufMetadataValue::Int64(i) => Some(*i as f32),
            _ => None,
        })
    }

    /// Get metadata value as string
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| match v {
            GgufMetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        })
    }

    /// Get metadata value as u32
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| match v {
            GgufMetadataValue::Uint32(u) => Some(*u),
            GgufMetadataValue::Uint64(u) => Some(*u as u32),
            GgufMetadataValue::Int32(i) => Some(*i as u32),
            GgufMetadataValue::Int64(i) => Some(*i as u32),
            _ => None,
        })
    }

    /// Find tensor by name
    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get scale factor for a layer
    pub fn get_layer_scale(&self, layer_idx: usize) -> f32 {
        let key = format!("kiyengine.layers.{}.linear.scale", layer_idx);
        self.get_f32(&key).unwrap_or(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_type_from_raw() {
        assert_eq!(GgufValueType::from_raw(0), Some(GgufValueType::Uint8));
        assert_eq!(GgufValueType::from_raw(6), Some(GgufValueType::Float32));
        assert_eq!(GgufValueType::from_raw(8), Some(GgufValueType::String));
        assert_eq!(GgufValueType::from_raw(99), None);
    }

    #[test]
    fn test_quantization_type_size() {
        assert_eq!(GgmlQuantizationType::F32.type_size(), 4);
        assert_eq!(GgmlQuantizationType::F16.type_size(), 2);
        assert_eq!(GgmlQuantizationType::I8.type_size(), 1);
    }
}
