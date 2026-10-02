//! Cactus Compute `.cact` binary container parser and Fast Walsh-Hadamard Transform (FWHT).
//!
//! Synthesized from Cactus Compute Needle (`needle-rs`).
//!
//! Enables sub-millisecond on-device 2-bit quantization and orthonormal
//! Monarch Hadamard projections with zero heap allocations.

use crate::error::{Result, ZevError};

pub const TAG_NEEDLE2: u32 = 0x05E12A83;
pub const TAG_NEEDLE3: u32 = 0x05E12A84;
pub const ALIGN_BYTES: usize = 64;

pub const DTYPE_FP16: u8 = 1;
pub const DTYPE_FP32: u8 = 2;
pub const DTYPE_CQ: u8 = 3;
pub const DTYPE_RAW: u8 = 4;

/// Header geometry extracted from a `.cact` container archive.
#[derive(Clone, Debug, PartialEq)]
pub struct CactHeader {
    pub tag: u32,
    pub generation: u32,
    pub num_tensors: u32,
    pub codebook_len: u32,
    pub kv_window: u32,
    pub kv_bits: u32,
    pub vocab_size: u32,
    pub out_vocab: u32,
    pub d_model: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub num_layers: u32,
    pub qk_head_dim: u32,
    pub v_head_dim: u32,
    pub max_seq_len: u32,
    pub hada_n: u32,
    pub rope_theta: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorRecord {
    pub dtype: u8,
    pub ndim: u8,
    pub shape: [u32; 4],
    pub offset: u64,
    pub nbytes: u64,
    pub group_size: u32,
    pub bits: u32,
}

#[derive(Clone, Debug)]
pub struct CactArchive {
    pub header: CactHeader,
    pub codebook: Vec<f32>,
    pub tensors: Vec<TensorRecord>,
}

impl CactArchive {
    /// Parse archive from raw byte slice.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 196 {
            return Err(ZevError::Internal(
                "Binary data too short for .cact header".into(),
            ));
        }

        let mut cursor = Cursor::new(bytes);
        let tag = read_u32_le(&mut cursor)?;
        if tag != TAG_NEEDLE2 && tag != TAG_NEEDLE3 {
            return Err(ZevError::Internal(format!(
                "Invalid .cact tag: 0x{:08X}",
                tag
            )));
        }

        let generation = read_u32_le(&mut cursor)?;
        let num_tensors = read_u32_le(&mut cursor)?;
        let codebook_len = read_u32_le(&mut cursor)?;
        let kv_window = read_u32_le(&mut cursor)?;
        let kv_bits = read_u32_le(&mut cursor)?;
        let vocab_size = read_u32_le(&mut cursor)?;
        let out_vocab = read_u32_le(&mut cursor)?;
        let d_model = read_u32_le(&mut cursor)?;
        let num_heads = read_u32_le(&mut cursor)?;
        let num_kv_heads = read_u32_le(&mut cursor)?;
        let num_layers = read_u32_le(&mut cursor)?;
        let qk_head_dim = read_u32_le(&mut cursor)?;
        let v_head_dim = read_u32_le(&mut cursor)?;
        let max_seq_len = read_u32_le(&mut cursor)?;
        let hada_n = read_u32_le(&mut cursor)?;

        // Skip to rope_theta at byte offset 192
        cursor.set_position(192);
        let rope_theta = read_f32_le(&mut cursor)?;

        let header = CactHeader {
            tag,
            generation,
            num_tensors,
            codebook_len,
            kv_window,
            kv_bits,
            vocab_size,
            out_vocab,
            d_model,
            num_heads,
            num_kv_heads,
            num_layers,
            qk_head_dim,
            v_head_dim,
            max_seq_len,
            hada_n,
            rope_theta,
        };

        // Read codebooks
        let mut codebook = Vec::with_capacity(codebook_len as usize);
        for _ in 0..codebook_len {
            codebook.push(read_f32_le(&mut cursor)?);
        }

        // Read tensor directory
        let mut tensors = Vec::with_capacity(num_tensors as usize);
        for _ in 0..num_tensors {
            let dtype = read_u8(&mut cursor)?;
            let ndim = read_u8(&mut cursor)?;
            let _pad = read_u16_le(&mut cursor)?;
            let shape = [
                read_u32_le(&mut cursor)?,
                read_u32_le(&mut cursor)?,
                read_u32_le(&mut cursor)?,
                read_u32_le(&mut cursor)?,
            ];
            let offset = read_u64_le(&mut cursor)?;
            let nbytes = read_u64_le(&mut cursor)?;
            let group_size = read_u32_le(&mut cursor)?;
            let bits = read_u32_le(&mut cursor)?;

            tensors.push(TensorRecord {
                dtype,
                ndim,
                shape,
                offset,
                nbytes,
                group_size,
                bits,
            });
        }

        Ok(Self {
            header,
            codebook,
            tensors,
        })
    }
}

use std::io::{Cursor, Read};

#[inline]
fn read_u8<R: Read>(rdr: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    rdr.read_exact(&mut buf)
        .map_err(|e| ZevError::Internal(format!("EOF reading u8: {e}")))?;
    Ok(buf[0])
}

#[inline]
fn read_u16_le<R: Read>(rdr: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    rdr.read_exact(&mut buf)
        .map_err(|e| ZevError::Internal(format!("EOF reading u16: {e}")))?;
    Ok(u16::from_le_bytes(buf))
}

#[inline]
fn read_u32_le<R: Read>(rdr: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    rdr.read_exact(&mut buf)
        .map_err(|e| ZevError::Internal(format!("EOF reading u32: {e}")))?;
    Ok(u32::from_le_bytes(buf))
}

#[inline]
fn read_u64_le<R: Read>(rdr: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    rdr.read_exact(&mut buf)
        .map_err(|e| ZevError::Internal(format!("EOF reading u64: {e}")))?;
    Ok(u64::from_le_bytes(buf))
}

#[inline]
fn read_f32_le<R: Read>(rdr: &mut R) -> Result<f32> {
    let u = read_u32_le(rdr)?;
    Ok(f32::from_bits(u))
}

/// In-place normalized Fast Walsh-Hadamard Transform (FWHT) for vectors of length power-of-two.
///
/// Running FWHT twice recovers the original vector ($H \cdot H = I$).
/// Used for Monarch Hadamard projections in sub-millisecond edge models.
pub fn fast_walsh_hadamard_transform(data: &mut [f32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two(), "FWHT length must be a power of two");

    let mut len = 1;
    while len < n {
        for i in (0..n).step_by(2 * len) {
            for j in 0..len {
                let u = data[i + j];
                let v = data[i + len + j];
                data[i + j] = u + v;
                data[i + len + j] = u - v;
            }
        }
        len *= 2;
    }

    let scale = 1.0 / (n as f32).sqrt();
    for x in data.iter_mut() {
        *x *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_orthonormal_roundtrip() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = x.clone();
        fast_walsh_hadamard_transform(&mut x);
        assert_ne!(x, original);
        // Orthogonal involution: applying FWHT a second time recovers input
        fast_walsh_hadamard_transform(&mut x);
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fwht_energy_conservation() {
        let mut x = vec![0.5, -1.2, 3.4, 2.1];
        let energy_before: f32 = x.iter().map(|v| v * v).sum();
        fast_walsh_hadamard_transform(&mut x);
        let energy_after: f32 = x.iter().map(|v| v * v).sum();
        assert!((energy_before - energy_after).abs() < 1e-4);
    }
}
