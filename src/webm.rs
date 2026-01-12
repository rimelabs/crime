pub const EBML_ID: u32 = 0x1A45DFA3;
pub const SEGMENT_ID: u32 = 0x18538067;
pub const INFO_ID: u32 = 0x1549A966;
pub const TRACKS_ID: u32 = 0x1654AE6B;
pub const TRACK_ENTRY_ID: u32 = 0xAE;
pub const CLUSTER_ID: u32 = 0x1F43B675;
pub const SIMPLE_BLOCK_ID: u32 = 0xA3;

pub const TIMECODE_SCALE_ID: u32 = 0x2AD7B1;
pub const MUXING_APP_ID: u32 = 0x4D80;
pub const WRITING_APP_ID: u32 = 0x5741;

pub const TRACK_NUMBER_ID: u32 = 0xD7;
pub const TRACK_UID_ID: u32 = 0x73C5;
pub const TRACK_TYPE_ID: u32 = 0x83;
pub const CODEC_ID_ID: u32 = 0x86;
pub const CODEC_PRIVATE_ID: u32 = 0x63A2;
pub const CODEC_DELAY_ID: u32 = 0x56AA;
pub const SEEK_PRE_ROLL_ID: u32 = 0x56BB;

pub const AUDIO_ID: u32 = 0xE1;
pub const SAMPLING_FREQUENCY_ID: u32 = 0xB5;
pub const CHANNELS_ID: u32 = 0x94;

pub const TIMECODE_ID: u32 = 0xE7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebmError {
  VintTooLarge(u64),
}

impl std::fmt::Display for WebmError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      WebmError::VintTooLarge(value) => write!(f, "value {value} is too large for VINT"),
    }
  }
}

impl std::error::Error for WebmError {}

/// Generates a VINT (Variable Integer) as bytes.
pub fn encode_vint(value: u64) -> Result<Vec<u8>, WebmError> {
  let mut bytes = Vec::new();
  let mut len = 1;
  let mut mask = 0x80;

  // Find required length
  loop {
    if value < mask - 1 {
      break;
    }
    mask <<= 7;
    len += 1;
    if len > 8 {
      return Err(WebmError::VintTooLarge(value));
    }
  }

  let mut marker = 0x80u8;
  for _ in 1..len {
    marker >>= 1;
  }

  // Construct bytes in big-endian
  let val_with_marker = value | ((marker as u64) << ((len - 1) * 8));

  for i in (0..len).rev() {
    bytes.push(((val_with_marker >> (i * 8)) & 0xFF) as u8);
  }

  Ok(bytes)
}

/// Encodes an Element ID (which is technically a VINT, but usually fixed).
pub fn encode_id(id: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    if id >= 0x10000000 {
        bytes.push(((id >> 24) & 0xFF) as u8);
    }
    if id >= 0x0010000 {
        bytes.push(((id >> 16) & 0xFF) as u8);
    }
    if id >= 0x0000100 {
        bytes.push(((id >> 8) & 0xFF) as u8);
    }
    bytes.push((id & 0xFF) as u8);
    bytes
}

/// Writes a master element with specified children data.
pub fn make_element(id: u32, data: &[u8]) -> Vec<u8> {
    let mut out = encode_id(id);
    let size = encode_vint(data.len() as u64)
        .expect("element size should always fit within an 8-byte VINT");
    out.extend_from_slice(&size);
    out.extend_from_slice(data);
    out
}

/// Writes an unsigned integer element.
pub fn make_uint_element(id: u32, value: u64) -> Vec<u8> {
    let mut data = Vec::new();
    let bytes = value.to_be_bytes();
    // Skip leading zeros
    let skip = bytes.iter().position(|&x| x != 0).unwrap_or(bytes.len());
    if skip == bytes.len() {
        data.push(0);
    } else {
        data.extend_from_slice(&bytes[skip..]);
    }

    make_element(id, &data)
}

/// Writes a float element (f32).
pub fn make_float_element(id: u32, value: f32) -> Vec<u8> {
    make_element(id, &value.to_be_bytes())
}

/// Writes a string element (ASCII).
pub fn make_string_element(id: u32, value: &str) -> Vec<u8> {
    make_element(id, value.as_bytes())
}

pub fn make_simple_block(track_number: u64, timecode: i16, data: &[u8]) -> Vec<u8> {
    let mut block = encode_vint(track_number).expect("track number must fit in VINT");
    block.extend_from_slice(&timecode.to_be_bytes());
    // Flags: Keyframe (0x80) | Invisible (0x08) | Lacing (0x06) | Discardable (0x01)
    // Opus packets are always keyframes in WebM.
    block.push(0x80);
    block.extend_from_slice(data);

    make_element(SIMPLE_BLOCK_ID, &block)
}
