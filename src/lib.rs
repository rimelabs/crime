//! Concurrent real-time interface for multimedia engines.
//!
//! This crate provides abstractions and utilities for working with audio streams in Rust,
//! typically server-side applications that provides real-time audio streaming backed by
//! machine-learning models via gRPC, WebSocket, etc.
//!
//! This crate supports:
//!   - Audio inputs as `f32` samples
//!   - Resampling audio streams to different sample rates
//!   - Encoding audio streams to various formats (PCM, WAV, MP3)
//!
//! The main entry point is [`AudioStream`].

use async_stream::stream;
use futures::StreamExt;
use futures::stream::Stream;
use half::f16;
use rubato::Resampler;
use std::boxed::Box;
use std::pin::Pin;
mod wsola;
use wsola::time_scale;

pub type Mp3BitRate = mp3lame_encoder::Bitrate;
pub type Mp3Quality = mp3lame_encoder::Quality;

/// Represents a PCM encoding.
/// A PCM encoding can be linear or companding (only G.711 μ-law is supported).
pub enum PcmEncoding {
    /// Linear PCM.
    LinearPcm(LinearPcmEncoding),
    /// G.711 μ-law.
    G711MuLaw,
}

impl Default for PcmEncoding {
    fn default() -> Self {
        Self::LinearPcm(LinearPcmEncoding::default())
    }
}

/// Represents a linear PCM encoding.
///
/// All values are in little-endian format.
/// Float values are clamped between [-1.0, 1.0].
pub enum LinearPcmEncoding {
    /// IEEE 754 half-precision floating point.
    Float16,
    /// IEEE 754 single-precision floating point.
    Float32,
    /// 16-bit signed integer.
    Int16,
}

impl Default for LinearPcmEncoding {
    fn default() -> Self {
        Self::Int16
    }
}

pub enum Encoding {
    /// Raw stream of PCM samples.
    Pcm(PcmEncoding),
    /// Stream of Linear PCM samples with WAV header.
    Wav(LinearPcmEncoding),
    /// Stream of MP3 samples.
    Mp3(Mp3BitRate, Mp3Quality),
}

impl Default for Encoding {
    fn default() -> Self {
        Self::Pcm(PcmEncoding::default())
    }
}

/// Represents an audio stream with a specific sample rate.
///
/// This struct wraps a stream of audio samples (`f32`) and associates it with a sample rate.
/// It provides methods for audio processing, such as resampling and encoding.
///
/// # Fields
///
/// * `stream` - The input audio samples as a pinned, boxed stream of `f32` values.
/// * `sample_rate` - The sample rate of the audio stream, in Hz.
pub struct AudioStream<'a> {
    stream: Pin<Box<dyn Stream<Item = f32> + Send + 'a>>,
    sample_rate: u32,
}

impl<'a> AudioStream<'a> {
    /// Creates a new `AudioStream` from the given sample rate and input stream.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - The sample rate of the input audio stream, in Hz.
    /// * `stream` - The input audio samples as a stream of 32-bit floating point values (`f32`).
    ///
    /// # Returns
    ///
    /// A new `AudioStream` instance wrapping the provided stream and sample rate.
    pub fn new(sample_rate: u32, stream: impl Stream<Item = f32> + Send + 'a) -> AudioStream<'a> {
        AudioStream {
            sample_rate,
            stream: Box::pin(stream),
        }
    }

    /// Commits the audio manipulation on the original audio stream.
    ///
    /// This method applies resampling and encoding to the audio stream according to the specified
    /// output sample rate and encoding format. It consumes the `AudioStream` and returns a stream
    /// of encoded audio bytes.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - The desired output sample rate in Hz.
    /// * `time_scale_factor` - The time scale factor to apply to the audio stream. A value above
    ///   1.0 speeds up the audio, a value below 1.0 slows it down.
    /// * `encoding` - The desired output encoding format.
    ///
    /// # Returns
    ///
    /// A pinned, boxed stream of encoded audio bytes (`u8`).
    pub async fn commit(
        self,
        sample_rate: u32,
        time_scale_factor: f32,
        encoding: Encoding,
    ) -> Pin<Box<dyn Stream<Item = u8> + Send + 'a>> {
        let time_scaled_stream = if (time_scale_factor - 1.0).abs() > f32::EPSILON {
            time_scale(self.stream, time_scale_factor, self.sample_rate).await
        } else {
            self.stream
        };

        let resampled_stream = if self.sample_rate != sample_rate {
            resample(time_scaled_stream, self.sample_rate, sample_rate).await
        } else {
            time_scaled_stream
        };

        encode(resampled_stream, sample_rate, encoding).await
    }
}

async fn resample<'a>(
    samples: impl Stream<Item = f32> + Send + 'a,
    input_rate: u32,
    output_rate: u32,
) -> Pin<Box<dyn Stream<Item = f32> + Send + 'a>> {
    if input_rate == output_rate {
        return Box::pin(samples);
    }
    let chunk_size = (20 * input_rate / 1000) as usize;
    let mut resampler =
        rubato::FftFixedInOut::<f32>::new(input_rate as usize, output_rate as usize, chunk_size, 1)
            .expect("creating resampler");
    let output_delay = resampler.output_delay();
    let mut input = Box::pin(samples.chain(futures::stream::repeat(0.0).take(output_delay)));
    Box::pin(
        stream! {
          let mut in_buffer = resampler.input_buffer_allocate(false);
          let mut out_buffer = resampler.output_buffer_allocate(true);
          loop {
            let frames_needed = resampler.input_frames_next();

            in_buffer[0].clear();
            for _ in 0..frames_needed {
              match input.next().await {
                Some(sample) => in_buffer[0].push(sample),
                None => break,
              }
            }
            if in_buffer[0].len() < frames_needed {
              // No more pending samples. Not going to process the remainder.
              break;
            }

            let (_in_frames, out_frames) = resampler.process_into_buffer(
              &in_buffer, &mut out_buffer, None).unwrap();
            for sample in out_buffer[0][..out_frames].iter() {
              yield *sample;
            }
          }
        }
        .skip(output_delay),
    )
}

async fn encode<'a>(
    samples: impl Stream<Item = f32> + Send + 'a,
    sample_rate: u32,
    encoding: Encoding,
) -> Pin<Box<dyn Stream<Item = u8> + Send + 'a>> {
    match encoding {
        Encoding::Pcm(PcmEncoding::LinearPcm(pcm_encoding)) => {
            encode_as_linear_pcm(samples, pcm_encoding).await
        }
        Encoding::Pcm(PcmEncoding::G711MuLaw) => encode_as_g711_mu_law(samples).await,
        Encoding::Wav(linear_pcm_encoding) => {
            encode_as_wav(samples, sample_rate, linear_pcm_encoding).await
        }
        Encoding::Mp3(bit_rate, quality) => {
            encode_as_mp3(samples, sample_rate, bit_rate, quality).await
        }
    }
}

/// Encode f32 samples in [-1.0, 1.0] to 8-bit G.711 μ-law bytes.
///
/// The implementation follows ITU-T G.711 with μ=255, mapping linear PCM to μ-law.
/// Input f32 is first clamped to [-1.0, 1.0], scaled to i16 range, then encoded.
async fn encode_as_g711_mu_law<'a>(
    samples: impl Stream<Item = f32> + Send + 'a,
) -> Pin<Box<dyn Stream<Item = u8> + Send + 'a>> {
    let mut samples = Box::pin(samples);
    Box::pin(stream! {
      while let Some(sample) = samples.next().await {
        let s = sample.clamp(-1.0, 1.0);
        // Scale to 16-bit linear PCM range
        let pcm16 = (s * 32767.0).round() as i16;
        yield linear_to_mu_law(pcm16);
      }
    })
}

// Convert 16-bit linear PCM to G.711 μ-law (8-bit).
// Reference: ITU-T G.711.
fn linear_to_mu_law(pcm_val: i16) -> u8 {
    // Constants
    const BIAS: i16 = 0x84; // 132
    const CLIP: i16 = 32635;

    // Get sign and magnitude
    let mut pcm = pcm_val;
    let sign = if pcm < 0 { 2 << 7 } else { 0 };

    pcm = pcm.abs();
    pcm = pcm.min(CLIP);
    pcm = pcm + BIAS;

    // Determine exponent (3 bits) and mantissa (4 bits).
    let exponent = mu_law_exponent((pcm as u16) >> 7);
    let mantissa = ((pcm >> (exponent + 3)) & 0x0F) as u8;
    let mu_law = !(sign | (exponent << 4) | mantissa);
    mu_law
}

#[inline]
fn mu_law_exponent(mut value: u16) -> u8 {
    let mut exp: u8 = 7;
    while exp > 0 {
        if value & 0x0100 != 0 {
            break;
        }
        value <<= 1;
        exp -= 1;
    }
    7 - exp
}

async fn encode_as_linear_pcm<'a>(
    samples: impl Stream<Item = f32> + Send + 'a,
    bit_depth: LinearPcmEncoding,
) -> Pin<Box<dyn Stream<Item = u8> + Send + 'a>> {
    let mut samples = Box::pin(samples);
    Box::pin(stream! {
        while let Some(sample) = samples.next().await {
            let sample = sample.clamp(-1.0, 1.0);
            match bit_depth {
                LinearPcmEncoding::Float16 => {
                    let sample = f16::from_f32(sample);
                    for byte in sample.to_le_bytes() {
                        yield byte;
                    }
                }
                LinearPcmEncoding::Float32 => {
                    for byte in sample.to_le_bytes() {
                        yield byte;
                    }
                }
                LinearPcmEncoding::Int16 => {
                    let sample = (sample * 32767.0).round() as i16;
                    for byte in sample.to_le_bytes() {
                        yield byte;
                    }
                }
            };
        }
    })
}

fn make_wav_header(sample_rate: u32, linear_pcm_encoding: &LinearPcmEncoding) -> [u8; 44] {
    let num_channels = 1u16;
    let bits_per_sample = match linear_pcm_encoding {
        LinearPcmEncoding::Float16 | LinearPcmEncoding::Int16 => 16u16,
        LinearPcmEncoding::Float32 => 32u16,
    };
    let audio_format = match linear_pcm_encoding {
        LinearPcmEncoding::Int16 => 1u16,
        LinearPcmEncoding::Float16 | LinearPcmEncoding::Float32 => 3u16,
    };
    let byte_rate = sample_rate * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = num_channels * (bits_per_sample / 8);
    let data_chunk_size = 0xFFFF_FFFFu32; // Unknown length for streaming
    let riff_chunk_size = 0xFFFF_FFFFu32; // Unknown length for streaming

    let mut header = [0u8; 44];
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&(riff_chunk_size).to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&0x10u32.to_le_bytes()); // Subchunk1Size
    header[20..22].copy_from_slice(&audio_format.to_le_bytes());
    header[22..24].copy_from_slice(&num_channels.to_le_bytes());
    header[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    header[32..34].copy_from_slice(&block_align.to_le_bytes());
    header[34..36].copy_from_slice(&bits_per_sample.to_le_bytes());
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&data_chunk_size.to_le_bytes());
    header
}

async fn encode_as_wav<'a>(
    samples: impl Stream<Item = f32> + Send + 'a,
    sample_rate: u32,
    linear_pcm_encoding: LinearPcmEncoding,
) -> Pin<Box<dyn Stream<Item = u8> + Send + 'a>> {
    Box::pin(stream! {
        for &header_byte in &make_wav_header(sample_rate, &linear_pcm_encoding) {
            yield header_byte;
        }
        let mut pcm_stream = encode_as_linear_pcm(samples, linear_pcm_encoding).await;
        while let Some(sample) = pcm_stream.next().await {
            yield sample;
        }
    })
}

async fn encode_as_mp3<'a>(
    samples: impl Stream<Item = f32> + Send + 'a,
    sample_rate: u32,
    bit_rate: Mp3BitRate,
    quality: Mp3Quality,
) -> Pin<Box<dyn Stream<Item = u8> + Send + 'a>> {
    let mut mp3_encoder = mp3lame_encoder::Builder::new().expect("Create LAME encoder");
    mp3_encoder.set_num_channels(1).expect("set channels");
    mp3_encoder.set_brate(bit_rate).expect("set bit_rate");
    mp3_encoder
        .set_sample_rate(sample_rate)
        .expect("set sample rate");
    mp3_encoder.set_quality(quality).expect("set quality");
    let mut mp3_encoder = mp3_encoder.build().expect("To initialize LAME encoder");

    let samples = Box::pin(samples);
    let mut sample_chunks = samples.ready_chunks(128);
    Box::pin(stream! {
      let mut mp3_out_buffer = Vec::new();

      while let Some(chunk) = sample_chunks.next().await {
        let chunk: Vec<f32> = chunk.into_iter().collect();
        let input = mp3lame_encoder::MonoPcm(&chunk);

        mp3_out_buffer.reserve(mp3lame_encoder::max_required_buffer_size(input.0.len()));
        mp3_encoder.encode_to_vec(input, &mut mp3_out_buffer).expect("To encode");

        for sample in &mp3_out_buffer {
          yield *sample;
        }
        mp3_out_buffer.clear();
      }
      mp3_encoder.flush_to_vec::<mp3lame_encoder::FlushNoGap>(&mut mp3_out_buffer).expect("to flush");
      for sample in &mp3_out_buffer {
        yield *sample;
      }
    })
}
