//! Decoding headerless audio into linear PCM as it arrives.

use std::f64::consts::PI;
use std::sync::LazyLock;

use audio_codec_algorithms::decode_ulaw;

/// The interpolation filter's delay, in 8 kHz samples: 3 ms.
const DELAY: usize = 24;
/// The 8 kHz samples each interpolated 16 kHz sample is made from.
const TAPS: usize = 2 * DELAY;

/// The weights that interpolate a 16 kHz sample between the 8 kHz samples
/// `TAPS / 2` before and after it, oldest first: the odd phase of a
/// half-band low-pass at 4 kHz, a Blackman-windowed sinc of `2 * TAPS + 1`
/// taps at 16 kHz, scaled to unit gain.
static WEIGHTS: LazyLock<[f32; TAPS]> = LazyLock::new(|| {
    let span = (2 * TAPS) as f64;
    let weight = |tap: usize| {
        // Every odd tap of the prototype: its distance from the centre, in
        // 8 kHz samples, is a half-integer.
        let prototype_tap = (2 * tap + 1) as f64;
        let offset = (prototype_tap - span / 2.0) / 2.0;
        let sinc = (PI * offset).sin() / (PI * offset);
        let phase = 2.0 * PI * prototype_tap / span;
        sinc * (0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos())
    };
    let total: f64 = (0..TAPS).map(weight).sum();
    std::array::from_fn(|tap| (weight(tap) / total) as f32)
});

/// Decodes G.711 μ-law at 8 kHz, as telephony carries it, into 16 kHz
/// linear PCM: `i16` samples, little-endian.
///
/// Unlike [`AudioStream`](crate::AudioStream) it is synchronous and holds
/// nothing back, so a caller can keep the audio in order with its other
/// events: each byte gives two samples at once, 3 ms late. Every sample is
/// kept and one is interpolated after it by a half-band low-pass at 4 kHz.
pub struct G711MuLawDecoder {
    /// The last `TAPS - 1` samples decoded, oldest first.
    history: Vec<f32>,
}

impl Default for G711MuLawDecoder {
    fn default() -> Self {
        Self {
            history: vec![0.0; TAPS - 1],
        }
    }
}

impl G711MuLawDecoder {
    /// The rate of the PCM it makes.
    pub const OUTPUT_SAMPLE_RATE: u32 = 16_000;

    /// The PCM for `codes`, the stream's next bytes.
    pub fn decode(&mut self, codes: &[u8]) -> Vec<u8> {
        let history = &mut self.history;
        history.extend(codes.iter().map(|&code| f32::from(decode_ulaw(code))));
        let mut pcm = Vec::with_capacity(4 * codes.len());
        for window in history.windows(TAPS) {
            let kept = window[TAPS - 1 - DELAY];
            let between: f32 = window
                .iter()
                .zip(WEIGHTS.iter())
                .map(|(sample, weight)| sample * weight)
                .sum();
            for sample in [kept, between] {
                let sample = sample
                    .round()
                    .clamp(f32::from(i16::MIN), f32::from(i16::MAX));
                pcm.extend_from_slice(&(sample as i16).to_le_bytes());
            }
        }
        history.drain(..history.len() - (TAPS - 1));
        pcm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_codec_algorithms::encode_ulaw;

    const INPUT_SAMPLE_RATE: u32 = 8_000;

    fn mu_law_tone(hz: f64, seconds: f64, amplitude: f64) -> Vec<u8> {
        let samples = (seconds * f64::from(INPUT_SAMPLE_RATE)) as usize;
        (0..samples)
            .map(|n| {
                let phase = 2.0 * PI * hz * n as f64 / f64::from(INPUT_SAMPLE_RATE);
                encode_ulaw((amplitude * phase.sin()).round() as i16)
            })
            .collect()
    }

    fn decoded(codes: &[u8]) -> Vec<f64> {
        G711MuLawDecoder::default()
            .decode(codes)
            .chunks_exact(2)
            .map(|pair| f64::from(i16::from_le_bytes([pair[0], pair[1]])))
            .collect()
    }

    /// The level of `samples` at `hz`, in dB relative to `amplitude`, by
    /// correlation over their middle, clear of the filter's start and end.
    fn level(samples: &[f64], hz: f64, amplitude: f64) -> f64 {
        let middle = &samples[samples.len() / 4..samples.len() * 3 / 4];
        let rate = f64::from(G711MuLawDecoder::OUTPUT_SAMPLE_RATE);
        let (mut real, mut imaginary) = (0.0, 0.0);
        for (n, sample) in middle.iter().enumerate() {
            let phase = 2.0 * PI * hz * n as f64 / rate;
            real += sample * phase.cos();
            imaginary += sample * phase.sin();
        }
        20.0 * (2.0 * real.hypot(imaginary) / middle.len() as f64 / amplitude).log10()
    }

    #[test]
    fn each_byte_gives_two_samples_at_once() {
        let mut decoder = G711MuLawDecoder::default();
        assert_eq!(decoder.decode(&[0xFF; 3]).len(), 3 * 2 * 2);
        assert!(decoder.decode(&[]).is_empty());
    }

    #[test]
    fn the_telephone_band_passes_and_its_image_does_not() {
        for hz in [1000.0, 3400.0] {
            let gain = level(&decoded(&mu_law_tone(hz, 1.0, 8000.0)), hz, 8000.0);
            assert!(gain.abs() < 0.2, "{hz} Hz: {gain} dB");
        }
        // Upsampling mirrors a tone around 4 kHz: 3 kHz to 5 kHz, which the
        // filter's window keeps far down.
        let image = level(&decoded(&mu_law_tone(3000.0, 1.0, 8000.0)), 5000.0, 8000.0);
        assert!(image < -60.0, "{image} dB");
    }

    #[test]
    fn the_output_does_not_depend_on_how_the_input_is_split() {
        let codes = mu_law_tone(440.0, 0.1, 12000.0);
        let whole = G711MuLawDecoder::default().decode(&codes);
        let mut decoder = G711MuLawDecoder::default();
        let mut split = Vec::new();
        let mut rest = codes.as_slice();
        for size in [1, 7, 0, 160, 3].into_iter().cycle() {
            if rest.is_empty() {
                break;
            }
            let (chunk, after) = rest.split_at(size.min(rest.len()));
            split.extend(decoder.decode(chunk));
            rest = after;
        }
        assert_eq!(split, whole);
    }

    #[test]
    fn the_filter_delays_the_audio_by_three_milliseconds() {
        let mut impulse = vec![encode_ulaw(0); 100];
        impulse[10] = encode_ulaw(16000);
        let heard = decoded(&impulse);
        let loudest = (0..heard.len())
            .max_by(|&a, &b| heard[a].abs().total_cmp(&heard[b].abs()))
            .unwrap();
        assert_eq!(loudest, 2 * (10 + DELAY));
    }
}
