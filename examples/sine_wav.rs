/// Generate a 2-second sine wave at 440 Hz, then slow it down to 0.5x and encode it as WAV with
/// 16-bit integer linear PCM.
use std::f32::consts::TAU;
use std::io::{self, Write};
use std::time::Duration;

use crime::{AudioFormat, AudioStream, LinearPcmEncoding};
use futures::StreamExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parameters
    const INPUT_SAMPLE_RATE: u32 = 24_000;
    const OUTPUT_SAMPLE_RATE: u32 = 8_000;
    const DURATION: Duration = Duration::from_secs(2);
    const FREQUENCY_HZ: f32 = 440.0; // A4
    const TIME_SCALE: f32 = 0.5;

    let total_samples: usize = (INPUT_SAMPLE_RATE as f32 * DURATION.as_secs_f32()) as usize;

    let input_stream = futures::stream::unfold(0usize, move |n| async move {
        if n >= total_samples {
            None
        } else {
            let t = n as f32 / INPUT_SAMPLE_RATE as f32;
            let s = (TAU * FREQUENCY_HZ * t).sin();
            Some((s, n + 1))
        }
    });

    // Process through crime
    let audio_stream = AudioStream::new(INPUT_SAMPLE_RATE, input_stream);

    futures::executor::block_on(async move {
        let output_stream = audio_stream
            .commit(
                OUTPUT_SAMPLE_RATE,
                TIME_SCALE,
                AudioFormat::Wav(LinearPcmEncoding::Int16),
            )
            .await;

        let mut out = io::BufWriter::new(io::stdout());
        let mut chunks = output_stream.ready_chunks(4096);
        while let Some(chunk) = chunks.next().await {
            out.write_all(&chunk)?;
        }
        out.flush()?;
        Ok::<(), io::Error>(())
    })?;

    Ok(())
}
