use async_stream::stream;
use futures::StreamExt;
use futures::stream::Stream;
use realfft::RealFftPlanner;
use std::collections::VecDeque;
use std::pin::Pin;

fn hann_window(length: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; length];
    if length > 1 {
        for n in 0..length {
            w[n] =
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * (n as f32) / ((length - 1) as f32)).cos();
        }
    }
    w
}

fn reverse_weighted_product(dst: &mut [f32], weights_squared: &[f32], x: &[f32]) {
    // dst[i] = weights_squared[rev(i)] * x[rev(i)]
    let len = dst.len().min(weights_squared.len()).min(x.len());
    let mut i = 0;
    while i + 8 <= len {
        let r0 = len - 1 - i;
        let tmp = [
            weights_squared[r0] * x[r0],
            weights_squared[r0 - 1] * x[r0 - 1],
            weights_squared[r0 - 2] * x[r0 - 2],
            weights_squared[r0 - 3] * x[r0 - 3],
            weights_squared[r0 - 4] * x[r0 - 4],
            weights_squared[r0 - 5] * x[r0 - 5],
            weights_squared[r0 - 6] * x[r0 - 6],
            weights_squared[r0 - 7] * x[r0 - 7],
        ];
        dst[i..i + 8].copy_from_slice(&tmp);
        i += 8;
    }
    while i < len {
        let j = len - 1 - i;
        dst[i] = weights_squared[j] * x[j];
        i += 1;
    }
}

fn reverse_copy(dst: &mut [f32], src: &[f32]) {
    let len = dst.len().min(src.len());
    let mut i = 0;
    while i + 8 <= len {
        let r0 = len - 1 - i;
        let tmp = [
            src[r0],
            src[r0 - 1],
            src[r0 - 2],
            src[r0 - 3],
            src[r0 - 4],
            src[r0 - 5],
            src[r0 - 6],
            src[r0 - 7],
        ];
        dst[i..i + 8].copy_from_slice(&tmp);
        i += 8;
    }
    while i < len {
        dst[i] = src[len - 1 - i];
        i += 1;
    }
}

fn square_to(dst: &mut [f32], src: &[f32]) {
    let len = dst.len().min(src.len());
    let mut i = 0;
    while i + 8 <= len {
        let mut tmp = [0.0f32; 8];
        for j in 0..8 {
            let v = src[i + j];
            tmp[j] = v * v;
        }
        dst[i..i + 8].copy_from_slice(&tmp);
        i += 8;
    }
    while i < len {
        dst[i] = src[i] * src[i];
        i += 1;
    }
}

async fn fill_until_abs(
    samples: &mut Pin<Box<dyn Stream<Item = f32> + Send + '_>>,
    buffer: &mut Vec<f32>,
    base: usize,
    needed_abs_len: usize,
) -> bool {
    // Returns true if EOF reached before condition satisfied
    while base + buffer.len() < needed_abs_len {
        match samples.next().await {
            Some(s) => buffer.push(s),
            None => return true,
        }
    }
    false
}

// Public WSOLA entry point
pub(crate) async fn time_scale<'a>(
    mut samples: Pin<Box<dyn Stream<Item = f32> + Send + 'a>>,
    time_scale_factor: f32,
    sample_rate: u32,
) -> Pin<Box<dyn Stream<Item = f32> + Send + 'a>> {
    Box::pin(stream! {
        const FRAME_DURATION_S: f32 = 0.030;
        const OVERLAP_FACTOR: f32 = 0.75;
        const NCCF_EARLY_EXIT_THRESHOLD: f32 = 0.985;
        const COMPACTION_THRESHOLD_SAMPLES: usize = 1 << 16;

        let frame_size = (sample_rate as f32 * FRAME_DURATION_S).round() as usize;
        let mut overlap = ((frame_size as f32) * OVERLAP_FACTOR).round() as usize;
        if overlap >= frame_size { overlap = frame_size.saturating_sub(1); }
        let synth_hop = frame_size - overlap;
        let analysis_hop = ((synth_hop as f32) * time_scale_factor).round() as usize;

        // Adaptive search window
        let min_delta = (sample_rate as f32 * 0.015).round() as usize;
        let mut delta = ((analysis_hop as f32) * 0.75).round() as usize;
        if delta < min_delta { delta = min_delta; }
        if delta > overlap - 1 { delta = overlap.saturating_sub(1); }

        // Windows
        let hann = hann_window(overlap);
        let fade_in = &hann[..];
        let fade_out: Vec<f32> = hann.iter().map(|v| 1.0 - *v).collect();

        // FFT setup
        let b_long_len = overlap + 2 * delta;
        let conv_len = overlap + b_long_len - 1;
        let n_fft = conv_len.next_power_of_two();
        let mut fft_planner = RealFftPlanner::<f32>::new();
        let r2c = fft_planner.plan_fft_forward(n_fft);
        let c2r = fft_planner.plan_fft_inverse(n_fft);
        let mut a_time = vec![0.0f32; n_fft];
        let mut b_time = vec![0.0f32; n_fft];
        let mut a_freq = r2c.make_output_vec();
        let mut b_freq = r2c.make_output_vec();
        let mut conv_freq = r2c.make_output_vec();
        let mut conv_time = vec![0.0f32; n_fft];
        let mut scratch_fwd_a = r2c.make_scratch_vec();
        let mut scratch_fwd_b = r2c.make_scratch_vec();
        let mut scratch_inv = c2r.make_scratch_vec();
        let inv_scale = 1.0f32 / (n_fft as f32);

        // NCCF weights
        let w2: Vec<f32> = hann.iter().map(|v| v * v).collect();
        let mut w2_rev_time = vec![0.0f32; n_fft];
        reverse_copy(&mut w2_rev_time[..overlap], &w2[..overlap]);
        let mut w2_rev_freq = r2c.make_output_vec();
        r2c.process_with_scratch(&mut w2_rev_time, &mut w2_rev_freq, &mut scratch_fwd_a).expect("FFT processing should not fail with correctly sized buffers");
        // Ey buffers
        let mut b_sq_time = vec![0.0f32; n_fft];
        let mut b_sq_freq = r2c.make_output_vec();
        let mut conv_sq_freq = r2c.make_output_vec();
        let mut conv_sq_time = vec![0.0f32; n_fft];

        // Input buffer and synthesis
        let mut in_buf: Vec<f32> = Vec::new();
        let mut base: usize = 0;
        let mut in_eof = false;
        let mut synth: VecDeque<f32> = VecDeque::new();
        let mut prev_frame_start: usize;

        // First frame
        let need_first = frame_size;
        let _ = fill_until_abs(&mut samples, &mut in_buf, base, need_first).await;
        if base + in_buf.len() < frame_size {
            for &s in &in_buf { yield s; }
            return;
        }
        let first = &in_buf[0..frame_size];
        for &s in first { synth.push_back(s); }
        for _ in 0..synth_hop { if let Some(v) = synth.pop_front() { yield v; } }
        prev_frame_start = 0;
        let mut next_analysis_pos: usize = prev_frame_start + analysis_hop;

        // Main loop
        'outer: loop {
            let cur_analysis_pos = next_analysis_pos;
            let search_start = cur_analysis_pos.saturating_sub(delta);
            let need = search_start + overlap + 2 * delta + (frame_size - overlap);
            if !in_eof && base + in_buf.len() < need {
                if fill_until_abs(&mut samples, &mut in_buf, base, need).await { in_eof = true; }
            }
            if base + in_buf.len() < search_start + overlap + 2 * delta { break 'outer; }

            let prev_tail = {
                let start = prev_frame_start + frame_size - overlap;
                if base + in_buf.len() < start + overlap { break 'outer; }
                &in_buf[start - base .. start - base + overlap]
            };

            let avail_abs = base + in_buf.len();
            let b_long_end = (search_start + overlap + 2 * delta).min(avail_abs);
            if b_long_end <= search_start { break 'outer; }
            let b_long = &in_buf[search_start - base .. b_long_end - base];
            if b_long.len() < overlap + 2*delta { break 'outer; }

            // Ex
            let ex: f32 = prev_tail.iter().zip(w2.iter()).map(|(x, w)| w * (*x) * (*x)).sum();
            let eps = 1e-12f32;

            // Numerator
            for v in a_time.iter_mut() { *v = 0.0; }
            for v in b_time.iter_mut() { *v = 0.0; }
            reverse_weighted_product(&mut a_time[..overlap], &w2[..overlap], prev_tail);
            b_time[..b_long.len()].copy_from_slice(b_long);
            r2c.process_with_scratch(&mut a_time, &mut a_freq, &mut scratch_fwd_a).unwrap();
            r2c.process_with_scratch(&mut b_time, &mut b_freq, &mut scratch_fwd_b).unwrap();
            for i in 0..a_freq.len() { conv_freq[i] = a_freq[i] * b_freq[i]; }
            c2r.process_with_scratch(&mut conv_freq, &mut conv_time, &mut scratch_inv).unwrap();
            for v in conv_time.iter_mut() { *v *= inv_scale; }

            // Ey
            for v in b_sq_time.iter_mut() { *v = 0.0; }
            square_to(&mut b_sq_time[..b_long.len()], b_long);
            r2c.process_with_scratch(&mut b_sq_time, &mut b_sq_freq, &mut scratch_fwd_b).unwrap();
            for i in 0..w2_rev_freq.len() { conv_sq_freq[i] = w2_rev_freq[i] * b_sq_freq[i]; }
            c2r.process_with_scratch(&mut conv_sq_freq, &mut conv_sq_time, &mut scratch_inv).unwrap();
            for v in conv_sq_time.iter_mut() { *v *= inv_scale; }

            // Select best k
            let corr_base = overlap - 1;
            let mut best_k = 0usize;
            let mut best_val = std::f32::MIN;
            for k in 0..=2*delta {
                let idx = corr_base + k;
                if idx >= conv_time.len() || idx >= conv_sq_time.len() { break; }
                let num = conv_time[idx];
                let ey = conv_sq_time[idx];
                let denom = (ex * ey + eps).sqrt();
                let nccf = if denom > 0.0 { num / denom } else { 0.0 };
                if nccf > best_val {
                  best_val = nccf;
                  best_k = k;
                  if best_val >= NCCF_EARLY_EXIT_THRESHOLD { break; }
                }
            }

            let best_input_start = search_start + best_k;
            let need_frame_end = best_input_start + frame_size;
            if !in_eof && base + in_buf.len() < need_frame_end {
                if fill_until_abs(&mut samples, &mut in_buf, base, need_frame_end).await { in_eof = true; }
            }
            if base + in_buf.len() < need_frame_end { break 'outer; }
            let candidate = &in_buf[best_input_start - base .. best_input_start - base + frame_size];

            // OLA
            while synth.len() < overlap { synth.push_back(0.0); }
            for n in 0..overlap {
                let prev_idx = synth.len() - overlap + n;
                let blended = synth[prev_idx] * fade_out[n] + candidate[n] * fade_in[n];
                if let Some(slot) = synth.get_mut(prev_idx) { *slot = blended; }
            }
            for &s in &candidate[overlap..] { synth.push_back(s); }
            for _ in 0..synth_hop { if let Some(v) = synth.pop_front() { yield v; } }

            // Advance and compact
            prev_frame_start = best_input_start;
            next_analysis_pos = next_analysis_pos + analysis_hop;
            let desired_base = prev_frame_start.saturating_sub(frame_size * 2);
            if desired_base > base {
                let pending_drop = desired_base - base;
                if pending_drop >= COMPACTION_THRESHOLD_SAMPLES && pending_drop > in_buf.len() / 2 {
                    in_buf.drain(0..pending_drop);
                    base = desired_base;
                }
            }
        }

        while let Some(v) = synth.pop_front() { yield v; }
    })
}
