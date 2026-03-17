# Concurrent real-time interface for multimedia engines

The `crime` crate is useful for real-time multimedia applications for providing
different output audio formats where the input is a stream of samples,
typically coming from a machine-learning model which generates a fixed amount
of samples in `fp32`.

This crate operates entirely on streams, provides delay compensation for each
supported operation, and thus eliminates the need for providing padding and/or
flushing the buffer. The first audio signal in the input is guaranteed to
correspond to the first audio signal in the output, and likewise for the last
signal.

Current functionalities:

* Input and outputs in `futures::Stream`.
* Resampling.
* Streaming audio codec (PCM, WAV, MP3, Ogg/Opus, WebM/Opus).
* Time scaling (using WSOLA).

## Feature flags

Each audio format is behind an optional feature flag. No formats are enabled by default.

| Feature | Format | Extra dependencies |
|---------|--------|--------------------|
| `pcm`   | Headerless PCM (linear + G.711 μ-law) | `audio-codec-algorithms`, `half` |
| `wav`   | WAV (linear PCM with header) | `half` |
| `mp3`   | MP3 | `mp3lame-encoder` |
| `ogg`   | Ogg/Opus | `opus`, `ogg`, `crc` |
| `webm`  | WebM/Opus | `opus` |

Example — enable MP3 and WAV only:

```toml
[dependencies]
crime = { version = "...", features = ["mp3", "wav"] }
```

## Supported codecs

* `pcm` — Headerless sample stream
  * Linear PCM (`i16`, `f16`, `f32`, all little-endian)
  * G.711 μ-law
* `wav` — WAV
  * Linear PCM (`i16`, `f16`, `f32`, all little-endian)
* `mp3` — MP3
* `ogg` — Ogg/Opus
* `webm` — WebM/Opus
