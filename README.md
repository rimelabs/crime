# Concurrent real-time interface for multimedia engines

The `crime` crate is useful for real-time multimedia applications for providing
different output audio formats where the input is a stream of samples,
typically coming from a machine-learning model which generates a fixed amount
of samples in `fp32`.

Concretely, the crate plans to support:

* Output codec in WAV and MP3 (more to be added).
* Resampling.
* Audio tempo changes.
