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
* Audio codec (PCM, WAV, MP3).

There are also concrete plans to support:

* Tempo adjustment for speech signals.
