use crate::{OpusApplication, OpusBitrate, webm};
use futures::{Stream, StreamExt};
use opus::Channels as OpusChannels;
use std::pin::Pin;

pub struct OpusPacket {
    pub data: Vec<u8>,
    pub frame_size_samples: usize,
}

pub async fn encode_opus_stream<'a>(
    samples: impl Stream<Item = f32> + Send + 'a,
    sample_rate: u32,
    application: OpusApplication,
    bitrate: OpusBitrate,
) -> impl Stream<Item = Result<OpusPacket, opus::Error>> + Send + 'a {
    // Opus supports 2.5, 5, 10, 20, 40, 60 ms.
    // We use 20ms as standard frame size.
    let frame_size = (sample_rate as usize * 20) / 1000;
    let samples = Box::pin(samples);
    let mut sample_chunks = samples.chunks(frame_size);

    async_stream::stream! {
        // Initialize encoder inside the stream so errors can be yielded
        let mut encoder = match opus::Encoder::new(sample_rate, OpusChannels::Mono, application) {
            Ok(enc) => enc,
            Err(e) => {
                yield Err(e);
                return;
            }
        };

        if let Err(e) = encoder.set_bitrate(bitrate) {
            yield Err(e);
            return;
        }

        while let Some(chunk) = sample_chunks.next().await {
            let chunk: Vec<f32> = chunk.into_iter().collect();
            let mut output = [0u8; 4000];

            // Handle partial frames by padding
            let encode_result = if chunk.len() == frame_size {
                encoder.encode_float(&chunk, &mut output)
            } else {
                let mut padded = chunk;
                padded.resize(frame_size, 0.0);
                encoder.encode_float(&padded, &mut output)
            };

            match encode_result {
                Ok(len) => {
                    yield Ok(OpusPacket {
                        data: output[..len].to_vec(),
                        frame_size_samples: frame_size,
                    });
                }
                Err(e) => {
                    yield Err(e);
                    return;
                }
            }
        }
    }
}

const OPUS_HEAD_MAGIC: &[u8] = b"OpusHead";
const OPUS_TAGS_MAGIC: &[u8] = b"OpusTags";

pub struct OpusHeader {
    pub version: u8,
    pub channels: u8,
    pub pre_skip: u16,
    pub input_sample_rate: u32,
    pub output_gain: i16,
    pub channel_mapping_family: u8,
    // Optional: channel mapping table
}

impl OpusHeader {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(OPUS_HEAD_MAGIC);
        bytes.push(self.version);
        bytes.push(self.channels);
        bytes.extend_from_slice(&self.pre_skip.to_le_bytes());
        bytes.extend_from_slice(&self.input_sample_rate.to_le_bytes());
        bytes.extend_from_slice(&self.output_gain.to_le_bytes());
        bytes.push(self.channel_mapping_family);
        // Mapping family 0 implies mono or stereo (L, R) - no table needed
        bytes
    }
}

pub fn make_opus_comment_header() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(OPUS_TAGS_MAGIC);

    // Vendor String Length (u32 le)
    let vendor = "rust-crime-crate";
    bytes.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    bytes.extend_from_slice(vendor.as_bytes());

    // User Comment List Length (u32 le) - 0 for now
    bytes.extend_from_slice(&0u32.to_le_bytes());

    bytes
}

pub async fn encode_opus_as_ogg<'a>(
    samples: impl Stream<Item = f32> + Send + 'a,
    sample_rate: u32,
    application: OpusApplication,
    bitrate: OpusBitrate,
) -> Pin<Box<dyn Stream<Item = u8> + Send + 'a>> {
    let opus_packets = encode_opus_stream(samples, sample_rate, application, bitrate).await;
    let mut opus_packets = Box::pin(opus_packets);

    Box::pin(async_stream::stream! {
        // Create Ogg Packet Writer manually to avoid stream/RefCell borrow checker/ICE issues.
        // We will just construct the pages ourselves.
        // ID Header Page
        let id_header = OpusHeader {
            version: 1,
            channels: 1,
            pre_skip: 0,
            input_sample_rate: sample_rate,
            output_gain: 0,
            channel_mapping_family: 0,
        };
        let id_packet = id_header.to_bytes();
        // Ogg CRC algorithm:
        // Width=32, Poly=0x04C11DB7, Init=0, RefIn=False, RefOut=False, XorOut=0
        // This corresponds to CRC_32_MPEG_2 in some catalogs, or we can define it.
        // crc crate's CRC_32_ISO_HDLC is reflected (poly reversed 0xEDB88320 effectively).
        // We need non-reflected 0x04C11DB7.
        // Let's use custom definition to be safe.
        const OGG_CRC: crc::Algorithm<u32> = crc::Algorithm {
            width: 32,
            poly: 0x04c11db7,
            init: 0,
            refin: false,
            refout: false,
            xorout: 0,
            check: 0,
            residue: 0,
        };

        let mut id_page = Vec::new();
        id_page.extend_from_slice(b"OggS");
        id_page.push(0); // version
        id_page.push(0x02); // type: BOS
        id_page.extend_from_slice(&0u64.to_le_bytes()); // granule pos
        id_page.extend_from_slice(&1u32.to_le_bytes()); // serial
        id_page.extend_from_slice(&0u32.to_le_bytes()); // sequence
        id_page.extend_from_slice(&0u32.to_le_bytes()); // checksum placeholder
        id_page.push(1); // segments
        id_page.push(id_packet.len() as u8); // segment table (assuming packet < 255 bytes)
        id_page.extend_from_slice(&id_packet);

        // Calculate CRC
        // Ogg CRC checksum is calculated over the entire page with the checksum field set to 0.
        let crc = crc::Crc::<u32>::new(&OGG_CRC).checksum(&id_page);
        id_page[22..26].copy_from_slice(&crc.to_le_bytes());

        for byte in &id_page { yield *byte; }

        // Comment Header Page
        let comment_packet = make_opus_comment_header();
        let mut comment_page = Vec::new();
        comment_page.extend_from_slice(b"OggS");
        comment_page.push(0);
        comment_page.push(0); // type: Normal (continuation not needed for small comment)
        comment_page.extend_from_slice(&0u64.to_le_bytes()); // granule pos
        comment_page.extend_from_slice(&1u32.to_le_bytes()); // serial
        comment_page.extend_from_slice(&1u32.to_le_bytes()); // sequence
        comment_page.extend_from_slice(&0u32.to_le_bytes()); // checksum
        comment_page.push(1);
        comment_page.push(comment_packet.len() as u8); // segment (assuming < 255)
        comment_page.extend_from_slice(&comment_packet);

        let crc = crc::Crc::<u32>::new(&OGG_CRC).checksum(&comment_page);
        comment_page[22..26].copy_from_slice(&crc.to_le_bytes());

        for byte in &comment_page { yield *byte; }

        // Audio Pages
        let mut granule_position: u64 = 0;
        let mut page_sequence: u32 = 2;

        while let Some(packet_result) = opus_packets.next().await {
            let packet = match packet_result {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Opus encoding error: {:?}", e);
                    return;
                }
            };
            let opus_data = packet.data;
            granule_position += packet.frame_size_samples as u64;

            // Construct Audio Page (one packet per page for simplicity)
            let mut page = Vec::new();
            page.extend_from_slice(b"OggS");
            page.push(0);
            page.push(0); // type: Normal (or End of Stream if last?)
            // We don't know if it's last easily here without peeking or checking stream end.
            // Let's assume normal. If sample_chunks.is_done() ... tricky with async iterator.

            page.extend_from_slice(&granule_position.to_le_bytes());
            page.extend_from_slice(&1u32.to_le_bytes()); // serial
            page.extend_from_slice(&page_sequence.to_le_bytes());
            page.extend_from_slice(&0u32.to_le_bytes()); // checksum
            page.push(1); // segments

            // Lacing values for packet
            let mut len_remaining = opus_data.len();
            while len_remaining >= 255 {
                page.push(255);
                len_remaining -= 255;
            }
            page.push(len_remaining as u8);

            // Update segment count in header (offset 26)
            // But we already wrote it as 1... wait.
            // If packet > 255, we have multiple segments for ONE packet.
            let num_segments = page.len() - 27; // 27 is header start to segments count (26) + 1
            page[26] = num_segments as u8;

            page.extend_from_slice(&opus_data);

            let crc = crc::Crc::<u32>::new(&OGG_CRC).checksum(&page);
            page[22..26].copy_from_slice(&crc.to_le_bytes());

            for byte in &page { yield *byte; }

            page_sequence += 1;
        }
    })
}

pub fn make_webm_header() -> Vec<u8> {
    let mut header = Vec::new();
    header.extend_from_slice(&webm::make_string_element(0x4286, "webm")); // EBMLDocType
    header.extend_from_slice(&webm::make_uint_element(0x42F7, 1)); // EBMLReadVersion
    header.extend_from_slice(&webm::make_uint_element(0x4287, 1)); // EBMLDocTypeVersion
    webm::make_element(webm::EBML_ID, &header)
}

pub fn make_segment_header() -> Vec<u8> {
    let mut segment = Vec::new();
    segment.extend_from_slice(&webm::encode_id(webm::SEGMENT_ID));
    // Unknown size (all 1s, 8 bytes width -> 0x01FFFFFFFFFFFFFF)
    let unknown_size = [0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
    segment.extend_from_slice(&unknown_size);
    segment
}

pub fn make_info_element() -> Vec<u8> {
    let mut info = Vec::new();
    info.extend_from_slice(&webm::make_uint_element(webm::TIMECODE_SCALE_ID, 1_000_000)); // 1ms
    info.extend_from_slice(&webm::make_string_element(webm::MUXING_APP_ID, "crime"));
    info.extend_from_slice(&webm::make_string_element(webm::WRITING_APP_ID, "crime"));
    webm::make_element(webm::INFO_ID, &info)
}

pub fn make_tracks_element(sample_rate: u32) -> Vec<u8> {
    let mut tracks = Vec::new();
    let mut track_entry = Vec::new();
    track_entry.extend_from_slice(&webm::make_uint_element(webm::TRACK_NUMBER_ID, 1));
    track_entry.extend_from_slice(&webm::make_uint_element(webm::TRACK_UID_ID, 1));
    track_entry.extend_from_slice(&webm::make_uint_element(webm::TRACK_TYPE_ID, 2)); // Audio
    track_entry.extend_from_slice(&webm::make_string_element(webm::CODEC_ID_ID, "A_OPUS"));

    // CodecDelay (Opus Pre-Skip)
    // 6.5ms = 6,500,000ns
    // 312 samples at 48k.
    let pre_skip = ((sample_rate as u64 * 312) / 48000) as u16;
    let codec_delay = 6_500_000u64; // 6.5ms in ns
    let seek_pre_roll = 80_000_000u64; // 80ms in ns

    track_entry.extend_from_slice(&webm::make_uint_element(webm::CODEC_DELAY_ID, codec_delay));
    track_entry.extend_from_slice(&webm::make_uint_element(
        webm::SEEK_PRE_ROLL_ID,
        seek_pre_roll,
    ));

    // CodecPrivate: OpusHead
    let opus_head = OpusHeader {
        version: 1,
        channels: 1,
        pre_skip,
        input_sample_rate: sample_rate,
        output_gain: 0,
        channel_mapping_family: 0,
    };
    track_entry.extend_from_slice(&webm::make_element(
        webm::CODEC_PRIVATE_ID,
        &opus_head.to_bytes(),
    ));

    // Audio
    let mut audio = Vec::new();
    audio.extend_from_slice(&webm::make_float_element(
        webm::SAMPLING_FREQUENCY_ID,
        sample_rate as f32,
    ));
    audio.extend_from_slice(&webm::make_uint_element(webm::CHANNELS_ID, 1));
    track_entry.extend_from_slice(&webm::make_element(webm::AUDIO_ID, &audio));

    tracks.extend_from_slice(&webm::make_element(webm::TRACK_ENTRY_ID, &track_entry));

    webm::make_element(webm::TRACKS_ID, &tracks)
}

pub async fn encode_opus_as_webm<'a>(
    samples: impl Stream<Item = f32> + Send + 'a,
    sample_rate: u32,
    application: OpusApplication,
    bitrate: OpusBitrate,
) -> Pin<Box<dyn Stream<Item = u8> + Send + 'a>> {
    let opus_packets = encode_opus_stream(samples, sample_rate, application, bitrate).await;
    let mut opus_packets = Box::pin(opus_packets);

    Box::pin(async_stream::stream! {
        // 1. EBML Header
        let ebml_header = make_webm_header();
        for byte in ebml_header { yield byte; }

        // 2. Segment (Unknown Size)
        let segment_header = make_segment_header();
        for byte in segment_header { yield byte; }

        // 3. Info
        let info_el = make_info_element();
        for byte in info_el { yield byte; }

        // 4. Tracks
        let tracks_el = make_tracks_element(sample_rate);
        for byte in tracks_el { yield byte; }

        // 5. Clusters
        let mut cluster_timecode = 0u64; // Absolute time
        let mut current_cluster_start = 0u64;
        let mut cluster_data = Vec::new();

        // Start first cluster
        cluster_data.extend_from_slice(&webm::make_uint_element(webm::TIMECODE_ID, 0));

        while let Some(packet_result) = opus_packets.next().await {
            let packet = match packet_result {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Opus encoding error: {:?}", e);
                    return;
                }
            };
            let opus_data = packet.data;
            // Frame duration 20ms
            let block_duration = 20u64;

            // Check if cluster is full (e.g. >= 1000ms)
            if cluster_timecode - current_cluster_start >= 1000 {
                 // Flush
                 let cluster_el = webm::make_element(webm::CLUSTER_ID, &cluster_data);
                 for byte in cluster_el { yield byte; }

                 current_cluster_start = cluster_timecode;
                 cluster_data.clear();
                 cluster_data.extend_from_slice(&webm::make_uint_element(webm::TIMECODE_ID, current_cluster_start));
            }

            let relative_tc = (cluster_timecode - current_cluster_start) as i16;

            cluster_data.extend_from_slice(&webm::make_simple_block(1, relative_tc, &opus_data));

            cluster_timecode += block_duration;
        }

        // Flush last cluster
        if cluster_data.len() > 10 { // Ensure we have data beyond just Timecode
             let cluster_el = webm::make_element(webm::CLUSTER_ID, &cluster_data);
             for byte in cluster_el { yield byte; }
        }
    })
}
