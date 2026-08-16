# Real AVS3 fixture

`test.av3a` is a 21 MiB, 8701-frame, 12-channel 7.1.4 AVS3 sample used by
the CI real-sample timing job. It is decoded to `/dev/null`, so the reported
time measures decoder work rather than filesystem output speed.

- Sample rate: 44.1 kHz
- Bitrate: 832 kbps
- Duration: about 202 seconds
- SHA-256: `e8648fe7a67fdafe94bf6d9653d510a6f6a574aa293a24e29e8fc257d6e01574`
