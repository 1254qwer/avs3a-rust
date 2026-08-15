# avs3a-rust

AVS3-P3（AV3A / 臻音全景声）解码器的 Rust 实现。

项目提供裸 `.av3a` elementary stream 的增量解析、CRC 校验、metadata 解析和
PCM16 解码，并以 `#![forbid(unsafe_code)]` 保持核心解码路径不使用 `unsafe`。

> 当前处于开发阶段，公开 API 仍可能调整。不要将未经回归验证的输出用于生产环境。

## 支持情况

| Profile | 当前支持 | 验证情况 |
|---|---|---|
| Channel-based mono | 完整解码链路 | 构造码流与 C 模块对照；缺少公开真实长流 |
| Stereo | ordinary MS/ILD、24/32 kbps MCR | 构造码流端到端对照 |
| Channel-based MC | 4.0、5.1、7.1、5.1.2、5.1.4、7.1.2、7.1.4 | 多份真实 7.1.4 长流回归 |
| Mix | 纯对象、声床加对象 | 多份真实 stereo/5.1/7.1.4 声床加对象长流回归 |
| HOA | 一至三阶，输出 4/9/16 通道 | 构造码流端到端对照；缺少真实长流 |

所有 profile 均已接通 metadata、神经频谱解码、coupling/inverse-DMX、
BWE/TNS/FD/LFE、IMDCT/overlap-add 和 PCM16 输出。解码器不会以静音或伪造
PCM 代替尚未支持的配置；损坏 payload、CRC 失败和流中配置变化会返回错误。

## 快速开始

需要 Rust 1.96 或更高版本。

```bash
cargo build --release
cargo test --all-targets --locked

# 解码为交错 PCM16 WAV；自动选择 mono/stereo/MC/Mix/HOA backend
cargo run --release --bin avs3a-decode -- input.av3a output.wav

# 只解码前 100 帧
cargo run --release --bin avs3a-decode -- input.av3a output.wav --frames 100

# 检查裸码流和 CRC，不运行音频合成
cargo run --release --bin avs3a-inspect -- --verify-crc input.av3a
```

## 命令行工具

| 工具 | 用途 |
|---|---|
| `avs3a-decode` | 解码 channel-based、Mix 或 HOA 流并写出 PCM16 WAV |
| `avs3a-inspect` | 统计布局、码率、帧数、CRC、重同步和配置变化；`--mc-side-info` 可输出 MC side info |
| `avs3a-wav-compare` | 以常量内存比较两个同格式 PCM16 WAV，报告逐声道误差 |
| `avs3a-model-inspect` | 检查 hyper-prior/VAE 神经模型及确定性 CNN fingerprint |
| `avs3a-mono-decode` | 仅接受 channel-based mono 的窄接口 |

查看各工具的完整参数：

```bash
cargo run --release --bin avs3a-decode -- --help
cargo run --release --bin avs3a-inspect -- --help
```

## Rust API

一次性查找并解析首个完整帧头：

```rust
let info = avs3a::parse_header(bytes)?;
println!("{} Hz, {} channels", info.header.sample_rate, info.header.channels);
# Ok::<(), avs3a::HeaderError>(())
```

网络或文件分片可交给增量解析器；它能从前导垃圾或伪同步字恢复：

```rust
let mut parser = avs3a::FrameStream::new();

for event in parser.push(chunk)? {
    match event {
        avs3a::StreamEvent::Frame(frame) => {
            if frame.crc_is_valid() {
                consume(frame);
            }
        }
        avs3a::StreamEvent::Skipped { bytes } => log_resync(bytes),
    }
}
# Ok::<(), avs3a::StreamError>(())
```

PCM 解码使用 `PendingDecoder<B>` 配置具体 backend。`Decoder::decode()` 返回拥有
样本的 `AudioFrame`；长时间流式解码应使用 `decode_into()` 复用调用方缓冲区。
可用 backend 包括：

- `MonoDecoderBackend`
- `StereoDecoderBackend`（自动区分 ordinary 与 MCR）
- `McDecoderBackend`
- `MixDecoderBackend`
- `HoaDecoderBackend`

`decode_into()` 要求输出长度严格等于 `channels * samples_per_channel`，并在推进
解码状态前校验 CRC 和配置连续性。自动按 profile 选择 backend 的完整示例见
[`src/bin/avs3a-decode.rs`](src/bin/avs3a-decode.rs)。

## 实现特点

- MSB-first 有界位读取和增量 framing，可接受任意输入分片并限制重同步缓冲增长
- static Basic L1、VR extension L1、dynamic L1/L2 metadata 的公开值模型
- Range Coder、反量化、Main/LC 神经 QC 及内置 hyper-prior 模型
- 基于 `rustfft` 的 256/1024/2048 点 AVS3 MDCT/IMDCT
- decoder 私有的 PRNG、工作区、FFT plan 和多声道线程池；稳态逐帧路径避免分配
- 显式 little-endian PCM16 WAV 写出和 RIFF chunk 感知的 WAV 比较器
- 对模型维度、索引、码率、布局、对象数、截断数据及分配上限进行运行时检查

内置模型和规范表位于 [`assets/`](assets/)，其来源、尺寸与 SHA-256 记录在
[`assets/README.md`](assets/README.md)。

## 验证与限制

实现使用 C 参考解码器进行逐模块和端到端交叉验证。目前真实流回归覆盖三份
channel-based 7.1.4 流，以及 stereo、5.1、7.1.4 声床附带对象的 Mix 流；这些
样例均完成全流解码，帧数、声道数、采样率和输出长度一致。

`rustfft` 与 C 标量 FFT 的浮点累加顺序不同，因此完整 PCM 不承诺逐位一致。
现有真实流回归中的差异集中在 1 LSB，已知基线最大为 2 LSB，且未观察到状态漂移。
不同架构的 FFT 路径使用独立的精确测试 fingerprint。

当前主要验证缺口：

- 真实 mono 长流
- 真实纯对象 Mix 长流
- 真实 HOA 长流
- 7.1.4 以外更多 channel-based MC 布局与码率

## 开发与性能分析

```bash
cargo fmt --check
cargo test --all-targets --locked
cargo build --release
```

仓库提供保留调试符号的 `profiling` profile。Linux 上可用
[`cargo-flamegraph`](https://github.com/flamegraph-rs/flamegraph) 分析足够长的真实输入：

```bash
cargo install flamegraph --locked
cargo flamegraph --profile profiling --bin avs3a-decode -- \
  input.av3a /tmp/avs3a-profile.wav --frames 2000
```

三路及以上的 MC/Mix core 使用 decoder 私有 Rayon 线程池。默认 worker 数为可用
逻辑 CPU 的一半且最多 8 个，可在构造 decoder 前通过 `RAYON_NUM_THREADS` 覆盖：

```bash
RAYON_NUM_THREADS=8 cargo run --release --bin avs3a-decode -- input.av3a output.wav
```

CI 覆盖 Linux、macOS 和 Windows。crate 的 Rust 代码启用 `unsafe_code = "forbid"`。

## 许可证

[MIT License](LICENSE)
