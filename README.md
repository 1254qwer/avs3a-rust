# avs3a-rust

AVS3-P3（AV3A / 臻音全景声）解码器的 Rust 实现。

项目提供裸 `.av3a` elementary stream 的增量解析、MP4/M4A 中 AV3A 音轨的按需读取、
CRC 校验、metadata 解析，以及交错 PCM16 / 浮点 PCM 解码。核心解码路径以
`#![forbid(unsafe_code)]` 保持不使用 `unsafe`。

> 当前处于开发阶段，公开 API 仍可能调整。不要将未经回归验证的输出用于生产环境。

## 使用文档

完整用法从 [使用指南](<docs/README.md>) 开始，按任务分为：

- [命令行工具](<docs/cli.md>)：全部参数、默认值、输入限制与退出状态。
- [Rust 库使用](<docs/library.md>)：自动 backend、裸流分片、PCM/float、metadata、错误和线程。
- [MP4/M4A 与播放器集成](<docs/mp4.md>)：音轨索引、时间戳、seek 预热、分段下载与 N-API 封装约定。
- 可运行示例：[裸流转 WAV](<examples/decode_raw.rs>)、[MP4 转 WAV 与 seek](<examples/decode_mp4.rs>)。

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

# M4A/MP4 中的 AV3A 音轨也可直接解码；按文件内容识别
cargo run --release --bin avs3a-decode -- input.m4a output.wav

# 只解码前 100 帧
cargo run --release --bin avs3a-decode -- input.av3a output.wav --frames 100

# 检查裸码流和 CRC，不运行音频合成；也接受 MP4/M4A
cargo run --release --bin avs3a-inspect -- --verify-crc input.av3a
```

## 命令行工具

| 工具 | 用途 |
|---|---|
| `avs3a-decode` | 从裸流或 MP4/M4A 解码 channel-based、Mix 或 HOA 并写出 PCM16 WAV |
| `avs3a-inspect` | 统计布局、码率、帧数、CRC、重同步和配置变化；`--mc-side-info` 可输出 MC side info |
| `avs3a-wav-compare` | 以常量内存比较两个同格式 PCM16 WAV，报告逐声道误差 |
| `avs3a-model-inspect` | 检查 hyper-prior/VAE 神经模型及确定性 CNN fingerprint |
| `avs3a-mono-decode` | 裸流 mono 窄接口；也接受单对象纯对象 Mix，详见 [CLI 文档](<docs/cli.md>) |

完整说明见 [命令行指南](<docs/cli.md>)，也可查看帮助：

```bash
cargo run --release --bin avs3a-decode -- --help
cargo run --release --bin avs3a-inspect -- --help
```

## Rust API

常规应用优先使用 `BuiltinDecoder`，它按帧头自动选择 backend；不必自己在五种 backend 之间分派。
例如读取 M4A/MP4 中的 AV3A 帧并复用 PCM16 输出缓冲区：

~~~rust
use std::fs::File;
use avs3a::{BuiltinDecoder, Mp4FrameReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = Mp4FrameReader::open(File::open("input.m4a")?)?;
    let Some(first) = reader.next_frame()? else { return Ok(()); };
    let mut decoder = BuiltinDecoder::configure(first.header())?;
    let mut pcm = vec![0_i16; decoder.sample_count()?];
    decoder.decode_into(&first, &mut pcm)?;
    // 在这里消费第一帧 pcm，例如送到音频队列或 WavWriter。
    while let Some(frame) = reader.next_frame()? {
        decoder.decode_into(&frame, &mut pcm)?;
        // 消费本帧 pcm；下一次解码会覆盖同一个缓冲区。
    }
    Ok(())
}
~~~

裸流使用 `FrameStream::push` 处理任意分片，并在 EOF 调用 `finish`；完整程序见
[裸流示例](<examples/decode_raw.rs>)。参数、错误类型的实际导入路径，以及手动 backend 用法见
[Rust 库指南](<docs/library.md>)；完整 API 签名可通过 `cargo doc --no-deps --open` 查看。

`decode_into` 要求长度严格等于 `sample_count()`，输出交错 PCM16。
需要增益、混音或渲染时可用 `decode_into_f32`，注意它使用 PCM16 幅度单位，
要除以 `avs3a::decode::FLOAT_FULL_SCALE` 才是常见的归一化浮点值。
MP4 的声明声道数只是提示，应以帧头配置为准；对象/HOA 输出也不等于已经渲染好的扬声器信号。
Seek 必须同时处理读取位置、解码器 reset 和预热，详见 [MP4/M4A 指南](<docs/mp4.md>)。

## 实现特点

- MSB-first 有界位读取和增量 framing，可接受任意输入分片并限制重同步缓冲增长
- static Basic L1、VR extension L1、dynamic L1/L2 metadata 的公开值模型
- Range Coder、反量化、Main/LC 神经 QC 及内置 hyper-prior 模型
- 基于 `rustfft` 的 256/1024/2048 点 AVS3 MDCT/IMDCT
- decoder 私有的 PRNG、工作区、FFT plan 和多声道线程池；稳态逐帧路径避免分配
- 显式 little-endian PCM16 WAV 写出和 RIFF chunk 感知的 WAV 比较器
- 对模型维度、索引、码率、布局、对象数、截断数据及分配上限进行运行时检查

内置模型和规范表位于 [assets](<assets/>)，其来源、尺寸与 SHA-256 记录在
[assets 说明](<assets/README.md>)。

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
逻辑 CPU 数的一半向上取整且最多 8 个，可在构造 decoder 前通过正整数 `RAYON_NUM_THREADS` 覆盖：

```bash
RAYON_NUM_THREADS=8 cargo run --release --bin avs3a-decode -- input.av3a output.wav
```

CI 覆盖 Linux、macOS 和 Windows。crate 的 Rust 代码启用 `unsafe_code = "forbid"`。

## 许可证

[MIT License](<LICENSE>)
