# avs3a-rust

AVS3-P3（AV3A / 臻音全景声）解码器的渐进式 Rust 重写。

这个仓库的音频解码路径已经覆盖 channel-based mono、总码率高于 32 kbps 的普通 stereo（MS/ILD）、24/32 kbps stereo MCR、channel-based MC、纯对象/声床加对象 Mix，以及 HOA1/2/3，均已接通 metadata envelope → Main/LC 神经频谱 → coupling/inverse-DMX → BWE/TNS/FD/LFE → IMDCT/overlap-add → PCM16。HOA 还包含 512-hop 分析/合成后滤波、两帧延迟 basis 空间恢复和动态传输声道配置。static Basic L1、VR extension L1 与 dynamic L1/L2 metadata 已有完整公开值模型，programme/content/object/pack/channel、声学环境、渲染/EQ/DRC 和逐对象位置/扩展/增益/L2 控制均按 C 公式反量化并保留。当前已用三份完整的真实 Mix 流（stereo/5.1/7.1.4 声床分别附带 1/3/4 个对象）和三份真实 channel-based 7.1.4 流做长时间 PCM 回归；mono、纯对象 Mix 与 HOA 仍以构造码流和逐模块/端到端 C 对照为主。项目不会用静音或伪造 PCM 冒充未完成的布局。

## 当前已实现

- 无 `unsafe`、仅用标准库实现的 MSB-first 位读取器
- mono / stereo / MC / HOA / Mix 五类帧头的严格解析
- 码率、采样率、通道布局、对象数和帧尺寸的边界检查
- 可接受任意网络分片的增量帧解析器
- 从垃圾数据/伪同步字恢复，且不会无限增长缓冲区
- 与 C 参考实现兼容的 CRC（注意：不是标准 CRC-16/CCITT 的字节递推）
- 逐帧零分配 metadata parser：一次性准备固定容量值存储，有界解析 static Basic L1、VR extension L1、dynamic L1/L2 及全部条件字段，将非字节对齐音频 payload 恢复到 bit 0；非法 pack→channel 引用和过量对象返回结构化错误
- Range Coder 解码、overflow 映射和小端模型 CDF 加载（含 C 对照向量）
- latent 的 channel-major / entropy-interleaved 安全布局转换
- quantile median 反量化及 hyper-prior context scale → CDF 选择（含逐位 C 对照向量）
- 完整神经模型读取：XOR `0x55`、CNN 拓扑/权重、bias、GDN/IGDN 参数、quantizer、context scales 和 CDF
- 模型维度、层数、通道数、浮点有效性、截断、尾随数据和分配上限检查
- 内置 79,930 字节明文 hyper-prior 模型，使用 `include_bytes!` 和 `OnceLock` 每进程只校验/解析一次，decoder 共享只读权重
- 无 `unsafe` 的 CNN transpose-convolution 标量 decoder，覆盖 stride 1/2、kernel 3/5、ReLU、GDN/IGDN
- 精确复现 `GEMM_REFORM_ENC` 的四累加器及 tail 顺序，构造后每帧推理零分配
- Main/LC 神经 QC：context/base Range Decode、布局还原、反量化、context CDF 选择、noise filling、feature scale 与 1024 线输出
- 可复用 core side information：HBR/LBR LSF、两个有界 TNS Huffman filter、mono/stereo BWE 配置、短窗 grouping，以及未字节对齐的 QC payload
- ordinary stereo 按 C 线序解析“双 core → 双 grouping → MS/ILD/bit split → 双 QC”，字节分配使用无窄化整数运算；逆 MS/ILD 与 C 逐位一致
- 24/32 kbps MCR 按 C 线序解析“双 core → 左 grouping → 18 个 VQ index → 单 QC”，使用内置 8/9-bit rotation codebook 无分配 upmix，并完成双声道后处理
- channel-based MC 按 C 线序解析“全部 core prefix → 全部 grouping → silence/pair/ILD/ratio → 各声道 QC”，支持 4.0、5.1、7.1、5.1.2、5.1.4、7.1.2、7.1.4 的通用码率/BWE/LFE 配置
- MC coupling 使用规范 coupling-order 映射、逆序 MS、30 项 ILD 码本和 LFE 32-line 处理；`McCoreDecoder` 共享神经 decoder/PRNG，并为最多 16 路保留独立后处理与 overlap 状态
- HOA1/2/3 全部规范码率的传输声道、分组、core lines 和四档 BWE 配置；按 C 线序解析“全部 core prefix → 全部 grouping → scene/basis/group pair/SFB mask/ILD/ratio → 各传输声道 QC”，并完成分组字节分配与 inverse-DMX
- HOA 512-hop 后滤波：RustFFT 正向 MDCT、规范 sine window、两帧传输延迟、三阶球谐 basis 恢复、两帧 basis 延迟和逐输出声道 IMDCT/overlap-add；1343 对角度和 257 项正弦表以 6400 字节小端只读 asset 内置
- 短窗 spectrum degroup/interleave/deinterleave、BWE tile/Off-Mid-High whitening/SFB envelope、TNS synthesis lattice
- HBR/LBR LSF 反量化、LSP/LPC 与 49-band inverse FD shaping
- 基于 `rustfft` 的 256/1024/2048 点 AVS3 MDCT/IMDCT，构造期缓存 SIMD FFT plan 和 scratch；codec 折叠/twiddle/排列/归一化、窗函数、四种 core transform 和 overlap-add 状态完整迁移
- `MonoCoreDecoder` 和 `MonoDecoderBackend`：成功后才提交 decoder-local PRNG 状态，输出使用 C 的 `floor(x + 0.5)` PCM16 饱和规则
- `StereoCoreDecoder` 和 `StereoDecoderBackend`：双路共享神经 decoder/PRNG、各自持有 FFT/overlap 状态并输出交错 PCM16；24/32 kbps 按码率自动分派到 MCR
- `McCoreDecoder` 和 `McDecoderBackend`：逐帧无分配的多声道完整链路，输出按声明布局交错；三份真实 7.1.4 文件均已完整解码
- `HoaCoreDecoder` 和 `HoaDecoderBackend`：最多 16 路预分配状态，动态解码传输声道并输出 4/9/16 路交错 HOA PCM16；公开 CLI 已按 profile 自动分派
- `MixDecoderBackend`：保留声床/单对象独立码率，纯 1/2 对象按规范复用 mono/stereo（含 MCR），纯 3 路以上对象和声床加对象复用 MC；LFE 位于声床末尾、对象之前，ILD 仅作用于声床非 LFE 通道，配置时只分配实际 core
- decoder 私有、可显式共享的 glibc seed-1 PRNG，替代不可移植且跨实例耦合的全局 `rand()`
- feature-scale `pow()` 结果固化为按位查表，消除每帧 libm 调用和平台末位差异
- 解码状态与 DSP 后端之间的强类型接口
- RAII PCM16 WAV 写出，显式小端序并在结束时回填 RIFF 长度
- 纯标准库、常量内存的 PCM16 RIFF/WAVE 逐样本比较器，报告总计和逐声道差异率、最大绝对误差、RMS 及首个差异位置
- Linux / macOS / Windows CI 基线

Rust crate 在 crate 根启用了 `#![forbid(unsafe_code)]`。以后若某个平台的 SIMD 确实需要 `unsafe`，应放在独立、可关闭且有标量对照测试的 crate 中，不能污染位流和状态管理层。

## 构建与检查码流

```bash
cargo build --release
cargo test --all-targets
cargo run --release --bin avs3a-inspect -- --verify-crc input.av3a
cargo run --release --bin avs3a-model-inspect -- --cnn-fingerprint model.bin
cargo run --release --bin avs3a-decode -- input.av3a output.wav
cargo run --release --bin avs3a-decode -- input.av3a first-frame.wav --frames 1
cargo run --release --bin avs3a-mono-decode -- input.av3a output.wav
cargo run --release --bin avs3a-wav-compare -- output.wav reference.wav
```

`avs3a-inspect` 只解析裸 `.av3a` elementary stream，不调用合成后端。它会打印布局、总码率、Mix 声床/对象分项码率、帧数、CRC 失败数、跳过字节数和相邻帧完整配置变化数；配置判定与公开 `Decoder` 使用同一个 `DecoderConfig`。

`avs3a-decode` 自动选择 mono、ordinary/MCR stereo、channel-based MC、Mix 或 HOA1/2/3 backend，逐帧强制 CRC 和配置连续性检查，并复用调用方 PCM 缓冲输出 WAV。`--frames` 可限制回归帧数。它遇到 parser 重同步、损坏 payload 或配置变化会立即报错，不会悄悄生成部分可信的音频。

`avs3a-mono-decode` 保留为只接受 channel-based mono 的窄接口。由于尚无公开 mono 实流回归集，mono 合成当前仍应视为实验性。

`avs3a-wav-compare` 解析 RIFF chunk 而不是假定固定 44 字节头，以常量内存流式比较两个同格式 PCM16 WAV。长度不同时会比较共同前缀并明确打印双方 PCM frame 数；浮点 DSP 跨实现对照可直接查看逐声道 LSB 统计，不必生成巨大的文本 diff。

`avs3a-model-inspect` 读取原始神经模型块，默认按 C 内置模型的 hyper-prior + XOR `0x55` 格式解析；也支持 `--plain` 和 `--vae`。模型内容不会先被复制到一个可越界的解密缓冲区，而是在每次小端读取时按需去混淆。`--cnn-fingerprint` 会以确定性输入运行 base/context decoder，便于跨编译器和平台核对浮点路径。

crate 自带的明文模型位于 `assets/avs3a_hyper_model.bin`，SHA-256 为 `55d56a17dbfa22da21f2fa945ebe13824d246fe8b2669a63e0440316282ae068`。它省去了启动时的 XOR 和临时副本，但仍完整经过 Rust 的有界小端模型解析器。FD shaping 的 10,992 个规范 `f32` 码本值位于 `assets/avs3a_fd_tables.bin`（43,968 字节，SHA-256 `6b8e25a332edf722c81c494c85ab57d90f145d1524fd808e01333e1c9a6d39d5`），按需从显式小端字节读取。HOA 空间表位于 `assets/avs3a_hoa_spatial_tables.bin`（6400 字节，SHA-256 `641e93f65c86376815560119d6704064d33528ecddd331bab133f189164aec50`），并显式保留 C 声明中最后三行的隐式零初始化。

## Rust API

一次性解析帧头：

```rust
let info = avs3a::parse_header(bytes)?;
println!("{} Hz, {} channels", info.header.sample_rate, info.header.channels);
# Ok::<(), avs3a::HeaderError>(())
```

流式输入：

```rust
let mut stream = avs3a::FrameStream::new();
for event in stream.push(network_chunk)? {
    match event {
        avs3a::StreamEvent::Frame(frame) => {
            assert!(frame.crc_is_valid());
            consume(frame);
        }
        avs3a::StreamEvent::Skipped { bytes } => log_resync(bytes),
    }
}
# Ok::<(), avs3a::StreamError>(())
```

`DecoderBackend` 是算法迁移边界。后端必须先用不可变的 `DecoderConfig` 配置，然后只能写入长度固定为 `channels * 1024` 的输出切片。公开 `Decoder` 会先验证 CRC 和配置连续性（包括影响 DSP/side syntax 的码率），再调用后端。`decode()` 返回拥有样本的 `AudioFrame`；长时间流式解码可改用 `decode_into()` 重用调用方缓冲区。

神经 QC 可以独立运行。顶层多声道 decoder 应给所有声道传入同一个 `Avs3Random`，以保持 C 原实现的调用顺序，同时又不让两个 decoder 实例共享进程全局状态：

```rust
let streams = avs3a::NeuralBitstreams::new(context_bytes, base_bytes)?;
let noise = avs3a::NoiseFilling::single(num_lines_noise_fill, nf_index)?;
let side_info = avs3a::MainNeuralQc::new(streams, noise, amplified, scale_index)?;
let mut neural = avs3a::NeuralSpectrumDecoder::new_builtin()?;
let mut random = avs3a::Avs3Random::new();
let decoded = neural.decode_main(side_info, &mut random)?;
consume_1024_lines(decoded.spectrum());
# Ok::<(), avs3a::NeuralQcError>(())
```

实验性 mono PCM16 backend 可以直接接入公开的 framing `Decoder`：

```rust
let backend = avs3a::MonoDecoderBackend::new_builtin()?;
let mut decoder = avs3a::PendingDecoder::new(backend).configure(frame.header())?;
let pcm = decoder.decode(&frame)?;
consume_interleaved_pcm16(pcm.samples());
# Ok::<(), Box<dyn std::error::Error>>(())
```

普通 stereo 和 MCR stereo 使用同一接口，只需换成 `StereoDecoderBackend`；它返回按 `L, R, L, R, ...` 排列的 2048 个 PCM16 样本。总码率高于 32 kbps 走 MS/ILD，24/32 kbps 自动走 MCR。channel-based 多声道使用 `McDecoderBackend`，Mix 使用 `MixDecoderBackend`，HOA 使用 `HoaDecoderBackend`，都返回 `channels * 1024` 个交错样本。Mix backend 在 `configure` 时按对象数/声床类型惰性构造 mono、stereo 或 MC core，`core_kind()` 和对应的 backend accessor 可用于读取细分诊断。所有内置 backend 都会先按 C 顶层顺序解析 static metadata、dynamic presence flag 和 dynamic metadata，再把剩余位流交给音频 core；`last_metadata()` 保留轻量摘要兼容接口，`last_metadata_values()` 返回完整 `FrameMetadata`。直接使用 `MetadataPayloadParser` 时，可从 `ParsedMetadataPayload::metadata()` 读取同一对象模型；`parse_with_object_count` 接收 Mix 帧头中的对象数，channel-based/HOA 可直接使用对象数为 0 的 `parse`。

## 已验证的参考样例

使用相邻 C 仓库 `/home/zxq/Git/avs3a/test` 中的现有样例全量验证（样例文件不复制进本仓库）：

| 文件 | 帧数 | 字节数 | 布局 | CRC 失败 | 跳过字节 |
|---|---:|---:|---|---:|---:|
| `test.av3a` | 8701 | 21,012,915 | 7.1.4 / 44.1 kHz / 832 kbps | 0 | 0 |
| `test2.av3a` | 7726 | 18,658,290 | 7.1.4 / 44.1 kHz / 832 kbps | 0 | 0 |

两份流均解析为固定的 `7-byte header + 2408-byte payload`，总长度和 C 解码器处理的帧数一致。Rust 已完整输出对应的 12-channel PCM16 WAV：`test.av3a` 8701 帧耗时约 47.9 秒，`test2.av3a` 7726 帧约 43.3 秒（单次本机 Release 标量测试，仅作当前性能基线）。两份首帧均与 C WAV 逐字节一致。

全流比较中，RustFFT 与 C 标量 FFT 的累加顺序造成少量 PCM 末位差异，但未观察到状态漂移：`test.av3a` 的 106,917,888 个样本中 99.508% 完全一致，其余最大绝对差 2 LSB、RMS 差 0.070 LSB；`test2.av3a` 的 94,937,088 个样本中 99.148% 完全一致，其余最大绝对差 2 LSB、RMS 差 0.092 LSB。输出帧数、WAV 尺寸、CRC、metadata/side consumption 和全部声道时序均一致。

另用 `/home/zxq/Tmps/av3a_sample` 中四份 AV3A 与 C 解码器生成的配对 WAV 做了完整回归。三个看似由文件名表示声床布局的文件实际都是 Mix profile，WAV 声道数因此包含对象；`cjhyy.av3a` 是纯 channel-based 7.1.4。四份流共 22,442 个 codec frame，CRC 失败、跳过字节和完整配置切换均为 0，Rust/C 的首个 codec frame PCM 均逐字节一致。

| 文件 | codec 帧 | 实际布局 | 码率 | 交错 PCM 样本 | 不同样本 | 最大差 | RMS 差 |
|---|---:|---|---:|---:|---:|---:|---:|
| `stereo.av3a` | 2812 | stereo 声床 + 1 对象（3 ch） | 320 + 1×192 kbps | 8,638,464 | 24,255（0.280779%） | 1 LSB | 0.052989 LSB |
| `5.1.av3a` | 2812 | 5.1 声床 + 3 对象（9 ch） | 720 + 3×192 kbps | 25,915,392 | 120,114（0.463485%） | 1 LSB | 0.068080 LSB |
| `7.1.4.av3a` | 2812 | 7.1.4 声床 + 4 对象（16 ch） | 832 + 4×192 kbps | 46,071,808 | 160,204（0.347727%） | 1 LSB | 0.058968 LSB |
| `cjhyy.av3a` | 14006 | channel-based 7.1.4（12 ch） | 384 kbps | 172,105,728 | 301,774（0.175342%） | 1 LSB | 0.041874 LSB |

四个 Rust WAV 的 PCM frame 数、数据长度、采样率和声道数均与配对 C WAV 完全一致；全部非零差异都严格限制为 1 LSB，逐声道也未发现漂移。单次本机 Release 解码耗时约为 2.4、7.3、12.8 和 30 秒，仅作为当前标量性能基线。

相邻 C 仓库内置的 79,930 字节 hyper-prior 模型也已完整交叉检查：Rust 读取器正好消费全部字节，无尾随数据；base 模型为 4 层 encoder + 4 层 decoder、latent `64×16`、64 个 CDF，context 模型为 3+3 层、latent `16×16`、16 个 CDF。所有层的维度、通道、kernel、stride、activation 与 C 探针一致。

CNN 使用两组验证。合成双层网络直接调用 C 的 `InitCnnLayer`、`Conv1DTranspose`、`Conv1DTranspose2Part` 和 IGDN，在 `-O0/-O1/-O3` 下生成相同逐位向量；Rust Debug/Release 均完全匹配。完整真实模型的确定性 fingerprint 也一致：context 为 `68fbea61c73befc9`（首尾 `3e66f73f/bde0387b`），base 为 `011e720d9e643a65`（首尾 `43828b6b/c0968429`）。

完整神经 QC 对照使用真实内置模型和 C `RangeEncodeProcess` 生成的 context/base 码流，覆盖两组短窗 noise filling、feature amplification 及 LC 直出路径。C `-O0/-O1/-O3` 与 Rust Debug/Release 的 1024 个 `f32` 均逐位一致：Main fingerprint 为 `015bee238ed6728b`，LC 为 `9e969a842e4742b1`；noise filling 后的下一项随机数也一致。

mono side-info 使用 Main/HBR/short/BWE/two-group 与 LC/LBR/transition/no-BWE 两组固定 payload，对照 LSF、全部 8 阶 TNS Huffman 表、BWE、grouping、QC byte split、消费位数及 padding。BWE 三档 whitening、TNS 长短窗、degroup、HBR/LBR FD shaping、四种 window/OLA 时序均另有 C `-O0/-O1/-O3` 向量。

ordinary stereo 另有 1309-bit 完整 side-info 与合成向量。side parser 在 C `-O0/-O3` 下共同确认 `mode_end=191`、QC 可用 `1077` bits、左右 entropy 分配 `[64, 70]`、总消费 `1304` bits 和 `5` padding bits；MS/ILD、双声道神经 PRNG 顺序、degroup 与 BWE 中间结果均逐位一致。完整双声道向量受 FFT 运算顺序影响，浮点 PCM 的最大绝对误差为 `9.01e-3`（相对该声道峰值 `1.16e-5`），但全部 2048 个 PCM16 样本与 C 一致，交错 fingerprint 为 `a0362bb2f0ab465a`。公开 `FrameStream + CRC + Decoder` 也使用同一向量做端到端回归。

short MCR 另有 626-bit 完整 side-info 与合成向量。音频 core 独立对拍使用 `41` entropy bytes、消费 `625` bit；加入 C 顶层两个空 metadata flag 后，真实帧预算自动调整为 `40` entropy bytes、音频消费 `617` bit 和 `7` padding bits。Rust 内置明文 rotation asset 为 18,432 bytes（SHA-256 `9fe0ece1f78509f66847b9b31c60efed6a2185b5ee533e0b17e66cf3b5df61bc`）。短窗 deinterleave 后的左右 shaped spectrum 最大绝对误差分别为 `3.05e-4` 和 `1.66e-2`，最终浮点合成最大误差分别为 `7.63e-5` 和 `3.30e-3`；差异来自 RustFFT 与 C 标量 FFT 的运算顺序。metadata-aware 构造帧直接通过 C `Avs3DecoderLibProcess`，C PCM16 fingerprint 为 `8353a77e3acfc126`；Rust 为 `291bdf9d90779ad0`，仅索引 `1389` 和 `1989` 相差 1。公开 `FrameStream + CRC + Decoder` 也覆盖 32 kbps MCR；24 kbps 已覆盖 framing/configuration 分派。另一个 64 kbps stereo 向量在 1309-bit 帧预算内携带 90-bit static Basic L1，剩余 1219-bit 音频自动分为 `[60, 62]` entropy bytes；完整值可从 backend 读取，PCM fingerprint 仍为 `a0362bb2f0ab465a`。

HOA 端到端对照覆盖 192 kbps FOA 单帧和 320 kbps HOA3 三帧构造流，均经过完整 framing、CRC、metadata、core、两级时域变换和 PCM16 backend。FOA 的 C/Rust WAV 均为 8,236 字节并逐字节一致，SHA-256 为 `bff08ef91b7a8c0d5ff75f949c7c9a8b2edf6ec476760a0ccb444257e7d52f37`。HOA3 使用 9 路传输恢复 16 路输出并覆盖两帧 basis 延迟；49,152 个 PCM 样本中只有 7 个相差 1 LSB，其余完全一致。球谐 basis 向量（含 C 隐式零初始化的最后三行）逐位一致，三帧后滤波时序也与 C 浮点参考向量一致。由于相邻 C 仓库没有 HOA 实流，这些结果仍属于构造码流回归。

Mix 端到端对照覆盖 64 kbps 单对象、32 kbps 双对象 MCR，以及 384 kbps 5.1 声床加 64 kbps 单对象三种构造流，均经过完整 framing、CRC、static/dynamic metadata envelope、lazy backend 分派和 PCM16 写出。C/Rust 的 WAV 长度分别同为 2,092、4,140、14,380 字节；单对象 1,024 个样本中 31 个不同、最大 2 LSB，双对象 2,048 个样本中 2 个不同、最大 1 LSB，声床加对象 7,168 个样本中仅 1 个不同且相差 1 LSB。后者还覆盖声床 LFE 重排、跨声床/对象 coupling pair、仅声床 ILD 和未字节对齐的一对象 dynamic metadata；三份真实 channel-bed Mix 又补充了约 60 秒的长状态回归。

FFT 热路径使用成熟的 `rustfft 6.4.1`（MIT OR Apache-2.0，MSRV 1.61，默认 AVX/SSE/NEON）。codec-specific pre/post twiddle、归一化和布局仍由本仓库实现。它与 C 的标量 radix-2 运算顺序不同，因此不宣称逐位一致；完整 IMDCT 向量的最大绝对误差为 `3.58e-7`（N=256）、`4.77e-7`（N=1024）、`8.64e-7`（N=2048），对应峰值相对误差不超过 `3.28e-7`。完整构造 Main mono payload 的最终浮点 PCM、clipping 与 PRNG 下一项也已和 C 对拍。

RustFFT 会在运行时为 x86-64 选择 AVX/SSE、为 AArch64 选择 NEON；两组 SIMD kernel 的浮点累加顺序不同，但同一架构上的 Linux、Windows、macOS 和 Android 结果稳定。涉及完整 FFT 链路的回归测试因此分别维护 x86-64 与 AArch64 精确 fingerprint，仍使用 `assert_eq!`；未建立基线的新架构会明确失败，不会退化成有限值或非静音检查。Linux AArch64 数值路径可在 QEMU 下预检，原生 GitHub runner 继续作为各操作系统 CPU 路径的最终验证：

```bash
docker run --rm --platform linux/arm64 \
  --mount type=bind,src="$PWD",dst=/work,readonly \
  -w /work -e CARGO_TARGET_DIR=/tmp/avs3a-target \
  rust:1.96-slim-bookworm \
  cargo test --all-targets --locked --no-fail-fast
```

## 迁移顺序

1. ~~Range coder 与 C 对照向量。~~ 已完成解码路径和模型 CDF 加载。
2. ~~latent dequant 与完整安全模型加载。~~ 已完成 C 逐位向量和真实内置模型全量解析。
3. ~~CNN 标量后端。~~ 已复现 `GEMM_REFORM_ENC`，合成向量和完整真实模型均与 C 一致。
4. ~~串联 context/base entropy decode、noise filling 和 feature scale，输出 1024 线神经频谱。~~ Main/LC 已通过完整 C 对照向量。
5. mono core：side bits、FD/TNS/BWE、IMDCT、overlap-add 和 PCM16 backend 已完成 C 构造向量；仍需真实 mono 码流与长时间 PCM 回归集。
6. ~~stereo / channel-based MC。~~ 普通 stereo、24/32 kbps MCR、MC side bits/bit split/coupling/ILD/LFE、多路 core 和 PCM16 backend 已完成；MC 已通过三份完整 7.1.4 实流回归，仍需补充其他 channel-based 布局和码率的真实样本。
7. ~~HOA core 与 PCM backend。~~ HOA1/2/3 的码率配置、四档 BWE、完整 side/QC wire order、分组字节分配、inverse-DMX、多传输通道 neural/DSP、延迟 basis 恢复、时域后滤波与 PCM backend 已完成；仍需真实 HOA 码流与长时间 PCM 回归集。
8. ~~Mix core 与 PCM backend。~~ 已覆盖纯对象和 channel-bed + objects，按 C 规则复用 mono/stereo/MC core，并完成独立码率、LFE/ILD/coupling、metadata 和 CLI 分派；三份 48 kHz channel-bed + objects 实流已完成约 60 秒长状态回归，仍需补充真实纯对象 Mix。
9. ~~完整 metadata 值模型。~~ static Basic L1、VR extension L1、dynamic L1/L2 的全部 C 参考字段已保留并公开，固定容量存储接入所有内置 backend；三份真实 Mix 流已验证逐帧 metadata envelope 与音频 payload 边界，仍需补充覆盖更多 metadata 条件字段的实流。
10. RustFFT 已提供通用 AVX/SSE/NEON；额外的 codec-specific SIMD 只在标量结果稳定后增加，并与标量后端逐帧比较。

每一步的完成标准不是“能运行”，而是：正常样例逐帧匹配、截断/损坏输入不 panic、不越界、Release/Debug 结果一致，并至少通过 Linux、Windows、macOS 构建。

## 与 C 版本相比已经消除的风险

- `GetNextIndice` 没有输入长度参数；Rust `BitReader` 在读取前验证剩余位数。
- C 顶层使用固定 `payload[12300]` 和 `data[16 * 1024]`；Rust 从已验证配置计算精确长度，并在分配前检查上限与溢出。
- C 配置表列出 MC 10.2/22.2，但对应码率表是 `NULL`；Rust 返回结构化错误。
- C HOA side parser 依赖 `assert`，对 4 项 basis、8 对 group pair、1343 项 basis 表和 30 项 ILD 表没有运行时边界保护；Rust 在任何数组访问或 DSP 前返回结构化错误。
- C CLI 的 `fread` 返回短读时仍可能继续；Rust 只在完整帧到齐后产生 `EncodedFrame`。
- C 模型加载用固定 79,930 字节数组和无边界 `memcpy(data + nIndex)`；Rust 每个标量/数组读取都先检查剩余长度、乘法溢出和固定分配上限，失败时不推进游标。
- C 模型枚举、层数、stride、通道和浮点权重未经验证就参与分配/索引；Rust 在构造拥有所有权的模型对象前完成拓扑与数值验证。
- C noise filling 使用 libc 全局 `rand()`，序列因平台而异且多个 decoder/声道会互相推进状态；Rust 使用显式传递的固定 glibc 序列状态，并让调用方决定跨声道共享范围。
- C 每帧为 context/base latent、CDF index 和 overflow CDF 反复 `malloc/free`；Rust 在构造 `NeuralSpectrumDecoder` 时一次性分配固定上限工作区，逐帧路径不分配。
- C BWE/FD/TNS/IMDCT 在栈上反复放置多个 1024/2048 项临时数组；Rust DSP 对象持有固定 workspace，RustFFT plan/scratch 也只在构造期分配。
- C 的 synthesis buffer、Mix 声床/对象共享状态、HOA 两帧传输/basis 延迟和全局 `rand()` 会在错误传播不清晰时留下隐式时序；mono/stereo/MC/Mix/HOA pipeline 使用 decoder-owned 的逐声道 overlap、显式 Mix core 选择与 HOA 延迟状态，并先在克隆 PRNG 上处理整帧，成功后才提交随机状态。
- C WAV 每帧转换依赖宿主内存布局；Rust PCM16 和 WAV 均显式 little-endian，写出复用固定 8192-byte buffer。
- 计时、文件 I/O 和算法状态已从解码核心边界移除，后续不会用全局帧计数或平台时钟驱动音频状态。

## 许可证

MIT License
