# Rust 库使用

[返回使用指南](<README.md>) · [MP4/M4A、seek 与分段下载](<mp4.md>) · [命令行](<cli.md>)

## 1. 入口与导入路径

普通应用优先使用 crate 根部的高层入口：

~~~rust
use avs3a::{BuiltinDecoder, EncodedFrame, FrameStream, StreamEvent};
use avs3a::{Av3aTrack, Mp4FrameReader, WavWriter, parse_frames, parse_header, parse_sample};
~~~

不是所有类型都在根部重导出，常用的其他路径如下。精确的公开范围见 [模块入口](<../src/lib.rs>)。

| 模块 | 常用类型/函数 |
|---|---|
| `decode` | `Decoder`、`PendingDecoder`、`DecoderConfig`、`AudioFrame`、`warmup_frames_for`、`FLOAT_FULL_SCALE` |
| `backend` | `DecoderBackend` 与五种内置 backend |
| `header` | `HeaderError`、`HeaderInfo`、`CodecProfile`、`ChannelConfig`、`SoundBedType`、`BitDepth` |
| `stream` | `StreamError` |
| `mp4` | `Mp4Error`、`Mp4Sample`、`Mp4Edit`、`ISO_BMFF_SNIFF_BYTES` |
| `metadata` | `FrameMetadata`、`MetadataPayloadParser` 和公开的 metadata 值类型 |
| `wav` | `WavError` |

表中的模块均以 `avs3a::` 为前缀；`DecodeError` 也可直接从 `avs3a::DecodeError` 导入。
实现文件名不是导入路径，例如不能使用 `avs3a::builtin_decoder`。

## 2. 选择合适的帧读取方式

| 输入 | API | 行为 |
|---|---|---|
| 只想读取裸流头信息 | `parse_header(&bytes)` | 查找可解析的帧头，返回头及其字节 offset；不要求完整 payload，不校验 CRC，不解码 |
| 文件/网络任意大小的裸流分片 | `FrameStream::push` | 跨分片缓存，产出零到多个完整帧或 `Skipped` 事件 |
| 已知首尾恰好落在帧边界的连续裸流字节 | `parse_frames` | 严格解析，不重同步；前导垃圾、尾部残帧报错，空输入返回空列表 |
| MP4 中一个完整 sample 的字节 | `parse_sample(&bytes, index)` | 从 offset 0 解析一个压缩帧，允许 sample 尾部 padding；不允许前导垃圾 |
| 可 seek 的 MP4/M4A 文件 | `Mp4FrameReader` | 建立 sample 索引，按需读取；详见 [容器指南](<mp4.md>) |

不要把整个 MP4 文件交给裸流解析器；也不要假设多个 MP4 sample 的物理字节间没有 padding 或其他轨道数据。
`parse_sample` 的 `index` 只用于错误定位，不会替你读取或查找第 index 帧。

### 裸流：处理所有事件，并在真正 EOF 时收尾

完整可运行版本在 [裸流示例](<../examples/decode_raw.rs>)。关键流程：

~~~rust
use avs3a::{FrameStream, StreamEvent};
use avs3a::stream::StreamError;

fn inspect_chunks(chunks: &[&[u8]]) -> Result<usize, StreamError> {
    let mut parser = FrameStream::new();
    let mut count = 0;
    let mut accept = |event| match event {
        StreamEvent::Frame(frame) => {
            count += 1;
            println!("CRC valid: {}", frame.crc_is_valid());
        }
        StreamEvent::Skipped { bytes } => eprintln!("skipped {bytes} bytes"),
    };
    for chunk in chunks {
        for event in parser.push(chunk)? {
            accept(event);
        }
    }
    for event in parser.finish()? {
        accept(event);
    }
    Ok(count)
}
~~~

- `push` 返回空列表通常只是「还没凑齐一帧」，不等于 EOF。
- `finish` 只用于整个输入确实结束时；剩余数据不能组成完整帧时返回 `TrailingData`。
  它不是音频合成的 flush，不会产生额外的 overlap 尾帧。可跳过的垃圾也可能作为 `Skipped` 报告。
- `FrameStream` 可以从前导垃圾或伪同步字重同步，但**字节层重新对齐不等于音频状态恢复**。
  检查工具可只记录 `Skipped`；解码应用应停止或显式执行自己的恢复策略。CLI 和示例选择停止。
- 完整帧即使 CRC 错误也可能被产出。可用 `crc_is_valid()` 先检查；内置 decoder 在合成前还会强制校验。
- `with_buffer_limit` 限制内部待解析缓冲区，默认 12,309 字节。它**不限制**一次 `push`
  产生的事件列表或调用方保留的帧总内存。长流使用固定大小分片，处理完及时释放帧。
- `clear()` 只清除解析器字节缓存，不会 reset 解码器。
- `push` / `finish` 出错不是事务回滚：本次调用可能已消费部分字节，而局部事件列表不会返回。
  示例遇错即停止；不要原样重试整块输入并假定 parser 状态没变。

`EncodedFrame` 拥有头、完整帧字节和缓存的实际 CRC；用 `header()`、`bytes()`、
`header_bytes()`、`payload()` 获取只读视图。构造器不是公开接口，请通过上面的 parser 获得帧。

## 3. 解码器生命周期与缓冲区

推荐流程是：**第一帧配置一次 → 复用 decoder/PCM 缓冲 → 顺序解码每帧**。
下面的函数可接在任意一种帧读取器后；完整文件循环见两个 [示例入口](<README.md>)。

~~~rust
use avs3a::{BuiltinDecoder, DecodeError, EncodedFrame};

fn decode_frames(frames: &[EncodedFrame]) -> Result<(), DecodeError> {
    let Some(first) = frames.first() else { return Ok(()); };
    let mut decoder = BuiltinDecoder::configure(first.header())?;
    let mut pcm = vec![0_i16; decoder.sample_count()?];
    for frame in frames {
        decoder.decode_into(frame, &mut pcm)?;
        // 在这里播放、复制或写出 pcm；下次解码会覆盖同一缓冲区。
        let config = decoder.config();
        println!("{} Hz / {} channels / {} values",
            config.sample_rate, config.channels, pcm.len());
    }
    Ok(())
}
~~~

`configure` 根据 profile/layout 自动选择 mono、stereo、MC、Mix 或 HOA backend，并使用内置模型。
它并没有解码第一帧；第一帧仍须交给 `decode_into`。不要每帧重新 configure，否则会丢失时域历史，
也重复承担 decoder 工作区/线程池等构造开销（内置模型权重使用进程级缓存）。

| 方法 | 用法 |
|---|---|
| `config()` | 返回当前流配置，包括采样率、布局、对象数、声床通道数等 |
| `sample_count()` | 一个压缩帧输出的**交错采样值总数**：channels × samples_per_channel |
| `decode_into(frame, &mut [i16])` | 输出 PCM16，slice 长度必须**严格等于** sample_count；只预留 capacity 不够 |
| `decode_into_f32(frame, &mut [f32])` | 输出量化前的浮点数据，同样要求精确长度 |
| `frame_index()` | 自 configure/reset 后成功解码的帧数，不是 MP4 的绝对 sample index |
| `reset()` | 清空时域/解码历史和计数，保留原配置；不移动输入位置，不接受新配置 |
| `warmup_frames()` | reset/seek 后需回放的合成预热帧数，见 [seek 流程](<mp4.md>) |
| `last_metadata_values()` | 最近解码帧的 metadata 借用视图，应在成功解码后读取 |
| `last_clipped_samples()` / `total_clipped_samples()` | 最近一次 PCM16 转换/自配置或 reset 起的饱和计数；浮点路径不更新它们 |

同一帧不要在同一个 decoder 上先调 `decode_into_f32` 再调 `decode_into` 来取得两种格式——
这会推进状态两次。需要两份输出时，先解码一次 float，再用 `avs3a::decode::float_to_pcm16`
转换；传入等长的输入、输出 slice。该辅助函数返回 clipping 数量，但不会更新 decoder 的统计。

## 4. PCM 格式、声道与 WAV

### PCM16 和 float 的区别

- `decode_into` 输出有符号 i16，范围 −32768 到 32767；按采样时刻交错：
  `[t0.ch0, t0.ch1, ..., t1.ch0, t1.ch1, ...]`。
  索引公式是 `pcm[time_index * channels + channel_index]`。
- `decode_into_f32` 的幅度单位仍是 **PCM16 单位**，不是常见的 −1 到 1。对接普通浮点音频接口前，
  除以 `avs3a::decode::FLOAT_FULL_SCALE`（32768.0）。原始值可能超出满幅，归一化后也不保证在 −1 到 1 内；
  在适当的增益/混音之后按下游要求限幅。
- `config.bit_depth` 是码流的声明，不会让 `decode_into` 改成 PCM24。
- Rust 的 `Vec<i16>` 不是跨语言字节协议。输出 little-endian PCM16 时逐个用 `to_le_bytes()`，
  每帧字节数为 `sample_count * 2`。JS 的 ArrayBuffer 只是存储，音频接口可能还需要转换为平面化 Float32 数据。

~~~rust
fn pcm16_le(samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}
~~~

### 多声道不是自动渲染

Channel-based 输出保持解码器的声道顺序；Mix 输出声床和对象通道（声床在前、对象在后）；
HOA 输出 ACN 顺序的 4/9/16 路 HOA 分量，不是 4/9/16 个扬声器的现成播放信号。
使用 `channel_config`、`bed_channels`、`objects`、`hoa_order` 及 metadata 决定下游处理。
本库没有通用的设备声道重排、对象空间渲染或 HOA-to-speaker/binaural 渲染接口。
不能仅凭「12 声道」便假定所有音频设备或 WAV 播放器会正确理解布局。

### WAV 写出

`WavWriter::create(path, channels, sample_rate)` 创建并覆盖文件；
`WavWriter::new(writer, channels, sample_rate)` 可包装任意 `Write + Seek` 输出，
例如 `Cursor<Vec<u8>>`。参数顺序是**声道数在前、采样率在后**。

每次调用 `write_samples(&pcm)` 的采样值数必须是声道数的整数倍，不必恰好一整个压缩帧。
结束时显式调用 `finalize()`：它更新 RIFF 长度、flush，并返回底层 writer。
Drop 也会尽力修复头，但忽略收尾错误，不应依赖它代替显式检查。

当前 writer 是经典 PCM16 RIFF/WAVE：不写 WAVE_FORMAT_EXTENSIBLE 声道掩码、对象/HOA metadata，
也不提供 RF64；data 大小加 36 字节不能超过 u32 上限，接近 4 GiB 时会报 `SizeOverflow`。
长输出应分段或使用适合的其他封装。实现依据：[WAV writer](<../src/wav.rs>)。

## 5. Metadata：读取，而不是自动应用

所有内置 backend 都解析帧内 metadata。当前主要覆盖 static Basic L1、VR extension L1 和
dynamic L1/L2；不要把解析成功等同于支持任意 metadata 扩展。成功解码后可读取：

~~~rust
fn inspect_metadata(decoder: &avs3a::BuiltinDecoder) {
    if let Some(values) = decoder.last_metadata_values() {
        if let Some(dynamic) = values.dynamic_metadata() {
            for object in &dynamic.objects {
                println!("transport={} muted={}",
                    object.transport_channel_reference, object.muted);
            }
        }
        if let Some(static_values) = values.static_metadata() {
            println!("static metadata: {static_values:?}");
        }
    }
}
~~~

`FrameMetadata` 通过 `static_metadata()` / `dynamic_metadata()` 暴露可选数据，
`MetadataList` 可迭代，也有 `as_slice()`、`len()`、`get()`。
这里是**当前帧的存在标志和内容**，不是渲染器替你维护的整段节目状态。
下次解码会更新内部存储；需要跨帧/跨线程保留的内容应自行复制或克隆。
静态信息可能不在每帧重传，seek 的音频预热不能代替恢复渲染所需的 metadata 状态。

仅需检查 payload metadata 而不合成时，可使用 `MetadataPayloadParser`。
给 `parse_with_object_count` 传入 `frame.payload()`、`frame.header().payload_bits`
和 `usize::from(frame.header().objects)`；简便方法 `parse` 使用对象数 0，不适用于带对象的 Mix。
该 parser 不代替帧 CRC 校验。类型与解析约定见 [metadata API](<../src/metadata.rs>)。

## 6. 手动 backend：仅在确实需要时使用

普通应用不需要匹配 `BuiltinDecoder` 的 enum 变体。如果已知输入配置，或需要 backend 的诊断接口：

~~~rust
use avs3a::backend::MonoDecoderBackend;
use avs3a::decode::{AudioFrame, PendingDecoder};
use avs3a::{DecodeError, EncodedFrame};

fn decode_one_mono(frame: &EncodedFrame) -> Result<AudioFrame, DecodeError> {
    let backend = MonoDecoderBackend::new_builtin()?;
    let mut decoder = PendingDecoder::new(backend).configure(frame.header())?;
    decoder.decode(frame)
}
~~~

这是「从初始状态解一个帧」的示范，不是长流的推荐写法。长流应将 decoder 保存在循环外。
`Decoder<B>::decode` 返回拥有样本的 `AudioFrame`，通过 `config()`、`samples()`
或 `into_samples()` 取值；它每次分配输出。`BuiltinDecoder` 当前没有这个便利方法，使用其 `decode_into`。

其他内置类型为 `StereoDecoderBackend`、`McDecoderBackend`、`MixDecoderBackend`、
`HoaDecoderBackend`，同样从 `avs3a::backend` 导入。底层神经、DSP、side-info 模块面向阶段级测试和定制，
不要绕过高层 driver 期待它们自动完成 CRC/流连续性校验。

## 7. 错误、状态与线程

| 现象 | 含义及处理 |
|---|---|
| `HeaderError::NeedMoreData` | 帧头输入不够；累计更多原始字节，不要伪造补零 |
| `StreamError::TrailingData` | EOF 时有残留，或严格批量解析没有对齐完整帧；检查截断与输入类型 |
| `StreamError::BufferLimit` | 解析器缓冲上限不足以容纳声明的帧 |
| `DecodeError::CrcMismatch` | payload 与帧头 CRC 不一致；检查下载范围、文件损坏，勿当作成功 PCM 输出 |
| `DecodeError::SampleCount` | 输出 slice 长度不是 sample_count；resize 长度而非仅 reserve 容量 |
| `DecodeError::ConfigurationChanged` | 采样率、码率、布局、对象配置等与初始配置不同；reset 保留旧配置，需按新帧重新 configure |
| metadata/core/backend 错误 | 数据或配置无法解码；不要输出本次缓冲或假定可直接跳过后无缝继续 |
| MP4 I/O/索引错误 | 容器与下载问题，见 [MP4 指南](<mp4.md>) |

CRC、配置和输出长度检查发生在调用 backend 之前。`frame_index` 只在成功后递增，
但 backend 出错**不提供完整状态回滚保证**，`decode_into_f32` 的输出也可能已部分写入。
不能把「计数没变」推断为 decoder 完全没变。应用应区分可重试的输入读取失败与已进入合成的失败，
必要时重建/reset 后按既定策略回放；库不提供自动丢包隐藏或静音替代。

解码调用是同步 CPU 工作；`Mp4FrameReader<File>` 还会执行同步文件 I/O。
在 GUI、Electron、异步服务中，应将读取和解码放到专用 worker/阻塞任务，而不是 UI 线程或实时音频回调。
同一条流必须按顺序修改同一 decoder；锁只保证互斥，不会让调用变成异步，也不能把相邻帧独立并行解码。

三路及以上 MC/Mix core 会使用 decoder 私有 Rayon 池；默认 worker 数为可用逻辑 CPU 的一半向上取整、
最多 8。构造前的正整数 `RAYON_NUM_THREADS` 可覆盖默认值，多个 decoder 会各自占用资源。
这不消除外部 worker 的必要性。内置模型可共享缓存，时域历史与工作缓冲仍属于各 decoder。
部分解码回归测试显式使用 16 MiB 栈；这不是库为调用方自动设置的栈大小，也不是所有平台都需要的固定值。
嵌入受限栈环境、特别是 debug 构建时，应核实工作线程栈是否足够。

实现依据：[解码驱动](<../src/decoder.rs>)、[自动 backend](<../src/builtin_decoder.rs>)、
[裸流 parser](<../src/stream.rs>)。音频恢复和连续播放的具体例子继续看 [MP4/seek 指南](<mp4.md>)。
