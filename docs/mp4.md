# MP4/M4A、seek 与按需读取

[返回使用指南](<README.md>) · [Rust 库使用](<library.md>) · [完整 MP4 示例](<../examples/decode_mp4.rs>)

这条路径适合 M4A 音频播放器、带 AV3A 音轨的视频文件，以及「下载一部分、按 sample 解码」的集成。
库提供容器索引和同步读取；HTTP 请求、已下载区间跟踪、播放时钟、异步调度由调用方实现。

## 1. 支持范围与输入识别

当前实现读取经典、非 fragmented MP4 的 `moov/trak/mdia/minf/stbl`，寻找 `av3a` sample entry，
用 `stsz`、`stsc`、`stts` 和 `stco` 或 `co64` 展开索引。
可跳过非 AV3A 轨道，选择找到的第一条 AV3A 轨道；没有公开的多音轨枚举/按 track id 选择接口。
每个 sample 被当作一个完整的 AV3A access unit，允许压缩帧后附带 padding。

这些能力不等于通用 MP4 支持：不解析 `moof/trun` 中的 fragment 样本，不随着直播文件增长追加索引，
不实现加密媒体解密，也不实现 `stz2` 紧凑尺寸表。
M4A 后缀可能装的是 AAC/ALAC 等其他编码，没有 `av3a` entry 时会报 `NoAv3aTrack`。

`is_iso_bmff(prefix)` 仅检查首 8 字节中的第 4..8 字节是否为 `ftyp`：

- 少于 `avs3a::mp4::ISO_BMFF_SNIFF_BYTES`（8）字节时返回 false，网络场景应先等够字节。
- 返回 true 不代表容器有效，更不代表其中一定有 AV3A 音轨。
- 它不是所有 BMFF 变体的通用探测器；若已知输入是本库支持的 MP4，可直接尝试容器 API。
- 自行读取前缀探测后，记得将底层输入位置复原，再交给读取器。

## 2. 完整本地文件：open 一次，按需 next_frame

~~~rust
use std::fs::File;
use avs3a::{BuiltinDecoder, Mp4FrameReader};

fn decode_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = Mp4FrameReader::open(File::open(path)?)?;
    let Some(first) = reader.next_frame()? else { return Ok(()); };
    let mut decoder = BuiltinDecoder::configure(first.header())?;
    let mut pcm = vec![0_i16; decoder.sample_count()?];
    decoder.decode_into(&first, &mut pcm)?;
    // 消费第一帧 pcm；configure 本身不会消费第一帧。
    while let Some(frame) = reader.next_frame()? {
        decoder.decode_into(&frame, &mut pcm)?;
        // 消费 pcm，下次调用会覆盖它。
    }
    Ok(())
}
~~~

完整写 WAV、指定起始 sample 的程序见 [MP4 示例](<../examples/decode_mp4.rs>)。

`Mp4FrameReader<R>` 要求底层 `R: Read + Seek`，可使用 `File`、`BufReader<File>`
或完整文件字节的 `Cursor<Vec<u8>>`。它不是接受任意网络 chunk 的 parser。

- `open(reader)` 扫描顶层 box 头并读取完整 `moov`，跳过媒体主体，建立内存中的 sample 索引。
  内存/初始化开销跟元数据和 sample 数量相关，而不是把整个音频一次读入内存。
- `track()` 返回 `&Av3aTrack`；建索引成功不等于所有媒体 sample 已可读或 CRC 已通过。
- `position()` 是**下一次读取的零基 sample index**，初始为 0。
- `next_frame()` 读取该 sample 并解析帧，成功后 position 加 1；在索引末尾返回 `Ok(None)`。
  它复用读取暂存区，但返回的 `EncodedFrame` 仍拥有自己分配的字节。
- `seek_to_sample(index)` 仅更新逻辑位置，实际 seek/read 发生在后续 `next_frame`。
- `into_inner()` 消费 reader 并取回底层输入；直接 drop reader 即释放它持有的文件句柄。

### 音轨提示与真实音频格式

| 信息 | 来源与单位 |
|---|---|
| `track_id()` | 容器的轨道标识，不能当作 sample index |
| `samples().len()` | 压缩帧/sample 数量，不是 PCM 采样值总数 |
| `declared_channels()` | sample entry 的声道数提示；例如实际 7.1.4 也可能声明为 2 |
| `declared_sample_rate()` | sample entry 的 16.16 格式采样率整数部分，无法表示 65536 Hz 及以上的整数采样率 |
| `timescale()` | 媒体时间戳每秒的刻度数，不保证等于采样率 |
| `duration()` / `duration_seconds()` | 媒体轨道声明的时长；后者为 duration / timescale，timescale 为 0 时返回 0 |
| `data_range()` | 所有 sample 的最小起点和最大终点组成的包围区间；空轨道为 None |

真正配置 decoder 和音频设备时，以 `frame.header()` / `decoder.config()` 为准。
如果 UI 在首帧前显示容器提示，拿到首帧后应更新，不能持续把声明值当作实际格式。

## 3. 只取元数据或网络区间：三种索引入口

| 已有数据 | 入口 | 注意 |
|---|---|---|
| 可随机读的完整文件或等价 reader | `Av3aTrack::read_from(&mut reader)` | 从文件头扫描并读取 moov，会改变底层位置 |
| 从**原文件 offset 0** 起的连续前缀 | `Av3aTrack::from_prefix(&prefix)` | 不够时返回 NeedMoreData，可逐步补长前缀 |
| 单独取回的完整 moov box | `Av3aTrack::from_moov_box(&box_bytes)` | 参数要包含 box 头，不仅是 body；sample offset 仍是原文件绝对偏移 |

已有索引时可用 `Mp4FrameReader::with_track(reader, track)`，避免再次索引。
这个 reader 必须对应**同一份原文件的坐标空间**；不能把仅含 moov 的 Cursor 与原文件 offset 的索引配对。

### 如何理解 NeedMoreData

在 `from_prefix` 中，`Mp4Error::NeedMoreData { needed, available }` 的单位是字节：
`needed` 是下一次尝试所需的**总前缀长度**，不是「额外再读这么多」。例如当前有 1000 字节，
needed 为 1500，则还需补齐 500 字节。

拿到这些字节后再次调用，仍可能得到新的 NeedMoreData：此前缺少的可能只是下一个 box 头，
现在才知道整个 moov 的长度。调用方要结合远端总长度判断文件截断/缺少 moov，
不能在已到真实 EOF 时仍无限扩展请求。该 API 也不维护历史输入，重试需传入完整累计前缀。

对于 moov 在 mdat 后面的文件，连续前缀方案可能需要获取几乎整个文件；这不是最优的 Range 下载策略。
想跳过媒体主体，应由调用方按 box 头定位并单独下载 moov，然后交给 `from_moov_box`。
该方法的 NeedMoreData 长度相对于所传 moov box 缓冲区，而非原文件起点。

`track.first_frame_header(&prefix)` 也要求**原文件前缀**，而且检查首个 sample 的完整范围已在前缀内。
仅下载了第一 sample 的字节时，可用 `let header = *parse_sample(&sample_bytes, 0)?.header();` 复制出帧头；
它不会接受只有帧头而没有完整压缩帧的输入。两者解析头时都不等于验证音频 CRC。

## 4. 时间与 sample：不要混用单位

每个 `Mp4Sample` 包含：

- `offset: u64`：原文件中的绝对字节偏移。
- `size: u32`：sample 字节数，可能包含尾部 padding。
- `timestamp: u64`：解码时间戳，单位是媒体 timescale tick，由 stts 累加得到。
- `duration: u32`：该 sample 的媒体 tick 数。
- `end()?`：第一个不属于 sample 的字节位置，有溢出检查。

按**媒体时间**定位时，先校验秒数有限且非负，再计算 `floor(seconds * timescale)`，
调用 `track.sample_at_time(ticks)`。它返回时间戳不晚于参数的最后一个 sample：
空轨道或早于第一 sample 返回 None；超过轨道结尾仍会返回最后一个 sample，而不是 EOF。
因此播放器需要自己定义超出总时长时的行为。不要用整数秒直接作 ticks，也不要把 PCM 帧数作字节 offset。

这个 API 只做帧级定位。需要在帧内某个 PCM 时刻开始播放时，先找到并解码目标 sample，再按声道数裁掉前面
的交错采样值；不是要求解码器输出更短的 buffer。

### Edit list 不会自动应用

`track.edits()` 返回 `elst` 条目，但 reader、decoder 和 WAV CLI 都不自动做 encoder-delay 裁剪、
空编辑段补偿或变速。`Mp4Edit::media_time` 使用媒体 timescale（−1 表示 empty edit）；
`segment_duration` 则使用 **movie timescale**，不能直接除以 track.timescale。
当前 Av3aTrack 没有暴露 movie timescale，需要完整 presentation timeline 的调用方应另外解析相应信息。
`is_identity()` 只检查 media_time 为 0 且 rate 为 1，不代表已经验证了编辑段覆盖整条轨道。

## 5. Seek：位置移动 + reset + 预热

设总 sample 数为 N，目标为 T。合法位置是 `0..=N`，T=N 表示 EOF，最后一个实际 sample 是 N−1。
`seek_to_sample` 不会代替你 reset decoder，也不会预热。

对一条配置不变的流，建议按以下步骤实现：

1. 检查 T 不大于 N。T=N 时直接将 reader 定位到 N、清空旧输出并报告 EOF，不必回放预热帧。
2. 已有 decoder 时查询 `decoder.warmup_frames()`；只有帧头时可用
   `avs3a::decode::warmup_frames_for(&header)?`，无需为查询深度而构建 decoder。
3. 计算 `start = T.saturating_sub(warmup)`，将 reader 定位到 start。
4. 对原 decoder 调用 `reset()?`，或重新 configure。reset 保留配置；如果格式变化要重新 configure。
5. 依次读取并**解码** `start..=T` 的帧（包含 T），但丢弃 index < T 的 PCM；T 帧才交给播放器。
6. 之后继续顺序解码，保持 decoder 状态；应用层同步清空 seek 前的 PCM 队列并更新时钟。

当前 channel-based 与 Mix 的合成预热值是 1，HOA 是 2；这是实现当前值，应用应查询 API，
不要把调用方示例中的固定 32 帧当作库规定。接近流开头时只能从 0 起回放；正常从头播放也不应无条件丢掉首帧。

**「恢复合成历史」不等于「任意 seek 后与从头解码逐样本一致」。**
除了 overlap/basis 历史，解码器还有持续推进的伪随机状态；reset 会重置它。短预滚不一定复现原位置的随机状态，
更多预热帧也不构成任意码流精确相等的保证。若必须严格复现顺序输出，应在相同配置/实现下从流起点按原顺序回放。
库没有通用的完整解码状态快照 API。依据：[预热接口](<../src/builtin_decoder.rs>)、
[随机状态说明](<../src/random.rs>) 和 [特定测试样本的 seek 回归](<../tests/mp4_container.rs>)。

还要区分两套计数：

- 读取前的 `reader.position()` 是此次 sample 的绝对轨道编号；成功读帧后可用 position−1。
- `decoder.frame_index()` 从 configure/reset 后重新计数，包含已解码但丢弃的预热帧。

## 6. 边下载边解码：读取前确认所需字节真正可用

所有范围在本指南中都采用**左闭右开** `[start, end)`。
HTTP Range 使用闭区间，发请求时应转换成 `bytes=start-(end-1)`；只对非空范围发出请求。

### 单 sample 与一次 decodeNext 的窗口

普通顺序读取只需等待当前 sample 的 `[offset, end()?)`。
如果应用的 `decodeNext` 会内部循环吞掉预热帧再返回目标帧，那么一次调用可能读多个 sample：
只检查第一个 sample 就不够了。

下面是**应用层辅助函数**，不是库的方法。参数 `position` 是 reader 当前的位置，
`discard_until` 是 seek 的目标 sample（正常顺序播放可等于 position）。函数覆盖本次调用要读的每一个 sample，
不假设 sample 的物理 offset 随解码顺序单调递增。

~~~rust
use avs3a::Av3aTrack;
use avs3a::mp4::Mp4Error;

fn decode_window_range(
    track: &Av3aTrack,
    position: usize,
    discard_until: usize,
) -> Result<Option<(u64, u64)>, Mp4Error> {
    let samples = track.samples();
    for index in [position, discard_until] {
        if index > samples.len() {
            return Err(Mp4Error::SampleOutOfRange { index, count: samples.len() });
        }
    }
    let Some(first) = samples.get(position) else { return Ok(None); };
    // 若应用选择 seek 到 EOF 后仍从前面开始预热，最多覆盖到最后一个 sample。
    let last = position.max(discard_until).min(samples.len() - 1);
    let mut start = first.offset;
    let mut end = first.end()?;
    for sample in &samples[position..=last] {
        start = start.min(sample.offset);
        end = end.max(sample.end()?);
    }
    Ok(Some((start, end)))
}
~~~

这是包围区间，可能包含 padding、其他轨道或空隙，不保证其中每个字节都属于这次 AV3A 解码。
若下载器要减少流量，可逐个检查/合并 sample 区间。`track.data_range()` 则是**整条音轨**的包围区间，
不能当作「下次解码一定要下载完整的范围」。

### 部分文件的可用性不等于文件长度

- `Mp4FrameReader::open` 还需要扫描经过的顶层 box 头、完整 moov，以及足以通过已扫描 box 尺寸检查的逻辑文件长度。
  不能概括成「任意部分文件里只要有 moov 就行」。例如 moov 位于媒体之后时，缺失中间 box 头会妨碍扫描。
- moov 在前时可以先建索引、后取媒体；moov 单独下载时更适合 `from_moov_box + with_track` 或直接使用 `parse_sample`。
- 对稀疏文件或预分配文件，尚未下载的洞可能读出零而不是 EOF。库不能识别这些零是未下载内容。
  下载器必须维护已完成的字节区间，在读取前确认整个 sample/预热窗口已经写好，而不是只比较文件长度。
- 若用 File 边下载边读，不要在尚未完成的范围上抢读；BufReader 的预读缓存也可能留下旧的零洞内容。
  最直接的方案是只在确认范围已完成后通过 File 按需读取，或先取全单个 sample 字节再调用 parse_sample。
- `parse_sample` 的参数是下载到的**完整 sample 数据**；后面的 decoder 仍使用同一个有状态实例。
  不要为每个网络 chunk 或 sample 建一个新 decoder。

### 错误与重试的边界

`next_frame` 的读失败或 parse_sample 失败不递增 position。短读会使内部文件游标缓存失效，
下一次会重新定位读取同一 sample。确认下载补齐后可重试该读取。
但是在 `next_frame` 成功后，reader 已经前进；随后 decode 失败不会让 reader 自动退回，也不保证 backend 状态回滚。
用错误类型区分「等下载后重读」与「坏数据/解码失败」，不要无限重试所有错误。

| 错误 | 常见含义 |
|---|---|
| `NeedMoreData` | 前缀或独立 moov 缓冲不足，依该 API 的坐标语义补齐 |
| `Io(UnexpectedEof)` | 底层实际读取不足；只在知道下载未完成时把它当作可恢复 |
| `MissingBox` / `NoAv3aTrack` | 元数据/音轨不满足当前支持范围 |
| `Truncated` / `InvalidSampleTable` / `InconsistentIndex` | box 或 sample 表损坏/截断/不一致 |
| `FrameExceedsSample` | 帧头声明的压缩帧长度大于收到的 sample 字节 |
| `SampleOutOfRange` | 索引大于 sample 总数 |
| `DecodeError::CrcMismatch` | 已取得的压缩数据校验失败；也可能是误读了下载洞 |

## 7. N-API / Electron 等语言绑定的设计检查表

调用方可以封装 `info / position / seekToFrame / nextSampleRange / decodeWindowRange / decodeNext / close`，
但这些名字、JS 对象和 `done` 协议**不是本库自带 API**。
Rust 侧只需要 `Mp4FrameReader`、`BuiltinDecoder`、可复用 PCM buffer 和当前丢弃目标等状态。

- `info` 分开暴露容器提示与已知真实格式；逐帧返回的 sampleRate/channels 用 decoder.config。
- `seekToFrame` 按上面的流程实现。`decodeNext` 丢弃预热输出时，其返回 frameIndex 应为轨道编号。
- `nextSampleRange` 只覆盖下一次底层读帧；封装的 decodeNext 若吞多帧，下载器应查 decodeWindowRange。
- EOF 用清晰的 `done` 表达，不用零长度 PCM 冒充正常帧；位置可以合法等于总 sample 数。
- 在 worker/阻塞任务中执行同步读盘与解码；`Arc<Mutex<_>>` 只串行化访问，不会避免堵塞调用线程。
- 导出 little-endian PCM16 时显式转换字节，或直接使用浮点输出并按 [PCM 说明](<library.md>) 归一化。
  对象/HOA 数据还需渲染，不是封装 ArrayBuffer 就能正确播放。
- 文件 offset/timestamp 是 u64，不能无条件无损转换为 JS Number。若可能超过 2^53−1，使用 BigInt 或字符串等协议。
- `close` 是包装层定义的生命周期操作；drop reader 释放文件，移除 decoder 并按需释放缓存。
  若约定关闭后所有方法报错，每个方法（包括只返回缓存信息的 info）都要检查关闭状态，不能只依赖锁是否 poisoned。

源码入口：[容器 API 与实现](<../src/mp4.rs>)；可运行流程：[MP4 解码示例](<../examples/decode_mp4.rs>)。
