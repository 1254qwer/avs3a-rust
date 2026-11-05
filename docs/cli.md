# 命令行工具使用指南

[文档首页](<README.md>) · [项目说明](<../README.md>) · [Rust 库使用指南](<library.md>)

本文说明五个命令行工具的实际参数、输入要求和退出行为。它们用于检查或解码本地文件，不是播放器，也不提供音频编码、重采样或空间音频渲染。

## 构建、安装与运行

需要 Rust 1.96 或更新版本，以及当前平台的 Rust 编译、链接环境；版本要求见 [Cargo.toml](<../Cargo.toml>)。以下命令都从本地源码仓库根目录运行，不需要另行下载参考解码器或外部音频样例。首次构建仍可能需要 Cargo 获取依赖。

```sh
# 构建全部五个工具，使用仓库锁定的依赖版本
cargo build --release --locked --bins

# 可选：从当前本地源码安装全部工具到 Cargo 的可执行文件目录
cargo install --path . --locked --bins
```

安装后，将 Cargo 的可执行文件目录加入 `PATH`，即可直接使用工具名。无需安装也能通过 `cargo run --release --locked --bin 工具名 -- 工具参数` 运行；其中 `--` 是 **Cargo 与程序参数之间的分隔符**。Windows 直接运行构建产物时使用对应的 `.exe` 文件。

查看各工具帮助：

```sh
cargo run --release --locked --bin avs3a-decode -- --help
cargo run --release --locked --bin avs3a-inspect -- --help
cargo run --release --locked --bin avs3a-mono-decode -- --help
cargo run --release --locked --bin avs3a-wav-compare -- --help
cargo run --release --locked --bin avs3a-model-inspect -- --help
```

| 工具 | 输入 | 用途 |
|---|---|---|
| `avs3a-decode` | 裸 AV3A，或带 AV3A 轨道的 MP4/M4A | 自动选择解码器，输出多声道 PCM16 WAV |
| `avs3a-inspect` | 同上 | 检查帧头、CRC、重同步和配置变化，不合成音频 |
| `avs3a-mono-decode` | 裸 AV3A 单声道输入 | 输出单声道 PCM16 WAV；具体 profile 限制见下文 |
| `avs3a-wav-compare` | 两个声道数、采样率相同的 PCM16 WAV | 比较 PCM 数据并报告整体及逐声道误差 |
| `avs3a-model-inspect` | AVS3 神经模型文件，或标准输入 | 检查模型结构，可计算确定性 CNN fingerprint |

## 通用约定与退出行为

- 正常完成、正常显示帮助均返回退出码 **0**。由程序返回的参数、格式、解码及文件 I/O 错误均返回 **1**，并向标准错误输出 `error: ...`。没有独立的“参数错误”或“CRC 错误”退出码。
- 普通报告写入标准输出。`avs3a-model-inspect` 的帮助也写入标准输出；其余四个工具的帮助写入标准错误。错误发生前可能已经输出部分报告，不能仅以“有输出”判断成功。
- `avs3a-decode`、`avs3a-mono-decode`、`avs3a-wav-compare` 只在第一个程序参数为 `-h` 或 `--help` 时显示帮助；`avs3a-inspect`、`avs3a-model-inspect` 在依次解析参数时遇到帮助选项就立即返回成功。此前已经发生的参数错误不会被后面的帮助选项覆盖。
- 路径包含空格时使用 shell 引号。除 `avs3a-model-inspect` 的输入 `-` 外，`-` 都只是普通文件名，不表示标准输入或标准输出；这些工具也不把 URL 当作网络输入。
- 参数解析器不是统一的通用选项框架。请按各节给出的顺序传参；没有额外的 `--version`、通用选项终止符或 `--选项=值` 语法。需要传入以 `-` 开头的文件名时，可写成带目录前缀的相对路径。

**两个容易误判的成功状态：**

1. `avs3a-wav-compare` 即使发现差异或长度不同，也返回 **0**；退出码只表示比较过程是否成功。
2. `avs3a-inspect` 默认会统计 CRC 失败而不因此返回失败；需要遇到 CRC 错误即失败时，显式加 `--verify-crc`。

## 音频输入与 WAV 输出范围

### 裸 AV3A 与 MP4/M4A

`avs3a-decode` 和 `avs3a-inspect` 按文件内容而不是扩展名识别输入：文件第 4–7 字节（从 0 开始）的 box 类型为 `ftyp` 时走 MP4/M4A 路径，否则按裸 AV3A 解析。将其他文件改名为 `.av3a` 或 `.mp4` 不会转换其格式。

MP4/M4A 路径的使用边界：

- 自动使用第一个包含 `av3a` sample entry 的轨道；可以同时存在视频或其他非 AV3A 轨道，但没有轨道选择参数。普通 AAC/ALAC M4A 不在支持范围内。
- 读取 `moov` 中完整的样本表，支持 `moov` 在媒体数据之前或之后、32/64 位 chunk offset，以及固定或逐样本大小。它不是 fragmented MP4、DASH 或 HLS 播放器。
- 每个 MP4 sample 按一个 AV3A 帧读取；sample 内完整编码帧之后的填充字节忽略。sample 截断、缺少所需 box、无 AV3A 轨道或不一致的索引会报错。
- 输出声道数和采样率以 AV3A 帧头为准，不以容器声明值为准。
- 不按 edit list 做裁剪、补静音、变速或编码器延迟补偿。`avs3a-inspect` 会标出 `non-identity edit list`，但 `avs3a-decode` 仍顺序输出编码帧的 PCM。容器报告的时长不一定等于输出 WAV 时长。

裸流解析时，`avs3a-inspect` 可以跳过垃圾字节并报告 `skipped bytes`；两个解码工具则拒绝在这种重同步后继续合成，以免把不连续输入当作连续音频。遇到无法组成完整帧的尾部数据也可能返回错误。

### 通用解码工具的 profile 范围

| Profile | 输出范围 |
|---|---|
| Channel-based mono | 1 声道 |
| Channel-based stereo | 2 声道，自动选择普通立体声或低码率 MCR 路径 |
| Channel-based MC | 4.0、5.1、7.1、5.1.2、5.1.4、7.1.2、7.1.4 |
| Mix | 纯对象，或 stereo/受支持 MC 声床加对象；声床与对象合计不超过 16 声道 |
| HOA | 一、二、三阶，分别输出 4、9、16 声道 |

帧头解析识别 Main/LowComplexity、8/16/24 bit 源位深，以及 8、16、22.05、24、32、44.1、48、96、192 kHz 采样率；这并不代表任意 profile、码率和采样率组合都有效。非法配置、码率索引、过大 payload 或不支持的编码内容仍会报错。真实流覆盖范围与已知限制见 [项目说明](<../README.md>)。

两个解码工具的输出固定为 **little-endian、交错、有符号 PCM16 RIFF/WAVE**，不保留源位深，也不附带 AVS3 metadata、空间位置或扬声器 channel mask。Mix 输出声床和对象的离散通道，HOA 输出相应 HOA 通道，不自动渲染成双耳或指定扬声器布局。没有音量、重采样、下混、浮点 WAV 或 RF64 选项。

输出文件在首次成功配置解码器后创建，**同名文件会直接截断覆盖**，因此不要让输入和输出指向同一文件。运行中出错时可能留下已写出的 PCM 前缀；程序会尽力修复其 WAV 长度字段，但不会自动删除部分输出。是否完整成功仍以退出码为准。普通 RIFF 大小字段为 32 位，`data` 字节数加 36 不能超过 `u32::MAX`，超限会报错而不是自动切换到 RF64。

## `avs3a-decode`：通用解码

```text
avs3a-decode <input.av3a|input.mp4|input.m4a> <output.wav> [--frames <count>]
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| 输入路径 | 必填 | 裸流或 MP4/M4A 本地文件 |
| 输出路径 | 必填 | 要创建或覆盖的 WAV 文件；父目录须已存在 |
| `--frames <count>` | 不限制 | 最多输出多少个 AV3A 帧；必须是大于 0、可表示为 `u64` 的十进制整数 |
| `-h`、`--help` | 关闭 | 仅作为第一个参数时显示帮助并退出 |

`--frames` 必须放在两个路径之后，只能出现一次；缺值、0、负数、非整数、整数溢出、未知选项或多余参数都会失败。

仓库自带 [test.av3a](<../tests/fixtures/test.av3a>)，是一份 12 声道 7.1.4 裸流，说明见 [fixture 说明](<../tests/fixtures/README.md>)。可直接做短输出试用：

```sh
cargo run --release --locked --bin avs3a-decode -- tests/fixtures/test.av3a target/cli-example.wav --frames 100
```

若已有自己的 MP4/M4A 输入，将命令中的输入路径替换为该本地文件即可，不必另加容器选项。成功报告包含输入、容器、输出、声道数、采样率、实际写出的 AV3A 帧数以及 `clipped samples`（PCM16 转换时被限制到满幅范围的样本数）。出现削波统计本身不会导致失败。

### frame、skip 与 warmup 的含义

- `--frames N` 的单位是 **AV3A 编码帧**，不是单个 PCM 采样点，也不是秒。当前每帧输出每声道 1024 个样本；成功写出 `N` 帧时，PCM 数据字节数为 `N × 1024 × 声道数 × 2`，时长为 `N × 1024 / 采样率` 秒。输入不足 `N` 帧时，只要正常到达结尾且至少成功解码一帧，仍可成功输出较短结果。
- 达到帧数上限后停止处理，不保证读取或验证文件剩余部分，也不再对裸流执行完整的 EOF 收尾检查。因此短解码成功不等于全文件验证成功。MP4 报告中的 sample 总数和时长仍是整条轨道的信息。
- 检查报告里的 `skipped bytes` 是裸流寻找合法同步头时丢弃的**字节数**，不是用户选择跳过的音频帧。
- CLI 从头顺序解码，保留最初的输出，没有自动丢弃 warmup 帧，也没有 `--skip` 或 `--warmup` 参数。库中的 warmup 用于跳转后补足有限合成历史，不是此 CLI 默认裁掉的开头，也不保证任意 seek 与从头解码逐样本一致；应用级跳转及随机状态边界请参阅 [MP4/M4A 指南](<mp4.md>)。

解码时 CRC 校验始终开启；任何已处理帧的 CRC 错误、解码错误或配置变化都会失败。配置变化不只包括采样率/声道数，还包括码率、profile、神经网络类型、位深、声床及对象配置。空输入或没有完整受支持帧时也返回失败，不生成“成功的静音替代品”。

### 多声道线程数

MC 和使用 MC 路径的 Mix 解码可通过环境变量 `RAYON_NUM_THREADS` 指定正整数 worker 数。默认值为可用逻辑 CPU 数的一半**向上取整**、最多 8 个；无法取得 CPU 数时使用 1。0、非法值或超出 `usize` 范围的值回退默认设置；有效显式值不受默认的 8 个上限限制。该变量不会让所有 profile 都采用同样的多声道并行路径。

POSIX shell：

```sh
RAYON_NUM_THREADS=4 cargo run --release --locked --bin avs3a-decode -- tests/fixtures/test.av3a target/cli-example.wav --frames 100
```

PowerShell：

```powershell
$env:RAYON_NUM_THREADS = "4"
cargo run --release --locked --bin avs3a-decode -- tests/fixtures/test.av3a target/cli-example.wav --frames 100
Remove-Item Env:RAYON_NUM_THREADS
```

## `avs3a-inspect`：检查帧与容器

```text
avs3a-inspect [--verify-crc] [--mc-side-info] <input.av3a|input.mp4|input.m4a>
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| 输入路径 | 必填 | 只允许一个本地输入文件 |
| `--verify-crc` | 关闭 | 遇到第一个 CRC 错误立即失败；错误信息中的 frame 编号从 0 开始 |
| `--mc-side-info` | 关闭 | 在正常摘要后解析并输出**首帧**的 metadata/MC side info |
| `-h`、`--help` | 关闭 | 参数解析遇到时显示帮助并退出 |

两个开关可出现在输入路径前后，重复指定仍是开启。未识别的参数会被当成输入路径，而不是统一报告“未知选项”；超过一个这样的参数会报 `only one input file may be specified`。

```sh
cargo run --release --locked --bin avs3a-inspect -- --verify-crc tests/fixtures/test.av3a
cargo run --release --locked --bin avs3a-inspect -- --verify-crc --mc-side-info tests/fixtures/test.av3a
```

报告中的 profile、布局、声道、采样率、源位深、码率和帧尺寸取自**首帧**；`frames`、`parsed bytes`、`skipped bytes`、`CRC failures`、`configuration changes` 是全程统计。`parsed bytes` 只统计编码帧，不包含 MP4 容器开销或 sample 尾部填充。`configuration changes` 统计相邻帧解码配置之间的变化次数，不是不同配置的种类数。

默认也会计算 CRC；不加 `--verify-crc` 时仅累计失败数。配置变化和成功重同步本身不使检查失败。一般摘要只验证容器、帧结构和 CRC，**并不验证全部音频 payload 能否成功合成**。没有完整帧、截断/非法容器、无法完成的裸流尾部或 I/O 错误仍会失败。

`--mc-side-info` 适用于采用 MC 编码的 channel-based 多声道帧，以及多对象/声床加对象的相应 Mix 帧：纯对象 Mix 要至少 3 个对象。它会报告首帧的 metadata 位数、side/QC 位数、padding 位数、逐通道 entropy 字节数、silence flags、配对、ILD 索引和 bit ratios；不是逐帧 metadata 导出工具。单声道、channel-based stereo（包括 MCR）、1/2 对象纯对象 Mix 和 HOA 不适用，相关解析会失败；即使摘要已输出，最终退出码也可能为 1。

## `avs3a-mono-decode`：单声道窄接口

```text
avs3a-mono-decode <input.av3a> <output.wav>
```

只有两个必填路径，以及位于第一个参数的 `-h`/`--help`；没有帧数限制或其他可选参数。它只读取裸 AV3A，不做 MP4/M4A 检测或解复用，正常运行会处理到输入结束。成功报告输入、输出、帧数和 clipped samples。CRC、配置连续性、重同步拒绝及部分 WAV 输出行为与通用解码工具相同。

**实际接受范围比帮助文本略宽：**除 channel-based mono 外，当前实现也接受只有 **1 个对象、无声床的 Mix**。帮助文本称所有 Mix 均拒绝，但 [mono 命令入口](<../src/bin/avs3a-mono-decode.rs>) 直接使用 [MonoDecoderBackend](<../src/mono_backend.rs>)，后者明确接受这种单对象配置。其他多声道输入仍会失败；这不是把任意输入下混为 mono 的工具。

仓库自带的 [test.av3a](<../tests/fixtures/test.av3a>) 是多声道输入，不能用作此工具的成功示例。对已有的单声道文件，按上述语法提供真实的本地输入和一个不同的输出路径；一般使用场景优先选择 `avs3a-decode`。

## `avs3a-wav-compare`：比较 PCM16 WAV

```text
avs3a-wav-compare <left.wav> <right.wav>
```

两个路径必填；`-h`/`--help` 仅在第一个参数位置有效，没有容差、对齐、声道映射或抽样区间选项。比较采用固定大小缓冲区，不把整段音频读入内存。

输入限制：

- 必须是 little-endian `RIFF/WAVE`，格式标记为 PCM（1）、16 bits/sample；不接受 float WAV、24/32 bit PCM、RF64 或 `WAVE_FORMAT_EXTENSIBLE`。
- 两个文件声道数、采样率必须相同且非零；不要求 RIFF 头逐字节相同，也不要求音频长度相同。
- 按 chunk 查找唯一的 `fmt ` 和 `data`，允许其他 chunk、合法的奇数字节 padding，以及 `data` 在 `fmt ` 之前。缺少或重复必要 chunk、截断 chunk、不一致的 byte rate/block align，或非完整交错采样帧长度都会报错。
- 只比较声明的 PCM 数据，不比较其他 metadata 或整个文件的字节一致性。

先运行上面的短解码命令，再把同一份生成的输出与自身比较，可验证最简单的相同比较流程：

```sh
cargo run --release --locked --bin avs3a-wav-compare -- target/cli-example.wav target/cli-example.wav
```

报告包括两边及共同前缀的 PCM 帧数、比较样本数、不同样本数/比例、最大绝对误差、RMS 误差、首个差异以及逐声道指标。这里的 **PCM frame 是每声道各一个采样点**，与 AV3A 编码帧不同；首差异的 frame 和 channel 编号均从 0 开始，`delta` 为左样本减右样本，误差以 PCM16 的 LSB 为单位。

长度不同时只比较共同前缀；较长一侧多出来的尾部不进入误差和差异样本数，但最终结果仍为 `result: PCM16 data differs`。只有长度相同且所有 PCM 样本相同时才报告 `result: identical PCM16 data`。**两种结果的退出码都是 0**；脚本判断音频一致性时须检查结果文字和长度，不能把该工具直接当作以退出码表示“相同/不同”的 `diff`。格式或读取失败则返回 1。

## `avs3a-model-inspect`：检查模型

```text
avs3a-model-inspect [--xor-55|--plain] [--hyper|--vae] [--cnn-fingerprint] <model.bin|->
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| 模型路径或 `-` | 必填 | 一个本地模型文件，或从标准输入读取原始二进制字节直至 EOF |
| `--xor-55` | 开启 | 将输入按逐字节 XOR `0x55` 编码形式解析 |
| `--plain` | 关闭 | 按未做 XOR 的明文二进制模型解析 |
| `--hyper` | 开启 | 按 hyper-prior 模型解析 |
| `--vae` | 关闭 | 按 VAE 模型解析 |
| `--cnn-fingerprint` | 关闭 | 使用确定性输入运行模型的 CNN decoder，输出 fingerprint 及首尾输出的位模式 |
| `-h`、`--help` | 关闭 | 参数解析遇到时显示帮助并退出 |

选项可放在模型路径前后。同组的编码/模型类型选项若重复或冲突，**最后一个生效**，不会报互斥冲突。未知的 `-` 开头选项（单独的 `-` 除外）或第二个模型路径会报错。工具不自动探测编码或模型类型，不接受任意 ONNX、PyTorch 等模型格式，也不会替换音频解码工具使用的内置模型。

仓库的 [avs3a_hyper_model.bin](<../assets/avs3a_hyper_model.bin>) 已经移除了 XOR，必须显式使用 `--plain`，不能直接套用 CLI 默认值：

```sh
cargo run --release --locked --bin avs3a-model-inspect -- --plain --hyper assets/avs3a_hyper_model.bin
cargo run --release --locked --bin avs3a-model-inspect -- --plain --hyper --cnn-fingerprint assets/avs3a_hyper_model.bin
```

在支持二进制输入重定向的 POSIX shell 中，也可从标准输入读取同一个模型：

```sh
cargo run --release --locked --bin avs3a-model-inspect -- --plain --hyper - < assets/avs3a_hyper_model.bin
```

PowerShell 用户可直接使用上面的文件路径形式，避免通过文本管道改变二进制内容。模型文件与 stdin 输入均有 **4 MiB（4,194,304 字节）**上限；超限、截断、尾随数据或非法结构/维度会失败。

正常报告包含模型类型、编码、输入字节数、base/context 的输入和 latent 形状、medians/CDF 数量，以及 encoder/decoder 各层信息。fingerprint 用于比较同一确定性 CNN 运算结果，不是文件 SHA-256，也不是音频解码正确性证明；该选项不会自动将输出与某个基准判定为通过或失败。

## 遇到问题时

| 现象 | 检查方向 |
|---|---|
| `no complete ... frame...` | 输入是否为空、是否真是 AV3A；单声道工具是否误收了容器或多声道文件 |
| `CRC mismatch` | 输入帧损坏或不完整；通用/单声道解码不能关闭此校验 |
| `refusing to synthesize after parser resynchronization...` | 裸流存在前导垃圾、插入字节或非法帧头；先用 `avs3a-inspect` 查看，不要把检查器能恢复误解为解码器会容错拼接 |
| `configuration changed` | 同一输出中出现码率、声道等配置变化；工具不会自动重新配置并拼接 |
| WAV 留下了文件但命令失败 | 可能只是失败之前的 PCM 前缀，或写入/收尾失败；不要仅凭文件存在判定成功 |
| WAV `data differs` 但退出码为 0 | 这是比较成功但内容不同，非工具执行失败 |
| 检查内置模型时出现结构错误 | 确认给了 `--plain --hyper`，而不是默认的 XOR 编码 |

需要复核本地构建时，可从仓库根目录执行：

```sh
cargo test --locked --bin avs3a-inspect --bin avs3a-wav-compare
cargo test --locked --test mp4_container
```

前一条覆盖检查器配置变化统计与 WAV 比较单元测试；后一条使用仓库自带样例覆盖容器提帧、解码一致性与跳转预热相关行为。它们不替代对自备输入执行完整检查和全流解码。
