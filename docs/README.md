# 使用指南

这份文档面向「把 AV3A 解码接入自己的程序」和「直接使用命令行工具」的读者。
它描述的是当前仓库的公开接口；项目仍处于开发阶段，升级依赖时请重新核对 API 和输出。

## 从哪里开始

| 你要做什么 | 阅读入口 |
|---|---|
| 先把一个文件转成 WAV | [命令行使用](<cli.md>) |
| 在 Rust 中解码裸流、网络分片或输出 PCM | [Rust 库使用](<library.md>) |
| 在 M4A/MP4 中定位音轨、按需读 sample、seek、边下载边解码 | [MP4/M4A 与播放器集成](<mp4.md>) |
| 复制一份完整的文件解码程序 | [裸流示例](<../examples/decode_raw.rs>)、[MP4 与 seek 示例](<../examples/decode_mp4.rs>) |
| 查精确类型、方法签名或更底层的 DSP 接口 | 运行 `cargo doc --no-deps --open`，再看 [公开模块入口](<../src/lib.rs>) |
| 了解 profile 支持和真实流验证范围 | [项目概览](<../README.md>) |

## 构建与引入依赖

需要 **Rust 1.96 或更新版本**，以 [Cargo 清单](<../Cargo.toml>) 的 `rust-version` 为准。
使用对应平台的 Rust 工具链和链接器；Windows MSVC 工具链需要相应的 C++ 构建工具。
模型与规范表已经内置，不需要另行安装 C 参考解码器或在运行时下载模型。资源说明见
[assets 说明](<../assets/README.md>)。

在仓库根目录运行：

~~~bash
cargo build --release --locked
cargo run --release --bin avs3a-decode -- input.m4a output.wav
cargo run --release --bin avs3a-inspect -- --verify-crc input.av3a
cargo doc --no-deps --open
~~~

以上输入名是占位路径，请换成自己的文件。输出会覆盖已有文件，**不要让输入与输出指向同一文件**。
要快速检查仓库附带的样本，可以运行：

~~~bash
cargo run --release --bin avs3a-decode -- tests/fixtures/test.av3a sample.wav --frames 10
~~~

该 [测试样本](<../tests/fixtures/test.av3a>) 是 44.1 kHz、12 声道的 7.1.4 流，不是普通双声道音乐；
样本来源信息见 [fixture 说明](<../tests/fixtures/README.md>)。

在自己的 Rust 工程中，以下二选一加入依赖。仓库目录名是 avs3a-rust，**crate 名是 avs3a**。

~~~toml
[dependencies]
avs3a = { path = "../avs3a-rust" }
~~~

或者使用 Git 来源：

~~~toml
[dependencies]
avs3a = { git = "https://github.com/1254qwer/avs3a-rust" }
~~~

集成时建议为 Git 依赖加上已验证提交的 `rev`，并保留调用方的 Cargo 锁文件。
这里不假设当前版本已经发布到 crates.io。

## 先分清四个层次

~~~text
裸 AV3A 字节 -> FrameStream / parse_frames ---+
                                            +-> EncodedFrame -> BuiltinDecoder -> PCM
MP4/M4A    -> Mp4FrameReader / parse_sample --+
~~~

1. **容器**负责从文件中定位压缩数据，不负责音频合成。MP4/M4A 扩展名不代表其中一定是 AV3A。
2. **压缩帧 / MP4 sample**：当前容器路径假定一个 sample 对应一个 AV3A access unit。
   它不是一个 PCM 采样点。
3. **PCM 帧**含多个声道的采样值。一个 AV3A 帧当前输出每声道 1024 点；程序应读
   `config.samples_per_channel` 和 `decoder.sample_count()`，不要硬编码。
4. **播放/渲染**是调用方工作。本库不打开声卡、不提供 HTTP 客户端、不做采样率转换，
   也不会自动把对象音频或 HOA 渲染成双声道/耳机音频。

## 推荐的集成顺序

1. 先用 CLI 确认输入能解析、CRC 正常，解码结果符合预期。
2. 用 `BuiltinDecoder::configure(frame.header())` 配置一次解码器，复用它和输出缓冲区。
3. 用实际帧头中的格式初始化音频设备；MP4 声道数、采样率声明只能作为提示。
4. 接入 seek 时同时处理读取位置、解码器 reset、预热帧、输出队列和播放时钟。
5. 最后再加按需下载、线程调度或语言绑定，分别处理 I/O 错误与解码错误。

[MP4/M4A 指南](<mp4.md>) 按这个顺序讲解，并单独列出 N-API/Electron 接入需要注意的约定。

## 示例与检查

两个示例都是完整的 Cargo example，没有额外依赖：

~~~bash
cargo check --examples --locked
cargo run --release --example decode_raw -- input.av3a output.wav
cargo run --release --example decode_mp4 -- input.m4a output.wav
# 从零基编号的第 100 个 sample 起输出，之前只解码必要的预热帧
cargo run --release --example decode_mp4 -- input.m4a from-100.wav 100
~~~

[裸流示例](<../examples/decode_raw.rs>) 演示分块读取、EOF、CRC 由解码器校验、PCM 缓冲复用和 WAV 收尾。
[MP4 示例](<../examples/decode_mp4.rs>) 演示容器索引、真实格式、sample 定位和预热；
它使用完整本地文件，启动时会读第一帧，不是 HTTP 下载器。

更新文档或示例后可运行：

~~~bash
cargo fmt --all -- --check
cargo check --examples --locked
cargo test --all-targets --locked
cargo doc --no-deps
~~~

不要把「能解析帧头」「能解出 PCM」或「有该布局的 enum 变体」等同于完整播放支持。
尤其是 Mix/HOA 的渲染、MP4 edit list、错误恢复和设备声道映射，需要调用方另外实现；
详见 [库使用的输出说明](<library.md>) 与 [容器边界](<mp4.md>)。
