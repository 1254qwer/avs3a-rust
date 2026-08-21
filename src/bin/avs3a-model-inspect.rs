use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{self, Read};
use std::process::ExitCode;

use avs3a::neural::{
    CnnNetwork, DEFAULT_MAX_MODEL_BYTES, ModelEncoding, NeuralCodecModel, NeuralModel,
    NeuralModelType, ScalarCnnDecoder,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut encoding = ModelEncoding::Xor55;
    let mut model_type = NeuralModelType::Hyper;
    let mut cnn_fingerprint = false;
    let mut input_path = None;

    for argument in env::args().skip(1) {
        match argument.as_str() {
            "--plain" => encoding = ModelEncoding::Plain,
            "--xor-55" => encoding = ModelEncoding::Xor55,
            "--vae" => model_type = NeuralModelType::Vae,
            "--hyper" => model_type = NeuralModelType::Hyper,
            "--cnn-fingerprint" => cnn_fingerprint = true,
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            _ if argument.starts_with('-') && argument != "-" => {
                return Err(format!("unknown option {argument}").into());
            }
            _ if input_path.is_none() => input_path = Some(argument),
            _ => return Err("only one model input may be supplied".into()),
        }
    }

    let input_path = input_path.ok_or("missing model input; use --help for usage")?;
    let bytes = if input_path == "-" {
        read_limited(io::stdin().lock())?
    } else {
        read_limited(File::open(&input_path)?)?
    };
    let model = NeuralModel::from_bytes(&bytes, model_type, encoding)?;

    println!(
        "model: {:?}, encoding: {:?}, bytes: {}",
        model.model_type(),
        encoding,
        bytes.len()
    );
    print_codec("base", model.base());
    if let Some(context) = model.context() {
        print_codec("context", context);
    }
    if cnn_fingerprint {
        if let Some(context) = model.context() {
            print_cnn_fingerprint("context", context, 17, 13, 31, 15, 16)?;
        }
        print_cnn_fingerprint("base", model.base(), 19, 11, 47, 23, 32)?;
    }
    Ok(())
}

fn read_limited(mut input: impl Read) -> Result<Vec<u8>, Box<dyn Error>> {
    let limit = DEFAULT_MAX_MODEL_BYTES
        .checked_add(1)
        .ok_or("model input limit overflow")?;
    let mut bytes = Vec::new();
    input.by_ref().take(limit as u64).read_to_end(&mut bytes)?;
    if bytes.len() > DEFAULT_MAX_MODEL_BYTES {
        return Err(format!("model input exceeds {} bytes", DEFAULT_MAX_MODEL_BYTES).into());
    }
    Ok(bytes)
}

fn print_codec(name: &str, codec: &NeuralCodecModel) {
    println!(
        "{name}: input {}x{}, latent {}x{}, medians {}, CDFs {}",
        codec.input_shape().dimensions(),
        codec.input_shape().channels(),
        codec.latent_shape().dimensions(),
        codec.latent_shape().channels(),
        codec.quantizer().channels(),
        codec.range_coder().cdfs().len()
    );
    if let Some(scales) = codec.context_scales() {
        println!("  context scales: {}", scales.scales().len());
    }
    print_network("encoder", codec.encoder());
    print_network("decoder", codec.decoder());
}

fn print_network(name: &str, network: &CnnNetwork) {
    println!("  {name}: {} layers", network.layers().len());
    for (index, layer) in network.layers().iter().enumerate() {
        println!(
            "    {index}: {}x{} -> {}x{}, kernel {}, stride {}, {:?}, {:?}, bias {}",
            layer.input_shape().dimensions(),
            layer.input_shape().channels(),
            layer.output_shape().dimensions(),
            layer.output_shape().channels(),
            layer.kernel_size(),
            layer.stride(),
            layer.padding(),
            layer.activation(),
            layer.bias().is_some()
        );
    }
}

fn print_cnn_fingerprint(
    name: &str,
    codec: &NeuralCodecModel,
    dimension_factor: usize,
    channel_factor: usize,
    modulus: usize,
    offset: usize,
    divisor: usize,
) -> Result<(), Box<dyn Error>> {
    let shape = codec.latent_shape();
    let mut input = vec![0.0_f32; shape.len()];
    for channel in 0..shape.channels() {
        for dimension in 0..shape.dimensions() {
            let integer = (dimension * dimension_factor + channel * channel_factor) % modulus;
            input[dimension + channel * shape.dimensions()] =
                (integer as f32 - offset as f32) / divisor as f32;
        }
    }
    let mut decoder = ScalarCnnDecoder::new(codec.decoder())?;
    let output = decoder.decode(&input)?;
    let hash = output
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3)
        });
    let first = output
        .first()
        .ok_or("CNN produced an empty output")?
        .to_bits();
    let last = output
        .last()
        .ok_or("CNN produced an empty output")?
        .to_bits();
    println!("{name} CNN fingerprint: {hash:016x}, first {first:08x}, last {last:08x}");
    Ok(())
}

fn print_usage() {
    println!(
        "Usage: avs3a-model-inspect [--xor-55|--plain] [--hyper|--vae] [--cnn-fingerprint] <model.bin|->\n\
         Defaults match the C decoder's embedded model: XOR 0x55 and hyper-prior."
    );
}
