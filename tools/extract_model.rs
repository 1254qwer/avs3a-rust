use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

const EXPECTED_MODEL_BYTES: usize = 79_930;
const XOR_MASK: u8 = 0x55;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let input = arguments
        .next()
        .ok_or("usage: extract_model <model.h> <output.bin>")?;
    let output = arguments
        .next()
        .ok_or("usage: extract_model <model.h> <output.bin>")?;
    if arguments.next().is_some() {
        return Err("usage: extract_model <model.h> <output.bin>".into());
    }

    let source = fs::read_to_string(&input)?;
    let mut model = Vec::with_capacity(EXPECTED_MODEL_BYTES);
    let mut cursor = 0;
    while let Some(relative) = source[cursor..].find("0x") {
        let start = cursor + relative + 2;
        let end = start.checked_add(2).ok_or("model offset overflow")?;
        let digits = source.get(start..end).ok_or("truncated hexadecimal byte")?;
        model.push(u8::from_str_radix(digits, 16)? ^ XOR_MASK);
        cursor = end;
    }
    if model.len() != EXPECTED_MODEL_BYTES {
        return Err(format!(
            "parsed {} model bytes; expected {EXPECTED_MODEL_BYTES}",
            model.len()
        )
        .into());
    }

    let fingerprint = model.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    });

    let output = Path::new(&output);
    fs::write(output, model)?;
    println!(
        "wrote {EXPECTED_MODEL_BYTES} decrypted bytes to {} (FNV-1a {fingerprint:016x})",
        output.display()
    );
    Ok(())
}
