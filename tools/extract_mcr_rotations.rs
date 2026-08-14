use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

const CODEBOOKS: [(&str, usize); 2] = [
    ("mcr_codebook_9bit", 512 * 3),
    ("mcr_codebook_8bit", 256 * 3),
];

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let input = arguments
        .next()
        .ok_or("usage: extract_mcr_rotations <avs3_rom_com.c> <output.bin>")?;
    let output = arguments
        .next()
        .ok_or("usage: extract_mcr_rotations <avs3_rom_com.c> <output.bin>")?;
    if arguments.next().is_some() {
        return Err("usage: extract_mcr_rotations <avs3_rom_com.c> <output.bin>".into());
    }

    let source = fs::read_to_string(&input)?;
    let expected_angles: usize = CODEBOOKS.iter().map(|(_, count)| count).sum();
    let mut bytes = Vec::with_capacity(expected_angles * 2 * std::mem::size_of::<f32>());
    for (name, expected_count) in CODEBOOKS {
        let marker = format!("const float {name}[");
        let declaration = source
            .find(&marker)
            .ok_or_else(|| format!("missing C table {name}"))?;
        let open = source[declaration..]
            .find('{')
            .map(|offset| declaration + offset + 1)
            .ok_or_else(|| format!("missing opening brace for {name}"))?;
        let close = source[open..]
            .find("};")
            .map(|offset| open + offset)
            .ok_or_else(|| format!("missing closing brace for {name}"))?;

        let mut count = 0;
        for token in source[open..close].split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            let token = token
                .strip_suffix('f')
                .or_else(|| token.strip_suffix('F'))
                .unwrap_or(token);
            let theta: f32 = token.parse()?;
            let cosine = f64::from(theta).cos() as f32;
            let sine = f64::from(theta).sin() as f32;
            bytes.extend_from_slice(&cosine.to_le_bytes());
            bytes.extend_from_slice(&sine.to_le_bytes());
            count += 1;
        }
        if count != expected_count {
            return Err(format!("parsed {count} values from {name}; expected {expected_count}").into());
        }
    }

    let expected_bytes = expected_angles * 2 * std::mem::size_of::<f32>();
    if bytes.len() != expected_bytes {
        return Err(format!(
            "serialized {} bytes; expected {expected_bytes}",
            bytes.len()
        )
        .into());
    }
    let fingerprint = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    });

    let output = Path::new(&output);
    fs::write(output, bytes)?;
    println!(
        "wrote {expected_angles} cos/sin pairs to {} (FNV-1a {fingerprint:016x})",
        output.display()
    );
    Ok(())
}
