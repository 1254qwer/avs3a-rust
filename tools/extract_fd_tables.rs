use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

const TABLES: [(&str, usize); 13] = [
    ("mean_lsf", 16),
    ("lsf_stage1_CB1_hbr", 256 * 9),
    ("lsf_stage1_CB2_hbr", 256 * 7),
    ("lsf_stage2_CB1_hbr", 128 * 3),
    ("lsf_stage2_CB2_hbr", 128 * 3),
    ("lsf_stage2_CB3_hbr", 64 * 3),
    ("lsf_stage2_CB4_hbr", 32 * 3),
    ("lsf_stage2_CB5_hbr", 32 * 4),
    ("lsf_stage1_CB1_lbr", 256 * 9),
    ("lsf_stage1_CB2_lbr", 256 * 7),
    ("lsf_stage2_CB1_lbr", 128 * 5),
    ("lsf_stage2_CB2_lbr", 128 * 4),
    ("lsf_stage2_CB3_lbr", 64 * 7),
];

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let input = arguments
        .next()
        .ok_or("usage: extract_fd_tables <avs3_rom_com.c> <output.bin>")?;
    let output = arguments
        .next()
        .ok_or("usage: extract_fd_tables <avs3_rom_com.c> <output.bin>")?;
    if arguments.next().is_some() {
        return Err("usage: extract_fd_tables <avs3_rom_com.c> <output.bin>".into());
    }

    let source = fs::read_to_string(&input)?;
    let expected_values: usize = TABLES.iter().map(|(_, count)| count).sum();
    let mut bytes = Vec::with_capacity(expected_values * 4);
    for (name, expected_count) in TABLES {
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
            let value: f32 = token.parse()?;
            bytes.extend_from_slice(&value.to_le_bytes());
            count += 1;
        }
        if count != expected_count {
            return Err(format!("parsed {count} values from {name}; expected {expected_count}").into());
        }
    }

    if bytes.len() != expected_values * 4 {
        return Err(format!(
            "serialized {} bytes; expected {}",
            bytes.len(),
            expected_values * 4
        )
        .into());
    }
    let fingerprint = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    });

    let output = Path::new(&output);
    fs::write(output, bytes)?;
    println!(
        "wrote {expected_values} f32 values to {} (FNV-1a {fingerprint:016x})",
        output.display()
    );
    Ok(())
}
