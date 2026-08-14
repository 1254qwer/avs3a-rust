use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

const BASIS_TABLE_LEN: usize = 1_343;
const SIN_TABLE_LEN: usize = 257;

fn table_body<'source>(source: &'source str, name: &str) -> Result<&'source str, Box<dyn Error>> {
    let marker = format!("const {} {name}[", if name == "avs3_hoa_sin_table" { "float" } else { "short" });
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
    Ok(&source[open..close])
}

fn parse_integers(body: &str) -> Result<Vec<i16>, Box<dyn Error>> {
    let mut values = Vec::new();
    let mut token = String::new();
    for character in body.chars().chain(std::iter::once(' ')) {
        if character.is_ascii_digit() || (character == '-' && token.is_empty()) {
            token.push(character);
        } else if !token.is_empty() {
            values.push(token.parse()?);
            token.clear();
        }
    }
    Ok(values)
}

fn parse_floats(body: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    body.split(',')
        .filter_map(|token| {
            let token = token.trim();
            (!token.is_empty()).then_some(token)
        })
        .map(|token| {
            token
                .strip_suffix('f')
                .or_else(|| token.strip_suffix('F'))
                .unwrap_or(token)
                .parse::<f32>()
                .map_err(Into::into)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let input = arguments
        .next()
        .ok_or("usage: extract_hoa_spatial_tables <avs3_rom_com.c> <output.bin>")?;
    let output = arguments
        .next()
        .ok_or("usage: extract_hoa_spatial_tables <avs3_rom_com.c> <output.bin>")?;
    if arguments.next().is_some() {
        return Err(
            "usage: extract_hoa_spatial_tables <avs3_rom_com.c> <output.bin>".into(),
        );
    }

    let source = fs::read_to_string(input)?;
    let mut angles = parse_integers(table_body(
        &source,
        "avs3_hoa_fixed_angle_basis_matrix",
    )?)?;
    if angles.len() > BASIS_TABLE_LEN * 2 {
        return Err(format!(
            "parsed {} HOA angle values; declared table only holds {}",
            angles.len(),
            BASIS_TABLE_LEN * 2
        )
        .into());
    }
    // The C definition explicitly initializes 1340 of the declared 1343
    // rows. Static storage zero-initializes the remaining three rows.
    angles.resize(BASIS_TABLE_LEN * 2, 0);
    let sine = parse_floats(table_body(&source, "avs3_hoa_sin_table")?)?;
    if sine.len() != SIN_TABLE_LEN {
        return Err(format!(
            "parsed {} HOA sine values; expected {SIN_TABLE_LEN}",
            sine.len()
        )
        .into());
    }

    let expected_bytes = angles.len() * std::mem::size_of::<i16>()
        + sine.len() * std::mem::size_of::<f32>();
    let mut bytes = Vec::with_capacity(expected_bytes);
    for value in angles {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in sine {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    debug_assert_eq!(bytes.len(), expected_bytes);

    let fingerprint = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    });
    let output = Path::new(&output);
    fs::write(output, bytes)?;
    println!(
        "wrote {BASIS_TABLE_LEN} HOA angle pairs and {SIN_TABLE_LEN} sine values to {} (FNV-1a {fingerprint:016x})",
        output.display()
    );
    Ok(())
}
