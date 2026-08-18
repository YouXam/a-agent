pub fn bound_stdin(input: &[u8], max_bytes: usize) -> String {
    if input.len() <= max_bytes {
        return String::from_utf8_lossy(input).into_owned();
    }
    let mut start = input.len().saturating_sub(max_bytes);
    while start < input.len() && (input[start] & 0b1100_0000) == 0b1000_0000 {
        start += 1;
    }
    format!(
        "[stdin truncated; showing last {max_bytes} bytes]\n{}",
        String::from_utf8_lossy(&input[start..])
    )
}
