use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::path::workspace_path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadArgs {
    pub path: String,
    #[serde(default)]
    pub offset: usize,
    pub limit: Option<usize>,
}

pub async fn read_text_file(root: &Path, args: &ReadArgs, max_lines: usize) -> Result<String> {
    read_text_file_bounded(root, args, max_lines, usize::MAX).await
}

pub async fn read_text_file_bounded(
    root: &Path,
    args: &ReadArgs,
    max_lines: usize,
    max_bytes: usize,
) -> Result<String> {
    let path = workspace_path(root, &args.path, true)?;
    let bytes = tokio::fs::read(&path)
        .await
        .with_context(|| format!("read {}", path.display()))?;
    if bytes.contains(&0) {
        anyhow::bail!("binary file cannot be read as text: {}", args.path);
    }
    let source = String::from_utf8(bytes)
        .with_context(|| format!("file is not valid UTF-8: {}", args.path))?;
    let lines = source.lines().collect::<Vec<_>>();
    let limit = args.limit.unwrap_or(max_lines).min(max_lines);
    let selected = lines.iter().skip(args.offset).take(limit);
    let mut output = selected
        .enumerate()
        .map(|(index, line)| format!("{}: {line}", args.offset + index + 1))
        .collect::<Vec<_>>()
        .join("\n");
    let consumed = args.offset.saturating_add(limit).min(lines.len());
    let remaining = lines.len().saturating_sub(consumed);
    if remaining > 0 {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(&format!(
            "[truncated; {remaining} more line{}]",
            if remaining == 1 { "" } else { "s" }
        ));
    }
    if output.len() > max_bytes {
        let mut end = max_bytes.min(output.len());
        while end > 0 && !output.is_char_boundary(end) {
            end -= 1;
        }
        output.truncate(end);
        output.push_str("\n[truncated by output byte limit]");
    }
    Ok(output)
}
