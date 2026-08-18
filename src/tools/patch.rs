use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::path::workspace_path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchSummary {
    pub files: Vec<PatchFileSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchFileSummary {
    pub path: String,
    pub added: usize,
    pub removed: usize,
}

#[derive(Debug)]
enum Operation {
    Add { path: String, content: String },
    Delete { path: String },
    Update { path: String, hunks: Vec<Hunk> },
}

#[derive(Debug)]
struct Hunk {
    anchor: Option<String>,
    lines: Vec<HunkLine>,
}

#[derive(Debug)]
enum HunkLine {
    Context(String),
    Add(String),
    Remove(String),
}

enum Prepared {
    Write {
        path: PathBuf,
        content: String,
        summary: PatchFileSummary,
    },
    Delete {
        path: PathBuf,
        summary: PatchFileSummary,
    },
}

pub fn affected_paths(patch: &str) -> Result<Vec<String>> {
    let lines = patch.lines().collect::<Vec<_>>();
    if lines.first() != Some(&"*** Begin Patch") || lines.last() != Some(&"*** End Patch") {
        anyhow::bail!("patch must start with '*** Begin Patch' and end with '*** End Patch'");
    }
    let mut paths = BTreeSet::new();
    for line in lines {
        let path = line
            .strip_prefix("*** Add File: ")
            .or_else(|| line.strip_prefix("*** Delete File: "))
            .or_else(|| line.strip_prefix("*** Update File: "));
        if let Some(path) = path {
            validate_patch_path(path)?;
            paths.insert(path.to_owned());
        }
    }
    Ok(paths.into_iter().collect())
}

pub async fn apply_patch(root: &Path, patch: &str) -> Result<PatchSummary> {
    let operations = parse_patch(patch)?;
    let mut prepared = Vec::with_capacity(operations.len());
    let mut seen = BTreeSet::new();
    for operation in operations {
        let requested = match &operation {
            Operation::Add { path, .. }
            | Operation::Delete { path }
            | Operation::Update { path, .. } => path,
        };
        if !seen.insert(requested.clone()) {
            anyhow::bail!("patch contains multiple operations for {requested}");
        }
        match operation {
            Operation::Add { path, content } => {
                let resolved = workspace_path(root, &path, false)?;
                if resolved.exists() {
                    anyhow::bail!("cannot add existing file: {path}");
                }
                let added = content.lines().count();
                prepared.push(Prepared::Write {
                    path: resolved,
                    content,
                    summary: PatchFileSummary {
                        path,
                        added,
                        removed: 0,
                    },
                });
            }
            Operation::Delete { path } => {
                let resolved = workspace_path(root, &path, true)?;
                let source = fs::read_to_string(&resolved)
                    .with_context(|| format!("read file before delete: {path}"))?;
                prepared.push(Prepared::Delete {
                    path: resolved,
                    summary: PatchFileSummary {
                        path,
                        added: 0,
                        removed: source.lines().count(),
                    },
                });
            }
            Operation::Update { path, hunks } => {
                let resolved = workspace_path(root, &path, true)?;
                let source = fs::read_to_string(&resolved)
                    .with_context(|| format!("read file before update: {path}"))?;
                let (content, added, removed) = apply_hunks(&source, &hunks, &path)?;
                prepared.push(Prepared::Write {
                    path: resolved,
                    content,
                    summary: PatchFileSummary {
                        path,
                        added,
                        removed,
                    },
                });
            }
        }
    }

    let mut summaries = Vec::new();
    for operation in prepared {
        match operation {
            Prepared::Write {
                path,
                content,
                summary,
            } => {
                let parent = path.parent().context("patch target has no parent")?;
                fs::create_dir_all(parent)?;
                let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
                temporary.write_all(content.as_bytes())?;
                temporary.as_file().sync_all()?;
                temporary
                    .persist(&path)
                    .map_err(|error| error.error)
                    .with_context(|| format!("atomically write {}", path.display()))?;
                summaries.push(summary);
            }
            Prepared::Delete { path, summary } => {
                fs::remove_file(&path).with_context(|| format!("delete {}", path.display()))?;
                summaries.push(summary);
            }
        }
    }
    Ok(PatchSummary { files: summaries })
}

fn parse_patch(patch: &str) -> Result<Vec<Operation>> {
    let lines = patch.lines().collect::<Vec<_>>();
    if lines.first() != Some(&"*** Begin Patch") || lines.last() != Some(&"*** End Patch") {
        anyhow::bail!("patch must start with '*** Begin Patch' and end with '*** End Patch'");
    }
    let mut operations = Vec::new();
    let mut index = 1;
    while index + 1 < lines.len() {
        let line = lines[index];
        if let Some(path) = line.strip_prefix("*** Add File: ") {
            validate_patch_path(path)?;
            index += 1;
            let mut content = Vec::new();
            while index + 1 < lines.len() && !lines[index].starts_with("*** ") {
                let value = lines[index].strip_prefix('+').with_context(|| {
                    format!("add-file line must start with '+': {}", lines[index])
                })?;
                content.push(value);
                index += 1;
            }
            let mut content = content.join("\n");
            if !content.is_empty() {
                content.push('\n');
            }
            operations.push(Operation::Add {
                path: path.into(),
                content,
            });
        } else if let Some(path) = line.strip_prefix("*** Delete File: ") {
            validate_patch_path(path)?;
            operations.push(Operation::Delete { path: path.into() });
            index += 1;
        } else if let Some(path) = line.strip_prefix("*** Update File: ") {
            validate_patch_path(path)?;
            index += 1;
            let mut hunks = Vec::new();
            while index + 1 < lines.len() && !lines[index].starts_with("*** ") {
                let header = lines[index]
                    .strip_prefix("@@")
                    .with_context(|| format!("expected hunk header, found: {}", lines[index]))?;
                let anchor = (!header.trim().is_empty()).then(|| header.trim().to_owned());
                index += 1;
                let mut hunk_lines = Vec::new();
                while index + 1 < lines.len()
                    && !lines[index].starts_with("@@")
                    && !lines[index].starts_with("*** ")
                {
                    let value = lines[index];
                    let (prefix, content) = value.split_at(1);
                    hunk_lines.push(match prefix {
                        " " => HunkLine::Context(content.into()),
                        "+" => HunkLine::Add(content.into()),
                        "-" => HunkLine::Remove(content.into()),
                        _ => anyhow::bail!("hunk line must start with ' ', '+' or '-': {value}"),
                    });
                    index += 1;
                }
                if hunk_lines.is_empty() {
                    anyhow::bail!("empty update hunk for {path}");
                }
                hunks.push(Hunk {
                    anchor,
                    lines: hunk_lines,
                });
            }
            if hunks.is_empty() {
                anyhow::bail!("update operation has no hunks: {path}");
            }
            operations.push(Operation::Update {
                path: path.into(),
                hunks,
            });
        } else {
            anyhow::bail!("unknown patch operation: {line}");
        }
    }
    if operations.is_empty() {
        anyhow::bail!("patch contains no operations");
    }
    Ok(operations)
}

fn validate_patch_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    if path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        anyhow::bail!("patch path is outside the workspace: {}", path.display());
    }
    Ok(())
}

fn apply_hunks(source: &str, hunks: &[Hunk], path: &str) -> Result<(String, usize, usize)> {
    let had_newline = source.ends_with('\n');
    let mut lines = source.lines().map(ToOwned::to_owned).collect::<Vec<_>>();
    let mut cursor = 0;
    let mut total_added = 0;
    let mut total_removed = 0;

    for hunk in hunks {
        if let Some(anchor) = &hunk.anchor
            && let Some(position) = lines[cursor..]
                .iter()
                .position(|line| line.contains(anchor))
        {
            cursor += position;
        }
        let old = hunk
            .lines
            .iter()
            .filter_map(|line| match line {
                HunkLine::Context(value) | HunkLine::Remove(value) => Some(value.as_str()),
                HunkLine::Add(_) => None,
            })
            .collect::<Vec<_>>();
        let new = hunk
            .lines
            .iter()
            .filter_map(|line| match line {
                HunkLine::Context(value) | HunkLine::Add(value) => Some(value.clone()),
                HunkLine::Remove(_) => None,
            })
            .collect::<Vec<_>>();
        let position = if old.is_empty() {
            cursor
        } else {
            find_sequence(&lines, &old, cursor)
                .with_context(|| format!("context not found in {path}"))?
        };
        total_added += hunk
            .lines
            .iter()
            .filter(|line| matches!(line, HunkLine::Add(_)))
            .count();
        total_removed += hunk
            .lines
            .iter()
            .filter(|line| matches!(line, HunkLine::Remove(_)))
            .count();
        lines.splice(position..position + old.len(), new.iter().cloned());
        cursor = position + new.len();
    }
    let mut result = lines.join("\n");
    if had_newline {
        result.push('\n');
    }
    Ok((result, total_added, total_removed))
}

fn find_sequence(lines: &[String], needle: &[&str], start: usize) -> Option<usize> {
    (start..=lines.len().saturating_sub(needle.len())).find(|&position| {
        lines[position..position + needle.len()]
            .iter()
            .map(String::as_str)
            .eq(needle.iter().copied())
    })
}
