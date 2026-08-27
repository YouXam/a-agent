use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::path::unrestricted_path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchSummary {
    pub files: Vec<PatchFileSummary>,
    /// What each touched file looked like beforehand, so the change can be
    /// undone. Absent for files too large to keep in the session.
    pub snapshots: Vec<FileSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchFileSummary {
    pub path: String,
    pub added: usize,
    pub removed: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileChange {
    Added,
    Modified,
    Deleted,
}

impl FileChange {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Modified => "modified",
            Self::Deleted => "deleted",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "added" => Some(Self::Added),
            "modified" => Some(Self::Modified),
            "deleted" => Some(Self::Deleted),
            _ => None,
        }
    }
}

/// One file as it was before a patch touched it, with a fingerprint of what the
/// patch left behind so a later restore can tell whether anything else has
/// changed the file since.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSnapshot {
    pub path: PathBuf,
    pub change: FileChange,
    pub before: Option<String>,
    pub after_len: u64,
    pub after_hash: u64,
    pub added: usize,
    pub removed: usize,
    /// False when the previous contents were too large to keep, which is
    /// reported rather than silently truncated.
    pub restorable: bool,
}

/// Collapses a rewind range into one entry per path, describing the whole range
/// rather than each patch in it.
///
/// A path the agent touched more than once has one snapshot per patch, and each
/// one's `after_*` describes only that patch. Checking them individually calls
/// every snapshot but the last "changed after the agent wrote it", which is both
/// useless and untrue — the agent's own later edit is what changed it. What
/// matters is the state at the start of the range and the state the agent last
/// left the file in.
///
/// `newest_first` must be ordered newest to oldest, as the store returns it.
pub fn coalesce_snapshots(newest_first: Vec<FileSnapshot>) -> Vec<FileSnapshot> {
    let mut order = Vec::new();
    let mut grouped: BTreeMap<PathBuf, (FileSnapshot, FileSnapshot)> = BTreeMap::new();
    for snapshot in newest_first {
        match grouped.get_mut(&snapshot.path) {
            // Later iterations are older, so this becomes the start of the range.
            Some((oldest, _)) => {
                oldest.added += snapshot.added;
                oldest.removed += snapshot.removed;
                let added = oldest.added;
                let removed = oldest.removed;
                *oldest = FileSnapshot {
                    added,
                    removed,
                    ..snapshot
                };
            }
            None => {
                order.push(snapshot.path.clone());
                grouped.insert(snapshot.path.clone(), (snapshot.clone(), snapshot));
            }
        }
    }

    let mut coalesced = Vec::new();
    for path in order {
        let Some((oldest, newest)) = grouped.remove(&path) else {
            continue;
        };
        let existed_before = oldest.change != FileChange::Added;
        let exists_after = newest.change != FileChange::Deleted;
        let change = match (existed_before, exists_after) {
            // Created and then removed again: the range left nothing behind.
            (false, false) => continue,
            (false, true) => FileChange::Added,
            (true, false) => FileChange::Deleted,
            (true, true) => FileChange::Modified,
        };
        coalesced.push(FileSnapshot {
            change,
            after_len: newest.after_len,
            after_hash: newest.after_hash,
            ..oldest
        });
    }
    coalesced
}

impl FileSnapshot {
    /// Why this snapshot cannot be put back, or `None` when it can. A file the
    /// agent wrote and something else then edited is left alone: reverting it
    /// would throw away the other edit.
    pub fn restore_blocker(&self) -> Option<String> {
        if !self.restorable {
            return Some("the previous contents were too large to keep".into());
        }
        let current = std::fs::read_to_string(&self.path);
        match self.change {
            FileChange::Deleted => match current {
                Ok(_) => Some("something recreated this file".into()),
                Err(_) => None,
            },
            FileChange::Added | FileChange::Modified => match current {
                Ok(current)
                    if current.len() as u64 == self.after_len
                        && content_hash(&current) == self.after_hash =>
                {
                    None
                }
                Ok(_) => Some("changed since the agent last wrote it".into()),
                Err(error) => Some(format!("unreadable: {error}")),
            },
        }
    }

    /// Puts the file back as it was. Callers are expected to have checked
    /// [`Self::restore_blocker`] first.
    pub fn restore(&self) -> Result<()> {
        match self.change {
            FileChange::Added => std::fs::remove_file(&self.path)
                .with_context(|| format!("remove {}", self.path.display())),
            FileChange::Modified | FileChange::Deleted => {
                let before = self
                    .before
                    .as_deref()
                    .context("snapshot kept no previous contents")?;
                if let Some(parent) = self.path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&self.path, before)
                    .with_context(|| format!("write {}", self.path.display()))
            }
        }
    }
}

/// Cheap content fingerprint. Only ever compared against another fingerprint
/// this program produced, so a non-cryptographic hash is enough.
pub fn content_hash(content: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
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
    Add {
        path: PathBuf,
        content: String,
        summary: PatchFileSummary,
    },
    Update {
        path: PathBuf,
        content: String,
        before: String,
        summary: PatchFileSummary,
    },
    Delete {
        path: PathBuf,
        before: String,
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
    apply_patch_with_snapshots(root, patch, usize::MAX).await
}

/// `snapshot_max_bytes` caps how much of a file's previous contents is kept for
/// undo; anything larger is recorded as unrestorable instead of truncated, since
/// a partial restore would corrupt the file.
pub async fn apply_patch_with_snapshots(
    root: &Path,
    patch: &str,
    snapshot_max_bytes: usize,
) -> Result<PatchSummary> {
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
                let resolved = unrestricted_path(root, &path, false)?;
                if resolved.exists() {
                    anyhow::bail!("cannot add existing file: {path}");
                }
                let added = content.lines().count();
                prepared.push(Prepared::Add {
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
                let resolved = unrestricted_path(root, &path, true)?;
                let source = fs::read_to_string(&resolved)
                    .with_context(|| format!("read file before delete: {path}"))?;
                prepared.push(Prepared::Delete {
                    path: resolved,
                    summary: PatchFileSummary {
                        path,
                        added: 0,
                        removed: source.lines().count(),
                    },
                    before: source,
                });
            }
            Operation::Update { path, hunks } => {
                let resolved = unrestricted_path(root, &path, true)?;
                let source = fs::read_to_string(&resolved)
                    .with_context(|| format!("read file before update: {path}"))?;
                let (content, added, removed) = apply_hunks(&source, &hunks, &path)?;
                prepared.push(Prepared::Update {
                    path: resolved,
                    content,
                    before: source,
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
    let mut snapshots = Vec::new();
    let keep = |before: String| {
        let restorable = before.len() <= snapshot_max_bytes;
        (restorable.then_some(before), restorable)
    };
    for operation in prepared {
        match operation {
            Prepared::Add {
                path,
                content,
                summary,
            } => {
                let parent = path.parent().context("patch target has no parent")?;
                fs::create_dir_all(parent)?;
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .with_context(|| format!("create {}", path.display()))?;
                let result = file
                    .write_all(content.as_bytes())
                    .and_then(|()| file.sync_all());
                if let Err(error) = result {
                    drop(file);
                    let _ = fs::remove_file(&path);
                    return Err(error)
                        .with_context(|| format!("write new file {}", path.display()));
                }
                snapshots.push(FileSnapshot {
                    path: path.clone(),
                    change: FileChange::Added,
                    before: None,
                    after_len: content.len() as u64,
                    after_hash: content_hash(&content),
                    added: summary.added,
                    removed: summary.removed,
                    restorable: true,
                });
                summaries.push(summary);
            }
            Prepared::Update {
                path,
                content,
                before,
                summary,
            } => {
                let mut file = OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(&path)
                    .with_context(|| format!("open {} for update", path.display()))?;
                file.write_all(content.as_bytes())?;
                file.sync_all()?;
                let (before, restorable) = keep(before);
                snapshots.push(FileSnapshot {
                    path: path.clone(),
                    change: FileChange::Modified,
                    before,
                    after_len: content.len() as u64,
                    after_hash: content_hash(&content),
                    added: summary.added,
                    removed: summary.removed,
                    restorable,
                });
                summaries.push(summary);
            }
            Prepared::Delete {
                path,
                before,
                summary,
            } => {
                fs::remove_file(&path).with_context(|| format!("delete {}", path.display()))?;
                let (before, restorable) = keep(before);
                snapshots.push(FileSnapshot {
                    path: path.clone(),
                    change: FileChange::Deleted,
                    before,
                    after_len: 0,
                    after_hash: content_hash(""),
                    added: summary.added,
                    removed: summary.removed,
                    restorable,
                });
                summaries.push(summary);
            }
        }
    }
    Ok(PatchSummary {
        files: summaries,
        snapshots,
    })
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
    if path.trim().is_empty() {
        anyhow::bail!("patch path must not be empty");
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
