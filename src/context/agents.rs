use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentsFile {
    pub path: PathBuf,
    pub content: String,
}

pub fn discover_agents(cwd: &Path, global: Option<&Path>) -> Result<Vec<AgentsFile>> {
    let mut files = Vec::new();
    if let Some(path) = global.filter(|path| path.is_file()) {
        files.push(read_agents(path)?);
    }

    let boundary = repository_boundary(cwd);
    let mut candidates = Vec::new();
    let mut current = Some(cwd);
    while let Some(path) = current {
        candidates.push(path.join("AGENTS.md"));
        if path == boundary {
            break;
        }
        current = path.parent();
    }
    candidates.reverse();
    for path in candidates.into_iter().filter(|path| path.is_file()) {
        files.push(read_agents(&path)?);
    }
    Ok(files)
}

pub fn discover_agents_for_targets(
    cwd: &Path,
    global: Option<&Path>,
    targets: &[PathBuf],
) -> Result<Vec<AgentsFile>> {
    let global_path = global.map(Path::to_path_buf);
    let mut unique = BTreeMap::new();
    for item in discover_agents(cwd, global)? {
        unique.insert(item.path.clone(), item);
    }
    for target in targets {
        let directory = if target.is_dir() {
            target.as_path()
        } else {
            target.parent().unwrap_or(cwd)
        };
        for item in discover_agents(directory, None)? {
            unique.insert(item.path.clone(), item);
        }
    }
    let mut files = unique.into_values().collect::<Vec<_>>();
    files.sort_by_key(|item| {
        (
            usize::from(global_path.as_ref() != Some(&item.path)),
            item.path.components().count(),
            item.path.clone(),
        )
    });
    Ok(files)
}

fn repository_boundary(cwd: &Path) -> &Path {
    cwd.ancestors()
        .find(|path| path.join(".git").exists())
        .unwrap_or(cwd)
}

fn read_agents(path: &Path) -> Result<AgentsFile> {
    Ok(AgentsFile {
        path: path.to_path_buf(),
        content: fs::read_to_string(path)
            .with_context(|| format!("read AGENTS.md {}", path.display()))?,
    })
}
