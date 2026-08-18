use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result};

pub fn workspace_path(root: &Path, requested: &str, must_exist: bool) -> Result<PathBuf> {
    let root = root
        .canonicalize()
        .with_context(|| format!("resolve workspace {}", root.display()))?;
    let requested = Path::new(requested);
    let candidate = if requested.is_absolute() {
        normalize(requested)?
    } else {
        normalize(&root.join(requested))?
    };
    if !candidate.starts_with(&root) {
        anyhow::bail!("path is outside the workspace: {}", requested.display());
    }

    if must_exist {
        let resolved = candidate
            .canonicalize()
            .with_context(|| format!("resolve path {}", requested.display()))?;
        if !resolved.starts_with(&root) {
            anyhow::bail!(
                "path resolves outside the workspace: {}",
                requested.display()
            );
        }
        return Ok(resolved);
    }

    let mut ancestor = candidate.as_path();
    while !ancestor.exists() {
        ancestor = ancestor
            .parent()
            .context("path has no existing parent inside workspace")?;
    }
    let resolved_parent = ancestor.canonicalize()?;
    if !resolved_parent.starts_with(&root) {
        anyhow::bail!(
            "path resolves outside the workspace: {}",
            requested.display()
        );
    }
    Ok(candidate)
}

fn normalize(path: &Path) -> Result<PathBuf> {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => result.push(prefix.as_os_str()),
            Component::RootDir => result.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !result.pop() {
                    anyhow::bail!("path escapes its root: {}", path.display());
                }
            }
            Component::Normal(value) => result.push(value),
        }
    }
    Ok(result)
}
