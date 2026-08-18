use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result};

pub fn unrestricted_path(root: &Path, requested: &str, must_exist: bool) -> Result<PathBuf> {
    let root = root
        .canonicalize()
        .with_context(|| format!("resolve workspace {}", root.display()))?;
    let requested = Path::new(requested);
    let candidate = if requested.is_absolute() {
        normalize(requested)?
    } else {
        normalize(&root.join(requested))?
    };
    if must_exist {
        candidate
            .canonicalize()
            .with_context(|| format!("resolve path {}", requested.display()))
    } else {
        Ok(candidate)
    }
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
