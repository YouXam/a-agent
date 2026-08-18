use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Take};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const METADATA_LIMIT: u64 = 8192;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillMetadata {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

pub fn discover_skills(global_root: &Path, project_root: &Path) -> Result<Vec<SkillMetadata>> {
    let mut skills = BTreeMap::new();
    index_root(global_root, &mut skills)?;
    index_root(project_root, &mut skills)?;
    Ok(skills.into_values().collect())
}

fn index_root(root: &Path, skills: &mut BTreeMap<String, SkillMetadata>) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    let mut entries = fs::read_dir(root)
        .with_context(|| format!("read skills directory {}", root.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path().join("SKILL.md");
        if !path.is_file() {
            continue;
        }
        let mut source = String::new();
        let mut reader: Take<File> = File::open(&path)?.take(METADATA_LIMIT);
        reader.read_to_string(&mut source)?;
        let metadata = parse_skill_metadata(&source, &path)?;
        skills.insert(metadata.name.clone(), metadata);
    }
    Ok(())
}

pub fn parse_skill_metadata(source: &str, path: &Path) -> Result<SkillMetadata> {
    let fallback_name = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .context("skill path has no UTF-8 directory name")?;
    let mut name = None;
    let mut description = None;

    if source.starts_with("---\n") || source.starts_with("---\r\n") {
        for line in source
            .lines()
            .skip(1)
            .take_while(|line| line.trim() != "---")
        {
            if let Some((key, value)) = line.split_once(':') {
                let value = value.trim().trim_matches(['\'', '"']);
                match key.trim() {
                    "name" => name = Some(value.to_owned()),
                    "description" => description = Some(value.to_owned()),
                    _ => {}
                }
            }
        }
    }
    if description.is_none() {
        description = source.lines().find_map(|line| {
            let line = line.trim();
            (!line.is_empty() && line != "---" && !line.starts_with('#')).then(|| line.to_owned())
        });
    }
    Ok(SkillMetadata {
        name: name.unwrap_or_else(|| fallback_name.to_owned()),
        description: description.unwrap_or_else(|| "No description provided.".into()),
        path: path.to_path_buf(),
    })
}
