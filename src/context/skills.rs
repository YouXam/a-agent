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

/// Directories scanned for Agent Skills, lowest precedence first.
///
/// The Agent Skills specification defines what a skill directory contains but
/// not where it lives. Only the shared `.agents/skills` convention is used, so a
/// skill installed by any compliant client is visible here and vice versa. The
/// user scope is scanned before the project scope, so a project skill takes
/// precedence over a user skill with the same name.
pub fn skill_roots(home: &Path, project_root: &Path) -> Vec<PathBuf> {
    vec![
        home.join(".agents/skills"),
        project_root.join(".agents/skills"),
    ]
}

pub fn discover_skills(roots: &[PathBuf]) -> Result<Vec<SkillMetadata>> {
    let mut skills = BTreeMap::new();
    for root in roots {
        index_root(root, &mut skills)?;
    }
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
        match parse_skill_metadata(&source, &path) {
            Ok(metadata) => {
                skills.insert(metadata.name.clone(), metadata);
            }
            Err(error) => eprintln!("warning: skipping skill {}: {error}", path.display()),
        }
    }
    Ok(())
}

/// Reads the `name` and `description` of a skill from its YAML frontmatter.
///
/// Only top-level keys are read, so a nested `metadata` map cannot shadow them.
/// Values may be plain, quoted, or block scalars. An unquoted colon is kept as
/// part of the value instead of failing, because skills written for other
/// clients frequently contain them.
pub fn parse_skill_metadata(source: &str, path: &Path) -> Result<SkillMetadata> {
    let directory = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .context("skill path has no UTF-8 directory name")?;
    let frontmatter =
        frontmatter(source).context("no YAML frontmatter delimited by --- was found")?;
    let fields = top_level_fields(frontmatter);

    let description = fields
        .get("description")
        .map(|value| value.trim().to_owned())
        .unwrap_or_default();
    if description.is_empty() {
        anyhow::bail!("description is required and must not be empty");
    }

    let name = match fields.get("name").map(|name| name.trim()) {
        Some(name) if !name.is_empty() => {
            if name != directory {
                eprintln!(
                    "warning: skill {} declares name {name:?} but its directory is {directory:?}",
                    path.display()
                );
            }
            if name.chars().count() > 64 {
                eprintln!("warning: skill name {name:?} exceeds 64 characters");
            }
            name.to_owned()
        }
        _ => directory.to_owned(),
    };

    Ok(SkillMetadata {
        name,
        description,
        path: path.to_path_buf(),
    })
}

fn frontmatter(source: &str) -> Option<&str> {
    let rest = source
        .strip_prefix("---\n")
        .or_else(|| source.strip_prefix("---\r\n"))?;
    let mut offset = 0;
    for line in rest.lines() {
        if line.trim_end() == "---" {
            return Some(&rest[..offset]);
        }
        offset += line.len() + 1;
    }
    None
}

fn top_level_fields(frontmatter: &str) -> BTreeMap<String, String> {
    let mut fields = BTreeMap::new();
    let mut lines = frontmatter.lines().peekable();
    while let Some(line) = lines.next() {
        // An indented line continues the value of an enclosing key; it is never
        // a key of its own.
        if line.starts_with(' ') || line.starts_with('\t') || line.trim().is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_owned();
        let value = value.trim();
        let value = if matches!(value, "|" | "|-" | "|+" | ">" | ">-" | ">+") {
            let folded = value.starts_with('>');
            let mut parts = Vec::new();
            while let Some(next) = lines.peek() {
                let indented = next.starts_with(' ') || next.starts_with('\t');
                if !indented && !next.trim().is_empty() {
                    break;
                }
                parts.push(lines.next().unwrap_or_default().trim().to_owned());
            }
            while parts.last().is_some_and(|part| part.is_empty()) {
                parts.pop();
            }
            if folded {
                parts.join(" ")
            } else {
                parts.join("\n")
            }
        } else {
            unquote(value)
        };
        fields.insert(key, value);
    }
    fields
}

fn unquote(value: &str) -> String {
    for quote in ['"', '\''] {
        if value.len() >= 2 && value.starts_with(quote) && value.ends_with(quote) {
            return value[1..value.len() - 1].to_owned();
        }
    }
    value.to_owned()
}
