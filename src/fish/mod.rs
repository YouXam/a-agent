use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub fn fish_script() -> &'static str {
    include_str!("../../assets/fish/a.fish")
}

pub fn install() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME is not set")?;
    install_to(&home)
}

pub fn install_to(home: &Path) -> Result<PathBuf> {
    let directory = home.join(".config/fish/conf.d");
    fs::create_dir_all(&directory)?;
    let destination = directory.join("a.fish");
    let mut temporary = tempfile::NamedTempFile::new_in(&directory)?;
    temporary.write_all(fish_script().as_bytes())?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(&destination)
        .map_err(|error| error.error)?;
    Ok(destination)
}
