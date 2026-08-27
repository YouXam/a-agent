use std::io::IsTerminal;

/// Whether stdin holds input for this turn.
///
/// `a` consumes stdin so `cargo test 2>&1 | a "fix this"` works. Consuming it
/// whenever it merely is not a terminal is wrong: a supervisor or sandbox hands
/// its child a socket that may never carry anything, and reading it blocks
/// forever.
///
/// A pipe or a redirected file is read to end of file, however long the producer
/// takes, because writing the pipe is how the user asked for that. Waiting is
/// what every filter does; giving up early would silently drop the input of a
/// producer that is simply slow to start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdinSource {
    Terminal,
    /// Not a pipe or file: an inherited channel that is not this turn's input.
    Foreign,
    Stream,
}

pub fn stdin_source() -> StdinSource {
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        return StdinSource::Terminal;
    }
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        classify_fd(stdin.as_raw_fd())
    }
    #[cfg(not(unix))]
    StdinSource::Stream
}

/// A shell pipeline hands over a pipe, and a redirection hands over a file.
/// Anything else, a socket in particular, came from a supervisor or sandbox that
/// is still using it for something else.
#[cfg(unix)]
pub fn classify_fd(fd: std::os::fd::RawFd) -> StdinSource {
    let mut stat = unsafe { std::mem::zeroed::<libc::stat>() };
    if unsafe { libc::fstat(fd, &mut stat) } != 0 {
        return StdinSource::Foreign;
    }
    let kind = stat.st_mode & libc::S_IFMT;
    if kind == libc::S_IFIFO || kind == libc::S_IFREG {
        StdinSource::Stream
    } else {
        StdinSource::Foreign
    }
}

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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;

    #[test]
    fn only_pipes_and_files_count_as_this_turn_input() {
        let file = tempfile::NamedTempFile::new().unwrap();
        assert_eq!(classify_fd(file.as_file().as_raw_fd()), StdinSource::Stream);

        let mut fds = [0_i32; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        assert_eq!(classify_fd(fds[0]), StdinSource::Stream);

        // A socket is how a supervisor or sandbox hands over stdio; draining it
        // would block on a writer that is not talking to us.
        let mut pair = [0_i32; 2];
        assert_eq!(
            unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, pair.as_mut_ptr()) },
            0
        );
        assert_eq!(classify_fd(pair[0]), StdinSource::Foreign);

        assert_eq!(classify_fd(-1), StdinSource::Foreign);
        for fd in fds.into_iter().chain(pair) {
            unsafe { libc::close(fd) };
        }
    }
}
