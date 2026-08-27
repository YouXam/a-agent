use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::time::{Instant, sleep_until};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashArgs {
    pub command: String,
    /// What the model asked for. A build or a full test run needs minutes, while
    /// most commands need seconds, and only the caller knows which this is.
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct BashOptions {
    /// Used when the call does not ask for a timeout.
    pub timeout: Duration,
    /// Ceiling for what a call may ask for, so a hung command cannot run
    /// unbounded.
    pub max_timeout: Duration,
    pub max_output_bytes: usize,
}

impl BashOptions {
    /// The timeout this call runs under, and the requested value if it had to be
    /// capped. Callers report the cap so the model can see why its request did
    /// not take effect.
    fn resolve_timeout(&self, args: &BashArgs) -> (Duration, Option<u64>) {
        let Some(requested) = args.timeout_seconds.filter(|seconds| *seconds > 0) else {
            return (self.timeout.min(self.max_timeout), None);
        };
        let requested_duration = Duration::from_secs(requested);
        if requested_duration > self.max_timeout {
            (self.max_timeout, Some(requested))
        } else {
            (requested_duration, None)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BashResult {
    pub output: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub cancelled: bool,
}

pub type OutputSink = Arc<dyn Fn(String) + Send + Sync>;

pub async fn execute_bash(
    root: &Path,
    args: &BashArgs,
    options: &BashOptions,
    output_sink: Option<OutputSink>,
) -> Result<BashResult> {
    execute_bash_cancellable(root, args, options, output_sink, CancellationToken::new()).await
}

pub async fn execute_bash_cancellable(
    root: &Path,
    args: &BashArgs,
    options: &BashOptions,
    output_sink: Option<OutputSink>,
    cancel: CancellationToken,
) -> Result<BashResult> {
    let mut command = Command::new("bash");
    command
        .arg("-o")
        .arg("pipefail")
        .arg("-lc")
        .arg(&args.command)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .spawn()
        .with_context(|| format!("start bash command: {}", args.command))?;

    let stdout = child.stdout.take().context("capture bash stdout")?;
    let stderr = child.stderr.take().context("capture bash stderr")?;
    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(32);
    tokio::spawn(pump(stdout, tx.clone()));
    tokio::spawn(pump(stderr, tx));

    let (timeout, capped_from) = options.resolve_timeout(args);
    let deadline = Instant::now() + timeout;
    let mut bounded = BoundedOutput::new(options.max_output_bytes);
    let mut timed_out = false;
    let mut cancelled = false;
    let status = loop {
        tokio::select! {
            status = child.wait() => break status.context("wait for bash command")?,
            chunk = rx.recv() => {
                if let Some(chunk) = chunk {
                    if let Some(sink) = &output_sink {
                        sink(String::from_utf8_lossy(&chunk).into_owned());
                    }
                    bounded.push(&chunk);
                }
            }
            _ = sleep_until(deadline) => {
                timed_out = true;
                break terminate(&mut child).await.context("terminate timed out bash command")?;
            }
            _ = cancel.cancelled() => {
                cancelled = true;
                break terminate(&mut child).await.context("terminate cancelled bash command")?;
            }
        }
    };
    while let Some(chunk) = rx.recv().await {
        if let Some(sink) = &output_sink {
            sink(String::from_utf8_lossy(&chunk).into_owned());
        }
        bounded.push(&chunk);
    }

    let mut output = bounded.finish();
    if timed_out {
        // Say what the limit was and how to raise it: a bare "timed out" leaves
        // the model guessing whether the command or the limit was wrong.
        output.push_str(&format!(
            "\n[bash timed out after {:.1}s",
            timeout.as_secs_f64()
        ));
        match capped_from {
            Some(requested) => output.push_str(&format!(
                "; timeout_seconds {requested} was capped at {}]",
                options.max_timeout.as_secs()
            )),
            None => output.push_str(&format!(
                "; pass timeout_seconds up to {} for slow commands]",
                options.max_timeout.as_secs()
            )),
        }
    } else if cancelled {
        output.push_str("\n[bash cancelled]");
    }
    Ok(BashResult {
        output,
        exit_code: status.code(),
        timed_out,
        cancelled,
    })
}

async fn terminate(child: &mut tokio::process::Child) -> Result<std::process::ExitStatus> {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // SAFETY: this negative PID addresses the dedicated process group created above.
        let result = unsafe { libc::kill(-(pid as i32), libc::SIGTERM) };
        if result != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error.into());
            }
        }
        match tokio::time::timeout(Duration::from_secs(1), child.wait()).await {
            Ok(status) => return Ok(status?),
            Err(_) => {
                // SAFETY: the same process group is still owned by this child.
                unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
                return Ok(child.wait().await?);
            }
        }
    }
    child.kill().await?;
    Ok(child.wait().await?)
}

async fn pump(mut stream: impl tokio::io::AsyncRead + Unpin, tx: mpsc::Sender<Vec<u8>>) {
    let mut buffer = vec![0_u8; 8192];
    loop {
        match stream.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(count) if tx.send(buffer[..count].to_vec()).await.is_err() => break,
            Ok(_) => {}
        }
    }
}

struct BoundedOutput {
    bytes: Vec<u8>,
    max: usize,
    dropped: usize,
}

impl BoundedOutput {
    fn new(max: usize) -> Self {
        Self {
            bytes: Vec::new(),
            max,
            dropped: 0,
        }
    }

    fn push(&mut self, chunk: &[u8]) {
        self.bytes.extend_from_slice(chunk);
        if self.bytes.len() > self.max {
            let remove = self.bytes.len() - self.max;
            self.bytes.drain(..remove);
            self.dropped += remove;
        }
    }

    fn finish(self) -> String {
        let tail = String::from_utf8_lossy(&self.bytes);
        if self.dropped == 0 {
            tail.into_owned()
        } else {
            format!("[output truncated; dropped {} bytes]\n{tail}", self.dropped)
        }
    }
}
