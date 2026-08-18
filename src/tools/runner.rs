use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::future::join_all;
use serde::Deserialize;
use tokio::sync::{Mutex, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::model::{StreamEvent, ToolCall, ToolResult};
use crate::provider::EventSink;

use super::bash::{BashArgs, BashOptions, OutputSink, execute_bash_cancellable};
use super::patch::{affected_paths, apply_patch};
use super::read::{ReadArgs, read_text_file_bounded};

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, call: ToolCall) -> ToolResult;

    async fn execute_with(
        &self,
        call: ToolCall,
        _events: EventSink,
        _cancel: CancellationToken,
    ) -> ToolResult {
        self.execute(call).await
    }
}

pub struct CoreToolExecutor {
    cwd: PathBuf,
    read_max_lines: usize,
    max_output_bytes: usize,
    bash_options: BashOptions,
}

impl CoreToolExecutor {
    pub fn new(
        cwd: PathBuf,
        read_max_lines: usize,
        bash_timeout: Duration,
        max_output_bytes: usize,
    ) -> Self {
        Self {
            cwd,
            read_max_lines,
            max_output_bytes,
            bash_options: BashOptions {
                timeout: bash_timeout,
                max_output_bytes,
            },
        }
    }
}

#[derive(Deserialize)]
struct PatchArgs {
    patch: String,
}

#[async_trait]
impl ToolExecutor for CoreToolExecutor {
    async fn execute(&self, call: ToolCall) -> ToolResult {
        self.execute_with(call, EventSink::default(), CancellationToken::new())
            .await
    }

    async fn execute_with(
        &self,
        call: ToolCall,
        events: EventSink,
        cancel: CancellationToken,
    ) -> ToolResult {
        if cancel.is_cancelled() {
            return ToolResult::error(call.id, "tool execution cancelled");
        }
        let call_id = call.id.clone();
        let result = match call.name.as_str() {
            "read" => match serde_json::from_str::<ReadArgs>(&call.arguments) {
                Ok(args) => {
                    read_text_file_bounded(
                        &self.cwd,
                        &args,
                        self.read_max_lines,
                        self.max_output_bytes,
                    )
                    .await
                }
                Err(error) => Err(error.into()),
            },
            "apply_patch" => match patch_text(&call.arguments) {
                Ok(patch) => apply_patch(&self.cwd, &patch).await.map(|summary| {
                    summary
                        .files
                        .iter()
                        .map(|file| format!("{} (+{} -{})", file.path, file.added, file.removed))
                        .collect::<Vec<_>>()
                        .join("\n")
                }),
                Err(error) => Err(error),
            },
            "bash" => match serde_json::from_str::<BashArgs>(&call.arguments) {
                Ok(args) => {
                    let output_events = events.clone();
                    let output_id = call_id.clone();
                    let output_sink: OutputSink = Arc::new(move |delta| {
                        output_events.emit(StreamEvent::ToolExecutionOutput {
                            id: output_id.clone(),
                            delta,
                        });
                    });
                    execute_bash_cancellable(
                        &self.cwd,
                        &args,
                        &self.bash_options,
                        Some(output_sink),
                        cancel,
                    )
                    .await
                    .map(|result| {
                        format!(
                            "{}\n[exit code: {}]",
                            result.output,
                            result
                                .exit_code
                                .map_or_else(|| "signal".into(), |code| code.to_string())
                        )
                    })
                }
                Err(error) => Err(error.into()),
            },
            name => Err(anyhow::anyhow!(
                "unknown tool '{name}'; available tools: read, apply_patch, bash"
            )),
        };
        match result {
            Ok(output)
                if call.name == "bash"
                    && (output.contains("[bash cancelled]")
                        || output.contains("[bash timed out after ")) =>
            {
                ToolResult::error(call.id, output)
            }
            Ok(output) => ToolResult::success(call.id, output),
            Err(error) => ToolResult::error(call.id, error.to_string()),
        }
    }
}

pub struct ToolRunner {
    executor: Arc<dyn ToolExecutor>,
    semaphore: Arc<Semaphore>,
    path_locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
}

impl ToolRunner {
    pub fn new(executor: Arc<dyn ToolExecutor>, max_parallel: usize) -> Self {
        Self {
            executor,
            semaphore: Arc::new(Semaphore::new(max_parallel.max(1))),
            path_locks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn execute(&self, calls: Vec<ToolCall>) -> Vec<ToolResult> {
        self.execute_with(calls, EventSink::default(), CancellationToken::new())
            .await
    }

    pub async fn execute_with(
        &self,
        calls: Vec<ToolCall>,
        events: EventSink,
        cancel: CancellationToken,
    ) -> Vec<ToolResult> {
        let futures = calls.into_iter().map(|call| {
            let executor = self.executor.clone();
            let semaphore = self.semaphore.clone();
            let path_locks = self.path_locks.clone();
            let events = events.clone();
            let cancel = cancel.clone();
            async move {
                let _permit = semaphore.acquire_owned().await.expect("semaphore closed");
                let paths = scheduling_paths(&call);
                let locks = {
                    let mut registry = path_locks.lock().await;
                    paths
                        .into_iter()
                        .map(|path| registry.entry(path).or_default().clone())
                        .collect::<Vec<_>>()
                };
                let mut guards = Vec::with_capacity(locks.len());
                for lock in locks {
                    guards.push(lock.lock_owned().await);
                }
                let result = executor.execute_with(call, events, cancel).await;
                drop(guards);
                result
            }
        });
        join_all(futures).await
    }
}

fn scheduling_paths(call: &ToolCall) -> Vec<String> {
    if call.name != "apply_patch" {
        return Vec::new();
    }
    patch_text(&call.arguments)
        .and_then(|patch| affected_paths(&patch))
        .unwrap_or_default()
}

fn patch_text(arguments: &str) -> anyhow::Result<String> {
    if arguments.trim_start().starts_with("*** Begin Patch") {
        return Ok(arguments.to_owned());
    }
    Ok(serde_json::from_str::<PatchArgs>(arguments)?.patch)
}
