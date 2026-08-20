use std::fs;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use a_agent::model::{ToolCall, ToolResult};
use a_agent::provider::EventSink;
use a_agent::tools::bash::{BashArgs, BashOptions, execute_bash, execute_bash_cancellable};
use a_agent::tools::patch::{affected_paths, apply_patch};
use a_agent::tools::read::{ReadArgs, read_text_file};
use a_agent::tools::runner::{CoreToolExecutor, ToolExecutor, ToolRunner};
use async_trait::async_trait;
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn read_returns_numbered_bounded_lines() {
    let temp = tempdir().unwrap();
    fs::write(temp.path().join("a.txt"), "one\ntwo\nthree\nfour\n").unwrap();
    let result = read_text_file(
        temp.path(),
        &ReadArgs {
            path: "a.txt".into(),
            offset: 1,
            limit: Some(2),
        },
        10,
    )
    .await
    .unwrap();
    assert_eq!(result, "2: two\n3: three\n[truncated; 1 more line]");
}

#[tokio::test]
async fn read_rejects_binary_and_allows_paths_outside_the_workspace() {
    let temp = tempdir().unwrap();
    let workspace = temp.path().join("workspace");
    fs::create_dir(&workspace).unwrap();
    fs::write(workspace.join("binary"), [0, 1, 2]).unwrap();
    assert!(
        read_text_file(
            &workspace,
            &ReadArgs {
                path: "binary".into(),
                offset: 0,
                limit: None
            },
            10
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("binary")
    );
    fs::write(temp.path().join("outside.txt"), "outside\n").unwrap();
    let relative = read_text_file(
        &workspace,
        &ReadArgs {
            path: "../outside.txt".into(),
            offset: 0,
            limit: None,
        },
        10,
    )
    .await
    .unwrap();
    assert_eq!(relative, "1: outside");

    let absolute = read_text_file(
        &workspace,
        &ReadArgs {
            path: temp
                .path()
                .join("outside.txt")
                .to_string_lossy()
                .into_owned(),
            offset: 0,
            limit: None,
        },
        10,
    )
    .await
    .unwrap();
    assert_eq!(absolute, "1: outside");
}

#[tokio::test]
async fn patch_adds_updates_and_deletes() {
    let temp = tempdir().unwrap();
    fs::create_dir(temp.path().join("src")).unwrap();
    fs::write(temp.path().join("src/old.txt"), "old\nline\n").unwrap();
    fs::write(temp.path().join("remove.txt"), "bye\n").unwrap();
    let patch = "*** Begin Patch\n*** Update File: src/old.txt\n@@\n-old\n+new\n line\n*** Add File: src/new.txt\n+created\n*** Delete File: remove.txt\n*** End Patch";
    let summary = apply_patch(temp.path(), patch).await.unwrap();
    assert_eq!(
        fs::read_to_string(temp.path().join("src/old.txt")).unwrap(),
        "new\nline\n"
    );
    assert_eq!(
        fs::read_to_string(temp.path().join("src/new.txt")).unwrap(),
        "created\n"
    );
    assert!(!temp.path().join("remove.txt").exists());
    assert_eq!(summary.files.len(), 3);
}

#[cfg(unix)]
#[tokio::test]
async fn patch_updates_preserve_inode_permissions_owner_and_hard_links() {
    let temp = tempdir().unwrap();
    let script = temp.path().join("script.sh");
    let hard_link = temp.path().join("script-link.sh");
    fs::write(&script, "#!/bin/sh\necho old\n").unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o751)).unwrap();
    fs::hard_link(&script, &hard_link).unwrap();
    let before = fs::metadata(&script).unwrap();

    let patch =
        "*** Begin Patch\n*** Update File: script.sh\n@@\n-echo old\n+echo new\n*** End Patch";
    apply_patch(temp.path(), patch).await.unwrap();

    let after = fs::metadata(&script).unwrap();
    assert_eq!(after.ino(), before.ino());
    assert_eq!(after.mode() & 0o7777, 0o751);
    assert_eq!(after.uid(), before.uid());
    assert_eq!(after.gid(), before.gid());
    assert_eq!(
        fs::read_to_string(&hard_link).unwrap(),
        "#!/bin/sh\necho new\n"
    );
}

#[tokio::test]
async fn patch_rejects_stale_context_and_allows_paths_outside_the_workspace() {
    let temp = tempdir().unwrap();
    let workspace = temp.path().join("workspace");
    fs::create_dir(&workspace).unwrap();
    fs::write(workspace.join("a.txt"), "current\n").unwrap();
    let stale = "*** Begin Patch\n*** Update File: a.txt\n@@\n-old\n+new\n*** End Patch";
    assert!(
        apply_patch(&workspace, stale)
            .await
            .unwrap_err()
            .to_string()
            .contains("context not found")
    );
    assert_eq!(
        fs::read_to_string(workspace.join("a.txt")).unwrap(),
        "current\n"
    );
    let escape = "*** Begin Patch\n*** Add File: ../escape.txt\n+no\n*** End Patch";
    apply_patch(&workspace, escape).await.unwrap();
    assert_eq!(
        fs::read_to_string(temp.path().join("escape.txt")).unwrap(),
        "no\n"
    );

    let absolute_path = temp.path().join("absolute.txt");
    let absolute = format!(
        "*** Begin Patch\n*** Add File: {}\n+absolute\n*** End Patch",
        absolute_path.display()
    );
    apply_patch(&workspace, &absolute).await.unwrap();
    assert_eq!(fs::read_to_string(absolute_path).unwrap(), "absolute\n");
}

#[test]
fn patch_paths_are_extracted_for_scheduling() {
    let patch = "*** Begin Patch\n*** Update File: b.rs\n*** Add File: a.rs\n+x\n*** End Patch";
    assert_eq!(affected_paths(patch).unwrap(), ["a.rs", "b.rs"]);
}

#[tokio::test]
async fn bash_captures_output_and_exit_status() {
    let temp = tempdir().unwrap();
    let result = execute_bash(
        temp.path(),
        &BashArgs {
            command: "printf out; printf err >&2; exit 7".into(),
        },
        &BashOptions {
            timeout: Duration::from_secs(2),
            max_output_bytes: 100,
        },
        None,
    )
    .await
    .unwrap();
    assert!(result.output.contains("out"));
    assert!(result.output.contains("err"));
    assert_eq!(result.exit_code, Some(7));
}

#[tokio::test]
async fn bash_uses_pipefail_for_pipeline_status() {
    let temp = tempdir().unwrap();
    let result = execute_bash(
        temp.path(),
        &BashArgs {
            command: "false | true".into(),
        },
        &BashOptions {
            timeout: Duration::from_secs(2),
            max_output_bytes: 100,
        },
        None,
    )
    .await
    .unwrap();
    assert_ne!(result.exit_code, Some(0));
}

#[tokio::test]
async fn bash_honors_cancellation() {
    let temp = tempdir().unwrap();
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        trigger.cancel();
    });
    let result = execute_bash_cancellable(
        temp.path(),
        &BashArgs {
            command: "sleep 10".into(),
        },
        &BashOptions {
            timeout: Duration::from_secs(30),
            max_output_bytes: 100,
        },
        None,
        cancel,
    )
    .await
    .unwrap();
    assert!(result.cancelled);
}

#[cfg(unix)]
#[tokio::test]
async fn bash_cancellation_terminates_the_process_group() {
    let temp = tempdir().unwrap();
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    let pid_path = temp.path().join("child.pid");
    let watched_pid_path = pid_path.clone();
    tokio::spawn(async move {
        for _ in 0..200 {
            if tokio::fs::metadata(&watched_pid_path).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        trigger.cancel();
    });
    execute_bash_cancellable(
        temp.path(),
        &BashArgs {
            command: "sleep 10 & echo $! > child.pid; wait".into(),
        },
        &BashOptions {
            timeout: Duration::from_secs(30),
            max_output_bytes: 100,
        },
        None,
        cancel,
    )
    .await
    .unwrap();
    let pid = fs::read_to_string(pid_path).unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    let pid = pid.trim().parse::<i32>().unwrap();
    // SAFETY: signal 0 only checks whether the recorded child PID still exists.
    let alive = unsafe { libc::kill(pid, 0) } == 0;
    assert!(!alive, "background child {pid} survived cancellation");
}

#[tokio::test]
async fn core_bash_marks_user_cancellation_as_a_tool_error() {
    let temp = tempdir().unwrap();
    let executor = CoreToolExecutor::new(
        temp.path().to_path_buf(),
        1000,
        Duration::from_secs(30),
        1024,
    );
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        trigger.cancel();
    });
    let result = executor
        .execute_with(
            ToolCall::new("cancelled", "bash", r#"{"command":"sleep 10"}"#),
            EventSink::default(),
            cancel,
        )
        .await;
    assert!(result.is_error);
    assert!(result.output.contains("[bash cancelled]"));
}

#[tokio::test]
async fn core_read_output_obeys_the_global_byte_limit() {
    let temp = tempdir().unwrap();
    fs::write(temp.path().join("large.txt"), "x".repeat(10_000)).unwrap();
    let executor =
        CoreToolExecutor::new(temp.path().to_path_buf(), 1000, Duration::from_secs(1), 128);
    let result = executor
        .execute(ToolCall::new("read-1", "read", r#"{"path":"large.txt"}"#))
        .await;
    assert!(!result.is_error);
    assert!(result.output.len() < 256);
    assert!(result.output.contains("truncated"));
}

struct CountingExecutor {
    active: AtomicUsize,
    max_active: AtomicUsize,
}

#[async_trait]
impl ToolExecutor for CountingExecutor {
    async fn execute(&self, call: ToolCall) -> ToolResult {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(15)).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        ToolResult::success(call.id, "ok")
    }
}

#[tokio::test]
async fn runner_parallelizes_independent_calls_and_preserves_order() {
    let executor = Arc::new(CountingExecutor {
        active: AtomicUsize::new(0),
        max_active: AtomicUsize::new(0),
    });
    let runner = ToolRunner::new(executor.clone(), 2);
    let results = runner
        .execute(vec![
            ToolCall::new("one", "read", "{}"),
            ToolCall::new("two", "read", "{}"),
        ])
        .await;
    assert_eq!(executor.max_active.load(Ordering::SeqCst), 2);
    assert_eq!(
        results
            .iter()
            .map(|item| item.call_id.as_str())
            .collect::<Vec<_>>(),
        ["one", "two"]
    );
}

#[tokio::test]
async fn runner_serializes_patches_that_touch_the_same_path() {
    let executor = Arc::new(CountingExecutor {
        active: AtomicUsize::new(0),
        max_active: AtomicUsize::new(0),
    });
    let runner = ToolRunner::new(executor.clone(), 8);
    let patch = "*** Begin Patch\n*** Update File: same.rs\n@@\n-a\n+b\n*** End Patch";
    let arguments = serde_json::json!({"patch": patch}).to_string();
    runner
        .execute(vec![
            ToolCall::new("one", "apply_patch", &arguments),
            ToolCall::new("two", "apply_patch", &arguments),
        ])
        .await;
    assert_eq!(executor.max_active.load(Ordering::SeqCst), 1);
}
