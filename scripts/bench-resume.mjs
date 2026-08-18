import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const binary = path.resolve(process.argv[2] ?? path.join(root, "target/release/a"));
const iterations = Number.parseInt(process.argv[3] ?? "100", 10);
const historyTurns = Number.parseInt(process.argv[4] ?? "1", 10);

if (!Number.isInteger(iterations) || iterations < 1) {
  throw new Error("iterations must be a positive integer");
}
if (!Number.isInteger(historyTurns) || historyTurns < 1) {
  throw new Error("history turns must be a positive integer");
}
if (!fs.existsSync(binary)) {
  throw new Error(`binary not found: ${binary}`);
}

const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "a-resume-bench-"));
const home = path.join(temporary, "home");
const repo = path.join(temporary, "repo");
const baselineState = path.join(temporary, "baseline-state");
fs.mkdirSync(path.join(home, ".config/a"), { recursive: true });
fs.mkdirSync(repo, { recursive: true });

let requestObserver;
const server = http.createServer((request, response) => {
  const arrivedAt = performance.now();
  request.resume();
  request.on("end", () => {
    requestObserver?.(arrivedAt);
    requestObserver = undefined;
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.end(
      'data: {"type":"response.output_text.delta","delta":"ok"}\n\n' +
        'data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1}}}\n\n' +
        "data: [DONE]\n\n",
    );
  });
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const address = server.address();
if (!address || typeof address === "string") {
  throw new Error("benchmark server did not expose a TCP address");
}

fs.writeFileSync(
  path.join(home, ".config/a/config.toml"),
  `[provider]\n` +
    `type = "responses"\n` +
    `base_url = "http://127.0.0.1:${address.port}/v1"\n` +
    `model = "benchmark-model"\n` +
    `api_key = "benchmark-key"\n`,
);

function runAgent(args, stateHome, debugTiming = false) {
  return new Promise((resolve, reject) => {
    let stderr = "";
    let requestTimeout;
    const startedAt = performance.now();
    const requestArrival = new Promise((resolveRequest, rejectRequest) => {
      requestObserver = (arrivedAt) => {
        clearTimeout(requestTimeout);
        resolveRequest(arrivedAt - startedAt);
      };
      requestTimeout = setTimeout(() => {
        requestObserver = undefined;
        rejectRequest(new Error("agent did not send a model request within 5 seconds"));
      }, 5_000);
    });
    const child = spawn(binary, args, {
      cwd: repo,
      env: {
        ...process.env,
        HOME: home,
        XDG_STATE_HOME: stateHome,
        ...(debugTiming ? { A_DEBUG_TIMING: "1" } : {}),
      },
      stdio: ["ignore", "ignore", "pipe"],
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.once("error", reject);
    child.once("close", async (code) => {
      try {
        const requestMs = await requestArrival;
        if (code !== 0) {
          throw new Error(`agent exited ${code}: ${stderr}`);
        }
        const internal = stderr.match(/pre-network\s+([\d.]+) ms/);
        resolve({
          requestMs,
          totalMs: performance.now() - startedAt,
          internalMs: internal ? Number.parseFloat(internal[1]) : undefined,
        });
      } catch (error) {
        reject(error);
      }
    });
  });
}

function percentile(values, percentileValue) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor((sorted.length - 1) * percentileValue)];
}

function summarize(values) {
  return {
    min: Math.min(...values),
    p50: percentile(values, 0.5),
    p95: percentile(values, 0.95),
    p99: percentile(values, 0.99),
    max: Math.max(...values),
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
  };
}

try {
  await runAgent(["-1", "seed"], baselineState);
  for (let index = 1; index < historyTurns; index += 1) {
    await runAgent(["-r", "-1", `seed-${index}`], baselineState);
  }
  const requestTimes = [];
  const totalTimes = [];
  const internalTimes = [];

  for (let index = 0; index < iterations; index += 1) {
    const stateHome = path.join(temporary, `run-${index}`);
    fs.cpSync(baselineState, stateHome, { recursive: true });
    const result = await runAgent(["-r", "-1", "continue"], stateHome, true);
    requestTimes.push(result.requestMs);
    totalTimes.push(result.totalMs);
    if (result.internalMs !== undefined) internalTimes.push(result.internalMs);
  }

  const result = {
    iterations,
    historyTurns,
    baselineDatabaseBytes: fs.statSync(path.join(baselineState, "a/sessions.db")).size,
    processToRequestMs: summarize(requestTimes),
    completeTurnMs: summarize(totalTimes),
    reportedPreNetworkMs: summarize(internalTimes),
  };
  console.log(JSON.stringify(result, null, 2));
} finally {
  await new Promise((resolve) => server.close(resolve));
  fs.rmSync(temporary, { recursive: true, force: true });
}
