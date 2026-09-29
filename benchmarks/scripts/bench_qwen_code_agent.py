#!/usr/bin/env python3
"""Harbor adapter for L^(0), L^(1), and L^(2) with qwen-code 0.20.0.
Records each request through the logging proxy and retries infrastructure failures."""
import sys
import asyncio
import gzip
import json
import os
import shlex
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, override

from pydantic import Field

from harbor.agents.installed.base import ApiConnectionClosedError
from harbor.agents.installed.qwen_code import QwenCode, QwenCodeOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

HERE = Path(__file__).resolve().parent
QWEN_MOUNT = "/opt/qwen-code"
AGENT_LOGS = "/logs/agent"  # harbor mounts <trial>/agent here
INFRA_MARKERS = ("Connection error", "ECONNREFUSED", "Request timeout", "fetch failed", "ECONNRESET", "socket hang up",
                 "No stream activity", "terminated", "Service Unavailable", "502", "503")
PROXY_PORT_RANGE = range(9100, 9900)


class BenchInfraApiError(ApiConnectionClosedError):
    """The model endpoint failed under the agent (runtime subtype api_error_infra): an infrastructure error, re-run, never a final outcome."""


class BenchQwenCodeOptions(QwenCodeOptions):
    model_dir: str = Field(description="served model id = the checkpoint directory")
    upstream_ports: str = Field(description="comma-separated vLLM replica ports on 127.0.0.1")
    seed_map: str = Field(description="json file {task_name: seed}")
    exclude_agent_tool: bool = Field(default=False, description="true -> settings tools.exclude=[agent] (the ablation)")
    condition: str = Field(default="", description="free-text label recorded in run_status.json")
    proxy_host: str = Field(default="172.17.0.1", description="host address the containers reach (docker0 = host-gateway)")
    proxy_port_base: int = Field(default=9100, description="first proxy port; each harbor job (process) must get its own 200-port block")
    proxy_python: str = Field(default=sys.executable)


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y")


class BenchQwenCode(QwenCode):
    options_model = BenchQwenCodeOptions
    _lock = threading.Lock()
    _active: dict[int, int] = {}      # upstream port -> trials currently using it (this process)
    _used_ports: set[int] = set()     # proxy ports reserved (this process)

    @staticmethod
    @override
    def name() -> str:
        return "bench-qwen-code"

    @property
    def opts(self) -> BenchQwenCodeOptions:
        return self.options  # type: ignore[return-value]

    @override
    def get_version_command(self) -> str | None:
        return "qwen --version"

    # ---- install: a 2-line wrapper to the read-only bundle; nothing downloaded ----
    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                f"test -x {QWEN_MOUNT}/bin/qwen || {{ echo 'qwen-code bundle is not mounted at {QWEN_MOUNT}' >&2; exit 69; }}; "
                f"printf '#!/bin/sh\\nexec {QWEN_MOUNT}/bin/qwen \"$@\"\\n' > /usr/local/bin/qwen && chmod 755 /usr/local/bin/qwen && "
                "qwen --version"
            ),
        )

    # ---- helpers ----
    def _task_name(self) -> str:
        trial_dir = self.logs_dir.parent
        try:
            cfg = json.loads((trial_dir / "config.json").read_text())
            task = cfg.get("task") or {}
            p = task.get("path") or task.get("name")
            if p:
                return str(p).rstrip("/").split("/")[-1]
        except Exception:  # noqa: BLE001
            pass
        return trial_dir.name.split("__")[0]

    def _seed(self, task_name: str) -> int:
        seeds = json.loads(Path(self.opts.seed_map).expanduser().read_text())
        if task_name not in seeds:
            raise ValueError(f"no seed for task {task_name!r} in {self.opts.seed_map}")
        return int(seeds[task_name])

    def _pick_upstream(self) -> int:
        ports = [int(p) for p in self.opts.upstream_ports.split(",") if p.strip()]
        with self._lock:
            for p in ports:
                self._active.setdefault(p, 0)
            best = min(ports, key=lambda p: (self._active[p], ports.index(p)))
            self._active[best] += 1
            return best

    def _release_upstream(self, port: int) -> None:
        with self._lock:
            self._active[port] = max(0, self._active.get(port, 1) - 1)

    def _pick_port(self) -> int:
        # The bookkeeping below is per process; concurrent harbor jobs are separate processes, so they get disjoint blocks
        # (proxy_port_base) — otherwise two jobs can pick one port in the same instant and the loser's trial dies at start.
        base = int(self.opts.proxy_port_base)
        with self._lock:
            for port in range(base, base + 200):
                if port in self._used_ports:
                    continue
                s = socket.socket()
                try:
                    s.bind((self.opts.proxy_host, port))
                except OSError:
                    continue
                finally:
                    s.close()
                self._used_ports.add(port)
                return port
        raise RuntimeError("no free proxy port in 9100-9899")

    def _release_port(self, port: int) -> None:
        with self._lock:
            self._used_ports.discard(port)

    def _http_json(self, url: str, timeout: float = 3.0) -> dict | None:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception:  # noqa: BLE001
            return None

    async def _start_proxy(self, port: int, upstream: int, logs: Path) -> subprocess.Popen:
        proxy_log = open(logs / "proxy.log", "ab")
        proc = subprocess.Popen(
            [self.opts.proxy_python, str(HERE / "bench_api_proxy.py"), "--listen", str(port), "--host", self.opts.proxy_host,
             "--upstream", f"http://127.0.0.1:{upstream}", "--log", str(logs / "api_requests.jsonl")],
            stdout=proxy_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        url = f"http://{self.opts.proxy_host}:{port}/v1/models"
        for _ in range(75):
            data = await asyncio.to_thread(self._http_json, url)
            if data is not None:
                ids = [m.get("id") for m in data.get("data") or []]
                if self.opts.model_dir in ids:
                    return proc
                proc.terminate()
                # a foreign proxy on our port (port collision with another job) or a replica serving the wrong model: retryable
                raise BenchInfraApiError(f"proxy port {port} answers but not with {self.opts.model_dir}: {ids} (collision or wrong replica)")
            if proc.poll() is not None:
                break
            await asyncio.sleep(0.2)
        proc.terminate()
        raise BenchInfraApiError(f"logging proxy on {self.opts.proxy_host}:{port} -> 127.0.0.1:{upstream} did not come up (see {logs / 'proxy.log'})")

    def _stop_proxy(self, proc: subprocess.Popen | None) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            proc.kill()

    def _settings(self, seed: int, client_endpoint: str, exclude_agent: bool) -> dict:
        # byte-for-byte the exp6_run_episode.sh settings (plus tools.exclude for the ablation only)
        tools: dict[str, Any] = {"shell": {"defaultTimeoutMs": 600000}}
        if exclude_agent:
            tools["exclude"] = ["agent"]
        return {
            "$version": 4,
            "env": {"VLLM_API_KEY": "EMPTY"},
            "modelProviders": {"openai": [{
                "id": self.opts.model_dir, "envKey": "VLLM_API_KEY", "baseUrl": client_endpoint,
                "generationConfig": {
                    "timeout": 1800000, "maxRetries": 0, "contextWindowSize": 262144,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
                    "samplingParams": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0,
                                       "repetition_penalty": 1, "max_tokens": 32768, "seed": seed},
                },
            }]},
            "security": {"auth": {"selectedType": "openai"}},
            "model": {"name": self.opts.model_dir},
            "tools": tools,
            "telemetry": {"enabled": False},
        }

    async def _kill_agent(self, environment: BaseEnvironment) -> None:
        # after harbor's timeout the docker exec client is gone but the node process would keep running during the tests
        cmd = ('pat="cli-ent""ry.js"; for p in /proc/[0-9]*; do tr "\\0" " " < "$p/cmdline" 2>/dev/null | grep -q "$pat" '
               '&& kill -9 "${p#/proc/}" 2>/dev/null; done; true')
        try:
            await asyncio.wait_for(environment.exec(command=cmd, user="root"), timeout=30)
        except Exception:  # noqa: BLE001
            pass

    # ---- the episode ----
    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        logs = self.logs_dir
        logs.mkdir(parents=True, exist_ok=True)
        task_name = self._task_name()
        seed = self._seed(task_name)
        exclude_agent = _truthy(self.opts.exclude_agent_tool)
        upstream = self._pick_upstream()
        port = self._pick_port()
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        proxy: subprocess.Popen | None = None
        timed_out = False
        status: dict = {}
        try:
            proxy = await self._start_proxy(port, upstream, logs)
            client_endpoint = f"http://host.docker.internal:{port}/v1"
            qhome = logs / "qwen_home"
            qhome.mkdir(exist_ok=True)
            (qhome / "settings.json").write_text(json.dumps(self._settings(seed, client_endpoint, exclude_agent), indent=2) + "\n")
            (logs / "prompt.txt").write_text(instruction)
            for p in (logs, qhome, qhome / "settings.json", logs / "prompt.txt"):
                os.chmod(p, 0o777 if p.is_dir() else 0o666)
            (logs / "trial_meta.json").write_text(json.dumps({
                "task": task_name, "seed": seed, "model": self.opts.model_dir, "model_label": self.model_name,
                "upstream_port": upstream, "proxy_port": port, "client_endpoint": client_endpoint,
                "exclude_agent_tool": exclude_agent, "condition": self.opts.condition, "started_at": started_at}, indent=1))
            env = {
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
                "OPENMM_CPU_THREADS": "2", "LOKY_MAX_CPU_COUNT": "4", "QWEN_STREAM_IDLE_TIMEOUT_MS": "600000",
                "QWEN_HOME": f"{AGENT_LOGS}/qwen_home", "VLLM_API_KEY": "EMPTY", "QWEN_TELEMETRY_ENABLED": "false",
                "QWEN_CODE_SUPPRESS_YOLO_WARNING": "1", "NO_PROXY": "host.docker.internal,127.0.0.1,localhost",
            }
            cmd = (f"qwen --model {shlex.quote(self.opts.model_dir)} --approval-mode yolo --output-format stream-json "
                   f"< {AGENT_LOGS}/prompt.txt > {AGENT_LOGS}/qwen_stream.jsonl 2> {AGENT_LOGS}/qwen_stderr.log; "
                   f"echo $? > {AGENT_LOGS}/qwen_exit_code")
            try:
                await self.exec_as_agent(environment, command=cmd, env=env)
            except asyncio.CancelledError:
                timed_out = True
                await self._kill_agent(environment)
                raise
        finally:
            self._stop_proxy(proxy)
            self._release_upstream(upstream)
            self._release_port(port)
            try:
                status = await asyncio.to_thread(self._finish, logs, started_at, seed, upstream, port, task_name, timed_out, exclude_agent)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"run_status.json could not be written: {exc!r}")
        failure = status.get("failure") or {}
        if failure.get("subtype") == "api_error_infra":
            raise BenchInfraApiError(failure.get("message") or "api_error_infra")

    # ---- exp6 classification, proxy-log usage, run_status.json ----
    @staticmethod
    def _events(stream: Path) -> list[dict]:
        out = []
        if not stream.exists():
            return out
        with open(stream, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def _finish(self, logs: Path, started_at: str, seed: int, upstream: int, port: int, task_name: str,
                timed_out: bool, exclude_agent: bool) -> dict:
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        events = self._events(logs / "qwen_stream.jsonl")
        result = next((e for e in reversed(events) if e.get("type") == "result"), None)
        trace_has_success = bool(result and result.get("subtype") == "success" and result.get("is_error") is False)
        # exp6: subtype=success with a final assistant turn that is an API-error string = infrastructure failure
        last = next((e for e in reversed(events) if e.get("type") == "assistant"), None)
        txt = ""
        if last:
            c = (last.get("message") or {}).get("content")
            txt = c if isinstance(c, str) else " ".join(b.get("text", "") for b in (c or []) if isinstance(b, dict))
        txt = txt.strip()
        infra = txt.startswith("[API Error") and any(k in txt for k in INFRA_MARKERS)
        failure: dict | None = None
        if result is not None and result.get("is_error"):
            error = result.get("error") or {}
            calls = []
            for e in reversed(events):
                if e.get("type") != "assistant":
                    continue
                for c in reversed((e.get("message") or {}).get("content") or []):
                    if isinstance(c, dict) and c.get("type") == "tool_use":
                        calls.append({"tool": c.get("name"), "args": json.dumps(c.get("input") or {}, ensure_ascii=False)[:200]})
                if len(calls) >= 5:
                    break
            message = error.get("message") if isinstance(error, dict) else str(error)
            failure = {"subtype": result.get("subtype"), "message": (message or "")[:1500] or None,
                       "last_tool_calls_most_recent_first": calls[:5]}
        if infra and not timed_out:
            trace_has_success = False
            failure = {"subtype": "api_error_infra", "message": txt[:300], "last_tool_calls_most_recent_first": []}
        if timed_out:
            trace_has_success = False
            failure = failure or {"subtype": "agent_timeout", "message": "harbor agent timeout", "last_tool_calls_most_recent_first": []}
        # proxy log: count + token usage, then gzip (as exp6)
        api_log = logs / "api_requests.jsonl"
        n_req = 0
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "chat_requests": 0, "max_prompt_tokens": 0}
        api_log_final = ""
        if api_log.exists():
            with open(api_log, errors="replace") as f, gzip.open(str(api_log) + ".gz", "wt", compresslevel=6) as g:
                for line in f:
                    g.write(line)
                    n_req += 1
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("path", "").startswith("/v1/chat/completions"):
                        usage["chat_requests"] += 1
                        u = (r.get("response") or {}).get("usage") or {}
                        usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                        usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                        usage["max_prompt_tokens"] = max(usage["max_prompt_tokens"], int(u.get("prompt_tokens") or 0))
            api_log.unlink()
            api_log_final = str(api_log) + ".gz"
        exit_code = None
        try:
            exit_code = int((logs / "qwen_exit_code").read_text().strip())
        except Exception:  # noqa: BLE001
            pass
        self._usage = usage
        status = {
            "started_at": started_at, "finished_at": finished_at, "model": self.opts.model_dir, "model_label": self.model_name,
            "endpoint": f"http://127.0.0.1:{upstream}/v1", "seed": seed, "qwen_code_version": self._version or "0.20.0",
            "exit_code": exit_code, "trace_has_success_result": trace_has_success, "failure": failure,
            "execution_mode": "harbor_container_yolo",
            "operational_limits": {"http_response_timeout_ms": 1800000, "foreground_shell_timeout_ms": 600000,
                                   "context_window_tokens": 262144, "max_tokens_per_model_response": 32768},
            "bench": {"task": task_name, "condition": self.opts.condition, "exclude_agent_tool": exclude_agent,
                      "timed_out": timed_out, "client_endpoint": f"http://host.docker.internal:{port}/v1",
                      "api_log": api_log_final, "api_requests": n_req, "usage": usage,
                      "env_policy": "OMP=2,MKL=2,OPENBLAS=2,NUMEXPR=2,OPENMM=2,LOKY=4,STREAM_IDLE_MS=600000"},
        }
        (logs / "run_status.json").write_text(json.dumps(status, indent=1) + "\n")
        return status

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        u = getattr(self, "_usage", None)
        if not u:
            return
        context.n_input_tokens = u["prompt_tokens"]
        context.n_output_tokens = u["completion_tokens"]
        context.n_cache_tokens = 0
        context.cost_usd = None
