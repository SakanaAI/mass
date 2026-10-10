"""Execute the shell boundary and proxy against local fixtures, never paid APIs."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("jq"), "requires jq")
class EpisodeTests(unittest.TestCase):
    def run_episode(self, *, backend="openrouter", message="DONE", key="fixture-secret"):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        base = Path(folder.name)
        binaries = base / "bin"
        binaries.mkdir()
        for name, source in {
            "curl": 'import json,sys\nsys.stdin.read()\nprint(json.dumps({"data":[{"id":"openai/gpt-oss-120b"},{"id":"local-model"}]}))\n',
            "qwen": '''import json,os,sys
if "--version" in sys.argv:
    print("fixture")
else:
    print(json.dumps({"type":"assistant","message":{"content":os.environ["FIXTURE_MESSAGE"]}}))
    print(json.dumps({"type":"result","subtype":"success","is_error":False}))
''',
        }.items():
            path = binaries / name
            path.write_text(f"#!{sys.executable}\n" + source)
            path.chmod(0o755)
        prompt = base / "prompt.txt"
        prompt.write_text("fixture")
        env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                   PROXY_PY=sys.executable, PROXY_PORT="", MASS_BACKEND=backend,
                   MODEL_ID="openai/gpt-oss-120b:nitro" if backend == "openrouter" else "local-model",
                   MODEL_BASE_URL="https://openrouter.ai/api/v1" if backend == "openrouter" else "http://127.0.0.1:8001/v1",
                   MODEL_CONTEXT_WINDOW="131072", MODEL_MAX_OUTPUT_TOKENS="8192",
                   MODEL_REASONING_EFFORT="medium", FIXTURE_MESSAGE=message)
        env.pop("OPENROUTER_API_KEY", None)
        if key is not None:
            env["OPENROUTER_API_KEY"] = key
        result = subprocess.run(["bash", str(ROOT / "runtime/run_episode.sh"), str(prompt),
                                 str(base / "workspace"), str(base / "logs"), "8001", "42"],
                                env=env, text=True, capture_output=True, timeout=15)
        return base, result

    def test_remote_settings_alias_and_secret_stays_in_environment(self):
        base, result = self.run_episode()
        self.assertEqual(result.returncode, 0, result.stderr)
        settings = json.loads((base / "logs/qwen_home/settings.json").read_text())
        provider = settings["modelProviders"]["openai"][0]
        self.assertEqual(provider["baseUrl"], "https://openrouter.ai/api/v1")
        self.assertEqual(provider["envKey"], "OPENROUTER_API_KEY")
        self.assertEqual(provider["id"], "openai/gpt-oss-120b:nitro")
        generation = provider["generationConfig"]
        self.assertEqual(generation["contextWindowSize"], 131072)
        self.assertEqual(generation["samplingParams"]["max_tokens"], 8192)
        self.assertEqual(generation["extra_body"]["reasoning"], {"effort": "medium"})
        self.assertNotIn("top_k", generation["samplingParams"])
        self.assertNotIn("fixture-secret", result.stdout + result.stderr)
        for path in (base / "logs").rglob("*"):
            if path.is_file():
                self.assertNotIn("fixture-secret", path.read_text())
        status = json.loads((base / "logs/run_status.json").read_text())
        self.assertTrue(status["trace_has_success_result"])
        self.assertEqual(status["operational_limits"]["context_window_tokens"], 131072)

    def test_missing_key_fails_before_workspace_creation(self):
        base, result = self.run_episode(key=None)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OPENROUTER_API_KEY", result.stderr)
        self.assertFalse((base / "workspace").exists())

    def test_api_errors_cannot_look_like_success(self):
        for message in ("[API Error: 401 Unauthorized]", "[API Error: 402 Payment Required]",
                        "[API Error: 429 Rate limit]", "[API Error: 500 Server error]",
                        "[API Error: Connection error]"):
            with self.subTest(message=message):
                base, result = self.run_episode(message=message)
                self.assertEqual(result.returncode, 0, result.stderr)
                status = json.loads((base / "logs/run_status.json").read_text())
                self.assertFalse(status["trace_has_success_result"])

    def test_local_runner_remains_usable(self):
        base, result = self.run_episode(backend="local-vllm", key=None)
        self.assertEqual(result.returncode, 0, result.stderr)
        provider = json.loads((base / "logs/qwen_home/settings.json").read_text())["modelProviders"]["openai"][0]
        self.assertEqual(provider["envKey"], "VLLM_API_KEY")
        self.assertEqual(provider["generationConfig"]["extra_body"]["chat_template_kwargs"], {"enable_thinking": True})


@unittest.skipUnless(importlib.util.find_spec("aiohttp"), "requires requirements-core.txt")
class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_exception_does_not_expose_authorization(self):
        import io
        from aiohttp import ClientResponseError, RequestInfo, web
        from aiohttp.test_utils import make_mocked_request
        from multidict import CIMultiDict
        from yarl import URL
        from runtime.api_proxy import handle

        secret = "fixture-secret"
        headers = CIMultiDict(Authorization=f"Bearer {secret}")
        info = RequestInfo(URL("https://openrouter.ai/api/v1/chat/completions"), "POST", headers)
        session = Mock()
        session.request.side_effect = ClientResponseError(info, (), status=502, message="Malformed response")
        log = io.StringIO()
        app = web.Application()
        app.update(upstream="https://openrouter.ai/api", log_fh=log, seq=0, session=session)
        request = make_mocked_request("POST", "/v1/chat/completions", headers=headers, app=app)
        request.read = AsyncMock(return_value=b'{}')
        response = await handle(request)
        self.assertEqual(response.status, 502)
        self.assertNotIn(secret, log.getvalue())
        self.assertNotIn(secret, response.text)
        self.assertIn("ClientResponseError", json.loads(log.getvalue())["proxy_error"])

    def test_stream_retains_tool_fragments_reasoning_usage_and_errors(self):
        from runtime.api_proxy import assemble_sse
        events = [
            {"choices": [{"index": 0, "delta": {"reasoning": "brief", "reasoning_details": [{"type": "reasoning.text", "text": "brief", "index": 0}], "tool_calls": [{"index": 0, "id": "call1", "function": {"name": "write", "arguments": '{"x":'}}]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": 0.01}},
            {"error": {"code": 429, "message": "Rate limited"}},
        ]
        raw = ("\n".join("data: " + json.dumps(e) for e in events) + "\ndata: [DONE]\n").encode()
        result = assemble_sse(raw)
        message = result["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], '{"x":1}')
        self.assertEqual(message["reasoning_details"][0]["text"], "brief")
        self.assertEqual(result["usage"]["cost"], 0.01)
        self.assertEqual(result["error"]["code"], 429)
        self.assertIsNone(assemble_sse(b"data: [DONE]\n")["usage"])

    async def test_proxy_prefix_auth_raw_stream_and_no_redirect(self):
        import io
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        from runtime.api_proxy import handle, on_startup
        requests = []

        async def upstream(request):
            requests.append((request.path, request.headers.get("Authorization")))
            if request.path.endswith("redirect"):
                return web.Response(status=307, headers={"Location": "/leak"})
            return web.Response(body=b'data: {"choices": [], "usage": {"cost": 0.1}}\n\ndata: [DONE]\n\n',
                                content_type="text/event-stream")

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", upstream)
        async with TestServer(app) as server:
            log = io.StringIO()
            proxy = web.Application()
            proxy.update(upstream=str(server.make_url("/api")).rstrip("/"), log_fh=log,
                         seq=0, keep_raw_sse=True)
            proxy.on_startup.append(on_startup)

            async def close(app):
                await app["session"].close()

            proxy.on_cleanup.append(close)
            proxy.router.add_route("*", "/{tail:.*}", handle)
            async with TestClient(TestServer(proxy)) as client:
                response = await client.post("/v1/chat/completions", json={"model": "fixture"},
                                             headers={"Authorization": "Bearer fixture-secret"})
                self.assertIn("[DONE]", await response.text())
                response = await client.get("/v1/redirect", allow_redirects=False,
                                            headers={"Authorization": "Bearer fixture-secret"})
                self.assertEqual(response.status, 307)
                await response.read()
            self.assertEqual(requests, [("/api/v1/chat/completions", "Bearer fixture-secret"),
                                        ("/api/v1/redirect", "Bearer fixture-secret")])
            self.assertNotIn("fixture-secret", log.getvalue())
            self.assertIn("sse_raw", json.loads(log.getvalue().splitlines()[0]))
