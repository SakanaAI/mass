"""Local API compatibility checks with HTTP fixtures and the real OpenAI client."""
import importlib.util
import json
import unittest


@unittest.skipUnless(importlib.util.find_spec("openai"), "requires requirements-core.txt")
class LocalBackendTests(unittest.TestCase):
    def backend(self, handler, **options):
        import httpx
        from harness_improvement.llm_backend import JsonLLMBackend

        settings = dict(backend="local-vllm", model="fixture-model",
                        api_key="EMPTY", base_url="http://127.0.0.1:1234/v1")
        settings.update(options)
        backend = JsonLLMBackend(**settings)
        backend.client.close()
        from openai import OpenAI
        backend.client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"],
                                max_retries=0,
                                http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(backend.client.close)
        return backend

    def completion(self, content):
        import httpx
        return httpx.Response(200, json={
            "id": "fixture", "object": "chat.completion", "created": 0,
            "model": "fixture-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
        })

    def test_lmstudio_format_fallback_preserves_json_validation(self):
        import httpx
        for content in ('{"winner":"A"}', '[]'):
            with self.subTest(content=content):
                requests = []

                def respond(request):
                    body = json.loads(request.content)
                    requests.append(body)
                    if len(requests) == 1:
                        return httpx.Response(400, json={
                            "error": "'response_format.type' must be 'json_schema' or 'text'"})
                    return self.completion(content)

                backend = self.backend(respond)
                if content == '[]':
                    with self.assertRaisesRegex(RuntimeError, "expected an object"):
                        backend.call_json(system_prompt="Return JSON", user_prompt="Choose")
                else:
                    self.assertEqual(backend.call_json(system_prompt="Return JSON",
                                                      user_prompt="Choose").payload,
                                     {"winner": "A"})
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
                self.assertNotIn("response_format", requests[1])
                self.assertEqual(requests[0]["messages"], requests[1]["messages"])

    def test_other_http_errors_are_not_retried(self):
        import httpx
        from openai import APIStatusError
        for status, message in ((400, "Context length exceeded"), (401, "Unauthorized"),
                                (500, "Server failed"),
                                (400, "'response_format.type' must be 'text'")):
            with self.subTest(status=status, message=message):
                requests = []

                def respond(request):
                    requests.append(request)
                    return httpx.Response(status, json={"error": message})

                with self.assertRaises(APIStatusError):
                    self.backend(respond).call_json(system_prompt="JSON", user_prompt="Choose")
                self.assertEqual(len(requests), 1)

    def test_vllm_json_and_plain_text_need_one_request(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            return self.completion('{"ok":true}' if "response_format" in body else "Design")

        backend = self.backend(respond)
        self.assertEqual(backend.call_json(system_prompt="JSON", user_prompt="Go").payload,
                         {"ok": True})
        self.assertEqual(backend.call_text(system_prompt="Text", user_prompt="Go").raw_text,
                         "Design")
        self.assertEqual(len(requests), 2)
        self.assertNotIn("response_format", requests[1])

    def test_format_fallback_is_bounded_and_only_for_json(self):
        import httpx
        from openai import BadRequestError
        for method, expected_requests in (("call_json", 2), ("call_text", 1)):
            with self.subTest(method=method):
                requests = []

                def respond(request):
                    requests.append(request)
                    return httpx.Response(400, json={
                        "error": "'response_format.type' must be 'json_schema' or 'text'"})

                with self.assertRaises(BadRequestError):
                    getattr(self.backend(respond), method)(system_prompt="JSON", user_prompt="Go")
                self.assertEqual(len(requests), expected_requests)


@unittest.skipUnless(importlib.util.find_spec("openai"), "requires requirements-core.txt")
class OpenRouterBackendTests(unittest.TestCase):
    def backend(self, handler):
        return LocalBackendTests.backend(
            self, handler, backend="openrouter", model="openai/gpt-oss-120b:nitro",
            api_key="fixture-secret", base_url="https://openrouter.ai/api/v1",
            reasoning_effort="medium", top_k=20)

    def test_routed_discovery_requests_and_reported_cost(self):
        import httpx
        requests = []

        def respond(request):
            self.assertEqual(request.headers["authorization"], "Bearer fixture-secret")
            if request.url.path == "/api/v1/models":
                return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-120b"}]})
            self.assertEqual(request.url.path, "/api/v1/chat/completions")
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, json={
                "id": "remote-id", "model": "openai/gpt-oss-120b", "provider": "Fixture",
                "choices": [{"message": {"content": '{"ok":true}' if "response_format" in body else "Design"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                          "cost": 0.0123},
            })

        backend = self.backend(respond)
        backend.verify_model()
        result = backend.call_json(system_prompt="JSON", user_prompt="Go")
        self.assertEqual(result.payload, {"ok": True})
        self.assertEqual(result.metrics["estimated_cost_usd"], 0.0123)
        self.assertEqual(result.metrics["raw_usage"]["cost"], 0.0123)
        self.assertEqual(result.metrics["served_model"], "openai/gpt-oss-120b")
        self.assertEqual(result.metrics["provider"], "Fixture")
        self.assertEqual(backend.call_text(system_prompt="Text", user_prompt="Go").raw_text, "Design")
        for body in requests:
            self.assertEqual(body["model"], "openai/gpt-oss-120b:nitro")
            self.assertEqual(body["reasoning"], {"effort": "medium"})
            self.assertNotIn("chat_template_kwargs", body)
            self.assertNotIn("top_k", body)
        self.assertNotIn("response_format", requests[1])

    def test_absent_cost_is_unknown_and_bad_final_answers_fail(self):
        import httpx
        for message in ({"content": "{}"}, {"content": "[]"}, {"content": "not JSON"},
                        {"content": "", "reasoning": "{}"},
                        {"content": "{}", "refusal": "Refused"}):
            with self.subTest(message=message):
                backend = self.backend(lambda request: httpx.Response(200, json={
                    "choices": [{"message": message, "finish_reason": "stop"}]}))
                if message == {"content": "{}"}:
                    self.assertIsNone(backend.call_json(system_prompt="JSON", user_prompt="Go").metrics["estimated_cost_usd"])
                else:
                    with self.assertRaises(RuntimeError):
                        backend.call_json(system_prompt="JSON", user_prompt="Go")

    def test_unknown_model_and_missing_key_fail(self):
        import httpx
        from harness_improvement.llm_backend import JsonLLMBackend
        backend = self.backend(lambda request: httpx.Response(200, json={"data": [{"id": "other"}]}))
        with self.assertRaises(RuntimeError):
            backend.verify_model()
        for key, url in (("", "https://openrouter.ai/api/v1"), ("secret", "http://other/v1")):
            with self.assertRaises(ValueError):
                JsonLLMBackend(backend="openrouter", model="m", api_key=key, base_url=url)

    def test_remote_errors_never_use_local_fallback_or_retry(self):
        import httpx
        from openai import APIStatusError
        for status in (400, 401, 402, 429, 500):
            requests = []

            def respond(request):
                requests.append(request)
                return httpx.Response(status, json={"error": "'response_format.type' must be 'json_schema' or 'text'"})

            with self.assertRaises(APIStatusError):
                self.backend(respond).call_json(system_prompt="JSON", user_prompt="Go")
            self.assertEqual(len(requests), 1)
