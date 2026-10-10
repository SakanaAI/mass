#!/usr/bin/env python3
"""Byte-faithful logging reverse proxy for an OpenAI-compatible vLLM replica.

    api_proxy.py --listen 9001 --upstream http://127.0.0.1:8001 --log api_requests.jsonl

Every request is forwarded unchanged except for hop-by-hop headers and
Accept-Encoding, which is set to identity so logged bodies can be parsed.
The upstream response is relayed unchanged (streaming
SSE is relayed chunk-by-chunk as it arrives). One JSON line per request is
appended to --log with the parsed request body and the response, where a
streamed chat completion is reassembled into a single message (content,
reasoning_content, tool_calls, finish_reason, usage).

Purpose: capture EXACTLY what qwen-code sends to the model (system prompt,
tool schemas, message history incl. reasoning replay, compression) so
training samples can be rendered from the very same messages the served
student will see. Request and response body contents are not modified.
"""
import argparse
import asyncio
import json
import os
import signal
import sys
import time

from aiohttp import ClientSession, ClientTimeout, web

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def assemble_sse(raw: bytes) -> dict:
    """Reassemble an OpenAI chat.completion.chunk SSE stream into one message."""
    choices = {}
    meta = {"id": None, "model": None, "created": None}
    usage = None
    error = None
    n_chunks = 0
    n_bad = 0
    for line in raw.decode("utf-8", "replace").split("\n"):
        line = line.strip("\r")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            n_bad += 1
            continue
        n_chunks += 1
        if obj.get("error"):
            error = obj["error"]
        for k in meta:
            if meta[k] is None and obj.get(k) is not None:
                meta[k] = obj[k]
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices") or []:
            idx = ch.get("index", 0)
            st = choices.setdefault(idx, {
                "index": idx, "role": None, "content": [], "reasoning_content": [],
                "reasoning_details": [],
                "tool_calls": {}, "finish_reason": None, "stop_reason": None,
            })
            delta = ch.get("delta") or {}
            if delta.get("role"):
                st["role"] = delta["role"]
            if delta.get("content"):
                st["content"].append(delta["content"])
            rc = delta.get("reasoning_content")
            if rc is None:
                rc = delta.get("reasoning")
            if rc:
                st["reasoning_content"].append(rc)
            st["reasoning_details"].extend(delta.get("reasoning_details") or [])
            for tc in delta.get("tool_calls") or []:
                ti = tc.get("index", 0)
                t = st["tool_calls"].setdefault(ti, {"index": ti, "id": None, "type": None,
                                                      "name": None, "arguments": []})
                if tc.get("id"):
                    t["id"] = tc["id"]
                if tc.get("type"):
                    t["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    t["name"] = (t["name"] or "") + fn["name"]
                if fn.get("arguments"):
                    t["arguments"].append(fn["arguments"])
            if ch.get("finish_reason"):
                st["finish_reason"] = ch["finish_reason"]
            if ch.get("stop_reason") is not None:
                st["stop_reason"] = ch["stop_reason"]
    out_choices = []
    for idx in sorted(choices):
        st = choices[idx]
        tool_calls = []
        for ti in sorted(st["tool_calls"]):
            t = st["tool_calls"][ti]
            tool_calls.append({
                "index": ti, "id": t["id"], "type": t["type"] or "function",
                "function": {"name": t["name"], "arguments": "".join(t["arguments"])},
            })
        msg = {
            "role": st["role"] or "assistant",
            "content": "".join(st["content"]) if st["content"] else None,
            "reasoning_content": "".join(st["reasoning_content"]) if st["reasoning_content"] else None,
            "reasoning_details": st["reasoning_details"] or None,
            "tool_calls": tool_calls or None,
        }
        out_choices.append({"index": idx, "message": msg, "finish_reason": st["finish_reason"],
                            "stop_reason": st["stop_reason"]})
    return {**meta, "object": "chat.completion.reassembled", "choices": out_choices,
            "usage": usage, "error": error, "n_chunks": n_chunks, "n_bad_chunks": n_bad}


def write_log(app, rec):
    f = app["log_fh"]
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()


async def handle(request: web.Request):
    app = request.app
    body = await request.read()
    url = app["upstream"] + request.rel_url.path_qs
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in HOP_BY_HOP and k.lower() != "accept-encoding"}
    headers["Accept-Encoding"] = "identity"
    t0 = time.time()
    rec = {
        "ts_start": t0, "ts_start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "method": request.method, "path": request.rel_url.path_qs,
        "seq": app["seq"], "request_bytes": len(body),
    }
    app["seq"] += 1
    if body:
        try:
            rec["request"] = json.loads(body)
        except json.JSONDecodeError:
            rec["request"] = None
            rec["request_raw"] = body.decode("utf-8", "replace")
    else:
        rec["request"] = None
    session: ClientSession = app["session"]
    try:
        async with session.request(request.method, url, headers=headers, data=body,
                                   allow_redirects=False) as resp:
            resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP}
            rec["status"] = resp.status
            rec["response_headers"] = {k: v for k, v in resp_headers.items()
                                       if k.lower() in ("content-type", "x-request-id")}
            ctype = resp.headers.get("Content-Type", "")
            if "text/event-stream" in ctype:
                out = web.StreamResponse(status=resp.status, headers=resp_headers)
                await out.prepare(request)
                chunks = []
                client_gone = False
                async for chunk in resp.content.iter_any():
                    chunks.append(chunk)
                    if not client_gone:
                        try:
                            await out.write(chunk)
                        except (ConnectionResetError, asyncio.CancelledError, Exception) as e:  # noqa: BLE001
                            client_gone = True
                            rec["client_disconnected"] = type(e).__name__
                raw = b"".join(chunks)
                rec["stream"] = True
                rec["response_bytes"] = len(raw)
                rec["response"] = assemble_sse(raw)
                if app["keep_raw_sse"]:
                    rec["sse_raw"] = raw.decode("utf-8", "replace")
                rec["duration_s"] = round(time.time() - t0, 3)
                write_log(app, rec)
                if not client_gone:
                    await out.write_eof()
                return out
            data = await resp.read()
            rec["stream"] = False
            rec["response_bytes"] = len(data)
            try:
                rec["response"] = json.loads(data) if data else None
            except json.JSONDecodeError:
                rec["response_raw"] = data.decode("utf-8", "replace")
            rec["duration_s"] = round(time.time() - t0, 3)
            write_log(app, rec)
            return web.Response(status=resp.status, headers=resp_headers, body=data)
    except web.HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        # HTTP exception representations can contain authenticated request headers.
        rec["proxy_error"] = type(e).__name__
        rec["duration_s"] = round(time.time() - t0, 3)
        write_log(app, rec)
        return web.Response(status=502, text=f"proxy error: {type(e).__name__}")


async def on_startup(app):
    app["session"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=30, sock_read=None),
                                   auto_decompress=False)


async def on_cleanup(app):
    await app["session"].close()
    app["log_fh"].close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--upstream", required=True, help="e.g. http://127.0.0.1:8001")
    ap.add_argument("--log", required=True)
    ap.add_argument("--keep-raw-sse", action="store_true")
    args = ap.parse_args()

    app = web.Application(client_max_size=4 * 1024 ** 3)
    app["upstream"] = args.upstream.rstrip("/")
    app["log_fh"] = open(args.log, "a", encoding="utf-8")
    app["keep_raw_sse"] = args.keep_raw_sse
    app["seq"] = 0
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", handle)

    async def run():
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, args.host, args.listen)
        await site.start()
        print(f"READY proxy http://{args.host}:{args.listen} -> {app['upstream']} log={args.log}", flush=True)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        await stop.wait()
        await runner.cleanup()
        print("proxy stopped", flush=True)

    asyncio.run(run())


if __name__ == "__main__":
    main()
