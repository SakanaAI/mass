#!/usr/bin/env python3
"""Render recorded qwen-code conversations into masked SFT windows."""
import argparse
import copy
import glob
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

from transformers import AutoTokenizer

IGNORE = -100
SENT = {}   # md5(original prompt text) -> qwen-code sent form (loaded by load_sent_index)


def load_sent_index(index_path):
    """Load sent/INDEX.json: {md5(original): {"sent_file": ...}} with the sent texts next to it."""
    d = os.path.dirname(index_path)
    for h, v in json.load(open(index_path)).items():
        SENT[h] = open(os.path.join(d, v["sent_file"]), encoding="utf-8").read()
    return len(SENT)


def sent_form(prompt):
    return SENT.get(hashlib.md5(prompt.encode("utf-8")).hexdigest())


def task_of_cond(cond):
    import re as _re
    m = _re.match(r"t(\d+)_", cond or "")
    return int(m.group(1)) if m else None


def open_maybe_gz(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_requests(logs_dir):
    for cand in ("api_requests.jsonl.gz", "api_requests.jsonl"):
        p = os.path.join(logs_dir, cand)
        if os.path.exists(p):
            recs = []
            with open_maybe_gz(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            recs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return recs, p
    return None, None


def vllm_postprocess(messages):
    """Mimic vLLM's request post-processing: tool_call.function.arguments str -> object."""
    out = []
    for m in messages:
        m = copy.deepcopy(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a)
                    except json.JSONDecodeError:
                        pass
        out.append(m)
    return out


def response_to_replay_message(resp_msg):
    """Convert a proxy-reassembled response message to the replay (request) form."""
    m = {"role": "assistant", "content": resp_msg.get("content") or ""}
    if resp_msg.get("reasoning_content"):
        m["reasoning_content"] = resp_msg["reasoning_content"]
    tcs = resp_msg.get("tool_calls") or []
    if tcs:
        m["tool_calls"] = [{"id": t.get("id"), "type": t.get("type") or "function",
                            "function": {"name": t["function"]["name"],
                                         "arguments": t["function"]["arguments"]}} for t in tcs]
    return m


def thread_key(msgs):
    u = msgs[1]["content"] if len(msgs) > 1 else ""
    return hashlib.md5((msgs[0]["content"] + "\x00" + json.dumps(u, ensure_ascii=False)).encode()).hexdigest()[:10]


def messages_prefix(a, b):
    """True if message list a is a prefix of b (exact JSON equality per message)."""
    if len(a) > len(b):
        return False
    return all(json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(a, b))


def group_threads(chats):
    """Group chat requests into (thread_key, segment) conversations."""
    by_key = defaultdict(list)
    for r in chats:
        by_key[thread_key(r["request"]["messages"])].append(r)
    convs = []
    for key, rs in by_key.items():
        rs.sort(key=lambda r: r["seq"])
        seg = [rs[0]]
        for r in rs[1:]:
            if messages_prefix(seg[-1]["request"]["messages"], r["request"]["messages"]):
                seg.append(r)
            else:
                convs.append((key, seg))
                seg = [r]
        convs.append((key, seg))
    return convs


def user_text_items(content):
    if isinstance(content, str):
        return [content]
    return [it.get("text", "") for it in content if isinstance(it, dict)]


def qwen_metric_at_prompt(prompt):
    """qwen-code 0.20.0's atCommandProcessor rewrites unresolved @-paths ('precision@k' -> 'precision @k') and trims the query.
    Only the two forms present in the frozen prompts are handled (task 202); anything else must match exactly."""
    if "precision@k" not in prompt and "recall@k" not in prompt:
        return prompt
    return prompt.replace("precision@k", "precision @k").replace("recall@k", "recall @k").strip()


def prompt_variants(prompt):
    out = [(prompt, "exact")]
    sf = sent_form(prompt)
    if sf and sf != prompt:
        out.append((sf, "qwen_sent"))
    for cand, mode in ((prompt.strip(), "strip"), (qwen_metric_at_prompt(prompt), "qwen_metric_at"), (qwen_metric_at_prompt(prompt).strip(), "qwen_metric_at_strip")):
        if cand and all(cand != c for c, _ in out):
            out.append((cand, mode))
    return out


def prompt_match(user_content, prompt):
    """Return (matched_text, mode) if exactly one variant of `prompt` occurs in the user message, else None."""
    if not prompt:
        return None
    texts = user_text_items(user_content)
    for cand, mode in prompt_variants(prompt):
        hits = sum(t.count(cand) for t in texts)
        if hits == 1:
            return cand, mode
        if hits > 1:
            raise ValueError(f"ambiguous prompt match: {hits} occurrences ({mode})")
    return None


def substitute_prompt(user_content, teacher_prompt, bare_prompt):
    """Replace the (possibly qwen-normalized) teacher prompt with the bare prompt, normalized the way qwen-code will present it."""
    hit = prompt_match(user_content, teacher_prompt)
    assert hit is not None, "teacher prompt not found in user message"
    matched, mode = hit
    if mode == "qwen_sent":
        bare = sent_form(bare_prompt) or bare_prompt   # what qwen-code presents for the bare prompt (identical unless it has @-tokens)
    else:
        bare = qwen_metric_at_prompt(bare_prompt) if mode.startswith("qwen_metric_at") else bare_prompt
    if isinstance(user_content, str):
        return user_content.replace(matched, bare, 1)
    out = copy.deepcopy(user_content)
    hits = 0
    for it in out:
        if isinstance(it, dict) and it.get("type") == "text" and matched in it.get("text", ""):
            it["text"] = it["text"].replace(matched, bare, 1)
            hits += 1
    assert hits == 1, f"expected exactly one prompt item, found {hits}"
    return out


def thread_role(system_content, is_main):
    s = system_content if isinstance(system_content, str) else " ".join(x.get("text", "") for x in (system_content or []) if isinstance(x, dict))
    if is_main or s.startswith("You are Qwen Code"):
        return "orchestrator"
    if s.startswith("You are a general-purpose subagent"):
        return "worker"
    return "auxiliary"


class Renderer:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.gen_prefix_ids = self.tok("<|im_start|>assistant\n<think>\n", add_special_tokens=False)["input_ids"]
        self.im_end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.think_end_id = self.tok.convert_tokens_to_ids("</think>")
        self.n_empty_reasoning = 0

    def render(self, messages, tools, enable_thinking=True):
        return self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=False,
                                            tokenize=False, enable_thinking=enable_thinking)

    @staticmethod
    def block_plan(msgs):
        """Expected block structure: (start_idx, end_idx_exclusive, kind). Consecutive
        tool messages share one block (the template merges them into one user turn)."""
        bounds = []
        i = 0
        while i < len(msgs):
            role = msgs[i]["role"]
            if role == "tool":
                j = i
                while j < len(msgs) and msgs[j]["role"] == "tool":
                    j += 1
                bounds.append((i, j, "tool"))
                i = j
            else:
                bounds.append((i, i + 1, role))
                i += 1
        return bounds

    def segments(self, messages, tools):
        """Split the conversation into query segments (one per user query). The Qwen
        template keeps <think> only for assistant turns after the LAST user query, so
        turns of segment s were generated with the context rendered from msgs[:end_s]
        (earlier segments' reasoning stripped). Returns a list of block lists, one per
        segment; loss masks are set only on that segment's assistant blocks."""
        msgs = vllm_postprocess(messages)
        q_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        if not q_idx or msgs[0]["role"] != "system" or q_idx[0] != 1:
            raise ValueError("conversation must start with system, user")
        out = []
        for s, q in enumerate(q_idx):
            end = q_idx[s + 1] if s + 1 < len(q_idx) else len(msgs)
            sub = msgs[:end]
            plan = self.block_plan(sub)
            text = self.render(sub, tools)
            pieces = text.split("<|im_start|>")
            if pieces[0] != "":
                raise ValueError("render does not start with <|im_start|>")
            pieces = ["<|im_start|>" + p for p in pieces[1:]]
            if len(pieces) != len(plan):
                raise ValueError(f"segment {s}: {len(pieces)} <|im_start|> pieces vs {len(plan)} planned blocks "
                                 f"(content containing the literal marker?)")
            blocks = []
            for piece, (a, b, kind) in zip(pieces, plan):
                head = {"system": "<|im_start|>system\n", "user": "<|im_start|>user\n",
                        "assistant": "<|im_start|>assistant\n", "tool": "<|im_start|>user\n<tool_response>"}[kind]
                if not piece.startswith(head):
                    raise ValueError(f"segment {s}: block {kind}@{a} does not start with {head!r}: {piece[:40]!r}")
                ids = self.tok(piece, add_special_tokens=False)["input_ids"]
                mask = [0] * len(ids)
                in_segment = a >= q
                if kind == "assistant" and in_segment:
                    n = len(self.gen_prefix_ids)
                    if ids[:n] == self.gen_prefix_ids:
                        start = n
                    elif ids[:n - 1] == self.gen_prefix_ids[:-1]:
                        # empty reasoning: "<think>\n\n</think>" merges the newlines into one
                        # token; mask it and start the loss at </think>
                        try:
                            start = ids.index(self.think_end_id, n - 1)
                        except ValueError:
                            raise ValueError(f"segment {s}: assistant block @{a} has no </think>")
                        self.n_empty_reasoning += 1
                    else:
                        raise ValueError(f"segment {s}: assistant block @{a} lacks the generation prefix")
                    try:
                        last = len(ids) - 1 - ids[::-1].index(self.im_end_id)
                    except ValueError:
                        raise ValueError("assistant block has no <|im_end|>")
                    for t in range(start, last + 1):
                        mask[t] = 1
                blocks.append({"kind": kind, "ids": ids, "mask": mask, "msg_range": [a, b],
                               "segment": s, "is_query": (kind == "user" and a == q)})
            whole = self.tok(text, add_special_tokens=False)["input_ids"]
            cat = [t for b in blocks for t in b["ids"]]
            if whole != cat:
                raise ValueError(f"segment {s}: block-wise tokenization differs from whole ({len(whole)} vs {len(cat)})")
            out.append(blocks)
        return out


def make_windows(blocks, window, overlap, max_header_frac=0.9):
    """Partition this segment's loss-bearing assistant blocks into windows:
    header (system + first user) + [<= overlap tokens of preceding blocks, masked;
    always incl. the segment's query block] + contiguous blocks from a loss block."""
    assert blocks[0]["kind"] == "system" and blocks[1]["kind"] == "user"
    header = blocks[:2]
    header_len = sum(len(b["ids"]) for b in header)
    if header_len > window * max_header_frac:
        raise ValueError(f"header ({header_len} tokens) exceeds {max_header_frac:.0%} of window {window}")
    body = blocks[2:]
    q_block = next((j for j, b in enumerate(body) if b.get("is_query")), None)  # None for segment 0
    has_loss = [any(b["mask"]) for b in body]
    windows, skipped = [], []
    i = 0
    while i < len(body):
        if not has_loss[i]:
            i += 1
            continue
        ctx, used = [], 0
        j = i - 1
        while j >= 0 and used + len(body[j]["ids"]) <= overlap:
            ctx.insert(0, j); used += len(body[j]["ids"]); j -= 1
        if q_block is not None and q_block < i and q_block not in ctx:
            # the segment's own query must be visible: drop the oldest ctx blocks to make room
            ctx = [c for c in ctx if c > q_block]
            ctx.insert(0, q_block)
            used = sum(len(body[c]["ids"]) for c in ctx)
        budget = window - header_len - used
        if len(body[i]["ids"]) > budget:
            if len(body[i]["ids"]) <= window - header_len:
                ctx = [q_block] if (q_block is not None and q_block < i and len(body[q_block]["ids"]) + len(body[i]["ids"]) <= window - header_len) else []
                used = sum(len(body[c]["ids"]) for c in ctx)
                budget = window - header_len - used
            else:
                skipped.append({"block": i + 2, "tokens": len(body[i]["ids"])})
                i += 1
                continue
        k, total, last_loss = i, 0, i
        while k < len(body) and total + len(body[k]["ids"]) <= budget:
            total += len(body[k]["ids"])
            if has_loss[k]:
                last_loss = k
            k += 1
        end = last_loss + 1
        ids, labels, loss_blocks = [], [], []
        for b in header:
            ids += b["ids"]; labels += [IGNORE] * len(b["ids"])
        for c in ctx:
            ids += body[c]["ids"]; labels += [IGNORE] * len(body[c]["ids"])
        for b_idx in range(i, end):
            b = body[b_idx]
            ids += b["ids"]
            labels += [t if m else IGNORE for t, m in zip(b["ids"], b["mask"])]
            if has_loss[b_idx]:
                loss_blocks.append(b_idx + 2)
        windows.append({"input_ids": ids, "labels": labels, "n_tokens": len(ids),
                        "n_loss_tokens": sum(1 for l in labels if l != IGNORE),
                        "blocks": [i + 2, end + 2], "ctx_blocks": [c + 2 for c in ctx],
                        "loss_blocks": loss_blocks, "header_tokens": header_len})
        i = end
    return windows, skipped


def process_episode(logs_dir, renderer, args, teacher_prompts, bare):
    bare_map, bare_default = bare if isinstance(bare, tuple) else ({}, bare)
    status_p = os.path.join(logs_dir, "run_status.json")
    status = json.load(open(status_p)) if os.path.exists(status_p) else None
    recs, log_path = load_requests(logs_dir)
    if recs is None:
        return None, "no api log"
    chats = [r for r in recs if r.get("path") == "/v1/chat/completions" and r.get("status") == 200
             and r.get("response") and (r["response"].get("choices") or [])]
    if not chats:
        return None, "no chat requests"
    cond = (status or {}).get("hiw", {}).get("condition") or os.path.basename(logs_dir).rsplit("_s", 1)[0]
    teacher_prompt = teacher_prompts.get(cond)
    bare_prompt = bare_map.get(cond, bare_default)
    if bare_prompt is None:
        return None, f"no bare prompt for condition {cond}"
    convs = group_threads(chats)
    ep_name = os.path.basename(logs_dir.rstrip("/"))
    samples, info = [], {"episode": ep_name, "condition": cond, "seed": (status or {}).get("seed"),
                         "success": (status or {}).get("trace_has_success_result"),
                         "n_requests": len(chats), "threads": [], "skipped_blocks": []}
    for key, seg in convs:
        last = seg[-1]
        req = last["request"]
        msgs = copy.deepcopy(req["messages"])
        choice = last["response"]["choices"][0]
        final = response_to_replay_message(choice["message"])
        truncated = choice.get("finish_reason") == "length"
        if not (truncated and args.drop_truncated_final):
            msgs.append(final)
        hit = prompt_match(msgs[1]["content"], teacher_prompt) if teacher_prompt else prompt_match(msgs[1]["content"], bare_prompt)
        is_main = hit is not None
        role = thread_role(msgs[0]["content"], is_main)
        if is_main and teacher_prompt and args.substitute:
            msgs[1]["content"] = substitute_prompt(msgs[1]["content"], teacher_prompt, bare_prompt)
        try:
            seg_blocks = renderer.segments(msgs, req.get("tools"))
        except ValueError as e:
            info["threads"].append({"key": key, "error": str(e)})
            continue
        n_tok = sum(len(b["ids"]) for b in seg_blocks[-1])
        n_asst = sum(1 for m in msgs if m["role"] == "assistant")
        wins, skipped = [], []
        try:
            for s_i, blocks in enumerate(seg_blocks):
                w_s, sk_s = make_windows(blocks, args.window, args.context_overlap)
                for w in w_s:
                    w["segment"] = s_i
                wins += w_s; skipped += sk_s
        except ValueError as e:  # e.g. a sub-agent thread whose header alone nearly fills the window
            info["threads"].append({"key": key, "is_main": is_main, "error": str(e)})
            continue
        for w_i, w in enumerate(wins):
            w.update({"episode": ep_name, "condition": cond, "seed": info["seed"], "thread": key, "task": task_of_cond(cond),
                      "is_main": is_main, "role": role, "match_mode": (hit[1] if hit else None), "window": w_i, "n_windows": len(wins), "success": info["success"]})
            samples.append(w)
        info["threads"].append({"key": key, "is_main": is_main, "role": role, "match_mode": (hit[1] if hit else None), "n_requests": len(seg), "n_messages": len(msgs),
                                "n_segments": len(seg_blocks), "n_tokens": n_tok, "n_assistant_blocks": n_asst,
                                "n_windows": len(wins), "final_truncated": truncated, "skipped": skipped})
        info["skipped_blocks"] += skipped
    return samples, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="*", default=[], help="runlog dirs or globs")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--bare-prompt", default=None, help="student prompt (single-task); or use --bare-prompt-map")
    ap.add_argument("--bare-prompt-map", action="append", default=[], help="condition=path (multi-task): bare prompt per condition")
    ap.add_argument("--episode-list", default=None, help="JSON list of {logs_dir, condition?, weight?} (from select_episodes.py); overrides --episodes")
    ap.add_argument("--teacher-prompt", action="append", default=[],
                    help="condition=path, e.g. teacher_v10=/path/query51_ourTeam-v10.txt")
    ap.add_argument("--window", type=int, default=32768)
    ap.add_argument("--context-overlap", type=int, default=8192)
    ap.add_argument("--require-success", action="store_true")
    ap.add_argument("--no-substitute", dest="substitute", action="store_false")
    ap.add_argument("--keep-truncated-final", dest="drop_truncated_final", action="store_false")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--strict", action="store_true", help="exit 1 if any used episode has a thread error, a skipped block, or a main thread without a prompt match")
    ap.add_argument("--sent-index", default=None, help="sent/INDEX.json (qwen-code sent forms of the prompts)")
    args = ap.parse_args()
    if args.sent_index:
        print(f"sent-form index: {load_sent_index(args.sent_index)} prompts", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    renderer = Renderer(tok)
    bare_default = open(args.bare_prompt, encoding="utf-8").read() if args.bare_prompt else None
    bare_map = {}
    for spec in args.bare_prompt_map:
        c, p = spec.split("=", 1)
        bare_map[c] = open(p, encoding="utf-8").read()
    teacher_prompts = {}
    for spec in args.teacher_prompt:
        c, p = spec.split("=", 1)
        teacher_prompts[c] = open(p, encoding="utf-8").read()

    weights = {}
    if args.episode_list:
        items = json.load(open(args.episode_list))
        dirs = [it["logs_dir"] for it in items]
        weights = {os.path.basename(it["logs_dir"].rstrip("/")): it.get("weight", 1.0) for it in items}
    else:
        dirs = []
        for pat in args.episodes:
            dirs += sorted(d for d in glob.glob(os.path.expanduser(pat)) if os.path.isdir(d))
    all_info, n_samples = [], 0
    agg = Counter()
    with open(args.out, "w", encoding="utf-8") as out:
        for d in dirs:
            samples, info = process_episode(d, renderer, args, teacher_prompts, (bare_map, bare_default))
            for smp in samples or []:
                smp["weight"] = weights.get(smp["episode"], 1.0)
            if samples is None:
                print(f"SKIP {d}: {info}", file=sys.stderr)
                continue
            if args.require_success and not info["success"]:
                print(f"SKIP {info['episode']}: not successful", file=sys.stderr)
                all_info.append({**info, "used": False})
                continue
            for s in samples:
                out.write(json.dumps(s) + "\n")
            n_samples += len(samples)
            agg["episodes"] += 1
            agg["windows"] += len(samples)
            agg["loss_tokens"] += sum(s["n_loss_tokens"] for s in samples)
            agg["total_tokens"] += sum(s["n_tokens"] for s in samples)
            agg["threads"] += len(info["threads"])
            agg["skipped_blocks"] += len(info["skipped_blocks"])
            all_info.append({**info, "used": True})
            print(f"{info['episode']}: cond={info['condition']} success={info['success']} threads={len(info['threads'])} "
                  f"windows={len(samples)} loss_tokens={sum(s['n_loss_tokens'] for s in samples)} "
                  f"skipped_blocks={len(info['skipped_blocks'])}", file=sys.stderr)
    agg["empty_reasoning_turns"] = renderer.n_empty_reasoning
    print(f"TOTAL: {dict(agg)} -> {args.out}", file=sys.stderr)
    if args.stats:
        json.dump({"args": vars(args), "aggregate": dict(agg), "episodes": all_info}, open(args.stats, "w"), indent=1)
    if args.strict:
        bad = []
        for e in all_info:
            if not e.get("used"):
                continue
            errs = [t for t in e["threads"] if "error" in t]
            no_main = not any(t.get("is_main") for t in e["threads"])
            if errs or e["skipped_blocks"] or no_main:
                bad.append({"episode": e["episode"], "thread_errors": [t.get("error") for t in errs], "skipped": e["skipped_blocks"], "no_main_thread": no_main})
        if bad:
            print("STRICT AUDIT FAILED:\n" + json.dumps(bad, indent=1), file=sys.stderr)
            sys.exit(1)
        print("strict audit passed: 0 thread errors, 0 skipped blocks, every episode has a matched main thread", file=sys.stderr)


if __name__ == "__main__":
    main()
