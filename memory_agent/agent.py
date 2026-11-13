"""A LangGraph chat agent for checking that a session keeps its memory.

The conversation lives in the LangGraph thread: the Agent Server checkpoints
`messages` and `turns` after every run, and each new turn on the same thread
sees everything said before. If the session is moved onto a new thread (for
example after the platform's harness process restarts), the history is gone
and `turns` starts again at 1.

Send `/history` at any time for a reply that does not use the model at all:
the thread id, the turn count and every earlier user message. Compare it
before and after a restart.

Send `/env` (or `printenv`), optionally with a name prefix (`/env TEST_`), for
the Agent Server's own environment, also without the model. It is the env this
process started with, so it shows whether a live env or secret update reached
the graph. Values of secret-looking names are masked.

Send `/tools` for the tools the `do.actions` Action Gateway advertises to this
graph, listed with its own MCP client from HARNESS_MCP_URL_DO_ACTIONS. The URL
is read on every call, so after a live tools update it shows the gateway
version the restarted Agent Server was handed.

Send `/notion <text>` to append a paragraph to the Notion page NOTION_PAGE_ID
through a Notion toolbelt on `do.actions`, or `/notion` alone to list the
Notion tools the session can reach. A toolbelt is not advertised in tools/list,
so the graph finds the append tool with action_search and calls it through
action_invoke.

With HARNESS_INFERENCE_* set, other messages go to the model with the whole
history (so "what is my secret word?" works). Without them the agent answers
deterministically, which is enough to test memory without an API key.
"""

# No `from __future__ import annotations`: LangGraph reads the state
# TypedDict's annotations at runtime.
import asyncio
import concurrent.futures
import hashlib
import json
import operator
import os
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

SYSTEM_PROMPT = (
    "You are a helpful assistant with a good memory. When the user asks about "
    "something they told you earlier in this conversation, answer from it."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # Summed across runs, so it counts turns on this thread.
    turns: Annotated[int, operator.add]


def _llm():
    """The chat model, or None when inference is not configured."""
    keys = ("HARNESS_INFERENCE_MODEL", "HARNESS_INFERENCE_BASE_URL", "HARNESS_INFERENCE_API_KEY")
    if not all(os.environ.get(k, "").strip() for k in keys):
        return None
    from langchain_openai import ChatOpenAI

    # streaming / stream_usage: per-token replies and token counts on the
    # platform (see the reactAgent example).
    return ChatOpenAI(
        model=os.environ["HARNESS_INFERENCE_MODEL"],
        base_url=os.environ["HARNESS_INFERENCE_BASE_URL"],
        api_key=os.environ["HARNESS_INFERENCE_API_KEY"],
        temperature=0,
        streaming=True,
        stream_usage=True,
    )


llm = _llm()


def _text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))


def _history_report(state: State, turn: int, thread_id: str) -> str:
    earlier = [_text(m) for m in state["messages"][:-1] if isinstance(m, HumanMessage)]
    lines = [
        f"thread_id: {thread_id}",
        f"turn: {turn}",
        f"agent server pid: {os.getpid()}",
        f"earlier user messages: {len(earlier)}",
    ]
    lines += [f"  {i}. {text}" for i, text in enumerate(earlier, 1)]
    return "\n".join(lines)


# A name containing any of these is masked by /env.
SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")


def _env_report(prefix: str = "") -> str:
    """The Agent Server's environment, as this process started with it.

    Values of secret-looking names are replaced by their length and a short
    sha256: enough to see a rotation, without the value. The reply travels
    through the session's event stream, so a real secret must never appear in
    it; an 8-hex digest of a short test value can still be guessed, so do not
    rely on the masking for real secrets.
    """
    lines = [f"agent server pid: {os.getpid()}"]
    for name in sorted(os.environ):
        if prefix and not name.startswith(prefix):
            continue
        value = os.environ[name]
        if any(hint in name.upper() for hint in SECRET_HINTS):
            digest = hashlib.sha256(value.encode()).hexdigest()[:8]
            value = f"<redacted len={len(value)} sha256={digest}>"
        lines.append(f"{name}={value}")
    if len(lines) == 1:
        lines.append(f"(no variable starts with {prefix!r})" if prefix else "(empty environment)")
    return "\n".join(lines)


DO_ACTIONS_URL_ENV = "HARNESS_MCP_URL_DO_ACTIONS"


async def _list_gateway_tools(url: str) -> list[str]:
    """The tool names the Action Gateway advertises in tools/list."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({"do_actions": {"url": url, "transport": "streamable_http"}})
    return sorted(tool.name for tool in await client.get_tools())


def _run_async(coro, timeout: float = 60):
    """Run coro to completion from this sync node, in a thread of its own so
    it works whether or not the caller already has an event loop running."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=timeout)


def _tools_report() -> str:
    url = os.environ.get(DO_ACTIONS_URL_ENV, "").strip()
    lines = [f"agent server pid: {os.getpid()}"]
    if not url:
        lines.append(f"do.actions is not attached ({DO_ACTIONS_URL_ENV} is unset)")
        return "\n".join(lines)
    lines.append(f"do_actions url: {url}")
    try:
        names = _run_async(_list_gateway_tools(url))
    except Exception as e:  # report it; a failed listing must not fail the turn
        lines.append(f"could not list tools: {type(e).__name__}: {e}")
        return "\n".join(lines)
    lines.append(f"tools: {len(names)}")
    lines += [f"  - {name}" for name in names]
    return "\n".join(lines)


NOTION_PAGE_ENV = "NOTION_PAGE_ID"
NOTION_TOOL_ENV = "NOTION_APPEND_TOOL"
NOTION_USE_CASE = "Append a paragraph block to the end of an existing Notion page"


async def _gateway_call(url: str, name: str, arguments: dict) -> dict:
    """Call one gateway tool and return its structured result."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({"do_actions": {"url": url, "transport": "streamable_http"}})
    async with client.session("do_actions") as session:
        result = await session.call_tool(name, arguments)
    text = "".join(getattr(c, "text", "") for c in result.content)
    if result.isError:
        raise RuntimeError(f"{name} failed: {text[:500]}")
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(text)


def _search_notion_tools(url: str) -> list[dict]:
    """The Notion tools action_search finds in this session's selection (the
    toolbelt), each with its name, version and inputSchema."""
    query = {"queries": [{"use_case": NOTION_USE_CASE}], "limit": 10}
    resp = _run_async(_gateway_call(url, "action_search", query))
    return [t for group in resp.get("results", []) for t in group.get("results", [])]


def _pick_append_tool(tools: list[dict]) -> dict | None:
    """NOTION_APPEND_TOOL when set, else the first tool that appends blocks."""
    wanted = os.environ.get(NOTION_TOOL_ENV, "").strip()
    if wanted:
        return next((t for t in tools if t.get("name") == wanted), {"name": wanted})
    for t in tools:
        if "append" in t.get("name", "").lower():
            return t
    for t in tools:
        name = t.get("name", "").lower()
        if "children" in name and any(verb in name for verb in ("patch", "add", "create", "update")):
            return t
    return None


def _page_id(value: str) -> str:
    """A Notion page id, also from a page URL (the last 32 hex digits of its path)."""
    tail = value.strip().split("?")[0].split("#")[0].rstrip("/").rsplit("/", 1)[-1]
    tail = tail.replace("-", "").lower()[-32:]
    return tail if re.fullmatch(r"[0-9a-f]{32}", tail) else value.strip()


def _place(props: dict, names: tuple, value, args: dict) -> None:
    """Put value under the first of names the schema has, at the top level or in
    a path/body group, else under names[0]."""
    for name in names:
        if name in props:
            args[name] = value
            return
    for group in ("path", "body", "params", "path_params"):
        sub = (props.get(group) or {}).get("properties") or {}
        for name in names:
            if name in sub:
                args.setdefault(group, {})[name] = value
                return
    args[names[0]] = value


def _append_arguments(schema: dict | None, page_id: str, text: str) -> dict:
    """Arguments for Notion's append-block-children call, laid out as the
    tool's inputSchema names them."""
    props = (schema or {}).get("properties") or {}
    paragraph = {"rich_text": [{"type": "text", "text": {"content": text}}]}
    args: dict = {}
    _place(props, ("block_id", "page_id", "id", "blockId", "pageId"), page_id, args)
    _place(props, ("children",), [{"object": "block", "type": "paragraph", "paragraph": paragraph}], args)
    return args


def _notion_report(text: str) -> str:
    url = os.environ.get(DO_ACTIONS_URL_ENV, "").strip()
    lines = [f"agent server pid: {os.getpid()}"]
    if not url:
        lines.append(f"do.actions is not attached ({DO_ACTIONS_URL_ENV} is unset)")
        return "\n".join(lines)
    page = _page_id(os.environ.get(NOTION_PAGE_ENV, ""))
    lines.append(f"notion page: {page or f'(unset; set {NOTION_PAGE_ENV})'}")
    try:
        tools = _search_notion_tools(url)
        tool = _pick_append_tool(tools)
        if not text:
            lines.append(f"notion tools found: {len(tools)}")
            for t in tools:
                args = ", ".join(sorted(((t.get("inputSchema") or {}).get("properties") or {})))
                lines.append(f"  - {t.get('name')}@{t.get('version', '?')} ({args})")
            lines.append(f"append tool: {tool['name'] if tool else f'(none found; set {NOTION_TOOL_ENV})'}")
            return "\n".join(lines)
        if not page:
            lines.append(f"nothing written: set {NOTION_PAGE_ENV}")
            return "\n".join(lines)
        if not tool:
            lines.append(f"nothing written: no append tool found; set {NOTION_TOOL_ENV}")
            return "\n".join(lines)
        call = {"tool": tool["name"], "arguments": _append_arguments(tool.get("inputSchema"), page, text)}
        if tool.get("version"):
            call["version"] = tool["version"]
        lines.append(f"tool: {tool['name']}@{tool.get('version', '?')}")
        invoke = {"rationale": "Append a paragraph to a Notion page", "tools": [call]}
        resp = _run_async(_gateway_call(url, "action_invoke", invoke))
    except Exception as e:  # report it; a failed call must not fail the turn
        lines.append(f"notion call failed: {type(e).__name__}: {e}")
        return "\n".join(lines)
    result = (resp.get("results") or [{}])[0].get("result") or {}
    lines.append(f"status: {result.get('status', '?')}")
    meta = result.get("_meta") or {}
    if meta.get("status"):
        lines.append(f"gateway: {meta['status']}")
    error = result.get("error") or {}
    if error:
        hint = f" ({error['recovery_hint']})" if error.get("recovery_hint") else ""
        lines.append(f"error: {error.get('class', '?')}: {error.get('message', '')}{hint}")
    output = result.get("output")
    # Notion answers an append with the new blocks under "results".
    blocks = output.get("results") if isinstance(output, dict) else None
    if isinstance(blocks, list) and blocks:
        lines += [f"new block: {b.get('id')}" for b in blocks if isinstance(b, dict)]
    elif output is not None:
        lines.append(f"output: {json.dumps(output)[:300]}")
    return "\n".join(lines)


def chat(state: State, config: RunnableConfig) -> dict:
    turn = state.get("turns", 0) + 1
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    last = _text(state["messages"][-1]).strip()
    words = last.split(maxsplit=1)
    command = words[0].lower() if words else ""

    if last.lower() in ("/history", "history"):
        reply = _history_report(state, turn, thread_id)
    elif command in ("/env", "printenv"):
        reply = _env_report(words[1].strip() if len(words) > 1 else "")
    elif command == "/tools":
        reply = _tools_report()
    elif command == "/notion":
        reply = _notion_report(words[1].strip() if len(words) > 1 else "")
    elif llm is not None:
        reply = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]]).content
    else:
        earlier = sum(isinstance(m, HumanMessage) for m in state["messages"][:-1])
        reply = (
            f"Turn {turn} on thread {thread_id}. You said: {last!r}. "
            f"I remember {earlier} earlier message(s); send /history to list them."
        )
    return {"messages": [AIMessage(reply)], "turns": 1}


workflow = StateGraph(State)
workflow.add_node("chat", chat)
workflow.add_edge(START, "chat")
workflow.add_edge("chat", END)
graph = workflow.compile()
