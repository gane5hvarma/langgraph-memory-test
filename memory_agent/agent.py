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

With HARNESS_INFERENCE_* set, other messages go to the model with the whole
history (so "what is my secret word?" works). Without them the agent answers
deterministically, which is enough to test memory without an API key.
"""

# No `from __future__ import annotations`: LangGraph reads the state
# TypedDict's annotations at runtime.
import asyncio
import concurrent.futures
import hashlib
import operator
import os
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
