"""Local check: two turns on one thread share memory; a new thread does not.

    pip install -r requirements.txt pytest && pytest -q
"""

import os

for k in ("HARNESS_INFERENCE_MODEL", "HARNESS_INFERENCE_BASE_URL", "HARNESS_INFERENCE_API_KEY"):
    os.environ.pop(k, None)

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from memory_agent.agent import workflow  # noqa: E402

graph = workflow.compile(checkpointer=MemorySaver())


def say(thread: str, text: str) -> str:
    """The turn's reply as the session shows it: every AI message after the
    user's, joined."""
    out = graph.invoke(
        {"messages": [("user", text)]},
        {"configurable": {"thread_id": thread}},
    )
    messages = out["messages"]
    start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage)) + 1
    return "".join(m.content for m in messages[start:] if isinstance(m, AIMessage))


def test_a_thread_remembers_earlier_turns():
    say("t1", "my secret word is pineapple")
    report = say("t1", "/history")
    assert "thread_id: t1" in report
    assert "turn: 2" in report
    assert "1. my secret word is pineapple" in report


def test_a_new_thread_starts_empty():
    say("t2", "hello")
    report = say("t3", "/history")
    assert "turn: 1" in report
    assert "earlier user messages: 0" in report


def test_env_lists_the_process_env_and_masks_secrets(monkeypatch):
    monkeypatch.setenv("TEST_FLAG", "hello")
    monkeypatch.setenv("TEST_SECRET", "s3cret-v1")
    report = say("t4", "/env TEST_")
    assert f"agent server pid: {os.getpid()}" in report
    assert "TEST_FLAG=hello" in report
    assert "s3cret-v1" not in report
    assert "TEST_SECRET=<redacted len=9 sha256=" in report


def test_env_shows_a_rotation_without_the_value(monkeypatch):
    monkeypatch.setenv("TEST_SECRET", "s3cret-v1")
    before = say("t5", "printenv TEST_SECRET")
    monkeypatch.setenv("TEST_SECRET", "s3cret-v2")
    after = say("t5", "/env TEST_SECRET")
    assert before != after
    assert "s3cret-v2" not in after


def test_env_with_an_unknown_prefix_says_so():
    report = say("t6", "/env NO_SUCH_PREFIX_")
    assert "(no variable starts with 'NO_SUCH_PREFIX_')" in report


def test_tools_without_do_actions_says_so(monkeypatch):
    monkeypatch.delenv("HARNESS_MCP_URL_DO_ACTIONS", raising=False)
    report = say("t7", "/tools")
    assert "do.actions is not attached" in report


def test_tools_lists_what_the_gateway_advertises(monkeypatch):
    from memory_agent import agent

    seen = {}

    async def fake_list(url):
        seen["url"] = url
        return ["action_invoke", "action_search", "web_search"]

    monkeypatch.setenv("HARNESS_MCP_URL_DO_ACTIONS", "https://ag.example/mcp/endpoint/e1/version/2")
    monkeypatch.setattr(agent, "_list_gateway_tools", fake_list)
    report = say("t8", "/tools")
    assert seen["url"] == "https://ag.example/mcp/endpoint/e1/version/2"
    assert "do_actions url: https://ag.example/mcp/endpoint/e1/version/2" in report
    assert "tools: 3" in report
    assert "  - web_search" in report


def test_tools_reports_a_listing_failure_without_failing_the_turn(monkeypatch):
    from memory_agent import agent

    async def boom(url):
        raise ConnectionError("gateway unreachable")

    monkeypatch.setenv("HARNESS_MCP_URL_DO_ACTIONS", "https://ag.example/mcp")
    monkeypatch.setattr(agent, "_list_gateway_tools", boom)
    report = say("t9", "/tools")
    assert "could not list tools: ConnectionError: gateway unreachable" in report


APPEND_SCHEMA = {"type": "object", "properties": {"block_id": {"type": "string"}, "children": {"type": "array"}}}
SEARCH_RESULT = {
    "results": [
        {
            "index": 1,
            "use_case": "append",
            "results": [
                {"name": "notion_retrieve_block_children", "version": "1", "inputSchema": {"properties": {"block_id": {}}}},
                {"name": "notion_append_block_children", "version": "2", "inputSchema": APPEND_SCHEMA},
            ],
        }
    ]
}
PAGE_URL = "https://www.notion.so/team/Test-page-0123456789abcdef0123456789abcdef"


def fake_gateway(monkeypatch, invoke_result):
    from memory_agent import agent

    calls = []

    async def call(url, name, arguments):
        calls.append((name, arguments))
        return SEARCH_RESULT if name == "action_search" else {"results": [{"index": 0, "result": invoke_result}]}

    monkeypatch.setenv("HARNESS_MCP_URL_DO_ACTIONS", "https://ag.example/mcp")
    monkeypatch.setattr(agent, "_gateway_call", call)
    return calls


def test_notion_without_do_actions_says_so(monkeypatch):
    monkeypatch.delenv("HARNESS_MCP_URL_DO_ACTIONS", raising=False)
    assert "do.actions is not attached" in say("n1", "/notion hello")


def test_notion_alone_lists_the_tools_and_writes_nothing(monkeypatch):
    monkeypatch.setenv("NOTION_PAGE_ID", PAGE_URL)
    calls = fake_gateway(monkeypatch, {})
    report = say("n2", "/notion")
    assert [name for name, _ in calls] == ["action_search"]
    assert "notion tools found: 2" in report
    assert "  - notion_append_block_children@2 (block_id, children)" in report
    assert "append tool: notion_append_block_children" in report


def test_notion_appends_a_paragraph_to_the_page(monkeypatch):
    monkeypatch.setenv("NOTION_PAGE_ID", PAGE_URL)
    monkeypatch.delenv("NOTION_APPEND_TOOL", raising=False)
    calls = fake_gateway(monkeypatch, {"status": "succeeded", "output": {"object": "list", "results": [{"id": "b1"}]}})
    report = say("n3", "/notion hello from langgraph")
    name, invoke = calls[-1]
    assert name == "action_invoke"
    call = invoke["tools"][0]
    assert (call["tool"], call["version"]) == ("notion_append_block_children", "2")
    assert call["arguments"]["block_id"] == "0123456789abcdef0123456789abcdef"
    rich_text = call["arguments"]["children"][0]["paragraph"]["rich_text"]
    assert rich_text[0]["text"]["content"] == "hello from langgraph"
    assert "status: succeeded" in report
    assert "new block: b1" in report


def test_notion_reports_a_tool_error(monkeypatch):
    monkeypatch.setenv("NOTION_PAGE_ID", "0123456789abcdef0123456789abcdef")
    fake_gateway(
        monkeypatch,
        {
            "status": "failed",
            "error": {"class": "unauthorized", "message": "connect Notion first", "recovery_hint": "refresh_auth"},
            "_meta": {"status": "requires_authorization"},
        },
    )
    report = say("n4", "/notion hello")
    assert "status: failed" in report
    assert "gateway: requires_authorization" in report
    assert "error: unauthorized: connect Notion first (refresh_auth)" in report


def test_notion_without_a_page_writes_nothing(monkeypatch):
    monkeypatch.delenv("NOTION_PAGE_ID", raising=False)
    calls = fake_gateway(monkeypatch, {})
    report = say("n5", "/notion hello")
    assert "nothing written: set NOTION_PAGE_ID" in report
    assert "action_invoke" not in [name for name, _ in calls]


def test_notion_arguments_follow_a_nested_schema():
    from memory_agent.agent import _append_arguments

    schema = {
        "properties": {
            "path": {"type": "object", "properties": {"block_id": {}}},
            "body": {"type": "object", "properties": {"children": {}}},
        }
    }
    args = _append_arguments(schema, "p1", "hi")
    assert args["path"] == {"block_id": "p1"}
    assert args["body"]["children"][0]["type"] == "paragraph"


def test_every_reply_ends_with_the_model_line():
    assert say("m1", "hello").endswith("\n\ncurrently I'm using model: none (HARNESS_INFERENCE_MODEL is unset)")
    assert say("m1", "/history").endswith("currently I'm using model: none (HARNESS_INFERENCE_MODEL is unset)")


def test_the_model_line_says_when_inference_is_half_configured(monkeypatch):
    monkeypatch.setenv("HARNESS_INFERENCE_MODEL", "kimi-k3")
    report = say("m2", "/env NO_SUCH_PREFIX_")
    assert report.endswith("currently I'm using model: none (kimi-k3 is set, but inference is not fully configured)")


class FakeLLM:
    model_name = "kimi-k3"

    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return AIMessage("the answer", response_metadata={"model_name": "kimi-k3-0711"})


def test_a_model_reply_is_sent_once_with_the_model_line(monkeypatch):
    from memory_agent import agent

    fake = FakeLLM()
    monkeypatch.setattr(agent, "llm", fake)
    first = say("m3", "what is 2+2?")
    assert first == "the answer\n\ncurrently I'm using model: kimi-k3 (reply from: kimi-k3-0711)"
    say("m3", "and 3+3?")
    # The model line stays out of the model's prompt.
    assert not any("currently I'm using model" in str(m.content) for m in fake.prompts[-1])
    assert any(m.content == "the answer" for m in fake.prompts[-1])
