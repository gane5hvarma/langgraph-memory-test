"""Local check: two turns on one thread share memory; a new thread does not.

    pip install -r requirements.txt pytest && pytest -q
"""

import os

for k in ("HARNESS_INFERENCE_MODEL", "HARNESS_INFERENCE_BASE_URL", "HARNESS_INFERENCE_API_KEY"):
    os.environ.pop(k, None)

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from memory_agent.agent import workflow  # noqa: E402

graph = workflow.compile(checkpointer=MemorySaver())


def say(thread: str, text: str) -> str:
    out = graph.invoke(
        {"messages": [("user", text)]},
        {"configurable": {"thread_id": thread}},
    )
    return out["messages"][-1].content


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
