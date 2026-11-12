# langgraph-memory-test

A minimal LangGraph chat agent for checking that a DigitalOcean Managed Agents
session keeps its conversation memory when the harness process (OHR)
restarts. That happens on fork, rollback and session restart.

The memory is the LangGraph thread. The Agent Server checkpoints `messages`
and a `turns` counter after every run, so every turn on the same thread sees
everything before it. If the session is moved onto a new thread, the history
is gone and `turns` starts again at 1.

## What it does

- **`/history`** (no model call): replies with the `thread_id`, the turn
  number, the Agent Server pid and every earlier user message. Use it before
  and after a restart.
- **`/env`** or **`printenv`** (no model call), optionally with a name prefix
  (`/env TEST_`): replies with the Agent Server pid and the environment that
  process started with. Values of names containing `KEY`, `TOKEN`, `SECRET`,
  `PASSWORD`, `PASSWD`, `CREDENTIAL` or `AUTH` are shown as
  `<redacted len=N sha256=XXXXXXXX>`, so a rotation is visible without the
  value. Use test values only: a short value can be guessed from 8 hex digits.
- **Anything else:** if `HARNESS_INFERENCE_*` is set, the message goes to the
  model with the whole history, so "what is my secret word?" works.
  Otherwise the agent replies deterministically, with the turn and thread and
  how many earlier messages it remembers. You can test memory without an API
  key.

## Test: memory across an OHR restart

1. Create a session from `agents.yaml` (set `repo_sha` first; see below).
2. Send `my secret word is pineapple`.
3. Send `/history` and note the `thread_id` and `turn: 2`.
4. Restart OHR in one of these ways:
   - fork or roll back the session (sandbox-agent restarts OHR after the
     restore);
   - from a shell in the sandbox: `s6-svc -r /run/s6-rc/servicedirs/ohr`.
5. Send `/history` again.

**Memory kept:** the same `thread_id`, `turn: 3`, and
`1. my secret word is pineapple` / `2. /history` are listed. With a model
configured, `what is my secret word?` answers `pineapple`.

**Memory lost:** a new `thread_id`, `turn: 1`, and `earlier user messages: 0`.

A restart of OHR alone leaves the Agent Server running, so its pid stays the
same.

## Test: a live env or secret update

A `PATCH /v2/agents/sessions/{id}` with `env` or `secrets` restarts the Agent
Server (the whole workload nest) and then OHR, on the new environment.

1. Create a session and send `my secret word is pineapple`.
2. Send `/env TEST_` (nothing listed yet) and `/history` (note the pid).
3. PATCH the session: `env` with `TEST_FLAG: hello`, `secrets` with
   `TEST_SECRET: s3cret-v1`. Keep every existing key, since a section sent
   replaces the whole section.
4. Wait for the session to leave `UPDATING`; `session.update.state` should be
   `APPLIED_RESTARTED`.
5. Send `/env TEST_` and `/history`.

**Update reached the graph:** `TEST_FLAG=hello` and `TEST_SECRET=<redacted
len=9 …>` are listed, and the Agent Server pid has changed.

**Memory kept across it:** `/history` shows the same `thread_id` and lists
`my secret word is pineapple`.

Rotate the secret with a second PATCH (`TEST_SECRET: s3cret-v2`): the
`sha256=` changes and the length stays 9.

## Deploy

```bash
git rev-parse HEAD          # put this in agents.yaml as repo_sha, commit, push
doctl harness-runtime validate agents.yaml
doctl harness-runtime create agents.yaml
# with a model:
doctl harness-runtime create agents.yaml --secret HARNESS_INFERENCE_API_KEY=@$HOME/.inference_key
```

`repo_sha` can be any pushed commit that has `langgraph.json`, so pinning the
previous commit is fine.

## Run locally

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "langgraph-cli[inmem]" -r requirements.txt pytest
pytest -q
langgraph dev --no-browser
```
