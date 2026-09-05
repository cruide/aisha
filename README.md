# AISHA

> 🇷🇺 [Русская версия](README_RU.md)

A local console AI agent in Python 3.11+. Works with an external
[llama-server](https://github.com/ggml-org/llama.cpp) (llama.cpp) via an
OpenAI-compatible REST API. This is **not a web app**: the entire logic is a loop
of "model request → tool calls → results → model again" in a single process.

Version: `0.2.5`.

## Features

- **Files** — read, write, inline editing, listing, `glob` search, and `grep`;
- **Shell** — run commands (`powershell`/`cmd`/`sh`), timeouts, output trimming, confirmation for dangerous commands;
- **Web** — DuckDuckGo search and page fetching with SSRF protection;
- **Memory** — persistent blocks (global and project-scoped) with project priority;
- **Skills** — reusable instructions in `SKILL.md`;
- **Multi-step tasks** — todo list and clarifying questions to the user;
- **Native tool calling** — the model invokes tools directly via API;
- **Automatic history compaction** when approaching the context limit;
- **Two modes** — one-shot (prompt passed as argument) and interactive REPL.

## Requirements

- Python **3.11+**;
- a running **llama-server** (llama.cpp) with OpenAI-compatible API and tool calling support.

### Option 1: from PyPI (recommended)

```bash
pip install aisha
```

The simplest way: the package is installed from https://pypi.org/project/aisha/ with all dependencies, the `aisha` command is available globally.

### Option 2: from source (repository)

```bash
pip install -e ".[dev]"   # src-layout: without this the `aisha` package won't import
```

Entry point — `aisha = "aisha.cli:main"` (see `pyproject.toml`).

## Running llama-server

Before using aisha you need to start the server, e.g.:

```bash
llama-server \
  --model ./models/Qwen3.5-9B-Q4_K_XL.gguf \
  --port 8088 \
  --n-gpu-layers 99
```

By default aisha expects the server at `http://localhost:8088`. Check connectivity:

```bash
aisha --doctor
```

## Quick Start

```bash
aisha "explain what this project does"   # one-shot: single request and exit
aisha                                    # no arguments — interactive REPL
python -m aisha ...                      # equivalent entry point
```

## CLI

```
aisha [prompt...] [flags]
```

| Flag | Description |
|---|---|
| `prompt` (positional) | query; without it REPL starts |
| `--server URL` | llama-server address (default `http://localhost:8088`) |
| `--model NAME` | model name on the server |
| `--api-key KEY` | API key for the server (if auth is required) |
| `--skip-health` | skip `/health` check (for incompatible servers) |
| `-r`, `--read-only` | read-only mode (safe tools only) |
| `--permission auto\|ask\|deny` | shell command execution mode |
| `--shell powershell\|cmd` | default shell |
| `--tools-only` | list tools and exit |
| `--doctor` | diagnose server connection |
| `--tool-call-test` | together with `--doctor`: test tool calling |
| `--no-color` | disable colors |
| `--debug` | debug mode: model reasoning, request/response dumps, tracebacks |
| `--version` | show version |

## Configuration

Priority (each layer deep-merges with the previous):

```
DEFAULTS  ←  ~/.aisha/config.toml  ←  <workspace>/aisha.toml  ←  env AISHA_*  ←  CLI flags
```

Full example of `~/.aisha/config.toml`:

```toml
[server]
base_url = "http://localhost:8088"
model = "Qwen3.5-9B-Q4_K_XL"
api_key = ""                  # optional; if server requires auth (Bearer)
skip_health = false           # true — skip /health check (for OpenRouter, vLLM, etc.)
connect_timeout = 5.0
request_timeout = 600.0

[llm]
temperature = 0.6
top_p = 0.9                 # optional; None = don't pass (server default used)
top_k = 40                  # optional; integer > 0
repeat_penalty = 1.1        # optional; > 0
frequency_penalty = 0.0     # optional; -2.0 .. 2.0
max_output_tokens = 32768
context_window = 32768
context_soft_limit = 0.85
max_tool_iterations = 25
tool_guide = false           # true — add "Tool Guide" to system prompt (for weak models)
communication_language = "Russian"  # agent's response language

[tools]
shell = true
web_search = true
permission = "ask"          # auto | ask | deny
shell_type = "powershell"
shell_timeout = 120
max_output_chars = 65536
allow_read_outside_workspace = false
allow_write_outside_workspace = false

[web]
timeout = 20
max_results = 8
max_page_bytes = 2097152
max_content_chars = 50000
allow_private_hosts = false

[memory]
enabled = true
max_block_chars = 30000

[ui]
theme = "dark"
stream = true
show_reasoning = false
debug = false                 # true — same as --debug: reasoning + request/response dumps
input_history = "~/.aisha/input_history.txt"
```

Environment variables:

| Variable | Maps to |
|---|---|
| `AISHA_SERVER_URL` | `server.base_url` |
| `AISHA_MODEL` | `server.model` |
| `AISHA_API_KEY` | `server.api_key` |
| `AISHA_SKIP_HEALTH` | `server.skip_health` (`true`/`false`) |
| `AISHA_PERMISSION` | `tools.permission` |
| `AISHA_SHELL` | `tools.shell_type` |
| `AISHA_CONTEXT_WINDOW` | `llm.context_window` |
| `AISHA_MAX_OUTPUT_TOKENS` | `llm.max_output_tokens` |

Config is strictly validated: unknown section or key raises an error.
**Project-level `aisha.toml` is restricted by security rules** — it cannot set
`permission = "auto"`, enable access outside the workspace, or enable `shell`
if it was disabled globally. This protects against a "trojan" config in a cloned repo.

## Tools

| Tool | Read-only | Purpose |
|---|---|---|
| `read_file` | yes | read UTF-8 file with offset/limit |
| `write_file` | no | create/overwrite (atomic) |
| `edit_file` | no | precise text fragment replacement |
| `list_dir` | yes | directory listing |
| `glob` | yes | file search by pattern |
| `grep` | yes | regex content search |
| `run_command` | no | run shell command |
| `web_search` | yes | DuckDuckGo search |
| `web_fetch` | yes | fetch web page |
| `todowrite` | yes | full replacement of session todo list |
| `ask_user` | yes | clarifying question (REPL only) |
| `memory_list` / `memory_get` | yes | list/read memory blocks |
| `memory_set` / `memory_replace` | no | write/edit memory blocks |
| `skill` | yes | load skill text by name |

File tools do not escape the workspace (path traversal is blocked)
unless the corresponding `allow_*_outside_workspace` is enabled.

## Memory and Skills

- **Memory** — JSON blocks in `~/.aisha/memory/` (global) and `<workspace>/.aisha/memory/`
  (project-scoped). Project block overrides global with the same `label`.
  `memory_get` call is not displayed in console (it's a background read of the agent's own memory).
- **Skills** — directories `~/.aisha/skills/<name>/SKILL.md` and
  `<workspace>/.aisha/skills/<name>/SKILL.md` with required YAML frontmatter
  (`name`, `description`).

## Custom System Prompt (SYSTEM.md)

If a file `<workspace>/.aisha/SYSTEM.md` exists in the project root, its content
**completely replaces** the built-in aisha system prompt (persona, environment, rules,
memory and skill sections). The "Tool Guide" (`tool_guide = true`), `AGENTS.md`,
and the current todo list are still appended after it. The file is truncated
to 64 KB, same as `AGENTS.md`.

## REPL

Commands inside interactive mode:

| Command | Action |
|---|---|
| `/help` | help |
| `/new` | new session (reset history) |
| `/status` | server, model, workspace, mode, tokens |
| `/tools` | tool list |
| `/skills` | skill index |
| `/memory` | memory blocks |
| `/compact` | force history compaction |
| `/doctor` | connection check |
| `/init` | explore project and create `AGENTS.md` |
| `/clear` | clear screen |
| `/quit`, `/exit`, `Ctrl+D` | exit |

Additionally: `Ctrl+C` cancels the current request (REPL does not exit),
`Ctrl+↑/↓` — request history, `Tab` — command and path autocompletion.

## Security

- Shell commands in `permission = "ask"` mode require confirmation;
  dangerous commands (`rm -rf`, `Remove-Item -Recurse`, `git reset --hard`, etc.)
  always require confirmation.
- `web_fetch` blocks private/localhost/loopback addresses (SSRF protection) unless
  `web.allow_private_hosts` is enabled.
- `find_danger` in `shell.py` is a **heuristic regex check, not a sandbox**: it can be bypassed.
  Do not run aisha as a privileged user in an untrusted environment.

## Development

```bash
pip install -e ".[dev]"

pytest                        # full suite; real server not needed
pytest tests/test_config.py   # single test

ruff check .                  # lint (E, F, I, W; line-length 100)
```

Tests do not hit a real server: `test_client.py` uses `httpx.MockTransport`,
config tests use `monkeypatch.setattr(Path, "home", ...)`.

## Project Structure

```
src/aisha/
├── cli.py        # entry point: args → config → registry → client → AgentLoop → UI
├── client.py     # async SSE client to llama-server, retries, tool-call assembly
├── agent.py      # AgentLoop: model↔tools loop, compaction
├── context.py    # system prompt, history, token estimation
├── config.py     # configuration, validation, security
├── memory.py     # persistent memory (blocks)
├── skills.py     # skills (SKILL.md)
├── ui.py         # ConsoleUI: rich + prompt_toolkit, REPL
├── fsutil.py     # atomic write, path checks, human_size
├── errors.py     # exception hierarchy
└── tools/        # tool implementations (base, files, shell, web, extras)
```
