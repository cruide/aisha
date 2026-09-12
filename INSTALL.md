# Installing the aisha Agent

> [Russian version](INSTALL_RU.md)

`aisha` — a local console AI agent in Python 3.11+. Works with an external `llama-server` (llama.cpp) via an OpenAI-compatible API and does not require cloud LLMs or API keys.

## 1. Requirements

- **OS:** Windows 11 (primary), also works on other OSes supporting Python 3.11+
- **Python 3.11+** (including `pip` and `venv`)
- **A running `llama-server`** from llama.cpp with OpenAI-compatible API support
- **A GGUF model** and, if needed, a chat-template (jinja)

## 2. Installing Python (if not already installed)

Check the version:

```powershell
python --version
```

You need `Python 3.11` or newer. If Python is missing — install it from https://www.python.org/downloads/ (be sure to check "Add python.exe to PATH").

## 3. Installing aisha

### Option 1: from PyPI (recommended)

```powershell
python -m pip install aisha
```

The simplest way: the package is installed from https://pypi.org/project/aisha/ with all dependencies, the `aisha` command is available globally.

### Option 2: from source (repository)

Clone or copy the repository to your working directory, then from the project root:

```powershell
python -m pip install .             # regular installation
python -m pip install -e ".[dev]"  # editable development installation
```

`-e` (editable) installs the agent in development mode: changes in `src/` are picked up immediately, and the `aisha` command is available globally. Dev dependencies (pytest, ruff) are installed as well.

### Option 3: isolated via pipx

```bash
pipx install aisha
```

Isolates the agent and its dependencies from the rest of the environment. The `aisha` command is also available globally.

> Note: project dependencies are `httpx`, `rich`, `prompt-toolkit`, `ddgs`, `beautifulsoup4`, `PyYAML`. They are installed automatically.

### Verifying installation

```bash
aisha --help
```

The command help should be displayed.

## 4. Installing and configuring llama-server

`aisha` does not load the model itself — the external `llama-server` does it.

1. Download the llama.cpp build (CUDA or CPU) from https://github.com/ggml-org/llama.cpp/releases.
2. Download a GGUF model, e.g., Qwen:
   `d:\models\llama\qwen\Qwen3.5-9B-UD-Q4_K_XL.gguf`
3. If needed, prepare a jinja chat-template:
   `d:\models\llama\qwen\chat_template.jinja`

### Starting the server

```powershell
z:\llamacpp\cuda\llama-server.exe `
  -m d:\models\llama\qwen\Qwen3.5-9B-UD-Q4_K_XL.gguf `
  --no-mmproj `
  --jinja `
  --chat-template-file "d:\models\llama\qwen\chat_template.jinja" `
  -c 65536 `
  -fa on `
  --fit off `
  --load-mode mlock `
  --host localhost `
  --port 8088 `
  -a Qwen3.5-9B-Q4_K_XL `
  -ngl 99 `
  -np 1 `
  -t 8 `
  -ctk q8_0 `
  -ctv q8_0 `
  -ub 1024 `
  -lv 4
```

Key parameters for working with `aisha`:

| Parameter | Value | Purpose |
| --- | --- | --- |
| `--host localhost` | local access | safer than `0.0.0.0` |
| `--port 8088` | default agent port | matches the default in config |
| `-a Qwen3.5-9B-Q4_K_XL` | model alias | just a hint for the agent |
| `-c 65536` | context | server-side context advertised as `meta.n_ctx` |

The model and chat template must support OpenAI-style function calling. Aisha sends its own
tool schemas and executes calls itself, so llama-server's built-in `--tools all` is not
required. Enabling it would create a second execution path outside Aisha's workspace,
read-only, and confirmation checks.

> Security: do not expose `llama-server` to the internet. Without a reverse proxy, authentication, and access restrictions, it should only listen on `localhost`.

### Checking the server

```powershell
Invoke-RestMethod http://localhost:8088/health
```

Expected response:

```text
{"status":"ok"}
```

## 5. Configuring aisha

The agent works with default settings right after installation, but for convenience you can create a global config.

### Global configuration

File: `%USERPROFILE%\.aisha\config.toml`

```toml
[server]
base_url = "http://localhost:8088"
model = "Qwen3.5-9B-Q4_K_XL"
connect_timeout = 5
request_timeout = 600

[llm]
temperature = 0.6
max_output_tokens = 32768  # fallback when the server does not publish meta.n_ctx
context_window = 32768     # fallback when the server does not publish meta.n_ctx
context_soft_limit = 0.75
max_tool_iterations = 25
# enable_thinking = true        # true/false; omit the key for the server default
compact_tool_schemas = false    # true — strip "Example:" blocks from tool descriptions

[tools]
shell = true
web_search = true
permission = "ask"
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
index_max_chars = 20000

[skills]
index_max_chars = 20000

[context]
agents_md_max_chars = 65536

[compaction]
summary_max_tokens = 4096
trim_user_messages = false
elide_large_tool_args = false
elide_threshold_chars = 4000
trim_head_chars = 2000
trim_tail_chars = 1000
trim_block_fraction = 0.25

[ui]
theme = "dark"
stream = true
show_reasoning = false
debug = false
```

> The model name in config is just a hint. If the declared name doesn't match `-a` on the server, `aisha` automatically connects to the first available model without crashing.

At normal startup Aisha reads a positive `meta.n_ctx` for the selected model from
`/v1/models` and assigns it to both `llm.context_window` and `llm.max_output_tokens`.
Configured values are fallbacks only. The actual per-request `max_tokens` is further limited
by the remaining context after prompts, history, tool schemas, and a safety margin.

### Project configuration

File `<workspace>/.aisha/aisha.toml` overrides global settings for a specific project. Special
checks prevent it from setting `permission = "auto"`, enabling outside-workspace file access,
or re-enabling shell when it was disabled globally. Other settings, including server and web
settings, remain project-overridable. Do not store API keys in project config; prefer
`AISHA_API_KEY` or the global config outside the workspace.

### Settings priority

1. Command-line arguments
2. Environment variables (`AISHA_SERVER_URL`, `AISHA_MODEL`, `AISHA_API_KEY`, `AISHA_SKIP_HEALTH`, `AISHA_PERMISSION`, `AISHA_SHELL`, `AISHA_CONTEXT_WINDOW`, `AISHA_MAX_OUTPUT_TOKENS`, `AISHA_COMMUNICATION_LANGUAGE`)
3. Project `aisha.toml`
4. Global `~/.aisha/config.toml`
5. Default values

Full list of keys and sections — see the config example in `README.md`.

## 6. Diagnostics

Checking server connection and model availability:

```bash
aisha --doctor
```

Extended tool calling check (sends a safe test request without shell commands or file writes):

```bash
aisha --doctor --tool-call-test
```

## 7. First launch

From any working directory:

```bash
aisha
```

An interactive REPL will open. Useful commands: `/status` — server, model, workspace, tokens; `/help` — help.

One-time request:

```bash
aisha "Explain the structure of this project"
```

Other useful flags:

```bash
aisha --server http://localhost:8088   # different llama-server
aisha --model Qwen3.5-9B-Q4_K_XL       # override model
aisha -r                               # read-only mode
aisha --permission deny                # deny shell commands
aisha --permission auto                # skip prompts only for commands not flagged as dangerous
aisha --shell cmd                      # cmd instead of PowerShell
aisha --tools-only                     # list available tools
```

## 8. Updating

From PyPI:

```powershell
python -m pip install --upgrade aisha
```

From repository (editable):

```powershell
git pull --ff-only
python -m pip install --upgrade -e ".[dev]"
```

For pipx:

```bash
pipx upgrade aisha
```

## 9. Troubleshooting

| Symptom | Solution |
| --- | --- |
| `ServerUnavailableError` at REPL start | Check `/health` for llama-server and `/v1/models`. For servers without compatible `/health`, use `--skip-health`. |
| Model is still loading | Wait until `llama-server` loads the GGUF, then repeat `aisha --doctor`. |
| Configuration error from project `aisha.toml` | Move restricted shell/outside-workspace settings to global config or an explicit CLI flag. |
| `aisha` command not found | Reinstall the package (`pip install -e .`) or check that the Python Scripts directory is in PATH. |
| `permission = "ask"` is annoying | `--permission auto` skips prompts for commands not recognized as dangerous; the regex check is not a sandbox. |

## 10. Testing and code review (development)

```bash
python -m pytest tests/ -q   # tests (do not require llama-server)
ruff check src/ tests/       # lint
```
