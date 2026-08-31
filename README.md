# aisha

Локальный консольный AI-агент для работы с исходным кодом, файлами, командной строкой и интернетом.

Использует OpenAI-совместимый API внешнего `llama-server` и не требует облачных LLM или API-ключей.

## Установка

```bash
pip install -e ".[dev]"
```

Или изолированно через pipx:

```bash
pipx install .
```

## Требования

- Python 3.11+
- Запущенный `llama-server` из llama.cpp с поддержкой OpenAI-compatible API

## Запуск сервера

```powershell
llama-server.exe `
  -m Qwen3.5-9B-UD-Q4_K_XL.gguf `
  --no-mmproj `
  --jinja `
  --chat-template-file "chat_template.jinja" `
  --tools all `
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

## Запуск агента

```bash
# Интерактивный REPL
aisha

# Одноразовый запрос
aisha "Объясни структуру этого проекта"

# Переопределение модели
aisha --model Qwen3.5-9B-Q4_K_XL

# Другой llama-server
aisha --server http://localhost:8088

# Режим только для чтения
aisha -r

# Запрет shell-команд
aisha --permission deny

# Автоматическое выполнение разрешённых команд
aisha --permission auto

# Использование cmd вместо PowerShell
aisha --shell cmd

# Список доступных инструментов
aisha --tools-only

# Диагностика
aisha --doctor
```

## Конфигурация

### Глобальная

`%USERPROFILE%\.aisha\config.toml`

```toml
[server]
base_url = "http://localhost:8088"
model = "Qwen3.5-9B-Q4_K_XL"
connect_timeout = 5
request_timeout = 600

[llm]
temperature = 0.2
max_output_tokens = 8192
context_window = 65536
context_soft_limit = 0.85
max_tool_iterations = 25

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

[ui]
theme = "dark"
stream = true
show_reasoning = false
```

### Проектная

`<workspace>/aisha.toml` — переопределяет глобальные настройки. Не может ослаблять безопасность (permission=auto, write outside workspace и т.д.).

### Приоритет

1. CLI-аргументы
2. Переменные окружения (`AISHA_SERVER_URL`, `AISHA_MODEL`, `AISHA_PERMISSION`, `AISHA_SHELL`, `AISHA_CONTEXT_WINDOW`, `AISHA_MAX_OUTPUT_TOKENS`)
3. Проектный `aisha.toml`
4. Глобальный `~/.aisha/config.toml`
5. Значения по умолчанию

## Команды REPL

| Команда | Назначение |
| --- | --- |
| `/help` | показать справку |
| `/new` | новая сессия, очистить историю |
| `/status` | сервер, модель, workspace, токены |
| `/tools` | доступные инструменты |
| `/skills` | индекс скиллов |
| `/memory` | блоки памяти |
| `/compact` | сжать историю |
| `/doctor` | проверить соединение |
| `/clear` | очистить экран |
| `/quit`, `/exit` | выход |

## Инструменты

| Инструмент | Назначение |
| --- | --- |
| `read_file` | чтение файла |
| `write_file` | создание/перезапись файла |
| `edit_file` | точная замена текста |
| `list_dir` | просмотр директории |
| `glob` | поиск файлов по маске |
| `grep` | regex-поиск по содержимому |
| `run_command` | выполнение PowerShell/cmd |
| `web_search` | поиск в интернете (DuckDuckGo) |
| `web_fetch` | загрузка веб-страницы |
| `todowrite` | список задач сессии |
| `ask_user` | вопрос пользователю |
| `memory_list` | список блоков памяти |
| `memory_get` | чтение блока памяти |
| `memory_set` | создание/запись блока |
| `memory_replace` | замена текста в блоке |
| `skill` | загрузка скилла |

## Скиллы

Размещаются в:
- `%USERPROFILE%\.aisha\skills\<name>\SKILL.md` (глобальные)
- `<workspace>\.aisha\skills\<name>\SKILL.md` (проектные)

Формат:

```markdown
---
name: my-skill
description: Описание скилла
---

Инструкции здесь.
```

## Память

- Глобальная: `%USERPROFILE%\.aisha\memory\`
- Проектная: `<workspace>\.aisha\memory\`

Каждый блок — JSON-файл. Проектная память имеет приоритет над глобальной с одинаковым label.

## Тестирование

```bash
pytest tests/ -v
```

## Проверка кода

```bash
ruff check src/ tests/
```
