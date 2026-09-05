# aisha

Локальный консольный AI-агент на Python 3.11+. Работает с внешним
[llama-server](https://github.com/ggml-org/llama.cpp) (llama.cpp) по
OpenAI-совместимому REST API. Это **не веб-приложение**: вся логика — цикл
«запрос модели → вызовы инструментов → результаты → снова модель» в одном процессе.

Версия: `0.2.4`.

## Возможности

- **Файлы** — чтение, запись, точечное редактирование, листинг, `glob`-поиск и `grep`;
- **Shell** — запуск команд (`powershell`/`cmd`/`sh`), таймауты, обрезка вывода, подтверждение опасных команд;
- **Веб** — поиск DuckDuckGo и загрузка страниц с SSRF-защитой;
- **Память** — постоянные блоки (глобальные и проектные) с приоритетом проекта;
- **Скиллы** — переиспользуемые инструкции в `SKILL.md`;
- **Многошаговые задачи** — todo-список и уточняющие вопросы к пользователю;
- **Нативный tool calling** — модель сама вызывает инструменты через API;
- **Авто-сжатие истории** при приближении к лимиту контекста;
- **Два режима** — one-shot (передан промпт) и интерактивный REPL.

## Требования

- Python **3.11+**;
- запущенный **llama-server** (llama.cpp) с поддержкой OpenAI-совместимого API и tool calling.

## Установка

### Вариант 1: из PyPI (рекомендуется)

```bash
pip install aisha
```

Самый простой способ: пакет устанавливается с https://pypi.org/project/aisha/ вместе со всеми зависимостями, команда `aisha` доступна глобально.

### Вариант 2: из исходников (репозиторий)

```bash
pip install -e ".[dev]"   # src-layout: без этого пакет `aisha` не импортируется
```

Точка входа — `aisha = "aisha.cli:main"` (см. `pyproject.toml`).

## Запуск llama-server

Перед использованием нужно поднять сервер, например:

```bash
llama-server \
  --model ./models/Qwen3.5-9B-Q4_K_XL.gguf \
  --port 8088 \
  --n-gpu-layers 99
```

По умолчанию aisha ждёт сервер на `http://localhost:8088`. Проверить подключение:

```bash
aisha --doctor
```

## Быстрый старт

```bash
aisha "объясни, что делает этот проект"   # one-shot: один запрос и выход
aisha                                       # без аргументов — интерактивный REPL
python -m aisha ...                         # эквивалентный вызов
```

## CLI

```
aisha [промпт...] [флаги]
```

| Флаг | Описание |
|---|---|
| `prompt` (позиционный) | запрос; без него запускается REPL |
| `--server URL` | адрес llama-server (по умолч. `http://localhost:8088`) |
| `--model NAME` | имя модели на сервере |
| `--api-key KEY` | API-ключ для сервера (если требуется авторизация) |
| `--skip-health` | пропустить проверку `/health` (для несовместимых серверов) |
| `-r`, `--read-only` | режим только для чтения (только безопасные инструменты) |
| `--permission auto\|ask\|deny` | режим запуска shell-команд |
| `--shell powershell\|cmd` | оболочка по умолчанию |
| `--tools-only` | показать список инструментов и выйти |
| `--doctor` | диагностика подключения к серверу |
| `--tool-call-test` | вместе с `--doctor`: проверить tool calling |
| `--no-color` | отключить цвета |
| `--debug` | режим отладки: reasoning модели, дампы запросов/ответов, traceback |
| `--version` | показать версию |

## Конфигурация

Приоритет (каждый слой глубоко сливается с предыдущим):

```
DEFAULTS  ←  ~/.aisha/config.toml  ←  <workspace>/aisha.toml  ←  env AISHA_*  ←  CLI-флаги
```

Полный пример `~/.aisha/config.toml`:

```toml
[server]
base_url = "http://localhost:8088"
model = "Qwen3.5-9B-Q4_K_XL"
api_key = ""                  # необязательно; если сервер требует авторизацию (Bearer)
skip_health = false           # true — пропустить проверку /health (для OpenRouter, vLLM и т.д.)
connect_timeout = 5.0
request_timeout = 600.0

[llm]
temperature = 0.6
top_p = 0.9                 # необязательно; None = не передавать (используется серверный умолчание)
top_k = 40                  # необязательно; целое > 0
repeat_penalty = 1.1        # необязательно; > 0
frequency_penalty = 0.0     # необязательно; -2.0 .. 2.0
max_output_tokens = 32768
context_window = 32768
context_soft_limit = 0.85
max_tool_iterations = 25
tool_guide = false           # true — добавить «Справочник инструментов» в системный промпт (для слабых моделей)

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
debug = false                 # true — то же, что --debug: reasoning + дампы запросов/ответов
input_history = "~/.aisha/input_history.txt"
```

Переменные окружения:

| Переменная | Куда |
|---|---|
| `AISHA_SERVER_URL` | `server.base_url` |
| `AISHA_MODEL` | `server.model` |
| `AISHA_API_KEY` | `server.api_key` |
| `AISHA_SKIP_HEALTH` | `server.skip_health` (`true`/`false`) |
| `AISHA_PERMISSION` | `tools.permission` |
| `AISHA_SHELL` | `tools.shell_type` |
| `AISHA_CONTEXT_WINDOW` | `llm.context_window` |
| `AISHA_MAX_OUTPUT_TOKENS` | `llm.max_output_tokens` |

Конфиг строго валидируется: неизвестная секция или ключ вызывает ошибку.
**Проектный `aisha.toml` ограничен в правах безопасности** — он не может ставить
`permission = "auto"`, включать доступ за пределы workspace или включать `shell`,
если тот отключён глобально. Это защита от «троянского» конфига в склонированном репозитории.

## Инструменты

| Инструмент | Read-only | Назначение |
|---|---|---|
| `read_file` | да | чтение файла UTF-8 с offset/limit |
| `write_file` | нет | создание/перезапись (атомарно) |
| `edit_file` | нет | точная замена фрагмента текста |
| `list_dir` | да | содержимое каталога |
| `glob` | да | поиск файлов по маске |
| `grep` | да | regex-поиск по содержимому |
| `run_command` | нет | запуск shell-команды |
| `web_search` | да | поиск DuckDuckGo |
| `web_fetch` | да | загрузка веб-страницы |
| `todowrite` | да | полная замена todo-списка сессии |
| `ask_user` | да | уточняющий вопрос (только в REPL) |
| `memory_list` / `memory_get` | да | список/чтение блоков памяти |
| `memory_set` / `memory_replace` | нет | запись/правка блоков памяти |
| `skill` | да | загрузить текст скилла по имени |

Файловые инструменты не выходят за пределы workspace (path-traversal блокируется),
если не включён соответствующий `allow_*_outside_workspace`.

## Память и скиллы

- **Память** — JSON-блоки в `~/.aisha/memory/` (глобально) и `<workspace>/.aisha/memory/`
  (проектно). Проектный блок перекрывает глобальный с тем же `label`.
  Вызов `memory_get` не выводится в консоль (это фоновое чтение собственной памяти агента).
- **Скиллы** — каталоги `~/.aisha/skills/<name>/SKILL.md` и
  `<workspace>/.aisha/skills/<name>/SKILL.md` с обязательным YAML-frontmatter
  (`name`, `description`).

## Кастомный системный промпт (SYSTEM.md)

Если в корне проекта есть файл `<workspace>/.aisha/SYSTEM.md`, его содержимое
**полностью заменяет** встроенный системный промпт aisha (persona, окружение, правила,
секции памяти и скиллов). При этом «Справочник инструментов» (`tool_guide = true`),
`AGENTS.md` и текущий todo-список по-прежнему добавляются после него. Файл обрезается
до 64 КБ, как и `AGENTS.md`.

## REPL

Команды внутри интерактивного режима:

| Команда | Действие |
|---|---|
| `/help` | справка |
| `/new` | новая сессия (сброс истории) |
| `/status` | сервер, модель, workspace, режим, токены |
| `/tools` | список инструментов |
| `/skills` | индекс скиллов |
| `/memory` | блоки памяти |
| `/compact` | принудительное сжатие истории |
| `/doctor` | проверка соединения |
| `/init` | изучить проект и создать `AGENTS.md` |
| `/clear` | очистка экрана |
| `/quit`, `/exit`, `Ctrl+D` | выход |

Дополнительно: `Ctrl+C` отменяет текущий запрос (REPL не завершается),
`Ctrl+↑/↓` — история запросов, `Tab` — автодополнение команд и путей.

## Безопасность

- Запуск shell-команд в режиме `permission = "ask"` требует подтверждения;
  опасные команды (`rm -rf`, `Remove-Item -Recurse`, `git reset --hard` и т.п.)
  подтверждаются всегда.
- `web_fetch` блокирует private/localhost/loopback-адреса (SSRF-защита), если не
  включён `web.allow_private_hosts`.
- `find_danger` в `shell.py` — **эвристика по regex, а не песочница**: обойти её можно.
  Не запускайте aisha от имени пользователя с повышенными правами в недоверенном окружении.

## Разработка

```bash
pip install -e ".[dev]"

pytest                        # весь набор; реальный сервер не нужен
pytest tests/test_config.py   # один тест

ruff check .                  # линт (E, F, I, W; line-length 100)
```

Тесты не ходят в реальный сервер: `test_client.py` подменяет транспорт
`httpx.MockTransport`, конфиг-тесты подменяют `Path.home`.

## Структура проекта

```
src/aisha/
├── cli.py        # entry point: args → config → registry → client → AgentLoop → UI
├── client.py     # async SSE-клиент к llama-server, ретраи, tool-call сборка
├── agent.py      # AgentLoop: цикл модель↔инструменты, компакция
├── context.py    # системный промпт, история, оценка токенов
├── config.py     # конфигурация, валидация, безопасность
├── memory.py     # постоянная память (блоки)
├── skills.py     # скиллы (SKILL.md)
├── ui.py         # ConsoleUI: rich + prompt_toolkit, REPL
├── fsutil.py     # атомарная запись, проверка путей, human_size
├── errors.py     # иерархия исключений
└── tools/        # реализации инструментов (base, files, shell, web, extras)
```
