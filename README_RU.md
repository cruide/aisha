# AISHA

> [English version](README.md)

Локальный консольный AI-агент на Python 3.11+. Работает с внешним
[llama-server](https://github.com/ggml-org/llama.cpp) (llama.cpp) по
OpenAI-совместимому REST API. Это **не веб-приложение**: вся логика — цикл
«запрос модели → вызовы инструментов → результаты → снова модель» в одном процессе.

Версия: `0.2.14`.

[![Aisha interface](aisha.jpg)](aisha.jpg)

## Возможности

- **Файлы** — чтение, запись, точечное редактирование, листинг, `glob`-поиск и `grep`;
- **Shell** — запуск команд (`powershell`/`cmd` в Windows; `/bin/sh` в других ОС), таймауты, обрезка вывода, подтверждение опасных команд;
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

### Вариант 1: из PyPI (рекомендуется)

```powershell
python -m pip install aisha
```

Самый простой способ: пакет устанавливается с https://pypi.org/project/aisha/ вместе со всеми зависимостями, команда `aisha` доступна глобально.

### Вариант 2: из исходников (репозиторий)

```powershell
python -m pip install .             # обычная установка
python -m pip install -e ".[dev]"  # editable-установка для разработки
```

Модель и chat template должны поддерживать OpenAI-style function calling. Aisha сама
передаёт и исполняет схемы инструментов; встроенный режим llama-server `--tools all` не
нужен и создаёт отдельный канал исполнения вне проверок Aisha. Проверить шаблон можно
командой `aisha --doctor --tool-call-test`.

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
DEFAULTS  ←  ~/.aisha/config.toml  ←  <workspace>/.aisha/aisha.toml  ←  env AISHA_*  ←  CLI-флаги
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
top_p = 0.9                 # необязательно; не указывайте ключ для серверного значения
top_k = 40                  # необязательно; целое > 0
repeat_penalty = 1.1        # необязательно; > 0
frequency_penalty = 0.0     # необязательно; -2.0 .. 2.0
max_output_tokens = 32768
context_window = 32768
context_soft_limit = 0.75
max_tool_iterations = 25
tool_guide = false           # true — добавить «Справочник инструментов» в системный промпт (для слабых моделей)
communication_language = "Russian"  # язык общения агента с пользователем
# enable_thinking = true        # true/false — управление thinking для Qwen-моделей;
                                # не указывайте ключ для серверного значения. Thinking выключен
                                # в запросах со схемами инструментов и при саммаризации.
compact_tool_schemas = false    # true — вырезать блоки «Example:» из описаний инструментов

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
index_max_chars = 20000     # лимит обрезки индекса памяти в системном промпте

[skills]
index_max_chars = 20000     # лимит обрезки индекса скиллов в системном промпте

[context]
agents_md_max_chars = 65536 # лимит обрезки AGENTS.md и SYSTEM.md

[compaction]
summary_max_tokens = 4096   # запрошенный бюджет сводки; эффективный минимум — 256 токенов
trim_user_messages = false  # разрешить обрезку user-сообщений в крайнем случае
elide_large_tool_args = false  # заменять крупные применённые аргументы превью
elide_threshold_chars = 4000
trim_head_chars = 2000      # head при обрезке крупных результатов инструментов
trim_tail_chars = 1000      # tail при обрезке
trim_block_fraction = 0.25  # fallback для внутренних обрезок без явного бюджета

[ui]
theme = "dark"
stream = true
show_reasoning = false
debug = false                 # reasoning и краткая диагностика запросов/ответов в консоли
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
| `AISHA_COMMUNICATION_LANGUAGE` | `llm.communication_language` |

Конфиг строго валидируется: неизвестная секция или ключ вызывает ошибку.
**Для проектного `aisha.toml` действуют отдельные ограничения shell и файлового доступа** — он не может ставить
`permission = "auto"`, включать доступ за пределы workspace или включать `shell`,
если тот отключён глобально. Это защита от «троянского» конфига в склонированном репозитории.

## Инструменты

| Инструмент | Read-only | Назначение |
|---|---|---|
| `read_file` | да | чтение текста как UTF-8 (невалидные байты заменяются), с offset/limit |
| `write_file` | нет | создание/перезапись (атомарно) |
| `edit_file` | нет | точная замена фрагмента текста |
| `list_dir` | да | содержимое каталога |
| `glob` | да | поиск файлов по маске |
| `grep` | да | regex-поиск по содержимому |
| `run_command` | нет | запуск shell-команды |
| `web_search` | да | поиск DuckDuckGo |
| `web_fetch` | да | загрузка веб-страницы |
| `todowrite` | да | полная замена todo-списка сессии |
| `ask_user` | да | уточняющий вопрос при интерактивных stdin/stdout |
| `memory_list` / `memory_get` | да | список/чтение блоков памяти |
| `memory_set` / `memory_replace` | нет | запись/правка блоков памяти |
| `skill` | да | загрузить текст скилла по имени |

Файловые инструменты не выходят за пределы workspace (path-traversal блокируется),
если не включён соответствующий `allow_*_outside_workspace`.
Разрешение внешней записи лишь делает путь допустимым: каждая такая запись всё равно
требует интерактивного подтверждения, поэтому в неинтерактивном режиме она отклоняется.
Проверки путей относятся к файловым инструментам и `cwd`, но не ограничивают действия
запущенного shell-процесса; `run_command` не работает в песочнице. `tools.permission`
управляет только `run_command`; для отключения файловых и memory-записей используйте
`--read-only`.
`run_command` регистрируется только при включённом `tools.shell`, `web_search` — только
при `tools.web_search`, memory-инструменты — только при `memory.enabled`;
`ask_user` исключается из схем в неинтерактивном режиме.

## Память и скиллы

- **Память** — JSON-блоки в `~/.aisha/memory/` (глобально) и `<workspace>/.aisha/memory/`
  (проектно). Проектный блок перекрывает глобальный с тем же `label`.
  `memory_set.scope` по умолчанию равен `global`; для памяти workspace укажите `project`.
  Вызов `memory_get` не выводится в консоль (это фоновое чтение собственной памяти агента).
- **Скиллы** — каталоги `~/.aisha/skills/<name>/SKILL.md` и
  `<workspace>/.aisha/skills/<name>/SKILL.md` с обязательным YAML-frontmatter
  (`name`, `description`). Поиск идёт по `name` из frontmatter; имя каталога может
  отличаться. Неизменённое тело скилла возвращается только один раз за сессию.

## Кастомный системный промпт (SYSTEM.md)

Системный промпт собирается из встроенной политики и окружения, необязательного Tool Guide,
workspace `AGENTS.md`, `<workspace>/.aisha/SYSTEM.md`, текущих todo и, в конце, индексов
памяти/скиллов. `AGENTS.md` и `SYSTEM.md` оформляются как недоверенные справочные данные и
не могут переопределить встроенную политику. Оба поддерживают UTF-8 BOM и независимо
ограничиваются настроенным и адаптивным лимитами.

## Совместимость сервера

Обязательны `GET /v1/models` и streaming `POST /v1/chat/completions`, включая OpenAI-style
tool-call deltas при использовании инструментов. `/health`, `/tokenize`, `meta.n_ctx`, usage
в stream options, `reasoning_content` и Qwen `chat_template_kwargs` необязательны и имеют
fallback. Положительный `meta.n_ctx` заменяет оба настроенных лимита контекста; значения
конфига используются для серверов, которые его не публикуют.

Chat-запросы повторяются через 0,5, 1 и 2 секунды при ошибках соединения и HTTP
429/502/503/504, но только до получения данных SSE. При обрыве начатого потока сохраняется
частичный текст без повторной отправки; частичные tool-call не выполняются. При обычном
старте явный 503 от `/health` запускает model discovery каждые четыре секунды до 120 секунд.
`--doctor` делает один диагностический проход и не ожидает.

`llm.max_output_tokens` — только верхняя граница. Каждый запрос резервирует оценку system
prompt, истории, схем инструментов и 5% запаса, затем использует оставшийся контекст.

## REPL

Команды внутри интерактивного режима:

| Команда | Действие |
|---|---|
| `/help` | справка |
| `/new` | новая сессия (сброс истории) |
| `/status` | сервер, модель, workspace, режим, токены, разбивка контекста |
| `/tools` | список инструментов |
| `/skills` | индекс скиллов |
| `/memory` | блоки памяти |
| `/compact` | принудительное сжатие истории |
| `/doctor` | проверка соединения |
| `/init` | изучить проект и создать `AGENTS.md` |
| `/clear` | очистка экрана |
| `/quit`, `/exit`, `Ctrl+D` | выход |

Дополнительно: `Ctrl+C` отменяет текущий запрос (REPL не завершается),
`Ctrl+↑/↓` — история запросов, `Tab` — автодополнение команд и путей,
`Alt+Enter` — вставка новой строки (многострочный ввод).

## Безопасность

- Запуск shell-команд в режиме `permission = "ask"` требует подтверждения. В режиме
  `auto` подтверждение также требуется для команд, распознанных эвристикой как опасные.
- `web_fetch` разрешает только HTTP(S), заново проверяет каждый redirect и блокирует
  localhost, `.local`, private, loopback, link-local, reserved, multicast, unspecified,
  mapped IPv6, опасный 6to4 и Teredo. `allow_private_hosts=true` отключает фильтрацию
  hostname/IP; DNS rebinding между проверкой и подключением остаётся известным ограничением.
- `find_danger` в `shell.py` — **эвристика по regex, а не песочница**: обойти её можно.
  Для недоверенного вывода модели используйте `permission = "ask"` или `deny` и не
  запускайте aisha от имени пользователя с повышенными правами.
- `--debug` записывает в `<workspace>/.aisha/logs/` промпты, историю, ответы/reasoning
  модели, аргументы инструментов и до 4000 символов каждого результата. Логи автоматически
  не удаляются; не включайте этот режим, если данные могут содержать секреты.

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
├── logger.py     # DebugLogger: traces LLM/инструментов в <workspace>/.aisha/logs/ (--debug)
├── fsutil.py     # атомарная запись, проверка путей, human_size
├── errors.py     # иерархия исключений
└── tools/        # реализации инструментов (base, files, shell, web, extras)
```
