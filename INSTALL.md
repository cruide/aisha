# Установка агента aisha

`aisha` — локальный консольный AI-агент на Python 3.11+. Он работает с внешним `llama-server` (llama.cpp) через OpenAI-совместимый API и не требует облачных LLM или API-ключей.

## 1. Требования

- **OS:** Windows 11 (основная), также работает на других ОС, поддерживающих Python 3.11+
- **Python 3.11+** (включая `pip` и `venv`)
- **Запущенный `llama-server`** из llama.cpp с поддержкой OpenAI-compatible API
- **GGUF-модель** и, при необходимости, chat-template (jinja)

## 2. Установка Python (если ещё нет)

Проверьте версию:

```powershell
python --version
```

Нужен `Python 3.11` или новее. Если Python отсутствует — установите его с https://www.python.org/downloads/ (обязательно отметьте «Add python.exe to PATH»).

## 3. Установка aisha

Склонируйте или скопируйте репозиторий в рабочую директорию, затем из корня проекта:

### Вариант 1: в текущее окружение Python

```bash
pip install -e ".[dev]"
```

`-e` (editable) устанавливает агент в режиме разработки: изменения в `src/` сразу подхватываются, а команда `aisha` доступна глобально. Dev-зависимости (pytest, ruff) ставятся вместе.

### Вариант 2: изолированно через pipx

```bash
pipx install .
```

Изолирует агент и его зависимости от остального окружения. Команда `aisha` также доступна глобально.

> Примечание: зависимости проекта — `httpx`, `rich`, `prompt-toolkit`, `ddgs`, `beautifulsoup4`, `PyYAML`. Они ставятся автоматически.

### Проверка установки

```bash
aisha --help
```

Должна отобразиться справка по команде.

## 4. Установка и настройка llama-server

`aisha` не загружает модель сам — за него это делает внешний `llama-server`.

1. Скачайте сборку llama.cpp (CUDA или CPU) с https://github.com/ggml-org/llama.cpp/releases.
2. Скачайте GGUF-модель, например Qwen:
   `d:\models\llama\qwen\Qwen3.5-9B-UD-Q4_K_XL.gguf`
3. При необходимости подготовьте jinja chat-template:
   `d:\models\llama\qwen\chat_template.jinja`

### Запуск сервера

```powershell
z:\llamacpp\cuda\llama-server.exe `
  -m d:\models\llama\qwen\Qwen3.5-9B-UD-Q4_K_XL.gguf `
  --no-mmproj `
  --jinja `
  --chat-template-file "d:\models\llama\qwen\chat_template.jinja" `
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

Ключевые параметры для работы с `aisha`:

| Параметр | Значение | Назначение |
| --- | --- | --- |
| `--host localhost` | локальный доступ | безопаснее, чем `0.0.0.0` |
| `--port 8088` | порт агента по умолчанию | совпадает с дефолтом в конфиге |
| `--tools all` | tool calling | нативный OpenAI-style tool calling |
| `-a Qwen3.5-9B-Q4_K_XL` | alias модели | только подсказка для агента |
| `-c 65536` | контекст | должен совпадать с `context_window` в конфиге |

> Безопасность: не публикуйте `llama-server` в интернет. Без reverse proxy, аутентификации и ограничения доступа он должен слушать только `localhost`.

### Проверка сервера

```powershell
Invoke-RestMethod http://localhost:8088/health
```

Ожидаемый ответ:

```text
{"status":"ok"}
```

## 5. Конфигурация aisha

Агент работает с настройками по умолчанию сразу после установки, но для удобства можно создать глобальный конфиг.

### Глобальная конфигурация

Файл: `%USERPROFILE%\.aisha\config.toml`

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

> Имя модели в конфиге — только подсказка. Если объявленное имя не совпадает с `-a` на сервере, `aisha` автоматически подключается к первой доступной модели и не падает.

### Проектная конфигурация

Файл `<workspace>/aisha.toml` переопределяет глобальные настройки для конкретного проекта. Он **не может ослабить безопасность**: `permission = "auto"`, чтение/запись за пределами workspace и т.п. приводят к ошибке конфигурации.

### Приоритет настроек

1. Аргументы командной строки
2. Переменные окружения (`AISHA_SERVER_URL`, `AISHA_MODEL`, `AISHA_PERMISSION`, `AISHA_SHELL`, `AISHA_CONTEXT_WINDOW`, `AISHA_MAX_OUTPUT_TOKENS`)
3. Проектный `aisha.toml`
4. Глобальный `~/.aisha/config.toml`
5. Значения по умолчанию

## 6. Диагностика

Проверка соединения с сервером и доступности модели:

```bash
aisha --doctor
```

Расширенная проверка tool calling (отправляет безопасный тестовый запрос без shell-команд и записи файлов):

```bash
aisha --doctor --tool-call-test
```

## 7. Первый запуск

Из любой рабочей директории:

```bash
aisha
```

Откроется интерактивный REPL. Полезные команды: `/status` — сервер, модель, workspace, токены; `/help` — справка.

Одноразовый запрос:

```bash
aisha "Объясни структуру этого проекта"
```

Другие полезные флаги:

```bash
aisha --server http://localhost:8088   # другой llama-server
aisha --model Qwen3.5-9B-Q4_K_XL       # переопределение модели
aisha -r                               # режим только для чтения
aisha --permission deny                # запрет shell-команд
aisha --permission auto                # выполнять разрешённые команды без подтверждения
aisha --shell cmd                      # cmd вместо PowerShell
aisha --tools-only                     # список доступных инструментов
```

## 8. Обновление

Обновите код из репозитория и переустановите пакет:

```bash
pip install -e ".[dev]" --upgrade
```

Для pipx:

```bash
pipx upgrade aisha
```

## 9. Устранение неполадок

| Симптом | Решение |
| --- | --- |
| `ServerUnavailableError` при старте REPL | `llama-server` не запущен или недоступен. Проверьте `http://localhost:8088/health`. |
| Модель ещё загружается | Подождите, пока `llama-server` не загрузит GGUF, затем повторите `aisha --doctor`. |
| Ошибка конфигурации из `aisha.toml` | Проектный конфиг не может ослаблять безопасность — переместите такие настройки в глобальный `config.toml` или передайте флагом CLI. |
| Команда `aisha` не найдена | Переустановите пакет (`pip install -e .`) или проверьте, что каталог Scripts Python в PATH. |
| `permission = "ask"` мешает | Запустите с `--permission auto` или задайте `AISHA_PERMISSION=auto`. |

## 10. Тестирование и проверка кода (разработка)

```bash
python -m pytest tests/ -q   # тесты (не требуют llama-server)
ruff check src/ tests/       # линт
```
