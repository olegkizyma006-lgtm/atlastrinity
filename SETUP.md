# AtlasTrinity - Quick Setup Guide

## Швидкий Старт (macOS)

### 1. Вимоги

- **macOS** 12.0+
- **Python** 3.12.12 (рекомендовано) або 3.10+
- **Homebrew** (для системних залежностей)
- **Bun** (встановиться автоматично)
- **Swift** 5.0+ (для компіляції macos-use MCP)

### 2. Автоматичне Налаштування

```bash
# 1. Клонуйте репозиторій
git clone https://github.com/olegkizima01/atlastrinity.git
cd atlastrinity

# 2. Запустіть повне налаштування (встановить всі залежності)
npm run setup
# або напряму:
python3 scripts/setup_dev.py
```

**Що робить setup:**
- ✅ Створює Python venv (.venv)
- ✅ Встановлює PostgreSQL, Redis через Homebrew
- ✅ **FORCED Build**: Завжди перекомпілює Swift MCP сервер (macos-use) для чистоти бінарника
- ✅ Встановлює Python залежності (requirements.txt)
- ✅ Встановлює NPM пакети та 8 основних MCP серверів
- ✅ Завантажує AI моделі (Faster-Whisper, Ukrainian TTS)
- ✅ **FORCED Sync**: Перезаписує глобальні конфігурації в ~/.config/atlastrinity/ актуальними темплейтами
- ✅ Ініціалізує базу даних PostgreSQL (таблиця `recovery_attempts`)
- ✅ Запускає Redis, PostgreSQL сервіси
- ✅ Налаштовує тайм-аути на **3600s** для стабільної роботи

### 3. Налаштування API Ключів

Додайте свій Copilot API ключ:

```bash
# Відредагуйте ~/.config/atlastrinity/.env
nano ~/.config/atlastrinity/.env
```

Додайте:
```env
COPILOT_API_KEY=your_api_key_here
GITHUB_TOKEN=your_github_token_here  # опціонально
```

### 4. Запуск

```bash
npm run dev
```

Це запустить:
- **Brain** (Python FastAPI): http://127.0.0.1:8000
- **Renderer** (Vite React): http://localhost:3000
- **Electron** (Desktop App)

### 5. Перевірка Системи

```bash
# Перевірка сервісів
brew services list | grep -E "redis|postgresql"

# Перевірка Python залежностей
.venv/bin/python -c "import faster_whisper, ukrainian_tts; print('OK')"

# Перевірка MCP серверів
cat ~/.config/atlastrinity/mcp/config.json | jq '.mcpServers | keys'

# Перевірка цілісності коду (Strict Mode)
.venv/bin/python -m pyrefly check src/brain src/mcp_server
```

## Доступні MCP Сервери

### Tier 1 (Критичні - Завжди Ввімкнені)
- **macos-use** - Нативний контроль macOS (Swift, **52 інструменти**)
- **filesystem** - Файлові операції
- **sequential-thinking** - Динамічне планування

### Tier 2 (Високий Пріоритет - Ввімкнені)
- **vibe** - AI Coding & Self-healing (**10 інструментів**)
- **memory** - Граф знань (Atlas, Tetyana, Grisha) + Namespace Filtering
- **graph** - Візуалізація зв'язків
- **redis** - Asyncio Persistence (Fast State management)
- **puppeteer** - Браузерна автоматизація
- **chrome-devtools** - Debugging

### Tier 3-4 (Опціональні - Вимкнені за замовчуванням)
- apple-mcp, github, duckduckgo-search, context7, whisper-stt, docker, postgres, slack, time, graph

## Архітектура Агентів

```
┌─────────────────────────────────────────────────────┐
│                     ATLAS                           │
│            (Стратег та Планувальник)                │
│  • Аналіз запитів                                   │
│  • Створення планів                                 │
│  • Читання звітів Гріші з memory/notes             │
│  • Recovery планування з врахуванням feedback       │
└─────────────────┬───────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼──────┐    ┌──────▼────────────────────────┐
│   TETYANA    │    │         GRISHA                │
│ (Виконавець) │    │   (Верифікатор/Аудитор)       │
│              │    │                               │
│ • Виконання  │◄───┤ • Перевірка результатів       │
│   кроків     │    │ • Vision аналіз (GPT-4o)      │
│ • MCP tools  │    │ • Детальні звіти відхилень   │
│ • Читання    │    │ • Збереження в memory + notes│
│   feedback   │    │ • Рекомендації виправлень     │
└──────────────┘    └───────────────────────────────┘
```

### 🆕 Новий Флоу Комунікації (з Notes)

1. **Tetyana** виконує крок → результат
2. **Grisha** перевіряє:
   - ✅ Підтверджує → продовжуємо
   - ❌ Відхиляє → **зберігає детальний звіт**:
     - Memory server (graph зв'язки)
     - Notes server (швидкий текстовий доступ)
3. **Atlas** читає звіт Гріші:
   - Бачить точні причини відхилення
   - Отримує рекомендації виправлень
   - Планує краще recovery
4. **Tetyana** читає той самий звіт:
   - Уникає повторення помилок
   - Використовує remediation suggestions

## Структура Проекту

```
atlastrinity/
├── src/
│   ├── brain/              # Python Brain (FastAPI)
│   │   ├── agents/         # Atlas, Tetyana, Grisha
│   │   ├── mcp_manager.py  # MCP клієнт
│   │   ├── memory.py       # ChromaDB long-term memory
│   │   └── server.py       # FastAPI endpoints
│   ├── main/               # Electron Main Process
│   ├── renderer/           # React UI
│   └── mcp_server/         # Кастомні MCP сервери
│       ├── memory_server.py
│       ├── notes_server.py  # 🆕
│       ├── terminal_server.py
│       └── ...
├── config/                 # Конфіги проекту
│   ├── config.yaml         # Основна конфігурація
│   └── config_sync.py      # Синхронізація з ~/.config
├── scripts/
│   ├── setup_dev.py        # 🔧 Головний setup скрипт
│   └── setup.sh            # Wrapper для setup_dev.py
└── ~/.config/atlastrinity/ # Глобальні конфіги (створюються setup)
    ├── .env                # API ключі
    ├── config.yaml         # Копія конфігурації
    ├── mcp/
    │   └── config.json     # MCP серверів конфігурація
    ├── memory/             # ChromaDB
    ├── models/             # AI моделі
    │   ├── tts/
    │   └── faster-whisper/
    ├── logs/               # Логи системи
    └── notes/              # 🆕 Текстові нотатки агентів
```

## Troubleshooting

### PostgreSQL не запускається
```bash
brew services restart postgresql@17
brew services list
```

### Redis не запускається
```bash
brew services restart redis
redis-cli ping  # має повернути PONG
```

### Помилка компіляції macos-use (Swift)
```bash
cd vendor/mcp-server-macos-use
swift build -c release
```

### Python залежності не встановлені
```bash
.venv/bin/pip install -r requirements.txt
```

### Запуск тестів / Running the test suite 🔬
- **Запускайте тести в активованому `.venv`** (щоб використовувати правильне середовище):

```bash
# у корені проєкту
. .venv/bin/activate
pytest
```

- **Проблеми з імпортом (наприклад `ModuleNotFoundError: No module named 'sqlalchemy'`)** свідчать про відсутні залежності у середовищі — це проблема конфігурації середовища, **не пов'язана** з рефактором промптів. Щоб вирішити, встановіть відсутні пакети у `.venv`:

```bash
.venv/bin/pip install sqlalchemy
# або
.venv/bin/pip install -r requirements.txt
```

- Якщо ви хочете запускати повний тест-сьют (включно із DB/Redis інтеграційними тестами), переконайтеся, що PostgreSQL та Redis запущені та доступні, або запускати тести в CI з налаштованими сервісами.

- Рекомендація: додати окремий `requirements-dev.txt` для тестів та CI, щоб уникнути таких помилок локально.

### MCP сервери не з'єднуються
```bash
# Перевірка конфігурації
cat ~/.config/atlastrinity/mcp/config.json | jq '.mcpServers | keys'

# Тест notes сервера
.venv/bin/python -m mcp_server.notes_server
```

## Розробка

### Додавання нового MCP сервера

1. Створіть файл `src/mcp_server/your_server.py`
2. Додайте конфігурацію в `~/.config/atlastrinity/mcp/config.json`
3. Перезапустіть: `npm run dev`

### Логи

```bash
# Brain логи
tail -f ~/.config/atlastrinity/logs/brain.log

# Electron логи
# Дивіться в консолі терміналу де запустили npm run dev
```

## Ліцензія

MIT

---

**Потрібна допомога?** Відкрийте issue на GitHub або перевірте docs/ папку для детальної документації.
