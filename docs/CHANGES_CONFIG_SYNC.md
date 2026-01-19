# Реалізовано: Гібридна система конфігурації

## Нещодавні зміни (2026-01-19)

- Делеговано валідацію результатів та голосові повідомлення про відновлення кроків агенту **Grisha**. Додано налаштування у `orchestrator`: `recovery_voice_agent` (default: `grisha`) та `validate_failed_steps_with_grisha` (default: true). Це унеможливлює, щоб Atlas самостійно оголошував, що "крок зупинився" — тепер це прерогатива валідатора.


## Що зроблено

### 1. Створено систему синхронізації .env → config.yaml

**Файл**: [src/brain/config_sync.py](src/brain/config_sync.py)

- `sync_env_to_config()` - синхронізує .env в config.yaml при старті
- `get_api_key()` - отримує API ключі з config.yaml
- Автоматична синхронізація при кожному запуску

### 2. Інтегровано в server.py

**Зміни в**: [src/brain/server.py](src/brain/server.py)

```python
# Замість load_dotenv():
from .config_sync import sync_env_to_config, get_api_key

# При старті
sync_env_to_config()
```

### 3. Виправлено моделі

#### raptor → raptor-mini

**Файли**:
- [config.yaml](config.yaml) - змінив на `raptor-mini`
- [src/brain/agents/atlas.py](src/brain/agents/atlas.py) - default parameter
- [src/brain/config_loader.py](src/brain/config_loader.py) - defaults

#### Оптимізовано вибір моделей

```yaml
agents:
  atlas:
    model: "raptor-mini"  # Планування
  
  tetyana:
    model: "gpt-4.1"      # Виконання
  
  grisha:
    vision_model: "gpt-4o"  # Vision

mcp:
  terminal:
    model: "gpt-4o"       # Tool calling
  
  filesystem:
    model: "gpt-4.1"      # Швидкість
  
  playwright:
    model: "gpt-4o"       # Browser automation
  
  computer_use:
    model: "gpt-4o"       # Vision control
  
  whisper:
    model: "base"         # STT
```

### 4. Оновлено документацію

**Створено**:
- [CONFIG_ARCHITECTURE.md](CONFIG_ARCHITECTURE.md) - повна документація архітектури
- [.env.example](.env.example) - шаблон з поясненнями

**Оновлено**:
- [setup_dev.py](setup_dev.py) - пояснення про синхронізацію
- [src/brain/production_setup.py](src/brain/production_setup.py) - коментарі

### 5. Додано PROJECT_ROOT

**Файл**: [src/brain/config.py](src/brain/config.py)

```python
PROJECT_ROOT = Path(__file__).parent.parent.parent
```

## Як це працює

### User perspective

```bash
# 1. Користувач редагує .env (звичний інтерфейс)
vim .env

# 2. Запускає програму
npm run dev

# 3. При старті .env автоматично синхронізується в config.yaml
# 4. Система працює ТІЛЬКИ з config.yaml
```

### System flow

```
┌──────────────┐
│  .env        │  ◄─── Користувач редагує
│  (project)   │
└──────┬───────┘
       │
       │ При старті
       ▼
┌──────────────────┐
│ sync_env_to_config()│
└──────┬───────────┘
       │
       ▼
┌──────────────────────┐
│ config.yaml          │  ◄─── Система читає
│ ~/.config/atlastrinity│
└──────────────────────┘
```

### Priority

```
config.yaml > .env > defaults
```

## Переваги

### 1. Зручність для користувача
- `.env` - стандарт для API ключів
- Знайомий інтерфейс

### 2. Системність
- Один YAML конфіг для всього
- Структурований формат
- Легко читати і редагувати

### 3. Гнучкість
- Advanced users можуть редагувати config.yaml напряму
- Пріоритет config.yaml > .env

### 4. Універсальність
- Працює однаково в dev та production
- Автоматична синхронізація

## Тестування

```bash
# Синхронізація працює
✓ .env → config.yaml
✓ API ключі копіюються
✓ Моделі оновлюються
✓ Система читає з config.yaml

# Перевірено
source .venv/bin/activate
python -c "from src.brain.config_sync import sync_env_to_config; sync_env_to_config()"

# Результат:
[ConfigSync] 📖 Reading .env from: /Users/.../atlastrinity/.env
[ConfigSync] 📖 Reading existing config from: ~/.config/atlastrinity/config.yaml
[ConfigSync] ✓ Added copilot_api_key to config
[ConfigSync] ✓ Added github_token to config
[ConfigSync] ✓ Config synchronized to: ~/.config/atlastrinity/config.yaml
```

## Моделі (виправлено)

### Доступні моделі

| Модель | ID | Призначення |
|--------|----|-----------| 
| **Raptor mini** | `raptor-mini` | Планування, reasoning ✓ |
| **GPT-4.1** | `gpt-4.1` | Виконання коду, швидкість ✓ |
| **GPT-4o** | `gpt-4o` | Vision, tool calling ✓ |
| GPT-5 mini | `gpt-5-mini` | Компактність |
| Grok Code Fast 1 | `grok-code-fast-1` | Швидкий coding |

### Виправлення

- ❌ `raptor` (не існує)
- ✅ `raptor-mini` (Raptor mini Preview)

### Оптимальний розподіл (реалізовано)

```yaml
# Агенти
Atlas:    raptor-mini  # найкраща для planning/reasoning
Tetyana:  gpt-4.1      # найкраща для code execution
Grisha:   gpt-4o       # обов'язково для vision

# MCP
Terminal:      gpt-4o   # tool calling + command interpretation
Filesystem:    gpt-4.1  # швидкість
Playwright:    gpt-4o   # browser automation
Computer Use:  gpt-4o   # vision-based control
Whisper:       base     # STT (offline)
```

## Наступні кроки

1. ✅ Протестувати в dev режимі
2. ⏳ Протестувати production build
3. ⏳ Перевірити всі агенти з новими моделями
4. ⏳ Оновити README.md з новою архітектурою

## Файли змінено

```
✓ src/brain/config_sync.py        (NEW) - синхронізація
✓ src/brain/server.py              (MOD) - інтеграція синхронізації
✓ src/brain/config.py              (MOD) - додано PROJECT_ROOT
✓ src/brain/config_loader.py       (MOD) - виправлено defaults
✓ src/brain/agents/atlas.py        (MOD) - raptor → raptor-mini
✓ config.yaml                      (MOD) - оптимізовано моделі
✓ setup_dev.py                     (MOD) - коментарі
✓ src/brain/production_setup.py   (MOD) - коментарі
✓ .env.example                     (NEW) - шаблон
✓ CONFIG_ARCHITECTURE.md           (NEW) - документація
✓ CHANGES_CONFIG_SYNC.md           (NEW) - цей файл
```
