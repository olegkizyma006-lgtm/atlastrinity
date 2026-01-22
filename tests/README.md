# 🧪 AtlasTrinity Tests

Тестові скрипти для перевірки функціональності AtlasTrinity.

## 📋 Доступні тести

### STT & Whisper

- **test_whisper_mps.py** - Тест Whisper на MPS (Apple Silicon GPU) vs CPU
- **verify_whisper.py** (in scripts/) - Колишній скрипт верифікації, тепер у папці scripts/.

### Agents

- **test_copilot.py** - Тест GitHub Copilot провайдера

  ```bash
  python tests/test_copilot.py
  ```

- **test_grisha_real.py** - Реальний тест агента Grisha (Computer Use)

  ```bash
  python tests/test_grisha_real.py
  ```

- **test_handoff.py** - Тест передачі задач між агентами
  ```bash
  python tests/test_handoff.py
  ```

## 🔧 Вимоги

Переконайтесь що виконано setup:

```bash
./setup.sh
# або
python setup_dev.py
```

## 📊 Результати

### Whisper MPS Test

Очікуваний результат на Apple Silicon:

- MPS: ~15s (для великих моделей)
- CPU: ~439s завантаження
- **Прискорення: ~14x**

### Verify Whisper

Має пройти всі 25 перевірок:

- ✅ Config files
- ✅ Directories
- ✅ Python imports
- ✅ STT initialization
- ✅ Config loader (MCP + Voice)
- ✅ Production/Dev setup
- ✅ Build configuration

## 🔗 Посилання

- [Документація](../docs/)
- [Конфігурація](../config/)
- [Setup Guide](../SETUP.md)
