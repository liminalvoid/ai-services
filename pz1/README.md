# Лабораторная работа №1. Проектирование AI-сервиса на FastAPI

## Запуск

```bash
# Создание и запуск виртуальной среды для Python
python3 -m venv .venv && source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск приложения
uvicorn main:app --reload


# Запуск тестов
python3 test_api.py
```
