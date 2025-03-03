ChatBot Tyrion

📌 Описание

Этот проект представляет собой чат-бота, обученного на репликах Тириона Ланнистера из "Игры престолов". Используется SentenceTransformer для дообучения модели на диалогах персонажа.

🛠️ Установка

1. Клонирование репозитория

git clone https://github.com/your-repo/chat_bot_tyrion.git
cd chat_bot_tyrion

2. Установка зависимостей

pip install -r requirements.txt

3. Запуск обучения

python train_model.py

📁 Структура проекта

chat_bot_tyrion/
│── app/                  # Основной код приложения
│── checkpoints/          # Контрольные точки модели
│── data/                 # Исходные данные и эмбеддинги
│   ├── game-of-thrones.csv  # Диалоги Тириона
│   ├── tyrion_embeddings.pkl  # Эмбеддинги фраз
│── fine_tuned_tyrion_model/ # Сохранённая модель
│── static/               # Статические файлы
│── templates/            # Шаблоны для веб-интерфейса
│── train_model.py        # Код обучения
│── chatbot.py            # Код бота
│── main.py               # Запуск сервера
│── requirements.txt      # Зависимости
│── .gitignore            # Исключения Git

🔥 Как использовать

Запусти сервер: python main.py

Открывай в браузере: http://127.0.0.1:5000

Чат-бот ответит в стиле Тириона 👑

