import telebot
from telebot import types
import datetime
import pytz
import sqlite3
import threading
import time
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import re
from textblob import TextBlob
from textblob.exceptions import NotTranslated


bot = telebot.TeleBot('YOUR TELEGRAM TOKEN')
logged_in_users = {}
user_data = {}
nltk.download('punkt')


timezone_map = {
    "UTC": "UTC",
    "UTC+1": "Etc/GMT-1",
    "UTC+2": "Etc/GMT-2",
    "UTC+3": "Etc/GMT-3",
    "UTC+4": "Etc/GMT-4",
    "UTC+5": "Etc/GMT-5",
    "UTC+6": "Etc/GMT-6",
    "UTC+7": "Etc/GMT-7",
    "UTC+8": "Etc/GMT-8",
    "UTC+9": "Etc/GMT-9",
    "UTC+10": "Etc/GMT-10",
    "UTC+11": "Etc/GMT-11",
    "UTC+12": "Etc/GMT-12",
    "UTC-1": "Etc/GMT+1",
    "UTC-2": "Etc/GMT+2",
    "UTC-3": "Etc/GMT+3",
    "UTC-4": "Etc/GMT+4",
    "UTC-5": "Etc/GMT+5",
    "UTC-6": "Etc/GMT+6",
    "UTC-7": "Etc/GMT+7",
    "UTC-8": "Etc/GMT+8",
    "UTC-9": "Etc/GMT+9",
    "UTC-10": "Etc/GMT+10",
    "UTC-11": "Etc/GMT+11",
    "UTC-12": "Etc/GMT+12",
}

day_responses = {
    "Понедельник": "Понедельник – отличный день для планирования недели! Составьте список задач и начните с самых приоритетных.",
    "Вторник": "Во вторник вы можете сосредоточиться на выполнении сложных задач. Это день продуктивности!",
    "Среда": "Среда – середина недели. Отличное время для анализа выполненной работы и корректировки планов.",
    "Четверг": "Четверг – хороший день для встреч и обсуждений. Попробуйте провести совещания или мозговые штурмы.",
    "Пятница": "Пятница – время подводить итоги недели. Завершите начатые дела и подготовьтесь к выходным.",
    "Суббота": "Суббота – время для отдыха и восстановления сил. Наслаждайтесь выходным и займитесь хобби.",
    "Воскресенье": "Воскресенье – отличный день для подготовки к новой неделе. Спланируйте задачи и установите цели."
}

days_of_week = ["Понедельник", "Вторник", "Среда",
                "Четверг", "Пятница", "Суббота", "Воскресенье"]


good_night = ["Доброй ночи, [user]!", "Спокойной ночи, [user]!", "Добрых снов, [user]!", "Сладких снов, [user]!", "Спи сладко, [user]!",
              "Мягкой постели, [user]!", "Набирайся сил, [user]!", "Наслаждайся отдыхом, [user]!", "Не засиживайся до поздна, [user]!"]

good_morning = ["Доброе утро, [user]!", "С добрым утром, [user]!", "Нежного пробуждения, [user]!", "Хорошего дня, [user]!", "Удачного дня, [user]!",
                "Продуктивного дня, [user]!", "Сегодня у тебя всё получится, [user]!", "Тебя ждут великие дела, [user]!", "Новый день – новые свершения!"]


def initialize_db():
    connection = sqlite3.connect('tasks.sql')
    cursor = connection.cursor()
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, pass TEXT, timezone TEXT DEFAULT "UTC", night TEXT, morning TEXT)')
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, task TEXT, time TEXT, day_of_week TEXT, date TEXT)')

    cursor.execute("PRAGMA table_info(tasks)")
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns]
    if 'date' not in column_names:
        cursor.execute("ALTER TABLE tasks ADD COLUMN date TEXT")

    connection.commit()
    cursor.close()
    connection.close()


initialize_db()


def get_user_timezone(user_id):
    connection = sqlite3.connect('tasks.sql')
    cursor = connection.cursor()
    cursor.execute("SELECT timezone FROM users WHERE pass=?", (str(user_id),))
    user_timezone = cursor.fetchone()[0]
    cursor.close()
    connection.close()
    return timezone_map.get(user_timezone, "UTC")


@bot.message_handler(commands=['reg'])
def reg(message):
    connection = sqlite3.connect('tasks.sql')
    cursor = connection.cursor()
    user_id_ = message.from_user.id
    cursor.execute("SELECT * FROM users WHERE pass=?", (str(user_id_),))
    user = cursor.fetchone()
    if user:
        bot.send_message(message.chat.id, 'Вы уже зарегистрированы.')
    else:
        bot.send_message(
            message.chat.id, 'Регистрация началась. Введи свое имя:')
        bot.register_next_step_handler(message, user_name)
    cursor.close()
    connection.close()


def user_name(message):
    name = message.text.strip()
    user_id_ = message.from_user.id
    connection = sqlite3.connect('tasks.sql')
    cursor = connection.cursor()
    cursor.execute("INSERT INTO users (name, pass) VALUES (?, ?)",
                   (name, str(user_id_)))
    connection.commit()
    cursor.close()
    connection.close()
    bot.send_message(
        message.chat.id, f'Отлично, теперь введи время, когда ты ложишься спать: ')
    bot.register_next_step_handler(message, user_night)


def user_night(message):
    night = message.text.strip()
    connection = sqlite3.connect('tasks.sql')
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE users SET night=? WHERE pass=?", (night, str(message.from_user.id)))
    connection.commit()
    cursor.close()
    connection.close()
    bot.send_message(
        message.chat.id, 'Отличо, теперь введи вермя, когда ты встаешь: ')
    bot.register_next_step_handler(message, user_morning)


def user_morning(message):
    morning = message.text.strip()
    connection = sqlite3.connect('tasks.sql')
    cursor = connection.cursor()
    cursor.execute("UPDATE users SET morning=? WHERE pass=?",
                   (morning, str(message.from_user.id)))
    connection.commit()
    cursor.close()
    connection.close()
    bot.send_message(
        message.chat.id, f'Регистрация прошла успешно! \nТвой уникальный ключ для входа: {message.from_user.id}')


class GrammarCorrector:
    def correct(self, text):
        try:
            blob = TextBlob(text)
            corrected_text = str(blob.correct())
        except NotTranslated:
            corrected_text = text
        return corrected_text


class Tokenizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
        self.fitted = False

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self.fitted = True

    def transform(self, text):
        if not self.fitted:
            raise ValueError("Tokenizer has not been fitted yet.")
        return self.vectorizer.transform([text]).toarray()[0]


class TaskDataset:
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.transform(text)
        return tokens, label


class ScheduleModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.model = Pipeline([
            ('vectorizer', CountVectorizer(tokenizer=nltk.word_tokenize)),
            ('classifier', MultinomialNB())
        ])

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


def generate_schedule(tasks, model, tokenizer, wake_up_time, sleep_time, meal_times):
    schedule = []
    current_time = wake_up_time

    meal_times.sort()

    if meal_times and current_time < meal_times[0][0]:
        current_time = meal_times[0][0] + timedelta(hours=1)

    free_slots = [(meal_times[i][0], meal_times[i + 1][0])
                  for i in range(len(meal_times) - 1)]
    free_slots.insert(0, (current_time, meal_times[0][0]))

    for start, end in free_slots:
        slot_duration = (end - start).seconds // 3600
        intervals = slot_duration // (len(tasks) + 1) if tasks else 0

        for i in range(len(tasks)):
            task = tasks.pop(0)
            start += timedelta(hours=intervals)
            schedule.append((start, task))

            if not tasks:
                break

    for meal_time in meal_times:
        schedule.append(meal_time)

    schedule.sort(key=lambda x: x[0])
    return schedule


@bot.message_handler(commands=['day'])
def start(message):
    user_data[message.chat.id] = {}
    bot.send_message(
        message.chat.id, "Привет! Я помогу тебе составить расписание на день. Введите время подъема (например, 7:00):")
    bot.register_next_step_handler(message, get_wake_up_time)


def get_wake_up_time(message):
    user_data[message.chat.id]['wake_up_time'] = message.text
    bot.send_message(
        message.chat.id, "Введите время, когда вы ложитесь спать (например, 23:00):")
    bot.register_next_step_handler(message, get_sleep_time)


def get_sleep_time(message):
    user_data[message.chat.id]['sleep_time'] = message.text
    bot.send_message(message.chat.id, "У вас будет завтрак? (да/нет):")
    bot.register_next_step_handler(message, get_breakfast_time)


def get_breakfast_time(message):
    meal_times = []
    if message.text.strip().lower() == 'да':
        bot.send_message(
            message.chat.id, "Введите время завтрака (например, 8:00):")
        bot.register_next_step_handler(message, lambda m: add_meal_time(
            m, meal_times, "завтрак", get_lunch_time))
    else:
        get_lunch_time(message)


def get_lunch_time(message):
    meal_times = user_data[message.chat.id].get('meal_times', [])
    bot.send_message(message.chat.id, "У вас будет обед? (да/нет):")
    bot.register_next_step_handler(message, lambda m: handle_meal_time(
        m, meal_times, "обед", get_dinner_time))


def get_dinner_time(message):
    meal_times = user_data[message.chat.id].get('meal_times', [])
    bot.send_message(message.chat.id, "У вас будет ужин? (да/нет):")
    bot.register_next_step_handler(
        message, lambda m: handle_meal_time(m, meal_times, "ужин", get_tasks))


def handle_meal_time(message, meal_times, meal_name, next_step):
    if message.text.strip().lower() == 'да':
        bot.send_message(
            message.chat.id, f"Введите время {meal_name}а (например, 12:00):")
        bot.register_next_step_handler(
            message, lambda m: add_meal_time(m, meal_times, meal_name, next_step))
    else:
        next_step(message)


def add_meal_time(message, meal_times, meal_name, next_step):
    meal_time_str = message.text
    meal_time = datetime.strptime(meal_time_str, "%H:%M").time()
    meal_times.append(
        (datetime.combine(datetime.today(), meal_time), meal_name))
    user_data[message.chat.id]['meal_times'] = meal_times
    next_step(message)


def get_tasks(message):
    bot.send_message(message.chat.id, "Введите ваши задачи на день:")
    bot.register_next_step_handler(message, generate_and_send_schedule)


def generate_and_send_schedule(message):
    tasks_text = message.text
    chat_id = message.chat.id

    wake_up_time_str = user_data[chat_id]['wake_up_time']
    sleep_time_str = user_data[chat_id]['sleep_time']
    meal_times = user_data[chat_id].get('meal_times', [])

    wake_up_time = datetime.strptime(wake_up_time_str, "%H:%M")
    sleep_time = datetime.strptime(sleep_time_str, "%H:%M")

    tasks = nltk.sent_tokenize(tasks_text)

    filtered_tasks = []
    for task in tasks:
        task = re.sub(r"(сегодня я хочу|а еще|также|сегодня я хотел бы|так же|еще| а | и |,а |,и |,я |сегодня| я |хотел|хочу| бы |еще я хотел бы|потом|следом|было бы неплохо|было бы хорошо|было бы славно)", "", task).strip()
        if task:
            filtered_tasks.extend(task.split(','))

    tokenizer = Tokenizer()
    tokenizer.fit(filtered_tasks)

    labels = list(range(len(filtered_tasks)))
    dataset = TaskDataset(filtered_tasks, labels, tokenizer)

    model = ScheduleModel(input_size=len(
        tokenizer.vectorizer.vocabulary_), hidden_size=128, output_size=24)

    X = filtered_tasks
    y = labels

    model.fit(X, y)

    grammar_corrector = GrammarCorrector()

    task_schedule = generate_schedule(
        filtered_tasks, model, tokenizer, wake_up_time, sleep_time, meal_times.copy())
    schedule = task_schedule
    schedule.sort()

    schedule_text = "Сгенерированное расписание:\n"
    for time_slot, task in schedule:
        corrected_task = grammar_corrector.correct(task.strip())
        schedule_text += f"{time_slot.strftime('%H:%M')} - {corrected_task}\n"

    bot.send_message(chat_id, schedule_text)


@bot.message_handler(commands=['night'])
def update_night_time_demo(message):
    if message.from_user.id in logged_in_users:
        bot.send_message(
            message.chat.id, 'Введи время, когда ты ложишься спать:')
        bot.register_next_step_handler(message, update_night_time)
    else:
        bot.send_message(message.chat.id, "Сначала войди в систему.")


def update_night_time(message):
    global night
    night = message.text.strip()
    connection = sqlite3.connect('tasks.sql')
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE users SET night=? WHERE pass=?", (night, str(message.from_user.id)))
    connection.commit()
    cursor.close()
    connection.close()
    bot.send_message(
        message.chat.id, 'Отлично, время, когда ты ложишься обновлено!')


@bot.message_handler(commands=['morning'])
def update_morning_time_demo(message):
    if message.from_user.id in logged_in_users:
        bot.send_message(
            message.chat.id, 'Введи время, когда ты встаешь:')
        bot.register_next_step_handler(message, update_morning_time)
    else:
        bot.send_message(message.chat.id, "Сначала войди в систему.")


def update_morning_time(message):
    morning = message.text.strip()
    connection = sqlite3.connect('tasks.sql')
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE users SET morning=? WHERE pass=?", (morning, str(message.from_user.id)))
    connection.commit()
    cursor.close()
    connection.close()
    bot.send_message(
        message.chat.id, 'Отличо, время, когда ты встаешь обновлено!')


@bot.message_handler(commands=['log'])
def log(message):
    user_id_ = message.from_user.id
    if user_id_ in logged_in_users:
        bot.send_message(message.chat.id, 'Ты уже вошел в систему.')
    else:
        bot.send_message(message.chat.id, '👤Введи свое имя для входа:')
        bot.register_next_step_handler(message, get_user_name)


def get_user_name(message):
    name = message.text.strip()
    user_id_ = message.from_user.id
    connection = sqlite3.connect('tasks.sql')
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE name=? AND pass=?",
                   (name, str(user_id_),))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    if user:
        logged_in_users[user_id_] = {
            'id': user[0], 'name': user[1], 'pass': user[2], 'timezone': user[3]}
        bot.send_message(message.chat.id, 'Секунду...')
        time.sleep(1.5)
        bot.send_message(message.chat.id, 'Ты успешно вошел в свой аккаунт!✅')
    else:
        bot.send_message(
            message.chat.id, 'Ошибка: это не твой аккаунт. Попробуй снова.❌')


@bot.message_handler(commands=['logout'])
def logout(message):
    user_id_ = message.from_user.id
    if user_id_ in logged_in_users:
        del logged_in_users[user_id_]
        bot.send_message(
            message.chat.id, 'Ты успешно вышел из своего аккаунта.✅')
    else:
        bot.send_message(message.chat.id, 'Ты не в системе.')


@bot.message_handler(commands=['users'])
def users(message):
    if message.from_user.id == 1458380959:
        connection = sqlite3.connect('tasks.sql')
        cursor = connection.cursor()
        cursor.execute("SELECT name, pass FROM users")
        users = cursor.fetchall()
        cursor.close()
        connection.close()
        if users:
            user_list = "Список пользователей:\n\n"
            for user in users:
                user_list += f"Имя: {user[0]}, Уникальный ключ: {user[1]}\n"
            bot.send_message(message.chat.id, user_list)
        else:
            bot.send_message(
                message.chat.id, 'В базе данных нет зарегистрированных пользователей.❌')
    else:
        bot.send_message(
            message.chat.id, 'У тебя нет прав на использование этой команды.❌')


@bot.message_handler(commands=['delete_user'])
def delete_user(message):
    if message.from_user.id == 1458380959:
        bot.send_message(
            message.chat.id, 'Введи уникальный ключ (user_id) пользователя, которого нужно удалить:')
        bot.register_next_step_handler(message, delete_user_by_id)
    else:
        bot.send_message(
            message.chat.id, 'У тебя нет прав на использование этой команды.❌')


def delete_user_by_id(message):
    user_id_to_delete = message.text.strip()
    connection = sqlite3.connect('tasks.sql')
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE pass=?", (user_id_to_delete,))
    user = cursor.fetchone()
    if user:
        cursor.execute("DELETE FROM users WHERE pass=?", (user_id_to_delete,))
        connection.commit()
        bot.send_message(
            message.chat.id, f'Пользователь с уникальным ключом {user_id_to_delete} был удален.✅')
    else:
        bot.send_message(
            message.chat.id, 'Пользователь с таким уникальным ключом не найден.❌')
    cursor.close()
    connection.close()


@bot.message_handler(commands=['timezone'])
def set_timezone(message):
    if message.from_user.id in logged_in_users:
        bot.send_message(
            message.chat.id, 'Введите ваш часовой пояс (например, UTC+3 или UTC-5):')
        bot.register_next_step_handler(message, save_timezone)
    else:
        bot.send_message(
            message.chat.id, 'Сначала войдите в систему командой /log')


def save_timezone(message):
    timezone = message.text.strip()
    if timezone in timezone_map:
        user_id_ = message.from_user.id
        connection = sqlite3.connect('tasks.sql')
        cursor = connection.cursor()
        cursor.execute("UPDATE users SET timezone=? WHERE pass=?",
                       (timezone, str(user_id_)))
        connection.commit()
        cursor.close()
        connection.close()
        bot.send_message(
            message.chat.id, f'Ваш часовой пояс установлен: {timezone} ✅')
    else:
        bot.send_message(
            message.chat.id, 'Некорректный формат часового пояса. Пожалуйста, попробуйте снова.❌')


@bot.message_handler(commands=['task_list'])
def task_list(message):
    user_id_ = message.from_user.id
    if user_id_ in logged_in_users:
        connection = sqlite3.connect('tasks.sql')
        cursor = connection.cursor()
        cursor.execute(
            "SELECT task, time, day_of_week, date FROM tasks WHERE user_id=?", (user_id_,))
        tasks = cursor.fetchall()
        cursor.close()
        connection.close()
        if tasks:
            task_list_msg = "📋 Список задач:\n\n"
            day_tasks = {}
            date_tasks = []
            for task in tasks:
                if task[3]:
                    date_tasks.append(
                        f"{task[0]} - {task[1]} (Дата: {task[3]})")
                elif task[2]:
                    day_tasks.setdefault(task[2], []).append(
                        f"{task[0]} - {task[1]}")
            for day, day_task_list in day_tasks.items():
                task_list_msg += f"📅 {day}:\n" + \
                    "\n".join(day_task_list) + "\n\n"
            if date_tasks:
                task_list_msg += "📅 Задачи на определенные даты:\n" + \
                    "\n".join(date_tasks) + "\n\n"
            bot.send_message(message.chat.id, task_list_msg)
        else:
            bot.send_message(message.chat.id, 'У вас нет задач.❌')
    else:
        bot.send_message(
            message.chat.id, 'Сначала войдите в систему командой /log')


def check_reminders():
    while True:
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_date = now.strftime("%d-%m-%Y")
        connection = sqlite3.connect('tasks.sql')
        cursor = connection.cursor()
        cursor.execute(
            "SELECT user_id, task, time, date FROM tasks WHERE date=? AND time=?", (current_date, current_time))
        tasks = cursor.fetchall()
        for task in tasks:
            user_id = task[0]
            task_description = task[1]
            bot.send_message(
                user_id, f"🚨🚨🚨 Напоминание: {task_description} 🚨🚨🚨")
        cursor.close()
        connection.close()
        time.sleep(60)


reminder_thread = threading.Thread(target=check_reminders)
reminder_thread.start()


@bot.message_handler(commands=['add_task_date'])
def add_task_date(message):
    if message.from_user.id in logged_in_users:
        bot.send_message(message.chat.id, 'Введите описание задачи:')
        bot.register_next_step_handler(message, get_task_description_date)
    else:
        bot.send_message(
            message.chat.id, 'Сначала войдите в систему командой /log')


def get_task_description_date(message):
    task_description = message.text.strip()
    user_id_ = message.from_user.id
    logged_in_users[user_id_]['task_description'] = task_description
    bot.send_message(
        message.chat.id, 'Введите дату для задачи (в формате ДД-ММ-ГГГГ):')
    bot.register_next_step_handler(message, get_task_date)


def validate_date(date_str):
    try:
        datetime.strptime(date_str, "%d-%m-%Y")
        return True
    except ValueError:
        return False


def get_task_date(message):
    task_date = message.text.strip()
    if validate_date(task_date):
        user_id_ = message.from_user.id
        logged_in_users[user_id_]['task_date'] = task_date
        bot.send_message(
            message.chat.id, 'Введите время для задачи (в формате ЧЧ:ММ):')
        bot.register_next_step_handler(message, get_task_time_date)
    else:
        bot.send_message(
            message.chat.id, 'Некорректный формат даты.❌ Попробуйте еще раз (в формате ДД-ММ-ГГГГ):')
        bot.register_next_step_handler(message, get_task_date)


def validate_time(time_str):
    try:
        datetime.strptime(time_str, "%H:%M")
        return True
    except ValueError:
        return False


def get_task_time_date(message):
    task_time = message.text.strip()
    if validate_time(task_time):
        user_id_ = message.from_user.id
        task_description = logged_in_users[user_id_].get('task_description')
        task_date = logged_in_users[user_id_].get('task_date')
        connection = sqlite3.connect('tasks.sql')
        cursor = connection.cursor()
        cursor.execute("INSERT INTO tasks (user_id, task, time, date) VALUES (?, ?, ?, ?)",
                       (user_id_, task_description, task_time, task_date))
        connection.commit()
        cursor.close()
        connection.close()
        bot.send_message(message.chat.id, 'Задача успешно добавлена!✅')
    else:
        bot.send_message(
            message.chat.id, 'Некорректный формат времени.❌ Попробуйте еще раз (в формате ЧЧ:ММ):')
        bot.register_next_step_handler(message, get_task_time_date)


@bot.message_handler(commands=['add_task'])
def add_task(message):
    if message.from_user.id in logged_in_users:
        bot.send_message(message.chat.id, 'Введите описание задачи:')
        bot.register_next_step_handler(message, get_task_description)
    else:
        bot.send_message(
            message.chat.id, 'Сначала войдите в систему командой /log')


def get_task_description(message):
    task_description = message.text.strip()
    user_id_ = message.from_user.id
    logged_in_users[user_id_]['task_description'] = task_description
    bot.send_message(
        message.chat.id, 'Введите время для задачи (в формате ЧЧ:ММ):')
    bot.register_next_step_handler(message, get_task_time)


def validate_time(time_str):
    try:
        datetime.strptime(time_str, "%H:%M")
        return True
    except ValueError:
        return False


def get_task_time(message):
    task_time = message.text.strip()
    if validate_time(task_time):
        user_id_ = message.from_user.id
        logged_in_users[user_id_]['task_time'] = task_time
        bot.send_message(
            message.chat.id, 'Введите день недели для задачи (например, Понедельник):')
        bot.register_next_step_handler(message, get_task_day_of_week)
    else:
        bot.send_message(
            message.chat.id, 'Некорректный формат времени.❌ Попробуйте еще раз (в формате ЧЧ:ММ):')
        bot.register_next_step_handler(message, get_task_time)


def validate_day_of_week(day_str):
    return day_str.capitalize() in days_of_week


def get_task_day_of_week(message):
    task_day_of_week = message.text.strip().capitalize()
    if validate_day_of_week(task_day_of_week):
        user_id_ = message.from_user.id
        task_description = logged_in_users[user_id_].get('task_description')
        task_time = logged_in_users[user_id_].get('task_time')
        connection = sqlite3.connect('tasks.sql')
        cursor = connection.cursor()
        cursor.execute("INSERT INTO tasks (user_id, task, time, day_of_week) VALUES (?, ?, ?, ?)",
                       (user_id_, task_description, task_time, task_day_of_week))
        connection.commit()
        cursor.close()
        connection.close()
        bot.send_message(message.chat.id, 'Задача успешно добавлена!✅')
    else:
        bot.send_message(
            message.chat.id, 'Некорректный день недели.❌ Пожалуйста, введите день недели заново:')
        bot.register_next_step_handler(message, get_task_day_of_week)


@bot.message_handler(commands=['delete_task'])
def delete_task(message):
    if message.from_user.id in logged_in_users:
        bot.send_message(
            message.chat.id, 'Введите название задачи, которую хотите удалить:')
        bot.register_next_step_handler(message, delete_task_by_name)
    else:
        bot.send_message(
            message.chat.id, 'Сначала войдите в систему командой /log')


def delete_task_by_name(message):
    task_name = message.text.strip()
    user_id_ = message.from_user.id
    connection = sqlite3.connect('tasks.sql')
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM tasks WHERE user_id=? AND task=?",
                   (str(user_id_), task_name))
    task = cursor.fetchone()
    if task:
        cursor.execute(
            "DELETE FROM tasks WHERE user_id=? AND task=?", (str(user_id_), task_name))
        connection.commit()
        bot.send_message(
            message.chat.id, f'Задача "{task_name}" была удалена.✅')
    else:
        bot.send_message(
            message.chat.id, 'Задача с таким названием не найдена.❌')
    cursor.close()
    connection.close()


@bot.message_handler(commands=['info'])
def user_info(message):
    user_id_ = message.from_user.id
    if user_id_ in logged_in_users:
        user = logged_in_users[user_id_]
        bot.send_message(
            message.chat.id, f'👤Ваш логин: {user["name"]}\n🔑Ваш уникальный ключ: {user["pass"]}\n⌚Ваш часовой пояс: {user["timezone"]}')
    else:
        bot.send_message(
            message.chat.id, 'Сначала войдите в систему командой /log')


"""
@bot.message_handler(commands=['test'])
def start(message):
    msg = bot.send_message(message.chat.id, 'введите что-то')
    bot.register_next_step_handler(msg, start_2)


def start_2(message):
    bot.edit_message_text(chat_id=message.chat.id,
                          message_id=message.message_id - 1, text='вы ввели ' + message.text)
"""


@bot.message_handler(commands=['clear_tasks'])
def clear_tasks(message):
    if message.from_user.id in logged_in_users:
        user_id_ = message.from_user.id
        connection = sqlite3.connect('tasks.sql')
        cursor = connection.cursor()
        cursor.execute("DELETE FROM tasks WHERE user_id=?", (str(user_id_),))
        connection.commit()
        cursor.close()
        connection.close()
        bot.send_message(message.chat.id, 'Все задачи удалены.✅')
    else:
        bot.send_message(
            message.chat.id, 'Сначала войдите в систему командой /log')


def send_task_reminders():
    day_map = {
        "Monday": "Понедельник",
        "Tuesday": "Вторник",
        "Wednesday": "Среда",
        "Thursday": "Четверг",
        "Friday": "Пятница",
        "Saturday": "Суббота",
        "Sunday": "Воскресенье"
    }

    while True:
        connection = sqlite3.connect('tasks.sql')
        cursor = connection.cursor()
        cursor.execute(
            "SELECT id, user_id, task, time, day_of_week FROM tasks")
        tasks = cursor.fetchall()
        cursor.close()
        connection.close()

        now_utc = datetime.now(pytz.utc)

        for task in tasks:
            task_id, user_id, task_description, task_time_str, task_day_of_week = task
            user_timezone = pytz.timezone(get_user_timezone(user_id))
            now_user = now_utc.astimezone(user_timezone)
            current_day_of_week = day_map[now_user.strftime('%A')]

            if current_day_of_week == task_day_of_week:
                task_time = datetime.strptime(
                    task_time_str, '%H:%M').time()
                if now_user.time() >= task_time and (now_user - timedelta(minutes=1)).time() < task_time:
                    bot.send_message(
                        user_id, f'🚨🚨🚨 Напоминание: {task_description} 🚨🚨🚨')
                    time.sleep(60)


def start_thread():
    reminder_thread = threading.Thread(target=send_task_reminders)
    reminder_thread.daemon = True
    reminder_thread.start()


start_thread()


@bot.message_handler(commands=['start'])
def handle_start(message):
    keyboard = types.ReplyKeyboardMarkup(row_width=2)
    button2 = types.KeyboardButton('Мой создатель')
    button3 = types.KeyboardButton('Меню')
    keyboard.add(button2, button3)

    bot.reply_to(message, 'Привет!\n\nЯ бот, который поможет тебе распоряжаться своим временем. \n\nС помощью меня ты сможешь поставить себе напоминания, запланировать какие-либо события и многое другое. \nПриятного использования.', reply_markup=keyboard)


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    if message.text == 'Мой создатель':
        bot.reply_to(
            message, 'Мой создатель: @leonid_baxmut \nВы можете написать ему с любым вопросом :)')
    elif message.text == "Меню":
        smile = "🛠️"
        bot.send_message(
            message.chat.id, f"🔑РЕГИСТРАЦИЯ🔑\nЗарегистрируйся по команде /reg\nУже зарегистрировался? Тогда /log - войти в свой аккаунт\n/logout - выйти из своего аккаунта\n\n{smile}ФУНКЦИОНАЛ{smile}\n/info - получить информацию о своем профиле\n/timezone - поменять/посмотреть твой часовой пояс.\n/task_list - список задач с возможностью выставить каждой из них время напоминания.\n/add_task - добавить новую задачу.\n/delete_task - удалить задачу.\n/clear_tasks - очистить все задачи.\n/day - система умных заметок(на стадии разработки)\n/add_task_date - добавить задачу по дате.")
    else:
        bot.reply_to(message, 'Напиши что-нибудь другое.')


bot.polling(none_stop=True)
