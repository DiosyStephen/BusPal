import os
import json
from google.oauth2 import service_account
import vertexai

# Load JSON string from env variable
key_json = os.environ["GOOGLE_CREDENTIALS"]

# Convert JSON string into Python dict
key_dict = json.loads(key_json)

# Create credentials object
credentials = service_account.Credentials.from_service_account_info(key_dict)

# Initialize Vertex AI
vertexai.init(
    project="bus-fare-predictor-ml",
    location="us-central1",
    credentials=credentials
)

#!/usr/bin/env python3
"""
routAfare_bot_fixed.py
A more-complete, fixed version of the user's Telegram + Flask webhook bot
Includes:
 - CSV-based bus loading
 - session-based conversational flow (route -> age -> traffic -> time -> fare -> select bus)
 - optional Vertex AI predictor integration (graceful fallback)
 - safe JSON persistence helpers
 - webhook setup for Telegram

Before running:
 - Create a .env with BOT_TOKEN, ADMIN_CHAT_ID (optional), WEBHOOK_URL_BASE, VERTEX_AI_* if used
 - Install dependencies: python-dotenv, flask, pyTelegramBotAPI (telebot), pandas, google-cloud-aiplatform (optional)

Note: adjust CSV_FILE_NAME to your actual CSV file placed next to this script.
"""

import os
import json
import re
import sys
import time
from datetime import datetime, timedelta

# Third-party libraries
try:
    from dotenv import load_dotenv
    from flask import Flask, request, jsonify
    import telebot
    from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
    # Vertex optional
    try:
        from google.cloud import aiplatform
    except Exception:
        aiplatform = None
    import pandas as pd
except ImportError as e:
    print(f"CRITICAL: Missing required package. Please install dependencies. Error: {e}")
    sys.exit(1)

# --- Configuration & Initialization ---
load_dotenv()

# Environment Variables (use correct env keys)
BOT_TOKEN = os.getenv('BOT_TOKEN')
ADMIN_CHAT_ID = os.getenv('ADMIN_CHAT_ID')
WEBHOOK_URL_BASE = os.getenv('WEBHOOK_URL_BASE')
WEBHOOK_URL_PATH = os.getenv('WEBHOOK_URL_PATH', '/')
SERVER_PORT = int(os.getenv('PORT', 8080))

if not BOT_TOKEN or not WEBHOOK_URL_BASE:
    print('CRITICAL: Missing BOT_TOKEN or WEBHOOK_URL_BASE environment variables.')
    # Not exiting so the user can run non-webhook tests locally, but warn
    # sys.exit(1)

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBSCRIBERS_FILE = os.path.join(BASE_DIR, 'subscribers.json')
SESSIONS_FILE = os.path.join(BASE_DIR, 'sessions.json')
BOOKINGS_FILE = os.path.join(BASE_DIR, 'bookings.json')
CSV_FILE_NAME = os.getenv('CSV_FILE_NAME', 'final routa dataset for bus routes.csv')
CSV_FILE_PATH = os.path.join(BASE_DIR, CSV_FILE_NAME)

# --- Utility Functions for Data Persistence ---

def safe_write_json(file_path, data):
    try:
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")


def safe_read_json(file_path, fallback):
    if not os.path.exists(file_path):
        return fallback
    try:
        with open(file_path, 'r', encoding='utf8') as f:
            content = f.read().strip()
            if not content:
                return fallback
            return json.loads(content)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return fallback

# Load state stores
store = safe_read_json(SUBSCRIBERS_FILE, {'chats': []})
sessions = safe_read_json(SESSIONS_FILE, {})
bookings = safe_read_json(BOOKINGS_FILE, [])

# Helpers to persist data
def save_store(): safe_write_json(SUBSCRIBERS_FILE, store)
def save_sessions(): safe_write_json(SESSIONS_FILE, sessions)
def save_bookings(): safe_write_json(BOOKINGS_FILE, bookings)

def clear_session(chat_id):
    s_k = str(chat_id)
    if s_k in sessions:
        del sessions[s_k]
        save_sessions()

# --- Time helper ---

def time_to_minutes(t: str):
    m = re.match(r'^(\d{1,2}):(\d{2})$', t)
    if not m: return None
    try:
        h = int(m.group(1))
        mm = int(m.group(2))
        return h * 60 + mm
    except ValueError:
        return None

# --- CSV Data Processing ---

def generate_departure_times(time_slot, interval_minutes=60):
    """Converts a time slot like '12:00-14:59' into discrete HH:MM departure times."""
    m = re.match(r'(\d{1,2}:\d{2})-(\d{1,2}:\d{2})', str(time_slot))
    if not m:
        return []

    start_str, end_str = m.groups()
    try:
        start_time = datetime.strptime(start_str, '%H:%M')
        end_time = datetime.strptime(end_str, '%H:%M')
    except ValueError:
        return []

    times = []
    current_time = start_time
    while current_time <= end_time:
        times.append(current_time.strftime('%H:%M'))
        current_time += timedelta(minutes=interval_minutes)

    return times


def load_bus_data(csv_file_path):
    global ROUTE_NAMES
    print(f"Loading bus data from {csv_file_path}...")

    if not os.path.exists(csv_file_path):
        print(f"WARNING: CSV file not found at {csv_file_path} - continuing with empty buses list")
        return []

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"CRITICAL: Failed to read CSV file: {e}")
        return []

    # Define columns used to uniquely identify a service. Adjust if your CSV differs.
    unique_bus_cols = [c for c in ['route_id', 'bus_route', 'bus_type_num', 'direction'] if c in df.columns]

    if not unique_bus_cols:
        print("CRITICAL: CSV missing expected columns like 'route_id' or 'bus_route'.")
        return []

    df_unique = df.groupby(unique_bus_cols).agg(
        times=('time_slot', lambda x: list(x.dropna().unique()))
    ).reset_index()

    bus_data_list = []
    bus_id_counter = 1

    for _, row in df_unique.iterrows():
        all_times = set()
        for slot in row['times']:
            all_times.update(generate_departure_times(slot, interval_minutes=60))

        if not all_times:
            continue

        sorted_times = sorted(list(all_times), key=lambda s: time_to_minutes(s))

        bus_data_list.append({
            'id': f'BUS-{bus_id_counter}',
            'route_id': row.get('route_id'),
            'name': row.get('bus_route') if 'bus_route' in row else str(row.get('route_id')),
            'bus_type_num': int(row.get('bus_type_num')) if 'bus_type_num' in row and pd.notna(row.get('bus_type_num')) else 1,
            'capacity': 50,
            'times': sorted_times
        })
        bus_id_counter += 1

    print(f"Successfully processed {len(bus_data_list)} unique bus services.")
    return bus_data_list

# Load buses (non-fatal if empty)
buses = load_bus_data(CSV_FILE_PATH)
ROUTE_NAMES = sorted(list({b.get('name') for b in buses if b.get('name')}))
AGE_GROUPS = {"Child (0-12)": 0, "Teenager (13-19)": 1, "Adult (20-59)": 2, "Senior (60+)": 3}
TRAFFIC_LEVELS = {"Low (1)": 1, "Medium (2)": 2, "High (3)": 3}

# --- Vertex AI Initialization (optional) ---
VERTEX_CONFIG = {
    'endpoint_id': os.getenv('VERTEX_AI_ENDPOINT_ID'),
    'project': os.getenv('VERTEX_AI_PROJECT'),
    'location': os.getenv('VERTEX_AI_LOCATION')
}

vertex_predictor = None
if aiplatform and all(VERTEX_CONFIG.values()):
    try:
        aiplatform.init(project=VERTEX_CONFIG['project'], location=VERTEX_CONFIG['location'])
        endpoint_name = VERTEX_CONFIG['endpoint_id'].split('/')[-1]
        vertex_predictor = aiplatform.Endpoint(endpoint_name=endpoint_name)
        print(f"✅ Vertex AI Predictor initialized for endpoint: {endpoint_name}")
    except Exception as e:
        print(f"Vertex AI initialization failed: {e}")
        vertex_predictor = None
else:
    print("Vertex AI not configured or not installed. Predictions will not be available.")

# --- Bot Initialization & Core Functions ---
if not BOT_TOKEN:
    print('WARNING: BOT_TOKEN not set; bot will not connect to Telegram but you can still run logic locally.')

bot = telebot.TeleBot(BOT_TOKEN, threaded=False) if BOT_TOKEN else None


def find_buses_by_route_name(route_name, all_buses):
    return [bus for bus in all_buses if bus.get('name') == route_name]


def find_bus_by_id(bus_id, all_buses):
    return next((bus for bus in all_buses if bus.get('id') == bus_id), None)


def get_fare_prediction_safe(data, predictor):
    if not predictor:
        # fallback: simple heuristic fare calculation when Vertex is unavailable
        try:
            distance_km = float(data.get('distance_km', 1.0))
        except Exception:
            distance_km = 1.0
        base = 20.0
        per_km = 5.0
        traffic_mult = 1.0 + (0.1 * int(data.get('traffic_level_num', 1)))
        fare = max(5.0, round((base + distance_km * per_km) * traffic_mult, 2))
        print(f"Using fallback heuristic fare: Rs. {fare}")
        return fare

    try:
        distance_km = float(data.get('distance_km', 1.0))
        bus_type_num = int(data.get('bus_type_num', 1))
        age_group_num = int(data.get('age_group_num', 2))
        traffic_level_num = int(data.get('traffic_level_num', 1))

        instance = [{
            'distance_km': distance_km,
            'bus_type_num': bus_type_num,
            'age_group_num': age_group_num,
            'traffic_level_num': traffic_level_num
        }]

        print(f"Submitting instance to Vertex AI: {instance[0]}")
        response = predictor.predict(instances=instance)
        predicted_fare = response.predictions[0]
        final_fare = max(5.0, round(float(predicted_fare), 2))
        print(f"Vertex AI Prediction successful: Rs. {final_fare}")
        return final_fare
    except Exception as e:
        raise RuntimeError(f"Vertex AI Prediction failed: {e}")

# --- Message and Callback Handlers ---
@bot.message_handler(commands=['start', 'help'])
def handle_commands(message):
    bot.send_message(message.chat.id, "Welcome! Type *ser* or *search* to start.", parse_mode='Markdown')


@bot.message_handler(func=lambda msg: True, content_types=['text'])
def handle_text_message(message):
    if bot is None:
        return

    chat_id = message.chat.id
    text = (message.text or '').strip()
    lc = text.lower()
    str_chat_id = str(chat_id)

    if lc == 'cancel':
        clear_session(chat_id)
        bot.send_message(chat_id, 'Flow *cancelled*. Type *ser* to start a new search.', parse_mode='Markdown')
        return

    session = sessions.get(str_chat_id, {'step': None, 'data': {}, 'user': {'username': message.from_user.username, 'first_name': message.from_user.first_name}})

    # --- Start flow ---
    if not session['step'] and lc in ('ser', 'search'):
        sessions[str_chat_id] = {
            'step': 'await_route_name',
            'data': {},
            'user': {'username': message.from_user.username, 'first_name': message.from_user.first_name}
        }
        save_sessions()

        if not ROUTE_NAMES:
            bot.send_message(chat_id, "No routes are loaded on the server. Please contact the admin.")
            return

        keyboard = InlineKeyboardMarkup()
        for r in ROUTE_NAMES:
            keyboard.add(InlineKeyboardButton(r, callback_data=f"route_{r}"))

        bot.send_message(chat_id, "🔍 Select a *route*: ", reply_markup=keyboard, parse_mode='Markdown')
        return

    # --- Await time ---
    if session.get('step') == 'await_time':
        # Validate time in HH:MM
        m = re.match(r'^([0-1]?\d|2[0-3]):([0-5]\d)$', text)
        if not m:
            bot.send_message(chat_id, "❌ Invalid time format. Use HH:MM (e.g., 13:45).")
            return

        # Save time
        sessions[str_chat_id]['data']['time'] = text
        save_sessions()

        bot.send_message(chat_id, 'Calculating fare and searching for matching buses... 🔎')
        s_data = sessions[str_chat_id]['data']

        # Provide a default distance_km if not present (in a real flow you'd ask)
        if 'distance_km' not in s_data:
            s_data['distance_km'] = 5.0

        # Fare prediction
        try:
            predicted_fare = get_fare_prediction_safe(s_data, vertex_predictor)
            s_data['predicted_fare'] = predicted_fare
            save_sessions()
        except RuntimeError as e:
            clear_session(chat_id)
            bot.send_message(chat_id, f"❌ *Prediction Error:* {e}\n\nThe external fare calculation service failed. Please try again later.", parse_mode='Markdown')
            print(f"Search aborted due to Prediction Error: {e}")
            return

        # Bus search using route name and time
        route_name = s_data.get('selected_route')
        time_input = s_data.get('time')

        matching = [bus for bus in buses if bus.get('name') == route_name and time_input in bus.get('times', [])]

        if not matching:
            bot.send_message(chat_id, "❌ No buses found for that route/time.")
            clear_session(chat_id)
            return

        keyboard = InlineKeyboardMarkup()
        for bus in matching:
            keyboard.add(InlineKeyboardButton(f"{bus['id']} — departures: {', '.join(bus['times'][:3])}...", callback_data=f"bus_{bus['id']}"))

        bot.send_message(chat_id, f"🚌 *Predicted Fare:* Rs. {s_data['predicted_fare']}\n\nSelect a bus:", parse_mode='Markdown', reply_markup=keyboard)
        sessions[str_chat_id]['step'] = 'await_bus_select'
        save_sessions()
        return

    # Default fallback message
    bot.send_message(chat_id, "Type *ser* to start a route search or *cancel* to exit.", parse_mode='Markdown')


@bot.callback_query_handler(func=lambda call: True)
def handle_callback_query(call):
    if bot is None:
        return

    data = call.data or ''
    chat_id = call.message.chat.id
    str_chat_id = str(chat_id)

    # Route selection
    if data.startswith('route_'):
        route_name = data.replace('route_', '', 1)
        sessions.setdefault(str_chat_id, {'step': None, 'data': {}, 'user': {}})
        sessions[str_chat_id]['data']['selected_route'] = route_name
        sessions[str_chat_id]['step'] = 'await_age'
        save_sessions()

        keyboard = InlineKeyboardMarkup()
        for label, num in AGE_GROUPS.items():
            keyboard.add(InlineKeyboardButton(label, callback_data=f"age_{num}"))

        bot.edit_message_text('👤 Select *age group*:', chat_id, call.message.message_id, reply_markup=keyboard, parse_mode='Markdown')
        bot.answer_callback_query(call.id)
        return

    # Age selection
    if data.startswith('age_'):
        age_group = int(data.replace('age_', '', 1))
        sessions.setdefault(str_chat_id, {'step': None, 'data': {}, 'user': {}})
        sessions[str_chat_id]['data']['age_group_num'] = age_group
        sessions[str_chat_id]['step'] = 'await_traffic'
        save_sessions()

        keyboard = InlineKeyboardMarkup()
        for label, num in TRAFFIC_LEVELS.items():
            keyboard.add(InlineKeyboardButton(label, callback_data=f"traffic_{num}"))

        bot.edit_message_text('🚦 Select *traffic level*:', chat_id, call.message.message_id, reply_markup=keyboard, parse_mode='Markdown')
        bot.answer_callback_query(call.id)
        return

    # Traffic selection
    if data.startswith('traffic_'):
        traffic_level = int(data.replace('traffic_', '', 1))
        sessions.setdefault(str_chat_id, {'step': None, 'data': {}, 'user': {}})
        sessions[str_chat_id]['data']['traffic_level_num'] = traffic_level
        sessions[str_chat_id]['step'] = 'await_time'
        save_sessions()

        bot.edit_message_text("⏰ Enter your time in *HH:MM* (example: 13:45):", chat_id, call.message.message_id, parse_mode='Markdown')
        bot.answer_callback_query(call.id)
        return

    # Bus selection
    if data.startswith('bus_'):
        bus_id = data.replace('bus_', '', 1)
        sessions.setdefault(str_chat_id, {'step': None, 'data': {}, 'user': {}})
        sessions[str_chat_id]['data']['selected_bus'] = bus_id
        sessions[str_chat_id]['step'] = 'done'
        save_sessions()

        bus = find_bus_by_id(bus_id, buses)
        fare = sessions[str_chat_id]['data'].get('predicted_fare', 'N/A')
        bot.edit_message_text(f"🎉 Your bus *{bus_id}* is confirmed.\n\nFare: Rs. {fare}", chat_id, call.message.message_id, parse_mode='Markdown')
        clear_session(chat_id)
        bot.answer_callback_query(call.id)
        return

    # Generic handler
    bot.answer_callback_query(call.id, text='Processing...')

# --- Flask Server for Webhook ---
app = Flask(__name__)

# Run bot webhook initialization once at app startup (Flask 3-safe)
with app.app_context():
    print("⚡ Initializing Telegram webhook (Render/Gunicorn mode)")
    set_initial_webhook()


@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return '', 200
    return 'Unsupported media type', 415

@app.route('/', methods=['GET'])
def index():
    return 'routAfare Bot Service is Running', 200

# --- Deployment ---

def set_initial_webhook():
    if bot is None:
        print('Bot not initialized — skipping webhook setup')
        return

    full_webhook_url = f"{WEBHOOK_URL_BASE}{WEBHOOK_URL_PATH}"
    try:
        bot.remove_webhook()
        time.sleep(0.1)
        bot.set_webhook(url=full_webhook_url)
        print(f"✅ Telegram Webhook set to: {full_webhook_url}")
    except Exception as e:
        print(f"FATAL: Failed to set webhook. Error: {e}")
        # Don't exit — let hosting platform decide

if __name__ == '__main__':
    print('Starting routAfare Bot...')
    set_initial_webhook()

    # For local testing, start Flask directly (uncomment when testing locally):
    # app.run(host='0.0.0.0', port=SERVER_PORT)

    # If running under Gunicorn/Render, the Procfile should start the app (e.g. gunicorn "routAfare_bot_fixed:app")
    print('Ready. If running locally uncomment app.run(...) in __main__.')


