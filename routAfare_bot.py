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
    from google.cloud import aiplatform
    import pandas as pd # <-- New dependency
except ImportError as e:
    print(f"CRITICAL: Missing required package. Please install all dependencies listed in requirements.txt. Error: {e}")
    sys.exit(1)


# --- Configuration & Initialization ---
load_dotenv()

# Environment Variables
BOT_TOKEN = os.getenv('BOT_TOKEN')
ADMIN_CHAT_ID = os.getenv('ADMIN_CHAT_ID')
WEBHOOK_URL_BASE = os.getenv('WEBHOOK_URL_BASE')
WEBHOOK_URL_PATH = os.getenv('WEBHOOK_URL_PATH', '/')
SERVER_PORT = int(os.getenv('PORT', 8080))

if not BOT_TOKEN or not WEBHOOK_URL_BASE:
    print('CRITICAL: Missing BOT_TOKEN or WEBHOOK_URL_BASE environment variables.')
    sys.exit(1)

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBSCRIBERS_FILE = os.path.join(BASE_DIR, 'subscribers.json')
SESSIONS_FILE = os.path.join(BASE_DIR, 'sessions.json')
BOOKINGS_FILE = os.path.join(BASE_DIR, 'bookings.json')
CSV_FILE_NAME = 'final routa dataset for bus routes.csv' # <-- CSV file name

# --- Utility Functions for Data Persistence ---

def safe_write_json(file_path, data):
    """Writes data to a file as formatted JSON."""
    try:
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

def safe_read_json(file_path, fallback):
    """Reads JSON from a file, returning fallback on error or if file is empty."""
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

# Load state stores (these will be stored on the host's ephemeral filesystem)
store = safe_read_json(SUBSCRIBERS_FILE, {'chats': []})
sessions = safe_read_json(SESSIONS_FILE, {})
bookings = safe_read_json(BOOKINGS_FILE, [])
buses = [] # Will be populated from CSV below

# Helpers to persist data
def save_store(): safe_write_json(SUBSCRIBERS_FILE, store)
def save_sessions(): safe_write_json(SESSIONS_FILE, sessions)
def save_bookings(): safe_write_json(BOOKINGS_FILE, bookings)

def clear_session(chat_id):
    if str(chat_id) in sessions:
        del sessions[str(chat_id)]
        save_sessions()

def time_to_minutes(t: str):
    """Converts 'HH:MM' string to total minutes from midnight."""
    m = re.match(r'^(\d{1,2}):(\d{2})$', t)
    if not m: return None
    try:
        h = int(m.group(1))
        mm = int(m.group(2))
        return h * 60 + mm
    except ValueError:
        return None

# --- CSV Data Processing (Replaces buses.json logic) ---

def generate_departure_times(time_slot, interval_minutes=60):
    """Converts a time slot range (e.g., '12:00-14:59') into discrete HH:MM times."""
    import re
    m = re.match(r'(\d{1,2}:\d{2})-(\d{1,2}:\d{2})', time_slot)
    if not m: return []

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
    
    return [t for t in times if t <= end_str]


def load_bus_data(csv_file_path):
    """Loads and transforms CSV data into the 'buses' list structure."""
    global buses, ROUTE_NAMES
    print(f"Loading bus data from {csv_file_path}...")
    
    if not os.path.exists(csv_file_path):
        print(f"CRITICAL: CSV file not found at {csv_file_path}")
        return []
        
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"CRITICAL: Failed to read CSV file: {e}")
        return []

    # Columns used for identifying a unique bus service and its time slots
    unique_bus_cols = ['route_id', 'bus_route', 'bus_type_num', 'direction']
    
    # Group by unique service attributes and aggregate unique time slots
    df_unique = df.groupby(unique_bus_cols).agg(
        times=('time_slot', lambda x: list(x.unique()))
    ).reset_index()

    bus_data_list = []
    bus_id_counter = 1
    
    for _, row in df_unique.iterrows():
        all_times = set()
        for slot in row['times']:
            # Generate departure times (assuming 60 min interval)
            all_times.update(generate_departure_times(slot, interval_minutes=60)) 
        
        if not all_times:
            continue
        
        # Sort times and define bus object structure
        sorted_times = sorted(list(all_times))
        
        bus_data_list.append({
            "id": f"BUS-{bus_id_counter}",
            "route_id": row['route_id'],
            "name": row['bus_route'],
            "bus_type_num": row['bus_type_num'],
            "capacity": 50, # Set default capacity
            "times": sorted_times
        })
        bus_id_counter += 1
        
    print(f"Successfully processed {len(bus_data_list)} unique bus services.")
    return bus_data_list

# Load the data right after defining the function
buses = load_bus_data(os.path.join(BASE_DIR, CSV_FILE_NAME))
if not buses:
    sys.exit(1)

ROUTE_NAMES = sorted(list({b.get('name') for b in buses if b.get('name')}))
AGE_GROUPS = {"Child (0-12)": 0, "Teenager (13-19)": 1, "Adult (20-59)": 2, "Senior (60+)": 3}
TRAFFIC_LEVELS = {"Low (1)": 1, "Medium (2)": 2, "High (3)": 3}

# --- Vertex AI Initialization ---
VERTEX_CONFIG = {
    'endpoint_id': os.getenv('VERTEX_AI_ENDPOINT_ID'),
    'project': os.getenv('VERTEX_AI_PROJECT'),
    'location': os.getenv('VERTEX_AI_LOCATION')
}

vertex_predictor = None
if all(VERTEX_CONFIG.values()):
    try:
        aiplatform.init(
            project=VERTEX_CONFIG['project'],
            location=VERTEX_CONFIG['location']
        )
        endpoint_name = VERTEX_CONFIG['endpoint_id'].split('/')[-1]
        vertex_predictor = aiplatform.Endpoint(endpoint_name=endpoint_name)
        print(f"✅ Vertex AI Predictor initialized for endpoint: {endpoint_name}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Vertex AI client/endpoint. Error: {e}")
        vertex_predictor = None
else:
    print("CRITICAL ERROR: Missing Vertex AI config. Fare prediction will fail.")

# --- Bot Initialization & Core Functions ---
bot = telebot.TeleBot(BOT_TOKEN, threaded=False)

def find_buses_by_route_id(route_id, all_buses):
    return [bus for bus in all_buses if bus.get('route_id') == route_id]

def find_bus_by_id(bus_id, all_buses):
    return next((bus for bus in all_buses if bus.get('id') == bus_id), None)

def get_fare_prediction_safe(data, predictor):
    """
    Core business logic using Vertex AI. Raises RuntimeError on failure.
    (Function body remains identical to the last safe version)
    """
    if not predictor:
        raise RuntimeError("Vertex AI Predictor is not configured or failed to initialize.")

    try:
        distance_km = float(data.get('distance_km', 1.0))
        bus_type_num = int(data.get('bus_type_num', 1))
        age_group_num = int(data.get('age_group_num', 2))
        traffic_level_num = int(data.get('traffic_level_num', 1))

        instance = [{
            "distance_km": distance_km,
            "bus_type_num": bus_type_num,
            "age_group_num": age_group_num,
            "traffic_level_num": traffic_level_num
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
# NOTE: The handlers for /start, handle_text_message, and handle_callback_query
# are highly dependent on the complete state machine logic.
# For conciseness, we include only the critical fare prediction section here.

@bot.message_handler(commands=['start', 'help'])
def handle_commands(message):
    bot.send_message(message.chat.id, "Welcome! Type **ser** to start a search.", parse_mode='Markdown')

@bot.message_handler(func=lambda msg: True, content_types=['text'])
def handle_text_message(message):
    chat_id = message.chat.id
    text = message.text.strip()
    lc = text.lower()
    str_chat_id = str(chat_id)

    if lc == 'cancel':
        clear_session(chat_id)
        bot.send_message(chat_id, 'Flow **cancelled**. Type **ser** to start a new search.', parse_mode='Markdown')
        return

    session = sessions.get(str_chat_id, {'step': None, 'data': {}, 'user': {'username': message.from_user.username, 'first_name': message.from_user.first_name}})

    if not session['step'] and lc in ('ser', 'search'):
        # Start flow: ask for route name
        # (Omitted: logic to ask for route name and set session['step'] = 'await_route_name')
        pass # Placeholder for start logic

    if session['step'] == 'await_time':
        # ... (Time format validation logic)
        
        # ... (Session update logic)

        bot.send_message(chat_id, 'Calculating fare and searching for matching buses... 🔎')
        s_data = session['data']

        # CORE: Fare prediction with strict error handling
        try:
            predicted_fare = get_fare_prediction_safe(s_data, vertex_predictor)
            s_data['predicted_fare'] = predicted_fare
        except RuntimeError as e:
            clear_session(chat_id)
            bot.send_message(chat_id, f"❌ **Prediction Error:** {e}\n\n"
                                     "The external fare calculation service failed. Please try again later.",
                                     parse_mode='Markdown')
            print(f"Search aborted due to Prediction Error: {e}")
            return

        # Bus search and result display logic follows here...
        # (Omitted: logic to find matches, build final markup, and send result)
        bot.send_message(chat_id, "Search complete. Proceed via inline buttons.")
        return
    
    # (Omitted: logic for other steps and default fallback)

@bot.callback_query_handler(func=lambda call: True)
def handle_callback_query(call):
    # (Omitted: full callback logic for select_route, select_age, select_traffic, select, confirm, cancel)
    bot.answer_callback_query(call.id, text='Processing...')


# --- Flask Server for Webhook ---
app = Flask(__name__)

@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def webhook():
    """Endpoint for Telegram to send updates."""
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return '', 200
    return 'Unsupported media type', 415

@app.route('/', methods=['GET'])
def index():
    """Simple status check for the hosting platform."""
    return 'Busly Bot Service is Running', 200

# --- Deployment ---

def set_initial_webhook():
    """Sets the Telegram Webhook on startup."""
    full_webhook_url = f"{WEBHOOK_URL_BASE}{WEBHOOK_URL_PATH}"
    bot.remove_webhook()
    time.sleep(0.1)
    try:
        bot.set_webhook(url=full_webhook_url)
        print(f"✅ Telegram Webhook set to: {full_webhook_url}")
    except Exception as e:
        print(f"FATAL: Failed to set webhook. Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    print("Starting Busly Bot...")
    set_initial_webhook()
    
    # Gunicorn (used by Render) will run the Flask app via the Procfile command
    # For local testing, you can uncomment this:
    # app.run(host='0.0.0.0', port=SERVER_PORT)
