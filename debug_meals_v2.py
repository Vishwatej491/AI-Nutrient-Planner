import sqlite3
import os
import json
import uuid
from datetime import datetime
import sys

# Redirect output to file
log_file = open("debug_log.txt", "w")
sys.stdout = log_file

DB_PATH = os.path.join('data', 'nutrition.db')
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 1. Get a user
cursor.execute('SELECT id, email FROM users LIMIT 1')
user = cursor.fetchone()
if not user:
    print("No users found.")
    log_file.close()
    exit()

user_id = user['id']
print(f"Testing for User: {user['email']} ({user_id})")

# 2. Insert a dummy meal using Python timestamp
log_id = str(uuid.uuid4())
ts = datetime.now()
print(f"Inserting meal at Python time: {ts} (ISO: {ts.isoformat()})")

cursor.execute("""
    INSERT INTO meal_logs (id, user_id, food_name, nutrition, source, confidence, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", (
    log_id,
    user_id,
    "Debug Apple",
    json.dumps({"calories": 95}),
    "debug",
    1.0,
    ts.isoformat()
))
conn.commit()

# 3. Try key retrieval methods

# Method A: The current implementation (date('now', 'localtime'))
print("\n--- Method A: date('now', 'localtime') ---")
cursor.execute("""
    SELECT count(*) as count FROM meal_logs 
    WHERE user_id = ? 
    AND date(timestamp) = date('now', 'localtime')
""", (user_id,))
count_a = cursor.fetchone()['count']
print(f"Found: {count_a}")

# Method B: date('now') (UTC?)
print("\n--- Method B: date('now') ---")
cursor.execute("""
    SELECT count(*) as count FROM meal_logs 
    WHERE user_id = ? 
    AND date(timestamp) = date('now')
""", (user_id,))
count_b = cursor.fetchone()['count']
print(f"Found: {count_b}")

# Method C: Filter by Python string prefixes (Robust)
print("\n--- Method C: Python Date Prefix ---")
today_prefix = ts.date().isoformat()
print(f"Searching for prefix: {today_prefix}")
cursor.execute("""
    SELECT count(*) as count FROM meal_logs 
    WHERE user_id = ? 
    AND timestamp LIKE ?
""", (user_id, f"{today_prefix}%"))
count_c = cursor.fetchone()['count']
print(f"Found: {count_c}")

# Check what the DB actually thinks date(timestamp) is
cursor.execute("SELECT timestamp, date(timestamp) as db_date, date('now') as now_utc, date('now', 'localtime') as now_local FROM meal_logs WHERE id=?", (log_id,))
row = cursor.fetchone()
if row:
    print("\n--- DB Inspection ---")
    print(f"Stored Timestamp: {row['timestamp']}")
    print(f"DB date(timestamp): {row['db_date']}")
    print(f"DB date('now'): {row['now_utc']}")
    print(f"DB date('now', 'localtime'): {row['now_local']}")

# Clean up
cursor.execute("DELETE FROM meal_logs WHERE id=?", (log_id,))
conn.commit()
conn.close()
log_file.close()
