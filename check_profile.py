import sqlite3
import os
import json

DB_PATH = os.path.join('data', 'nutrition.db')
if not os.path.exists(DB_PATH):
    print(f"Error: {DB_PATH} not found")
    exit(1)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

try:
    cursor.execute('SELECT * FROM medical_profiles ORDER BY created_at DESC LIMIT 1')
    row = cursor.fetchone()
    if row:
        print(json.dumps(dict(row), indent=2))
    else:
        print("No profile found in medical_profiles table")
except sqlite3.OperationalError as e:
    print(f"Error: {e}")

conn.close()
