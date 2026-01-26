import sqlite3
import os
import sys
import uuid
import json
from datetime import datetime

# Set output to file to ensure we see it
sys.stdout = open('db_diag_log.txt', 'w')
sys.stderr = sys.stdout

DB_PATH = os.path.join('data', 'nutrition.db')
print(f"Checking DB at: {os.path.abspath(DB_PATH)}")

if not os.path.exists(DB_PATH):
    print("CRITICAL: Database file does not exist!")
    sys.exit(1)

try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Check Tables
    print("\n--- TABLES ---")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"Tables found: {tables}")
    
    if 'meal_logs' not in tables:
        print("CRITICAL: 'meal_logs' table MISSING!")
        # Attempt to create it manually here
        print("Attempting to create table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meal_logs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                food_name TEXT,
                nutrition TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                confidence REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
        print("Table created.")
    else:
        print("'meal_logs' table exists.")
        
    # 2. Check Schema
    print("\n--- SCHEMA (meal_logs) ---")
    cursor.execute("PRAGMA table_info(meal_logs)")
    cols = cursor.fetchall()
    for col in cols:
        print(col)

    # 3. Test Insert/Read
    print("\n--- TEST INSERT ---")
    
    # Get a user
    cursor.execute("SELECT id FROM users LIMIT 1")
    user_row = cursor.fetchone()
    if not user_row:
        print("No users to test with.")
    else:
        user_id = user_row[0]
        print(f"Testing with user_id: {user_id}")
        
        test_id = str(uuid.uuid4())
        try:
            cursor.execute("""
                INSERT INTO meal_logs (id, user_id, food_name, nutrition, source, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                test_id,
                user_id,
                "Diagnostic Cookie",
                json.dumps({"calories": 500}),
                "diag",
                0.9,
                datetime.now().isoformat()
            ))
            conn.commit()
            print("Insert successful.")
            
            # Read back
            cursor.execute("SELECT * FROM meal_logs WHERE id=?", (test_id,))
            row = cursor.fetchone()
            print(f"Read back: {row}")
            
            # Cleanup
            cursor.execute("DELETE FROM meal_logs WHERE id=?", (test_id,))
            conn.commit()
            print("Cleanup successful.")
            
        except Exception as e:
            print(f"INSERT FAILED: {e}")
            import traceback
            traceback.print_exc()

except Exception as e:
    print(f"General Error: {e}")
finally:
    if 'conn' in locals():
        conn.close()
    print("Diagnostic complete.")
