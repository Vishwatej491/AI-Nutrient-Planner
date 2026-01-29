import sqlite3
import os

DB_PATH = os.path.join("data", "nutrition.db")

def fix_schema():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check current columns
    cursor.execute("PRAGMA table_info(daily_logs)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Current columns in daily_logs: {columns}")
    
    # Add missing columns
    needed_columns = [
        ("sugar_g", "REAL DEFAULT 0"),
        ("fiber_g", "REAL DEFAULT 0"),
        ("sodium_mg", "REAL DEFAULT 0")
    ]
    
    for col_name, col_spec in needed_columns:
        if col_name not in columns:
            try:
                cursor.execute(f"ALTER TABLE daily_logs ADD COLUMN {col_name} {col_spec}")
                print(f"Added column {col_name} to daily_logs")
            except Exception as e:
                print(f"Error adding column {col_name}: {e}")
        else:
            print(f"Column {col_name} already exists")
    
    conn.commit()
    conn.close()
    print("Schema fix complete")

if __name__ == "__main__":
    fix_schema()
