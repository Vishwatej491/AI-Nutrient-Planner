import sqlite3
import os
import json
import uuid
import datetime

DB_PATH = os.path.join('data', 'nutrition.db')
if not os.path.exists(DB_PATH):
    print(f"Error: {DB_PATH} not found")
    exit(1)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 1. Get the most recently created user
cursor.execute('SELECT id, email FROM users ORDER BY created_at DESC LIMIT 1')
user = cursor.fetchone()

if not user:
    print("No user found. Please register/login in the browser first.")
    exit(1)

user_id = user['id']
email = user['email']
print(f"Updating profile for User: {email} (ID: {user_id})")

# 2. Check if profile exists
cursor.execute('SELECT * FROM medical_profiles WHERE user_id = ?', (user_id,))
profile = cursor.fetchone()

# 3. Data to insert/update
bio_metrics = {
    'age': 30,
    'gender': 'Male',
    'weight_kg': 75.5,
    'height_cm': 180.0,
    'activity_level': 'Moderate',
    'fitness_goal': 'Maintain Weight'
}

if profile:
    print("Profile exists. Updating bio-metrics...")
    cursor.execute('''
        UPDATE medical_profiles 
        SET age=?, gender=?, weight_kg=?, height_cm=?, activity_level=?, fitness_goal=?, updated_at=?
        WHERE user_id=?
    ''', (
        bio_metrics['age'], 
        bio_metrics['gender'], 
        bio_metrics['weight_kg'], 
        bio_metrics['height_cm'], 
        bio_metrics['activity_level'], 
        bio_metrics['fitness_goal'],
        datetime.datetime.now().isoformat(),
        user_id
    ))
else:
    print("No profile found. Creating new profile with bio-metrics...")
    profile_id = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO medical_profiles (
            id, user_id, conditions, allergens, medications, daily_targets, 
            age, gender, weight_kg, height_cm, activity_level, fitness_goal,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        profile_id,
        user_id,
        json.dumps([]),
        json.dumps([]),
        json.dumps([]),
        json.dumps({}),
        bio_metrics['age'], 
        bio_metrics['gender'], 
        bio_metrics['weight_kg'], 
        bio_metrics['height_cm'], 
        bio_metrics['activity_level'], 
        bio_metrics['fitness_goal'],
        datetime.datetime.now().isoformat(),
        datetime.datetime.now().isoformat()
    ))

conn.commit()
print("SUCCESS: Profile updated with bio-metrics.")
conn.close()
