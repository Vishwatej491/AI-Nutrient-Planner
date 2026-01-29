import requests
import sys

BASE_URL = "http://localhost:8081"

def test_endpoint(name, path, method="GET", json=None):
    print(f"Testing {name} ({path})...", end=" ")
    try:
        if method == "GET":
            # We skip auth check for some endpoints if we can, 
            # but since they all depend on Depends(get_current_user), 
            # this will likely return 401. 
            # However, seeing a 401 proves the endpoint exists and auth middleware is active.
            response = requests.get(f"{BASE_URL}{path}")
        else:
            response = requests.post(f"{BASE_URL}{path}", json=json)
        
        if response.status_code in [200, 401]:
            print(f"PASS (Status: {response.status_code})")
            return True
        else:
            print(f"FAIL (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

endpoints = [
    ("Meal Prediction", "/api/analytics/meal-prediction"),
    ("Calorie Budget", "/api/analytics/calorie-budget"),
    ("Weight Trajectory", "/api/analytics/weight-trajectory"),
    ("Meal Plan Current", "/api/meal-plan/current"),
    ("Meal Plan Generate", "/api/meal-plan/generate", "POST"),
    ("Grocery List", "/api/grocery-list"),
]

print("--- API Verification ---")
success_count = 0
for name, path, *args in endpoints:
    method = args[0] if args else "GET"
    if test_endpoint(name, path, method):
        success_count += 1

print(f"\nSummary: {success_count}/{len(endpoints)} endpoints responded (expected 200 or 401).")
