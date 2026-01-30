
import requests
import datetime

BASE_URL = "http://localhost:8081"

def test_calorie_budget_endpoint():
    # We need a token. Since I can't easily get a real one without login, 
    # I'll rely on the server running and test the logic via internal call simulation 
    # if I could, but here I'll just check if the endpoint exists and accepts the Param.
    
    # Actually, I'll use the existing src code to test the logic directly.
    import sys
    import os
    sys.path.append(os.getcwd())
    from src.services.analytics_engine import AnalyticsEngine
    from src.auth.database import DailyLogRepository, MedicalProfileRepository
    
    user_id = "test_user_buffer"
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"Testing logic for user: {user_id}")
    
    # 1. Setup mock data for today
    DailyLogRepository.get_or_create(user_id, today)
    DailyLogRepository.update_nutrition(user_id, today, {"calories": 500})
    
    # 2. Setup mock data for yesterday
    DailyLogRepository.get_or_create(user_id, yesterday)
    DailyLogRepository.update_nutrition(user_id, yesterday, {"calories": 1500})
    
    engine = AnalyticsEngine(user_id)
    
    # Test Today
    forecast_today = engine.get_calorie_forecast(date_str=today)
    print(f"Today Remaining: {forecast_today['remaining_calories']}")
    
    # Test Yesterday
    forecast_yesterday = engine.get_calorie_forecast(date_str=yesterday)
    print(f"Yesterday Remaining: {forecast_yesterday['remaining_calories']}")
    
    # Assertions
    assert forecast_today['remaining_calories'] != forecast_yesterday['remaining_calories']
    print("\nSUCCESS: Calorie buffer logic is now date-aware!")

if __name__ == "__main__":
    try:
        test_calorie_budget_endpoint()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
