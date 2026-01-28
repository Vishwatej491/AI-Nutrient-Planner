import sys
import os
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from auth.database import init_database, UserRepository, MedicalProfileRepository, MealRepository, DailyLogRepository
from analytics.predictive_service import get_predictive_service

def verify():
    print("=== STARTING PREDICTIVE HEALTH VERIFICATION ===")
    init_database()
    
    # 1. Setup Mock User
    user_id = "test-forecaster"
    email = "forecast@test.com"
    UserRepository.create(user_id, email, "hashed", "Test Forecaster")
    
    # 2. Setup Profile (70kg, 175cm, 30yo, Sedentary)
    # TDEE for this user is ~2000 kcal
    MedicalProfileRepository.create(
        "prof-123", user_id, 
        conditions=["Diabetes"], 
        allergens=[],
        age=30,
        gender="male",
        weight_kg=70.0,
        height_cm=175.0,
        activity_level="sedentary"
    )
    
    # 3. Seed 7 days of "Bad" Data (Surplus + Increasing Sugar)
    # Intake: 2500 kcal per day (500 surplus)
    # Sugar: 20g, 22g, 25g, 28g, 30g, 35g, 40g (Increasing trend)
    print("Seedling 7 days of historical data...")
    today = datetime.now()
    for i in range(7):
        date_offset = 6 - i
        target_date = today - timedelta(days=date_offset)
        sugar = 20 + (i * 3) # Trend
        
        # Log a meal with surplus
        MealRepository.create(
            user_id=user_id,
            food_name=f"Heavy Lunch Day {i}",
            nutrition={
                "calories": 2500,
                "protein_g": 80,
                "carbs_g": 300,
                "fat_g": 100,
                "sugar_g": sugar,
                "sodium_mg": 2500, # High sodium
                "fiber_g": 10
            },
            timestamp=target_date
        )
        # Manually sync daily_logs if it doesn't happen automatically in this environment
        DailyLogRepository.update_nutrition(user_id, target_date.strftime("%Y-%m-%d"), {"calories": 2500, "protein_g": 80, "carbs_g": 300, "fat_g": 100})
    
    # 4. Run Prediction
    print("\nRunning Predictive Analysis...")
    service = get_predictive_service()
    
    weight_trend = service.predict_weight_trend(user_id)
    print(f"\n[WEIGHT FORECAST]")
    print(f"Current Weight: {weight_trend['current_weight']} kg")
    print(f"Projected Weight (30 days): {weight_trend['projected_weight']} kg")
    print(f"Change: {weight_trend['weight_change']} kg")
    print(f"Avg Daily Surplus: {weight_trend['avg_daily_surplus']} kcal")
    
    risks = service.detect_nutrient_risks(user_id)
    print(f"\n[NUTRIENT RISKS]")
    if not risks:
        print("No risks detected (FAIL - expected sugar/sodium risks)")
    for r in risks:
        print(f"- {r['nutrient'].upper()}: {r['message']} (Severity: {r['severity']})")
    
    # Validation
    if weight_trend['weight_change'] > 0 and len(risks) >= 2:
        print("\n✅ VERIFICATION SUCCESSFUL!")
    else:
        print("\n❌ VERIFICATION FAILED!")

if __name__ == "__main__":
    verify()
