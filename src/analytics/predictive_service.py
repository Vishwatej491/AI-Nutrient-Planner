"""
Predictive Health Service

Calculates future health trends and risks based on historical data.
Uses deterministic physiological models (TDEE, Mifflin-St Jeor) 
and statistical trend analysis.
"""

import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from auth.database import DailyLogRepository, MedicalProfileRepository
from models.user import ActivityLevel

class PredictiveHealthService:
    """
    Service for projecting future health outcomes.
    """
    
    # 1 kg body fat is approximately 7,700 calories
    CALORIES_PER_KG = 7700

    @staticmethod
    def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation."""
        if not weight_kg or not height_cm or not age:
            return 2000.0 # Default fallback
            
        if gender and gender.lower() == 'female':
            return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
        else:
            # Default to male / gender neutral if unspecified
            return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5

    @staticmethod
    def get_activity_multiplier(level: str) -> float:
        """Map activity level to TDEE multiplier."""
        multipliers = {
            "sedentary": 1.2,
            "lightly_active": 1.375,
            "moderately_active": 1.55,
            "very_active": 1.725,
            "extra_active": 1.9
        }
        return multipliers.get(level.lower(), 1.2)

    def predict_weight_trend(self, user_id: str, forecast_days: int = 30) -> Dict[str, Any]:
        """
        Forecast weight change over the next N days based on recent caloric balance.
        """
        profile = MedicalProfileRepository.get_by_user_id(user_id)
        if not profile or not profile.get('weight_kg'):
            return {"success": False, "error": "Incomplete user profile (missing weight/age)"}

        # Calculate TDEE
        weight = profile.get('weight_kg')
        height = profile.get('height_cm', 170)
        age = profile.get('age', 30)
        gender = profile.get('gender', 'male')
        activity = profile.get('activity_level', 'sedentary')
        
        bmr = self.calculate_bmr(weight, height, age, gender)
        tdee = bmr * self.get_activity_multiplier(activity)

        # Get last 7 days of logs to find average daily balance
        end_date = date.today()
        start_date = end_date - timedelta(days=6)
        
        daily_balances = []
        for i in range(7):
            curr_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            log = DailyLogRepository.get_or_create(user_id, curr_date)
            
            intake = log.get('calories_consumed', 0)
            burned = log.get('calories_burned', 0)
            
            # Daily balance = Intake - (Maintenance + Exercise)
            balance = intake - (tdee + burned)
            daily_balances.append(balance)

        avg_daily_surplus = sum(daily_balances) / len(daily_balances) if daily_balances else 0
        
        # Project weight change
        total_projected_surplus = avg_daily_surplus * forecast_days
        projected_weight_change = total_projected_surplus / self.CALORIES_PER_KG
        
        return {
            "success": True,
            "current_weight": weight,
            "projected_weight": round(weight + projected_weight_change, 2),
            "weight_change": round(projected_weight_change, 2),
            "forecast_days": forecast_days,
            "avg_daily_surplus": round(avg_daily_surplus, 0),
            "tdee": round(tdee, 0),
            "confidence": "medium" if len(daily_balances) >= 7 else "low"
        }

    def detect_nutrient_risks(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Analyze trends in sugar and sodium to predict upcoming target violations.
        """
        risks = []
        end_date = date.today()
        start_date = end_date - timedelta(days=6)
        
        sugar_logs = []
        sodium_logs = []
        
        # We need to get nutrition from meal_logs because daily_logs might not store sugar/sodium in the current DB schema
        # Actually daily_logs has calories, protein, carbs, fat but NOT sugar/sodium in the schema I saw.
        # Let's check meal_logs.
        from auth.database import MealRepository
        
        daily_totals = {}
        for i in range(7):
            date_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            meals = MealRepository.get_meals_by_date(user_id, date_str)
            
            day_sugar = 0
            day_sodium = 0
            for meal in meals:
                nut = meal.get('nutrition', {})
                day_sugar += nut.get('sugar_g', 0)
                day_sodium += nut.get('sodium_mg', 0)
            
            daily_totals[date_str] = {"sugar": day_sugar, "sodium": day_sodium}

        # Check for upward trends (naive simple slope)
        sugar_values = [v['sugar'] for v in daily_totals.values()]
        if len(sugar_values) >= 3:
            slope = (sugar_values[-1] - sugar_values[0]) / len(sugar_values)
            if slope > 2: # Increasing by 2g per day
                risks.append({
                    "nutrient": "sugar",
                    "type": "upward_trend",
                    "severity": "medium",
                    "message": "Your sugar intake is trending upwards. You may exceed your target by mid-next week."
                })

        # Check for consistent high sodium
        sodium_values = [v['sodium'] for v in daily_totals.values()]
        high_sodium_days = sum(1 for v in sodium_values if v > 2300) # RDA limit
        if high_sodium_days >= 4:
            risks.append({
                "nutrient": "sodium",
                "type": "chronic_excess",
                "severity": "high",
                "message": "Consistent high sodium intake detected. Predicting high water retention and hypertension risk."
            })

        return risks

def get_predictive_service():
    return PredictiveHealthService()
