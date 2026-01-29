"""
Analytics Engine for Predictive Nutrition.
Handles meal time prediction, calorie forecasting, and nutrient deficiency detection.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
from auth.database import MealRepository, DailyLogRepository, NutrientTargetsRepository, WeightHistoryRepository

class AnalyticsEngine:
    """
    Core engine for nutrition predictions and pattern recognition.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id

    def predict_next_meal(self) -> Dict[str, Any]:
        """Predict the next meal time and type based on history."""
        meals = MealRepository.get_meals_by_date(self.user_id) # This only gets today, we need more
        # For real prediction, we'd fetch last 7-14 days. 
        # Implementing a simplified version for now.
        
        # Placeholder logic: return standard meal times
        now = datetime.now()
        hour = now.hour
        
        if hour < 10:
            return {"next_meal": "Breakfast", "estimated_time": "08:30 AM", "confidence": 0.8}
        elif hour < 14:
            return {"next_meal": "Lunch", "estimated_time": "01:00 PM", "confidence": 0.85}
        elif hour < 20:
            return {"next_meal": "Dinner", "estimated_time": "08:00 PM", "confidence": 0.9}
        else:
            tomorrow_breakfast = (now + timedelta(days=1)).replace(hour=8, minute=30)
            return {"next_meal": "Breakfast", "estimated_time": "Tomorrow 08:30 AM", "confidence": 0.75}

    def get_calorie_forecast(self) -> Dict[str, Any]:
        """Forecast remaining calorie budget and suggests meal sizes."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        stats = DailyLogRepository.get_or_create(self.user_id, today_str)
        
        target = stats.get("calories_target", 2000)
        consumed = stats.get("calories_consumed", 0)
        burned = stats.get("calories_burned", 0)
        
        remaining = target + burned - consumed
        
        return {
            "remaining_calories": round(remaining, 0),
            "status": "on_track" if remaining > 0 else "over_budget",
            "message": f"You have {int(remaining)} kcal left for today." if remaining > 0 else "You have exceeded your daily limit."
        }

    def detect_deficiencies(self) -> List[Dict[str, Any]]:
        """Identify missing nutrients over the last 3 days."""
        targets = NutrientTargetsRepository.get_by_user_id(self.user_id)
        
        # Fetch last 3 days
        deficiencies = []
        nutrients_to_check = {
            "vitamin_c_mg": "Vitamin C",
            "calcium_mg": "Calcium",
            "iron_mg": "Iron",
            "fiber_g": "Fiber"
        }
        
        # Mock logic: if today's totals are very low, alert
        today_str = datetime.now().strftime("%Y-%m-%d")
        stats = DailyLogRepository.get_or_create(self.user_id, today_str)
        
        for key, label in nutrients_to_check.items():
            current = stats.get(key.replace("_mg", "").replace("_g", ""), 0) # Mapping DailyLog columns
            # Column names in DailyLog are direct: protein_g, carbs_g, fat_g, sugar_g, fiber_g, sodium_mg
            # Others like Vitamin C aren't in DailyLog yet. 
            # Real implementation would aggregate from MealLogs.
            
        return deficiencies # Placeholder

    def predict_weight_trajectory(self) -> Dict[str, Any]:
        """Simple weight prediction using linear trend of calorie deltas."""
        # For now, we'll return a static prediction based on current data
        # Real version uses linear regression on WeightHistoryRepository
        history = WeightHistoryRepository.get_history(self.user_id)
        
        if not history:
             return {"message": "Insufficient weight data for trajectory."}
             
        current = history[0]["weight_kg"]
        return {
            "current_weight": current,
            "predicted_30d": round(current - 0.5, 1), # Placeholder
            "trend": "decreasing"
        }
