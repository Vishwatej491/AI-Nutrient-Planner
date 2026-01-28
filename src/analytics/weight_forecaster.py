import numpy as np
from sklearn.linear_model import LinearRegression
import jobpy
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

class WeightForecaster:
    """
    ML-based weight loss forecasting service using Linear Regression.
    Predicts future weight based on calorie delta and activity level.
    """
    
    def __init__(self, model_path: str = "models/weight_forecast_model.joblib"):
        self.model_path = model_path
        self.model = LinearRegression()
        self.is_trained = False
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Try to load existing model
        self._load_model()
        
        if not self.is_trained:
            print("[WeightForecaster] No model found. Training with synthetic data...")
            self.train_with_synthetic_data()

    def _load_model(self):
        """Load the model if it exists."""
        if os.path.exists(self.model_path):
            try:
                import joblib
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                print(f"[WeightForecaster] Loaded model from {self.model_path}")
            except Exception as e:
                print(f"[WeightForecaster] Error loading model: {e}")

    def _save_model(self):
        """Save the trained model."""
        try:
            import joblib
            joblib.dump(self.model, self.model_path)
            print(f"[WeightForecaster] Saved model to {self.model_path}")
        except Exception as e:
            print(f"[WeightForecaster] Error saving model: {e}")

    def train_with_synthetic_data(self):
        """
        Generate synthetic data based on physiological principles:
        ~7700 calorie deficit = 1kg weight loss.
        Features: [Current_Weight, Daily_Calorie_Delta, Activity_Level]
        Target: Weight_After_30_Days
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Features
        weights = np.random.uniform(50, 120, n_samples)  # 50kg to 120kg
        calorie_deltas = np.random.uniform(-1000, 1000, n_samples)  # -1000 to +1000 kcal/day
        activity_levels = np.random.uniform(1.2, 1.9, n_samples)  # Sedentary to Extra Active
        
        X = np.stack([weights, calorie_deltas, activity_levels], axis=1)
        
        # Target: physiological weight change
        # Base metabolism is roughly 30 * weights (very simplified)
        # Weight change = (Calorie_Delta * 30) / 7700
        # We also add some noise and activity logic
        weight_change = (calorie_deltas * 30) / 7700
        # Activity level slightly modulates the efficiency or represents higher baseline burn
        weight_change -= (activity_levels - 1.2) * 2  # Active people lose more weight given the same delta
        
        y = weights + weight_change + np.random.normal(0, 0.2, n_samples)
        
        self.model.fit(X, y)
        self.is_trained = True
        self._save_model()

    def predict_30_day_forecast(
        self, 
        current_weight: float, 
        avg_daily_delta: float, 
        activity_level: float
    ) -> Dict[str, Any]:
        """
        Predict weight after 30 days.
        
        Args:
            current_weight: Current weight in kg
            avg_daily_delta: Average daily calorie intake minus TDEE
            activity_level: Physical activity level multiplier (1.2 to 1.9)
            
        Returns:
            Dict with predicted weight and insights
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        X_input = np.array([[current_weight, avg_daily_delta, activity_level]])
        predicted_weight = float(self.model.predict(X_input)[0])
        
        weight_change = predicted_weight - current_weight
        
        # Categorize the trend
        if weight_change < -0.5:
            trend = "Weight Loss"
            status = "On track"
        elif weight_change > 0.5:
            trend = "Weight Gain"
            status = "Surplus detected"
        else:
            trend = "Maintenance"
            status = "Stable"
            
        return {
            "current_weight": round(current_weight, 1),
            "predicted_weight": round(predicted_weight, 1),
            "weight_change": round(weight_change, 2),
            "trend": trend,
            "status": status,
            "days_forecast": 30,
            "timestamp": datetime.now().isoformat()
        }


# Global singleton
_forecaster = None

def get_weight_forecaster():
    global _forecaster
    if _forecaster is None:
        _forecaster = WeightForecaster()
    return _forecaster
