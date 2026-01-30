import pandas as pd
import os

try:
    df = pd.read_csv(r"c:\Users\hp\AI Nutrition\data\food_nutrition_with_serving_category (1).csv")
    print("Dataset loaded.")
    avg_cal = df.groupby('Category')['calories (kcal)'].mean().sort_values(ascending=False)
    print("\nAverage Calories by Category:")
    print(avg_cal)
    
    top_protein = df.sort_values('protein (g)', ascending=False)[['food name', 'protein (g)']].head(10)
    print("\nTop 10 Protein-rich Items:")
    print(top_protein)
except Exception as e:
    print(f"Error: {e}")
