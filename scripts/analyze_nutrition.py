import pandas as pd
import os
import sys

print("Script started...")
sys.stdout.flush()

def analyze_dataset(file_path, name):
    print(f"\n{'='*20} Analyzing: {name} {'='*20}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    # Basic Info
    print(f"\nShape: {df.shape}")
    print("\nColumns and Dtypes:")
    print(df.dtypes)
    
    # Identify numeric columns for analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    print(f"\nNumeric columns: {numeric_cols}")
    
    # Summary Statistics
    print("\nDescriptive Statistics (Numeric):")
    print(df[numeric_cols].describe().round(2))
    
    # Categorical Analysis
    cat_cols = ['Food_Group', 'Cuisine', 'Category']
    for col in cat_cols:
        if col in df.columns:
            print(f"\nDistribution by {col}:")
            print(df[col].value_counts().head(10))
            
            # Group by and find means
            print(f"\nAverage Calories by {col}:")
            # Try to find calorie column (might be 'Calories' or 'calories (kcal)')
            cal_col = next((c for c in df.columns if 'cal' in c.lower()), None)
            if cal_col:
                print(df.groupby(col)[cal_col].mean().sort_values(ascending=False).round(2))
    
    # Top 5 Highest Protein Dishes
    prot_col = next((c for c in df.columns if 'prot' in c.lower()), None)
    name_col = next((c for c in df.columns if 'dish' in c.lower() or 'name' in c.lower() or 'food' in c.lower()), None)
    
    if prot_col and name_col:
        print(f"\nTop 5 Highest Protein {name_col}s:")
        print(df.sort_values(by=prot_col, ascending=False)[[name_col, prot_col]].head(5))

    return df

def main():
    base_path = r"c:\Users\hp\AI Nutrition\data"
    
    # Dataset 1
    file1 = os.path.join(base_path, "FINAL_ACCURATE_FOOD_DATASET_WITH_CUISINE (1).csv")
    df1 = analyze_dataset(file1, "Cuisine Dataset")
    
    # Dataset 2
    file2 = os.path.join(base_path, "food_nutrition_with_serving_category (1).csv")
    df2 = analyze_dataset(file2, "Serving Category Dataset")

    # Cross-dataset check (optional)
    if df1 is not None and df2 is not None:
        print(f"\n{'='*20} Comparison {'='*20}")
        print(f"Total Unique Items in Dataset 1: {df1.iloc[:,0].nunique()}")
        print(f"Total Unique Items in Dataset 2: {df2.iloc[:,0].nunique()}")

if __name__ == "__main__":
    main()
