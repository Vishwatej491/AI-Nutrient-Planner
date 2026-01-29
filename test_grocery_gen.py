import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.services.meal_planner import GroceryGenerator

def test_grocery_gen():
    mock_plan = {
        "weekly_plan": {
            "Day 1": {
                "Breakfast": {
                    "name": "Oatmeal",
                    "ingredients": ["Oats", "Milk", "Honey"], # Correct list
                    "calories": 350
                },
                "Lunch": {
                    "name": "Salad",
                    "ingredients": "Tomato, Cucumber, Spinach", # String with commas
                    "calories": 200
                },
                "Dinner": {
                    "name": "Buggy Meal",
                    "ingredients": "a", # Single character bug
                    "calories": 100
                }
            }
        }
    }

    print("Testing GroceryGenerator with mixed formats...")
    result = GroceryGenerator.generate_list(mock_plan)
    
    all_items = []
    for cat in result:
        print(f"Category: {cat['category']}")
        for item in cat['items']:
            print(f"  - {item}")
            all_items.append(item)
    
    # Assertions
    assert "Oats" in all_items
    assert "Tomato" in all_items
    assert "Cucumber" in all_items
    assert "a" not in all_items
    assert len([i for i in all_items if len(i) == 1]) == 0
    
    print("\nSUCCESS: All items correctly parsed and single-character bugs filtered out.")

if __name__ == "__main__":
    test_grocery_gen()
