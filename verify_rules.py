import sys
import os
from pathlib import Path

# Add src to PYTHONPATH
sys.path.append(str(Path(__file__).parent / "src"))

from rules.engine import RuleEngine, Severity
from models.food import Food, NutritionInfo, FoodCategory
from models.user import UserProfile, HealthCondition, DailyTargets, ActivityLevel

def verify_rule_engine():
    print("--- Rule Engine Verification ---")
    
    # 1. Setup Rule Engine
    engine = RuleEngine()
    print("✓ RuleEngine initialized")
    
    # 2. Create Test User (Diabetic)
    user = UserProfile(
        user_id="test_user",
        name="Test",
        conditions=[HealthCondition.DIABETES],
        allergens=["Peanuts"],
        daily_targets=DailyTargets.for_diabetes(),
        activity_level=ActivityLevel.MODERATELY_ACTIVE
    )
    print(f"✓ Created User: Diabetes, Peanut Allergy")
    
    # 3. Test Case A: Safe Food
    safe_food = Food(
        food_id="safe_1",
        name="Cucumber Salad",
        serving_size=100,
        serving_unit="g",
        nutrition=NutritionInfo(calories=20, carbs_g=4, sugar_g=2, fiber_g=1),
        category=FoodCategory.VEGETABLE
    )
    violations_a = engine.evaluate(safe_food, user)
    print(f"Test A (Safe Food): {len(violations_a)} violations (Expected: 0)")
    if len(violations_a) == 0:
        print("✓ PASS")
    else:
        print("✗ FAIL")
        for v in violations_a:
            print(f"  - {v.message}")

    # 4. Test Case B: Allergen (Peanuts)
    allergen_food = Food(
        food_id="danger_1",
        name="Peanut Butter",
        serving_size=30,
        serving_unit="g",
        nutrition=NutritionInfo(calories=190, fat_g=16, protein_g=8),
        allergens=["Peanuts"],
        category=FoodCategory.PROTEIN
    )
    violations_b = engine.evaluate(allergen_food, user)
    verdict_b = engine.get_final_verdict(violations_b)
    print(f"Test B (Allergen): Verdict {verdict_b.value} (Expected: block)")
    if verdict_b == Severity.BLOCK:
        print("✓ PASS")
    else:
        print("✗ FAIL")

    # 5. Test Case C: Diabetes (High Sugar)
    sugar_food = Food(
        food_id="warn_1",
        name="Candy Bar",
        serving_size=50,
        serving_unit="g",
        nutrition=NutritionInfo(calories=250, carbs_g=30, sugar_g=25, fiber_g=0),
        category=FoodCategory.SNACK
    )
    violations_c = engine.evaluate(sugar_food, user)
    print(f"Test C (High Sugar): {len(violations_c)} violations (Expected: >0)")
    has_diabetes_warn = any(v.category == "diabetes" for v in violations_c)
    if has_diabetes_warn:
        print("✓ PASS (Found diabetes warning)")
    else:
        print("✗ FAIL (No diabetes warning found)")
        for v in violations_c:
            print(f"  - {v.rule_id}: {v.message}")

if __name__ == "__main__":
    try:
        verify_rule_engine()
    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
