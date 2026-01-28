import os
import sys
from typing import List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from intelligence.agent_service import DeterministicNutritionAgent, Intent

def test_agent():
    # Mock user ID
    user_id = "test_user_123"
    agent = DeterministicNutritionAgent(user_id)
    
    print("=== Testing Deterministic Agent ===\n")
    
    # 1. Test Activity Intent
    print("Test 1: Activity Intent (Walking)")
    res1 = agent.chat("I went for a 30 minute walk today")
    print(f"Response: {res1}\n")
    assert "burn" in res1.lower()
    assert "walking" in res1.lower()
    
    # 2. Test Food Lookup
    print("Test 2: Food Lookup (Apple)")
    res2 = agent.chat("How many calories in an apple?")
    print(f"Response: {res2}\n")
    assert "apple" in res2.lower()
    assert "calories" in res2.lower()
    
    # 3. Test Guardrail (Activity as Food)
    print("Test 3: Guardrail (Walking calories)")
    res3 = agent.chat("How many calories in walking?")
    print(f"Response: {res3}\n")
    assert "expenditure" in res3.lower() or "burn" in res3.lower()

    # 4. Test Stats Request
    print("Test 4: Stats Request")
    res4 = agent.chat("Show my stats for today")
    print(f"Response: {res4}\n")
    assert "summary" in res4.lower() or "consumed" in res4.lower()

    # 5. Test Recipe Request
    print("Test 5: Recipe Request")
    res5 = agent.chat("Make me a recipe with apple and yogurt")
    print(f"Response: {res5}\n")
    assert "ingredients" in res5.lower()
    assert "instructions" in res5.lower()

    print("=== Verification Complete! ===")

if __name__ == "__main__":
    test_agent()
