import os
import sys
from typing import List, Any
import json

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from intelligence.agent_service import NutritionAgent

def test_nutrition_agent():
    print("=== Testing LangChain NutritionAgent Integration ===\n")
    
    # Mock user ID
    user_id = "test_user_agent"
    
    try:
        agent = NutritionAgent(user_id)
        
        # Test Case 1: Simple Nutrition Question
        print("Test 1: Nutrition Lookup")
        response1 = agent.chat("How many calories in an apple?")
        print(f"Agent Response: {response1}\n")
        
        # Test Case 2: Multi-step reasoning (Daily stats + Advice)
        print("Test 2: Daily Stats & Advice")
        response2 = agent.chat("I ate an apple today. How many calories do I have left?")
        print(f"Agent Response: {response2}\n")
        
        # Test Case 3: Recipe Generation
        print("Test 3: Recipe Generation")
        response3 = agent.chat("Suggest a healthy recipe with spinach and chicken.")
        print(f"Agent Response: {response3}\n")
        
        print("=== Verification Script Finished ===")
        
    except Exception as e:
        print(f"ERROR during verification: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nutrition_agent()
