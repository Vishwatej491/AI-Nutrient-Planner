import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from intelligence.agent_service import NutritionAgent

def quick_test():
    user_id = "test_user"
    agent = NutritionAgent(user_id)
    # Use a prompt that forces a tool lookup
    print("Testing prompt: How many calories in chicken?")
    try:
        # We use a mocked/small model usually, but here we just want to see the formatting
        # If the LLM call succeeds and doesn't crash the parser, it's fixed.
        # We'll just look at the verbose output if possible or the final answer.
        res = agent.chat("How many calories in chicken breast?")
        print(f"Response: {res}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    quick_test()
