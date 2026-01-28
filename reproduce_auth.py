
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from auth.auth_service import auth_service
import time

def test_auth_flow():
    print("Testing Auth Flow...")
    
    # 1. Create token
    user_id = "test-user-123"
    email = "test@example.com"
    print(f"Creating token for {user_id}...")
    token = auth_service._create_token(user_id, email)
    print(f"Token: {token[:20]}...")
    
    # 2. Verify token
    print("Verifying token...")
    payload = auth_service.verify_token(token)
    
    if payload:
        print("✅ Token verification SUCCESS")
        print(f"Payload: {payload}")
    else:
        print("❌ Token verification FAILED")

    # 3. Test malformed token
    print("Testing malformed token...")
    bad_token = token + "junk"
    payload_bad = auth_service.verify_token(bad_token)
    if not payload_bad:
        print("✅ Malformed token correctly rejected")
    else:
        print("❌ Malformed token accepted (Unexpected)")

if __name__ == "__main__":
    test_auth_flow()
