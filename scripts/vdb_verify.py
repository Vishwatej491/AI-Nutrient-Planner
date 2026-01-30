import torch
import os

vdb_path = r"c:\Users\hp\AI Nutrition\data\continental_embeddings.pt"
print(f"Checking for VDB at: {vdb_path}")

if os.path.exists(vdb_path):
    print("File exists. Attempting to load...")
    try:
        data = torch.load(vdb_path)
        print("Model Name:", data.get('model_name'))
        print("Metadata count:", len(data.get('metadata', [])))
        print("Embeddings shape:", data.get('embeddings').shape)
        
        # Query test
        if len(data.get('metadata', [])) > 0:
            print("\nSample items:")
            for i in range(min(5, len(data['metadata']))):
                print(f"- {data['metadata'][i]['name']} ({data['metadata'][i]['category']})")
    except Exception as e:
        print(f"Error loading: {e}")
else:
    print("File does NOT exist yet.")
