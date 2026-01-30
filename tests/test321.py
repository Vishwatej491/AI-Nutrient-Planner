import torch

data = torch.load("continental_embedding.pt", map_location="cpu")

print(type(data))
print(data.keys() if isinstance(data, dict) else "Not a dict")
