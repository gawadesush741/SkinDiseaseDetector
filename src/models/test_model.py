import sys
sys.path.append("src")

import torch
from models.skin_model import create_model


model = create_model(num_classes=7)

print("Model created successfully!")

print("\nFinal layer:")
print(model.fc)

# Test with one fake image
test_image = torch.randn(1, 3, 224, 224)

output = model(test_image)

print("\nTest input shape:", test_image.shape)
print("Model output shape:", output.shape)