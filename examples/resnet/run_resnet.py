import torch.nn as nn
import sys
import os
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize
from torchvision.models import resnet50, ResNet50_Weights

# Load the model with pretrained weights
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode
# print(model)
# Get the image paths from command line arguments
img_paths = sys.argv[1:]
if not img_paths:
    print("Please provide at least one image path")
    sys.exit(1)

# Process each image path
for img_path in img_paths:
    # Read the image
    img = read_image(img_path).float()

    # Resize the image
    img = Resize((224, 224))(img)

    # Normalize the image
    img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img / 255.0)

    # Add batch dimension
    img = img.unsqueeze(0)
    print(f'Processing {img_path} with shape {img.shape}')
    
    # Perform inference
    with torch.no_grad():
        res = model(img)

    # Print top 5 scores
    _, indices = torch.topk(res, 5)
    percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100

    # Lookup the index from the labels
    labels = []
    with open('../vgg/imagenet/LOC_synset_mapping.txt') as f:
        for line in f:
            labels.append(line.strip())

    # Print the top 5 predictions
    print(f'Top 5 predictions for {img_path}:')
    for i in indices[0]:
        print(f'{labels[i]}: {percentage[i].item()}%')
    print()