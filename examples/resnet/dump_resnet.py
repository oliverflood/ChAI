import sys
import os
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=True, num_classes=1000)

# Add the scripts directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))

import chai

model.chai_dump('models/resnet50','resnet50', with_json=False, verbose=True)

