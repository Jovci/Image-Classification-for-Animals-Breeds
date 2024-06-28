import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights

def get_model(num_classes):
    # Load the pre-trained ResNet50 model with the recommended weights parameter
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
