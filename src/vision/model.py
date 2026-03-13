import torch.nn as nn
import torchvision.models as models
from typing import Optional
from src.utils.logging import log

class VisionModelFactory:
    """
    A factory class for creating modern vision architectures with optional pre-training.
    """
    
    _ARCHITECTURES = {
        "resnet50": models.resnet50,
        "mobilenet_v3": models.mobilenet_v3_small,
        "efficientnet_b0": models.efficientnet_b0
    }

    @staticmethod
    def create_model(
        arch_name: str, 
        num_classes: int, 
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ) -> nn.Module:
        """
        Creates a vision model with a custom classification head.
        """
        if arch_name not in VisionModelFactory._ARCHITECTURES:
            log.error(f"Architecture {arch_name} not supported.")
            raise ValueError(f"Unknown architecture: {arch_name}")
            
        log.info(f"Creating {arch_name} model (pretrained={pretrained}, classes={num_classes})")
        
        # Load base model
        weights = "DEFAULT" if pretrained else None
        model = VisionModelFactory._ARCHITECTURES[arch_name](weights=weights)
        
        # Modify final layer based on architecture
        if arch_name == "resnet50":
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        elif "mobilenet" in arch_name or "efficientnet" in arch_name:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            
        return model
