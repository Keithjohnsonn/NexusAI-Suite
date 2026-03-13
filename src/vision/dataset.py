import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Optional
from src.utils.logging import log

class ImageClassificationDataset(Dataset):
    """
    A professional, flexible PyTorch dataset for image classification tasks.
    """
    
    def __init__(
        self, 
        image_paths: List[str], 
        labels: List[int], 
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or self._default_transform()
        log.info(f"Initialized dataset with {len(image_paths)} images.")

    def _default_transform(self):
        """
        Returns a set of default transforms if none are provided.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            log.error(f"Error loading image at {image_path}: {e}")
            # Return zero tensor as fallback (or handle as preferred)
            return torch.zeros((3, 224, 224)), label
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
