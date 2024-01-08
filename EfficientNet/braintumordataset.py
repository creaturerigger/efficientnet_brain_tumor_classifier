from torchvision.datasets import VisionDataset
import torch
import os
from PIL import Image


class BrainTumorDataset(VisionDataset):
    
    def __init__(self, root_path, transform=None, target_transform=None, split='train'):
        super(BrainTumorDataset, self).__init__(root_path, transform=transform, target_transform=target_transform)        
        self.split = split
        self.root_path = os.path.join(root_path, "Training") if self.split == 'train' else os.path.join(root_path, "Testing")
        self.class_names = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
        self.images = []
        self.labels = []
        for class_name in os.listdir(self.root_path):
            class_dir = os.path.join(self.root_path, class_name)
            for img in os.listdir(class_dir):
                if img.endswith('jpg'):
                    image_path = os.path.join(class_dir, img)
                    self.images.append(image_path)
                    self.labels.append(self.class_names[class_name])
                else:
                    continue
        
        
    def __len__(self):
        return len(self.images)
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image, label = self._load_sample(idx)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    
    def _load_sample(self, idx):
        image = Image.open(self.images[idx]).convert('L')        
        label = self.labels[idx]
        return image, label