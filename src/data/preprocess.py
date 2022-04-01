
from albumentations import (
    Equalize, 
    CenterCrop,
    GaussianBlur,
    GridDistortion, 
    HorizontalFlip,
    Rotate, 
    RandomBrightnessContrast,
    Compose
)
import torch
import skimage
import numpy as np
from torchvision.transforms import Normalize

class Preprocessor:
    """
    A preproccessor for segmentation data which transforms the raw numpy arrays
    to torch tensors, optionally including various data augmentations.
    """
    
    def __init__(self, 
                 use_augmentations=True,
                 equalize_hist=True, 
                 random_rotate=True, 
                 grid_distortion=True,
                 horizontal_flip=True,
                 gaussian_blur=True,
                 random_brightness=True,
                 to_tensor=True
                ):
    
        self.eq = Equalize(always_apply=True) if equalize_hist else None
        self.normalize = Normalize((0.5, ), (0.25, ))
        
        self.use_augmentations = use_augmentations
        self.to_tensor = to_tensor
        
        augmentations = []
        if random_rotate:
            augmentations.append(Rotate(limit=15))
        if grid_distortion:
            augmentations.append(GridDistortion())
        if horizontal_flip:
            augmentations.append(HorizontalFlip())
        if gaussian_blur:
            augmentations.append(GaussianBlur())
        if random_brightness:
            augmentations.append(RandomBrightnessContrast())
        self.augmentations = Compose(augmentations)
        
        
    def __call__(self, mri, seg):
        
        mri = skimage.exposure.rescale_intensity(mri, out_range='uint8')
        
        if self.eq:
            mri, seg = self.__apply_transform(self.eq, mri, seg)
        
        if self.use_augmentations:
            mri, seg = self.__apply_transform(self.augmentations, mri, seg)
        
        mri, seg = self.__apply_transform(self.crop, mri, seg)
        
        if self.to_tensor:
            mri = skimage.exposure.rescale_intensity(mri, out_range='float32')
            mri = np.expand_dims(mri, axis=0)
            mri = torch.tensor(mri)
            mri = self.normalize(mri)
            seg = torch.tensor(seg).long()
            
        return mri, seg
        
    def __apply_transform(self, T, mri, seg):
            out = T(image=mri, mask=seg)
            return out['image'], out['mask']
