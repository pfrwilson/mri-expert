
from .dataset import PCASegmentationDataset
from .preprocess import Preprocessor
from torch.utils.data import DataLoader

def train_dataloader(root, batch_size, split_seed=0, val_size=5, target_size=224, 
                     use_augmentations=True, **preprocessor_kwargs):
    
    preprocess = Preprocessor(target_size=target_size, use_augmentations=use_augmentations, 
                              **preprocessor_kwargs)
    
    dataset = PCASegmentationDataset(
        root, 
        'train', 
        preprocess=preprocess, 
        split_seed=split_seed, 
        val_size=val_size,
    )
    
    return DataLoader(dataset, batch_size, shuffle=True)

def val_dataloader(root, batch_size, split_seed=0, val_size=5, target_size=224, 
                   **preprocessor_kwargs):
    
    preprocess = Preprocessor(target_size=target_size, use_augmentations=False,
                              **preprocessor_kwargs)
    
    dataset = PCASegmentationDataset(
        root, 'val', 
        preprocess=preprocess, 
        split_seed=split_seed,
        val_size=val_size
    )
    
