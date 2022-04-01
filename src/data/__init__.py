
from .dataset import PCASegmentationDataset
from .preprocess import Preprocessor
from torch.utils.data import DataLoader

def dataloaders(root, batch_size, split_seed=0, val_size=5, target_size=224, 
                     use_augmentations=True, **preprocessor_kwargs):
    
    train_preprocess = Preprocessor(target_size=target_size, use_augmentations=use_augmentations, 
                              **preprocessor_kwargs)
    
    val_preprocess = Preprocessor(target_size=target_size, use_augmentations=False, 
                              **preprocessor_kwargs)
    
    train = PCASegmentationDataset(
        root, 
        'train', 
        preprocess=train_preprocess, 
        split_seed=split_seed, 
        val_size=val_size,
    )
    
    val = PCASegmentationDataset(
        root,
        'val', 
        preprocess=val_preprocess,
        split_seed=split_seed, 
        val_size=val_size
    )
    
    return (DataLoader(train, batch_size, shuffle=True), 
                DataLoader(val, batch_size, shuffle=False))


