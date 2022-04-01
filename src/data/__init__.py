
from .dataset import PCASegmentationDataset
from .preprocess import Preprocessor
from torch.utils.data import DataLoader

def dataloaders(root, batch_size, split_seed=0, use_augmentations=True, 
                    **preprocessor_kwargs):
    
    train_preprocess = Preprocessor(use_augmentations=use_augmentations, 
                                        **preprocessor_kwargs)
    
    val_preprocess = Preprocessor(use_augmentations=False, 
                                    **preprocessor_kwargs)
    
    test_preprocess = Preprocessor(use_augmentations=False, 
                                    **preprocessor_kwargs)

    train = PCASegmentationDataset(
        root, 
        'train', 
        transform=train_preprocess, 
        split_seed=split_seed, 
    )
    
    val = PCASegmentationDataset(
        root,
        'val', 
        transform=val_preprocess,
        split_seed=split_seed, 
    )
    
    test = PCASegmentationDataset(
        root,
        'test', 
        transform=test_preprocess,
        split_seed=split_seed, 
    )

    return (DataLoader(train, batch_size, shuffle=True), 
                DataLoader(val, batch_size, shuffle=False), 
                DataLoader(test, batch_size, shuffle=False))


