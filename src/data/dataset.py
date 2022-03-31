

from torch.utils.data import Dataset
import os
import SimpleITK as sitk
from .utils import resample_volume
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

TEMP_STORAGE_DIR = os.path.join(
    os.path.expanduser('~'),         # home directory
    'promise_21_temp_datastore'
)

class PCASegmentationDataset(Dataset):
    
    def __init__(self, root, split='train', preprocess=None,  
                 split_seed=0, val_size: int=5):
        
        self.root = root
        assert split in ['train', 'val']
        self.split = split
        self.preprocess = preprocess
        
        # Creates a resampled version of the raw dataset, 
        # stored in a temporary location
        if not os.path.isdir(TEMP_STORAGE_DIR):
            os.mkdir(TEMP_STORAGE_DIR)
            print('Resampling images...')
            print(f'Saving slices as numpy files in {TEMP_STORAGE_DIR}')
            self.create_resampled_dataset(TEMP_STORAGE_DIR, root)

        # Generate train and validation splits for cases
        cases = range(50)
        train_cases, val_cases = train_test_split(
            cases,
            test_size=val_size, 
            random_state=split_seed
        )
        
        # Create a lookup table for file names
        indices = []
        data = {'mri': [], 'seg': []}
        for fname in [fname for fname in os.listdir(TEMP_STORAGE_DIR) \
            if 'segmentation' not in fname]:

            case = re.match('Case(\d+)', fname).groups()[0]
            slice = re.search('slice(\d+)', fname).groups()[0]

            data['mri'].append(fname)
            data['seg'].append(f'Case{case}_segmentation_slice{slice}.npy')

            indices.append([int(case), int(slice)])

        index = pd.MultiIndex.from_tuples(indices, names=['case', 'slice'])
        self.lookup_table = pd.DataFrame(data, index=index).sort_index()
        
        if self.split == 'train':
            self.lookup_table = self.lookup_table.loc[train_cases]
        else: 
            self.lookup_table = self.lookup_table.loc[val_cases]
    
    def __getitem__(self, idx):
        
        fnames = self.lookup_table.iloc[idx]
        
        mri = np.load(os.path.join(TEMP_STORAGE_DIR, fnames['mri']))
        seg = np.load(os.path.join(TEMP_STORAGE_DIR, fnames['seg']))
        
        if self.preprocess:
            mri, seg = self.preprocess(mri, seg)
            
        return mri, seg

    def __len__(self):
        return len(self.lookup_table)
        
    def __del__(self):
        if os.path.isdir(TEMP_STORAGE_DIR):
            shutil.rmtree(TEMP_STORAGE_DIR)
        
    @staticmethod
    def create_resampled_dataset(target_root, source_root, reference_spacing=(0.625, 0.625, 3.6)):
        """
        Creates a version of the dataset which is resampled to a specified reference spacing,
        and saved as .npy files of individual slices rather than whole volumes for more efficient 
        loading.
        """
        
        for fname in tqdm(os.listdir(source_root), desc='Processed file'):
                
            if fname.endswith('raw'):
                continue
            
            case_num = re.match('Case(\d+)', fname).groups()[0]
            segmentation = 'segmentation' in fname
            
            volume = sitk.ReadImage(os.path.join(source_root, fname))
            volume = resample_volume(
                volume, 
                reference_spacing,
                interpolator=sitk.sitkNearestNeighbor if segmentation else sitk.sitkLinear
            )
            
            array = sitk.GetArrayFromImage(volume)
            
            for idx, slice in enumerate(array):
                
                new_fname = f'Case{case_num}'
                if segmentation: 
                    new_fname += '_segmentation'
                new_fname += f'_slice{idx}.npy'
                
                np.save(os.path.join(target_root, new_fname), slice)