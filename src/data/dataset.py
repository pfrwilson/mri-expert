

from torch.utils.data import Dataset
import os
import SimpleITK as sitk
from .utils import resample_volume, center_crop
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import einops

REFERENCE_CASE = 33

TEMP_STORAGE_DIR = os.path.join(
    os.path.expanduser('~'),         # home directory
    'promise_21_temp_datastore'
)

REFERENCE_SPACING = (0.625, 0.625, 3.6)

REFERENCE_SLICE_DIM = (224, 224)

class PCASegmentationDataset(Dataset):
    
    def __init__(self, root, split='train', transform=None,  
                 split_seed=0):

        self.root = root
        self.transform = transform
        self.reference_case = REFERENCE_CASE
        self.referece_spacing = REFERENCE_SPACING
        self.reference_shape = REFERENCE_SLICE_DIM

        assert split in ['train', 'val', 'test']
        self.split = split

        # Creates a resampled version of the raw dataset, 
        # stored in a temporary location
        if os.path.isdir(TEMP_STORAGE_DIR):
            shutil.rmtree(TEMP_STORAGE_DIR)
        os.mkdir(TEMP_STORAGE_DIR)
        print('Resampling images...')
        print(f'Saving slices as numpy files in {TEMP_STORAGE_DIR}')
        self.create_processed_dataset()

        # Generate train and validation splits for cases
        cases = list(range(50))

        cases.remove(REFERENCE_CASE)
        train_cases, test_cases = train_test_split(
            cases,
            test_size=9, 
            random_state=0      # fixed random state for deterministic test-split
        )
        test_cases.append(REFERENCE_CASE)
        train_cases, val_cases = train_test_split(
            train_cases,
            test_size=10, 
            random_state=split_seed
        )
        
        self.splits = dict(
            train_cases=train_cases, 
            val_cases=val_cases, 
            test_cases=test_cases
        )

        print(
            f"""
            train cases: {train_cases}
            val cases: {val_cases}
            test cases: {test_cases}
            """
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
        elif self.split == 'val':
            self.lookup_table = self.lookup_table.loc[val_cases]
        elif self.split == 'test':
            self.lookup_table = self.lookup_table.loc[test_cases]
    
    def __getitem__(self, idx):
        
        fnames = self.lookup_table.iloc[idx]
        
        mri = np.load(os.path.join(TEMP_STORAGE_DIR, fnames['mri']))
        seg = np.load(os.path.join(TEMP_STORAGE_DIR, fnames['seg']))
        
        if self.transform:
            mri, seg = self.transform(mri, seg)
            
        return mri, seg

    def __len__(self):
        return len(self.lookup_table)
        
    def create_processed_dataset(self, 
                                 reference_spacing=REFERENCE_SPACING, 
                                 reference_shape=REFERENCE_SLICE_DIM, 
                                 use_k_best_slices=5):
        """
        Creates a version of the dataset which is resampled to a specified reference spacing and 
        reference shape. If none are specified, uses the class default. 

        Only the slices with the largest prostate volume are selecte to be part of the dataset -
        this is specified by the parameter use_k_best_slices.

        The dataset is saved as .npy files of individual slices in a temporary directory together
        with metadata for the dataset.
        """

        for case in tqdm(range(50), 'Processing Cases'):

            case = str(case).zfill(2)
            mri_fname = f'Case{case}.mhd'
            seg_fname = f'Case{case}_segmentation.mhd'    
            
            mri_volume = sitk.ReadImage(os.path.join(self.root, mri_fname))
            seg_volume = sitk.ReadImage(os.path.join(self.root, seg_fname))

            mri_volume = resample_volume(
                mri_volume, 
                reference_spacing,
                interpolator=sitk.sitkLinear
            )

            seg_volume = resample_volume(
                seg_volume, 
                reference_spacing, 
                interpolator=sitk.sitkNearestNeighbor
            )

            mri_array = sitk.GetArrayFromImage(mri_volume)
            seg_array = sitk.GetArrayFromImage(seg_volume)
            
            # get the prostate volumes
            prostate_volumes = einops.reduce(
                seg_array, 'slice h w -> slice', 'sum'
            )
            
            best_slice_indices = np.argsort(prostate_volumes)[-use_k_best_slices:]

            for idx in best_slice_indices:

                mri_slice = mri_array[idx]
                mri_slice = center_crop(reference_shape, mri_slice)
                seg_slice = seg_array[idx]
                seg_slice = center_crop(reference_shape, seg_slice)

                mri_slice_fname = f'Case{case}_slice{idx}.npy'
                seg_slice_fname = f'Case{case}_segmentation_slice{idx}.npy'

                np.save(
                    os.path.join(TEMP_STORAGE_DIR, mri_slice_fname), 
                    mri_slice
                )
                np.save(
                    os.path.join(TEMP_STORAGE_DIR, seg_slice_fname), 
                    seg_slice
                )

    def raw(self):

        class RawContext: 
            def __init__(self, dataset):
                self.dataset = dataset
                self.cached_transform = dataset.transform

            def __enter__(self):
                self.dataset.transform = None

            def __exit__(self, type, value, traceback):
                self.dataset.transform = self.cached_transform

        return RawContext(self)