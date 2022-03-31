
from calendar import c
import importlib
import re
import pandas as pd
import SimpleITK as sitk
import os
import numpy as np
import einops
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class ProstateMRIDataset:
    
    def __init__(self, root):
        
        self.root = root
        self.metadata = { 'units': 'mm' }
        self._lookup = None
        self._data = None
        self._data_resampled = None
        self._reference_spacing = None
        self._data_resampled = None
        
    @property
    def lookup(self):
        
        if self._lookup is None:
        
            # setup file lookup table
            self._lookup = pd.DataFrame(
                {'mri_filepath': pd.Series(),
                 'seg_filepath': pd.Series()},
                index=pd.Series(range(50), name='case')
            )
            
            with tqdm(os.listdir(self.root)) as pbar:
                pbar.set_description('Creating lookup table')
                
                for filename in pbar:
                    
                    if not re.match('.*\.mhd', filename):
                        continue
                        
                    case_number = int(re.match('Case(\d+)', filename).groups()[0])
                    filepath = os.path.join(self.root, filename)
                    
                    if re.search('segmentation', filename):
                        self._lookup['seg_filepath'][case_number] = filepath
                    else:
                        self._lookup['mri_filepath'][case_number] = filepath
                        
        return self._lookup

    @property
    def data(self):
        
        if self._data is None: 
            
            # setup data table
            self._data = pd.DataFrame(
                columns=self._parse_single_case(0).keys(),
                index=pd.Series(range(50), name='case'), 
                dtype='float32'
            )
            
            with tqdm(self._data.index) as pbar:
                pbar.set_description('Getting data statistics')
                
                for i in pbar:
                    self._data.iloc[i] = self._parse_single_case(i)
            
        return self._data
    
    @property
    def data_resampled(self):
        
        if self._data_resampled is None: 
            
            # setup data table
            self._data_resampled = pd.DataFrame(
                columns=self._parse_single_case(0).keys(),
                index=pd.Series(range(50), name='case'),
                dtype='float32'
            )
            
            with tqdm(self._data_resampled.index) as pbar:
                pbar.set_description('Getting resampled data statistics')
                
                for i in pbar:
                    self._data_resampled.iloc[i] = self._parse_single_case(
                        i, resample_spacing=self.reference_spacing
                    )
            
        return self._data_resampled
           
    @property
    def reference_spacing(self):
        
        if self._reference_spacing is None:
            # compute reference spacing
            sizing = pd.Series(list(zip(self.data['S_x'],
                                        self.data['S_y'], 
                                        self.data['S_z'])))
        
            self._reference_spacing = sizing.value_counts().index[0]
            
        return self._reference_spacing
              
    def _parse_single_case(self, case_number, resample_spacing = None):
        
        data = {}
        mri = self.get_sitk_image(case_number, resample_spacing)['mri']
        seg = self.get_sitk_image(case_number, resample_spacing)['seg']
        
        self._validate_mri_seg_pair(mri, seg)
        
        x, y, z = mri.GetSize()
        
        data['D_x'] = x
        data['D_y'] = y
        data['D_z'] = z
        
        x, y, z = mri.GetSpacing()

        data['S_x'] = x
        data['S_y'] = y
        data['S_z'] = z
        
        data['voxel_volume'] = self._get_voxel_volume(mri)
        
        mri_array = self.get_np_array(case_number, resample_spacing)['mri']
        seg_array = self.get_np_array(case_number, resample_spacing)['seg']
        
        data['max_mri_intensity'] = np.max(mri_array)
        
        data['min_mri_intensity'] = np.min(mri_array)
        
        data['mean_mri_intensity'] = np.mean(mri_array)
        
        data['std_mri_intensity'] = np.std(mri_array)
        
        data['seg_total_voxels'] = einops.reduce(seg_array, 'x y z -> ', 'sum')
        
        data['prostate_volume'] = data['voxel_volume'] * data['seg_total_voxels']
        
        data['total_image_volume'] = data['voxel_volume'] * data['D_x'] * data['D_y'] * data['D_z']
        
        return data
         
    def get_direction_matrix(self, case_number: int):
        sitk_images = self.get_sitk_image(case_number)
        mri_matrix = self._get_direction_matrix(sitk_images['mri']) 
        seg_matrix = self._get_direction_matrix(sitk_images['mri'])
        
        assert np.all(mri_matrix == seg_matrix), 'Validation error: direction matrices do not match'
        
        return mri_matrix
     
    def get_sitk_image(self, case_number, 
                       resample_spacing: Optional[Tuple[float, float, float]] = None):
         
        mri = sitk.ReadImage(self.lookup['mri_filepath'][case_number], sitk.sitkFloat32)
        seg = sitk.ReadImage(self.lookup['seg_filepath'][case_number])
        
        if resample_spacing:
            mri = self.resample_volume(mri, resample_spacing)
            seg = self.resample_volume(seg, resample_spacing, 
                                       interpolator=sitk.sitkNearestNeighbor)
                                            # use nearest neighbor interpolator for segmentation
                                            # to roughly preserve volumes
        return {
            'mri': mri, 
            'seg': seg
        }
      
    def get_np_array(self, case_number: int, 
                     resample_spacing: Optional[Tuple[float, float, float]] = None):
        
        sitk_images = self.get_sitk_image(case_number, resample_spacing)
        mri, seg = sitk_images['mri'], sitk_images['seg']
        
        return {
            'mri': sitk.GetArrayViewFromImage(mri), 
            'seg': sitk.GetArrayViewFromImage(seg)
        }
    
    @staticmethod  
    def _validate_mri_seg_pair(mri: sitk.Image, seg: sitk.Image):
        
        assert mri.GetSize() == seg.GetSize(), 'size mismatch'
        assert mri.GetSpacing() == seg.GetSpacing(), 'spacing mismatch'
        
        assert 'float' in str(sitk.GetArrayFromImage(mri).dtype), \
            f'incorrect type {str(sitk.GetArrayFromImage(mri).dtype)} for mri'
        assert 'int8' in str(sitk.GetArrayFromImage(seg).dtype), \
            f'incorrect type {str(sitk.GetArrayFromImage(seg).dtype)} for seg'
      
    @staticmethod
    def resample_volume(volume, new_spacing, interpolator = sitk.sitkLinear):
        """ 
        A method which uses sitk to resample an image to the specified pixel spacing, taken originally
        from : 
        {
            Title: Resample volume to specific voxel spacing - SimpleITK
            Author: Ziv Yaniv
            Date: Oct 2020
            Code version: 0.0
            Availability: https://discourse.itk.org/t/resample-volume-to-specific-voxel-spacing-simpleitk/3531
        }
        modified to accept sitk.Image type as first parameter rather than path to volume.
        """
        original_spacing = volume.GetSpacing()
        original_size = volume.GetSize()
        new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
        return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                             volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                             volume.GetPixelID())
    
    @staticmethod
    def _get_voxel_volume(image: sitk.Image):
        """returns 3-dimensional volume occupied by a voxel in the image"""
        
        det = np.linalg.det(ProstateMRIDataset._get_direction_matrix(image))
        s_x, s_y, s_z = image.GetSpacing()
        
        return np.abs(det) * s_x * s_y * s_z
    
    @staticmethod
    def _get_direction_matrix(image: sitk.Image):
        """returns the direction cosine matrix of the image"""
        
        return einops.rearrange(
            np.array(image.GetDirection()),
            '(i j) -> i j',
            i = 3
        )
    
    
    