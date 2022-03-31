from prostate_mri import ProstateMRIDataset
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import einops
import os

PLOT_DIR = './plots'


def plot_columns(dataset, *columns, units='mm'):
    
    fig, axes = plt.subplots(1, len(columns), figsize=(len(columns) * 4, 4))
    
    for i, axis in enumerate(axes):
        series: pd.Series = dataset.data[columns[i]]
        x = series.index,
        y = series
        axis.scatter(x, y)
        axis.set_xlabel('case no.')
        axis.set_ylabel(f'{columns[i]} {units}')
        axis.set_title(f'{columns[i].capitalize()} by case no.')

    fig.tight_layout()
    
    
def main():
    dataset = ProstateMRIDataset('./TrainingData')
    
    # =========================================================
    # plotting distributions of features across cases
    # =========================================================
    
    plot_columns(dataset, 'D_x', 'D_y', 'D_z')
    plt.savefig(os.path.join(PLOT_DIR, 'DimensionScatterPlot'))
    plt.close()
    
    columns = ['S_x', 'S_y', 'S_z']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, column in enumerate(columns):
        mask = np.abs(dataset.data[column] - dataset.reference_spacing[i]) < 1e-3
        x = dataset.data[column].index
        y_ref = dataset.data[column].where(mask, None)
        y = dataset.data[column].where(~mask, None)
        axes[i].scatter(x, y)
        axes[i].scatter(x, y_ref, color='red')
        axes[i].set_xlabel('case no.')
        axes[i].set_ylabel(column)
        axes[i].set_title(f'{column} by Case Number')
    
    plt.tight_layout()        
    plt.savefig(os.path.join(PLOT_DIR, 'SpacingScatterPlot'))
    plt.close()
    
    x = dataset.data.prostate_volume.index
    y = dataset.data.prostate_volume
    plt.scatter(x, y)
    plt.xlabel('Case number')
    plt.ylabel('prostate volume (mm^3)')
    plt.title('Total Prostate Volume by Case Number')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'ProstateVolumes'))
    plt.close()
    
    x = dataset.data['min_mri_intensity'].index
    y = dataset.data['min_mri_intensity']
    plt.scatter(x, y)
    plt.xlabel('Case number')
    plt.ylabel('Minimum pixel intensity')
    plt.title('Distribution of minimum pixel intensities')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'MinIntensityDistribution'))   
    plt.close()
    
    x = dataset.data['max_mri_intensity'].index
    y = dataset.data['max_mri_intensity']
    plt.scatter(x, y)
    plt.xlabel('Case number')
    plt.ylabel('Maximum pixel intensity')
    plt.title('Distribution of Maximum Pixel Intensity over Cases')
    plt.savefig(os.path.join(PLOT_DIR, 'MaxIntensityDistribution'))
    plt.close()
    
    # ===========================================
    # Analysing mri histogram within 1 case
    # =========================================
    
    # get mri and mask as np array
    arrays = dataset.get_np_array(33)
    mri_array = arrays['mri']
    seg_array = arrays['seg']
    
    # find index of largest prostate area
    areas = einops.reduce(seg_array, 'z y x -> z', 'sum')
    max_index = np.argmax(areas, axis=0)
    roi_mri_slice = mri_array[max_index]
    roi_seg_slice = seg_array[max_index]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    
    masked = np.where(roi_seg_slice == 1, roi_mri_slice, np.NAN)
    bins = range(0, 1200, 50)
    
    ax1.imshow(roi_mri_slice)
    ax1.set_xticks([])
    ax1.set_xticks([])
    ax1.set_title('MRI ROI Slice')

    ax2.hist(roi_mri_slice.flatten(), bins=bins)
    ax2.set_xlabel('Intensity Bins (pixels)')
    ax2.set_ylabel('Pixels per bin')
    ax2.set_title('Histogram of Pixel Intensities over ROI Slice')

    ax3.imshow(masked)
    ax3.set_xticks([])
    ax3.set_xticks([])
    ax3.set_title('MRI ROI Slice, Masked to Prostate Only')
    
    ax4.hist(masked.flatten(), bins=bins)
    ax4.set_xlabel('Intensity Bins (pixels)')
    ax4.set_ylabel('Pixels per bin')
    ax4.set_title('Histogram of Pixel Intensities in Prostate Region')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'Case33IntensityHistogram'))
    
    # ======================================================
    # Analyzing mri histogram within 1 case with resampling
    # =====================================================
    
    # get mri and mask as np array
    arrays = dataset.get_np_array(
        33, resample_spacing=dataset.reference_spacing
    )
    mri_array = arrays['mri']
    seg_array = arrays['seg']
    
    # find index of largest prostate area
    areas = einops.reduce(seg_array, 'z y x -> z', 'sum')
    max_index = np.argmax(areas, axis=0)
    roi_mri_slice = mri_array[max_index]
    roi_seg_slice = seg_array[max_index]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    
    masked = np.where(roi_seg_slice == 1, roi_mri_slice, np.NAN)
    bins = range(0, 1200, 50)
    
    ax1.imshow(roi_mri_slice)
    ax1.set_xticks([])
    ax1.set_xticks([])
    ax1.set_title('MRI ROI Slice')

    ax2.hist(roi_mri_slice.flatten(), bins=bins)
    ax2.set_xlabel('Intensity Bins (pixels)')
    ax2.set_ylabel('Pixels per bin')
    ax2.set_title('Histogram of Pixel Intensities over ROI Slice')

    ax3.imshow(masked)
    ax3.set_xticks([])
    ax3.set_xticks([])
    ax3.set_title('MRI ROI Slice, Masked to Prostate Only')
    
    ax4.hist(masked.flatten(), bins=bins)
    ax4.set_xlabel('Intensity Bins (pixels)')
    ax4.set_ylabel('Pixels per bin')
    ax4.set_title('Histogram of Pixel Intensities in Prostate Region')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'Case33IntensityHistogram_Resampled'))
    
    
if __name__ == '__main__':
    main()