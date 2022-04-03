
import wandb
from torch.utils.data import DataLoader
import torch
from .data.dataset import PCASegmentationDataset
import skimage
from .data.preprocess import Preprocessor
import numpy as np
from .predict import Report
from dataclasses import asdict

segmentation_classes = ['background', 'prostate']

def labels():
    l = {}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    return l


def wb_mask(bg_img, pred_mask, true_mask):
    return wandb.Image(bg_img, masks={
        "prediction" : {"mask_data" : pred_mask, "class_labels" : labels()},
        "ground truth" : {"mask_data" : true_mask, "class_labels" : labels()}})
  

def log_images_with_masks(name, model, dataloader, num_images, choose_randomly, device):

    ds = dataloader.dataset
    
    if choose_randomly:
        choices = np.random.choice(range(len(ds)), num_images)
    else:
        choices = range(num_images)

    preprocessor = Preprocessor(use_augmentations=False)
    raw_preprocessor = Preprocessor(use_augmentations=False, equalize_hist=False, to_tensor=False)

    # get batch for inference
    batch = torch.utils.data._utils.collate.default_collate(
        [ds[choice] for choice in choices]
    )

    # get raw batch to show unprocessed mri image
    ds.preprocess = raw_preprocessor
    
    raw_batch = torch.utils.data._utils.collate.default_collate(
        [ds[choice] for choice in choices]
    )

    mris, masks = batch
    raw_mris, _ = raw_batch
    
    with torch.no_grad():
        logits = model(mris.to(device))

    pred_masks = torch.argmax(logits, dim=1)
    
    images = []
    for raw_mri, pred_mask, mask in zip(raw_mris, pred_masks, masks):
        
        raw_mri = raw_mri.cpu().detach().numpy()
        pred_mask = pred_mask.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        
        images.append(wb_mask(raw_mri, pred_mask, mask))

    wandb.log({name: images})


def log_report(report: Report):
    
    wandb.log(
        {f'predict/{k}': v for k, v in asdict(report.overall_metrics).items()}
    )
    
    for case, case_report in report.case_reports.items():
        
        case_images = []
        
        for data in case_report.slice_predictions:
            
            case_images.append( 
                wb_mask(
                    data.mri, 
                    data.pred_mask, 
                    data.true_mask
                ) 
            ) 
            
        wandb.log({f'case_{case}': case_images})
            
            
