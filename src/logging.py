
import wandb
from torch.utils.data import DataLoader
import torch
import skimage
from .data.preprocess import Preprocessor
import numpy as np

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
  

class LogMasksCallback: 
    
    def __init__(self, num_images, choose_randomly=False):
        self.num_images=num_images
        self.choose_randomly=choose_randomly
        
    def __call__(self, model: torch.nn.Module, 
                 train_dl: DataLoader, val_dl: DataLoader, device):
        
        ds = val_dl.dataset
        
        preprocessor = Preprocessor(use_augmentations=False)
        raw_preprocessor = Preprocessor(use_augmentations=False, equalize_hist=False, to_tensor=False)

        ds.preprocess = preprocessor

        if self.choose_randomly:
            choices = np.random.choice(range(len(ds)), self.num_images)
        else:
            choices = range(self.num_images)
        
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
        
        logits = model(mris.to(device))
        pred_masks = torch.argmax(logits, dim=1)
        
        images = []
        for raw_mri, pred_mask, mask in zip(raw_mris, pred_masks, masks):
            
            raw_mri = raw_mri.cpu().detach().numpy()
            pred_mask = pred_mask.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
            
            images.append(wb_mask(raw_mri, pred_mask, mask))
        
        wandb.log({'predictions': images})
        