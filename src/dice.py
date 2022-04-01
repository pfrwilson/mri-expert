import einops
import torch

def dice_score(mask, target_mask, reduce_across_batch=True):
    
    intersection = mask * target_mask
    
    size_of_intersection = einops.reduce(
        intersection, 
        'b h w -> b', 
        reduction='sum'
    )
    
    size_of_mask = einops.reduce(
        mask, 
        'b h w -> b', 
        reduction='sum'
    )
    
    size_of_target_mask = einops.reduce(
        target_mask, 
        'b h w -> b', 
        reduction='sum'
    )
    
    dice_scores = 2 * size_of_intersection / ( size_of_mask + size_of_target_mask)
    
    if reduce_across_batch:
        dice_scores = torch.sum(dice_scores)
        
    return dice_scores
    
    