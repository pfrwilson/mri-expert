
from matplotlib.pyplot import hist
from tqdm import tqdm
import wandb
from torch import nn
from torch.optim import Optimizer 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torch
from typing import Optional
from .early_stopping import EarlyStopping
from .logging import log_images_with_masks


def train_epoch(epoch_idx: int, model: nn.Module, criterion, metrics: nn.Module,
          optim: Optimizer, dataloader: DataLoader, device,
          accumulate_grad_batches=1) -> dict:
    
    model = model.train().to(device)
    metrics = metrics.to(device)
    
    with tqdm(dataloader, desc=f'Train epoch {epoch_idx}') as pbar:
        for i, batch in enumerate(pbar):
        
            image, mask = batch
            image = image.to(device)
            mask = mask.to(device)
            
            logits = model(image)
            
            loss = criterion(logits, mask)
            loss.backward()

            if i % accumulate_grad_batches == 0: 
                optim.step()
                optim.zero_grad()
                
            if i % 10 == 0:
                pbar.set_postfix({'loss': loss.item()})
    
            metrics(logits, mask)
            
    history = metrics.compute()
    metrics.reset()
    return history
        
        
def eval_epoch(model: nn.Module, metrics: nn.Module, criterion, 
               dataloader: DataLoader, device) -> dict:
    
    model = model.eval().to(device)
    metrics = metrics.to(device)
    
    with torch.no_grad():
        with tqdm(dataloader, desc=f'Evaluating') as pbar:
            for i, batch in enumerate(pbar):
                
                image, mask = batch
                image = image.to(device)
                mask = mask.to(device)
                
                logits = model(image)
                
                loss = criterion(logits, mask)
                if i % 10 == 0: 
                    pbar.set_postfix({'loss': loss.item()})
                
                metrics(logits, mask)
    
    history = metrics.compute()
    metrics.reset()
    return history

    
def train(num_epochs, 
          model: nn.Module, 
          criterion, 
          train_metrics: nn.Module,
          eval_metrics: nn.Module, 
          test_metrics: nn.Module,
          optim: Optimizer, 
          train_dataloader: DataLoader, 
          val_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          device: torch.DeviceObjType,
          accumulate_grad_batches=1, 
          log_fn=None, 
          log_images=True,
          scheduler: Optional[_LRScheduler]=None, 
          early_stopping = None
          ):

    for epoch in range(num_epochs):
        
        train_history = train_epoch(
            epoch, 
            model, 
            criterion, 
            train_metrics, 
            optim, 
            train_dataloader, 
            device, 
            accumulate_grad_batches
        )

        if log_fn: 
            log_fn({f'train/{k}': v for k, v in train_history.items()})
        
        val_history = eval_epoch(
            model, 
            eval_metrics, 
            criterion, 
            val_dataloader, 
            device
        )
        
        if log_fn: 
            log_fn({f'val/{k}': v for k, v in val_history.items()})
        
        if log_images:
            log_images_with_masks('val/images', model, val_dataloader, 4, True, device)

        if early_stopping: 
            early_stopping(val_history['loss'], model)

        if scheduler:
            scheduler.step()
            if log_fn:     
                log_fn({'lr': scheduler.get_last_lr()[0]})
    
        if log_fn: 
            log_fn({'epoch': epoch})

        if early_stopping and early_stopping.early_stop: 
            break

    test_history = eval_epoch(model, test_metrics, criterion, test_dataloader, device)
    if log_fn: 
        log_fn({f'test/{k}': v for k, v in test_history.items()})

    if log_images:
        log_images_with_masks('test/images', model, test_dataloader, 12, True, device)