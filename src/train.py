
from matplotlib.pyplot import hist
from tqdm import tqdm
import wandb
from torch import nn
from torch.optim import Optimizer 
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import cross_entropy



def train_epoch(epoch_idx: int, model: nn.Module, metrics: nn.Module,
          optim: Optimizer, dataloader: DataLoader, device,
          accumulate_grad_batches=1) -> dict:
    
    model = model.train().to(device)
    metrics = metrics.to(device)
    
    with tqdm(dataloader, desc=f'Train epoch {epoch_idx}') as pbar:
        for (i, batch) in pbar:
        
            image, mask = batch
            image = image.to(device)
            mask = mask.to(device)
            
            logits = model(image)
            
            loss = cross_entropy(logits, mask)
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
        
        
def eval_epoch(model: nn.Module, metrics: nn.Module,
               dataloader: DataLoader, device) -> dict:
    
    model = model.eval().to(device)
    metrics = metrics.to(device)
    
    with torch.no_grad():
        with tqdm(dataloader, desc=f'Evaluating') as pbar:
            for (i, batch) in pbar:
                
                image, mask = batch
                image = image.to(device)
                mask = mask.to(device)
                
                logits = model(image)
                
                loss = cross_entropy(logits, mask)
                if i % 10 == 0: 
                    pbar.set_postfix({'loss': loss.item()})
                
                metrics(logits, mask)
    
    history = metrics.compute()
    metrics.reset()
    return history
    
    
def train(num_epochs, model: nn.Module, train_metrics: nn.Module,
          eval_metrics: nn.Module,
          optim: Optimizer, train_dataloader: DataLoader, 
          val_dataloader: DataLoader, device,
          accumulate_grad_batches=1, log_fn=None):
    
    for epoch in range(num_epochs):
        
        history = train_epoch(
            epoch, 
            model, 
            train_metrics, 
            optim, 
            train_dataloader, 
            device, 
            accumulate_grad_batches
        )

        if log_fn: 
            log_fn(history)
        
        history = eval_epoch(
            model, 
            eval_metrics, 
            val_dataloader, 
            device
        )
        
        if log_fn: 
            log_fn(history)
    
