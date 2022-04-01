
import hydra
from omegaconf import DictConfig
import wandb
import torch

@hydra.main(config_path='configs', config_name='config')
def main(config: DictConfig):
    
    # ======= LOGGING =======
    if config.log:
        wandb.init(
            config=config, 
            **config.wandb
        )
    if config.log_images:
        from src.logging import LogMasksCallback
        callback = LogMasksCallback(num_images=4, choose_randomly=True)
    else: 
        callback = None 
        
    # ======= MODEL =========
    from src.unet import UNet
    model = UNet(1, 2)
    
    # ======= DATA ==========
    from src.data import dataloaders
    train_dl, val_dl = dataloaders(
        config.data.root, 
        config.data.batch_size, 
        config.data.split_seed, 
        config.data.val_size, 
        config.data.target_size, 
        config.data.use_augmentations, 
        **config.data.preprocessor_kwargs
    )
    
    # ======= OPTIMIZER =========
    from torch.optim import Adam
    opt = Adam(model.parameters(), lr=config.optim.lr)
    
    if config.optim.scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=config.training.num_epochs)
    else: 
        sched = None
        
    # ======= TRAINING ==========
    from src.train import train
    from src.metrics import Metrics
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train(
        config.training.num_epochs, 
        model,
        Metrics(), 
        Metrics(), 
        opt, 
        train_dl, 
        val_dl, 
        device, 
        accumulate_grad_batches=config.training.accumulate_grad_batches, 
        log_fn = wandb.log if config.log else None, 
        scheduler = sched, 
        epoch_end_callback=callback
    )
    
    if config.save_model:
        torch.save('model.pt')
        wandb.save('model.pt')
    
        
if __name__ == '__main__':
    main()