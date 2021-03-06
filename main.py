
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
from datetime import datetime
import os

from src.predict import predict

@hydra.main(config_path='configs', config_name='config')
def main(config: DictConfig):
    
    # ======= LOGGING =======
    from datetime import datetime
    time = datetime.now().strftime('%d_%b_%H:%M:%S')
    exp_name = f'exp_{time}'

    if config.log:
        wandb.init(
            config=OmegaConf.to_object(config), 
            name=exp_name,
            reinit=True,
            **config.wandb
        )
        
    # ======= MODEL =========
    from src.unet import UNet
    model = UNet(1, 2)
    
    # ======= DATA ==========
    from src.data import dataloaders
    train_dl, val_dl, test_dl = dataloaders(
        config.data.root, 
        config.data.batch_size, 
        config.data.split_seed, 
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
        
    # ======= CRITERION ========= 
    from src.loss import SegmentationCriterion
    criterion = SegmentationCriterion(config.dice_loss_weight)

    # ======= EARLY STOPPING ====
    from src.early_stopping import EarlyStopping
    if config.early_stopping: 
        save_path = os.path.join(
            os.getcwd(), 
            'checkpoint.pt'
        )
        early_stopping = EarlyStopping(
            path=save_path, 
            **config.early_stopping
        )

    # ======= TRAINING ==========
    from src.train import train
    from src.metrics import Metrics
    
    if not config.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.gpu)
    
    train(
        config.training.num_epochs, 
        model,          
        criterion, 
        Metrics(), 
        Metrics(), 
        Metrics(), 
        opt, 
        train_dl, 
        val_dl, 
        test_dl, 
        device, 
        accumulate_grad_batches=config.training.accumulate_grad_batches, 
        log_fn = wandb.log if config.log else None, 
        log_images=config.log_images, 
        scheduler = sched, 
        early_stopping=early_stopping
    )
    
    if config.save_model:
        torch.save('model.pt')
        wandb.save('model.pt')
    
    # ======== PREDICT ON TEST DATA ===
    report = predict(model, test_dl.dataset, device) 
    from src.logging import log_report, log_mri_with_mask_overlay
    if config.log_images: 
        log_report(report)
    log_mri_with_mask_overlay(report, case_num=33)
    
if __name__ == '__main__':
    main()