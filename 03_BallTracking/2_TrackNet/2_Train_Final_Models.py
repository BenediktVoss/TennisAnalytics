import os
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np
from Dataset import BallDataSet
from BallTrackNet import BallTrackNet
from helpers import train, validate, WeightedBCELoss, BinaryFocalLoss
import wandb

##disable warnings
import warnings
warnings.filterwarnings("ignore")

os.environ["WANDB_API_KEY"] = "WANDB_API_KEY"

config = {
    'LR': 0.2,
    'Optimizer': "Adelta",
    'Loss': "WBCE",
    'HeatMap_size': 10,
    'path': '../../00_Dataset',
    'selected_gpus': [0, 1],
    'batch_size': 32,
    'num_epochs': 80,
    'steps_per_epoch': 200,
    'val_intervals': 4,
    'augmentation': True,
    'input_number': 3,
    'output_number': 3,
    'patience':4
}

# Seed for reproducibility
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def create_dataloaders(path, heatmap_size, augmentation, input_number, output_number, batch_size):
    training = BallDataSet(path=path, split='train', heatmap_size=heatmap_size, augment=augmentation, input_size=input_number, output_size=output_number, recreate_heatmap=False)
    validation = BallDataSet(path=path, split='validation', heatmap_size=heatmap_size, input_size=input_number, output_size=output_number, recreate_heatmap=False)

    train_loader = torch.utils.data.DataLoader(
        training,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4
    )

    val_loader = torch.utils.data.DataLoader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4
    )
    return train_loader, val_loader
    

def setup_paths(exp_id):
    exps_path = f'./exps_final/{exp_id}'
    if not os.path.exists(exps_path):
        os.makedirs(exps_path)
    
    paths = {
        "model_last": os.path.join(exps_path, 'model_last.pt'),
        "model_best": os.path.join(exps_path, 'model_best.pt'),
        "log": os.path.join(exps_path, 'training_log.txt')
    }
    return paths
    

def training(config=None):
    set_seed(SEED)

    with wandb.init(config=config, project='ball-tracking') as run:
        config = wandb.config
        best_f1 = 0
        no_improvement_steps = 0  # Counter for early stopping

    
        # load hyperparameters
        learning_rate = config.LR
        optimizer = config.Optimizer
        heatmap_size = config.HeatMap_size
        path = config.path
        batch_size = config.batch_size
        num_epochs = config.num_epochs
        steps_per_epoch = config.steps_per_epoch
        val_intervals = config.val_intervals
        augmentation = config.augmentation
        selected_gpus = config.selected_gpus
        input_number = config.input_number
        output_number = config.output_number
        patience = config.patience  # Maximum allowed steps without improvement

        #create name based aon augment, input and output number
        exp_id = "TrackNet_"+ str(input_number) + "-in-" + str(output_number) + "-out_aug_" + str(augmentation)
        

        # Set up paths for experiment and logging
        paths = setup_paths(exp_id)

        # Start logging to a file
        with open(paths["log"], 'w') as log_file:
            log_file.write("Run,Epoch,Train_Loss,Learning_Rate,Val_Loss,Accuracy,Precision,Recall,F1_Score\n")

            # Create data loaders
            train_loader, val_loader = create_dataloaders(
                path, heatmap_size, augmentation, input_number, output_number, batch_size
            )

            # Set up the model
            model = BallTrackNet(input_size=input_number, output_size=output_number)
            model = nn.DataParallel(model, device_ids=selected_gpus)
            model = model.to(f'cuda:{selected_gpus[0]}')

            # Set the primary device (first GPU in the list)
            device = torch.device(f'cuda:{selected_gpus[0]}')

            #set optimizer and criterion
            optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
            criterion = WeightedBCELoss()

            # Define the learning rate scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=patience//2)
            
            # train epochs
            for epoch in range(1, num_epochs + 1):
                # set epoch seed
                set_seed(SEED + epoch)
                
                # Training step
                train_loss = train(model, train_loader, optimizer, device, epoch, steps_per_epoch, criterion=criterion)
                print(f'Epoch {epoch} - Train Loss: {train_loss:.4f} - Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
                log_file.write(f"{run.id},{epoch},{train_loss},{optimizer.param_groups[0]['lr']}")
                wandb.log({"train_loss": train_loss, "learning_rate": optimizer.param_groups[0]["lr"], "epoch": epoch})

                # Validation step
                if epoch > 0 and epoch % val_intervals == 0:
                    val_loss, accuracy, precision, recall, f1 = validate(model, val_loader, device, criterion=criterion)
                    print(f'Epoch {epoch} - Val Loss: {val_loss:.4f} - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}')
                    log_file.write(f",{val_loss},{accuracy},{precision},{recall},{f1}\n")
                    wandb.log({"val_loss": val_loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "epoch": epoch})

                    scheduler.step(val_loss)

                    # Save the best model
                    if f1 > best_f1:
                        #save f1 and config
                        best_f1 = f1
                        no_improvement_steps = 0  # Reset counter
                        #save model
                        torch.save(model.state_dict(), paths["model_best"])
                    else:
                        no_improvement_steps += 1  # Increment counter

                    # Early stopping check
                    if no_improvement_steps >= patience:
                        print("Early stopping triggered. No improvement in F1 score for", patience, "validation steps.")
                        break
                else:
                    log_file.write("\n")  # Log training loss only if no validation

                # Save the latest model checkpoint
                torch.save(model.state_dict(), paths["model_last"])

    # Finish the WandB run
    wandb.finish()


if __name__ == '__main__':
    # define configurations
    versions = [
        {"augmentation": True, "input_number": 3, "output_number": 3},
        {"augmentation": False, "input_number": 3, "output_number": 3},
        {"augmentation": True, "input_number": 3, "output_number": 1},
        {"augmentation": False, "input_number": 3, "output_number": 1},
        {"augmentation": True, "input_number": 5, "output_number": 3},
        {"augmentation": False, "input_number": 5, "output_number": 3},
        {"augmentation": True, "input_number": 5, "output_number": 5},
        {"augmentation": False, "input_number": 5, "output_number": 5}
    ]

    for version in versions:
        #modify config
        config['augmentation'] = version['augmentation']
        config['input_number'] = version['input_number']
        config['output_number'] = version['output_number']

        #train model
        training(config)
        