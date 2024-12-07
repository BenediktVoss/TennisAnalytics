import os
import torch.nn as nn
import torch
import torch.optim as optim
from Dataset import BallDataSet
from BallTrackNet import BallTrackNet
from helpers import train, validate, WeightedBCELoss, BinaryFocalLoss
import wandb

##disable warnings
import warnings
warnings.filterwarnings("ignore")

os.environ["WANDB_API_KEY"] = "WANDB_API_KEY"

sweep_config = {
    'method': 'bayes',  
    'metric': {
        'name': 'f1_score',
        'goal': 'maximize'   
    },
    'parameters': {
        'LR': {'distribution': 'uniform', 'min': 0.001, 'max': 1},
        'Optimizer': {'values': ["Adam", "Adelta"]},
        'HeatMap_size': {'values': [5, 10]},
        'Loss': {'values': ["Focal", "WBCE"]},
        'path': {'value': '../../00_Dataset'},
        'selected_gpus': {'value': [0, 1]},
        'batch_size': {'value': 32},
        'num_epochs': {'value': 16},
        'steps_per_epoch': {'value': 200}, # use entire dataset
        'val_intervals': {'value': 4},
        'augmentation': {'value': True}
    }
}

bestF1 = 0
best_config = None
sweep_id = None

def hyperparameter_training(config=None):
    global bestF1
    global best_config
    global sweep_id

    with wandb.init(config=config) as run:
        config = wandb.config

        # load hyperparameters
        learning_rate = config.LR
        optimizer = config.Optimizer
        loss = config.Loss
        heatmap_size = config.HeatMap_size
        path = config.path
        batch_size = config.batch_size
        num_epochs = config.num_epochs
        steps_per_epoch = config.steps_per_epoch
        val_intervals = config.val_intervals
        augmentation = config.augmentation
        selected_gpus = config.selected_gpus
        exp_id = sweep_id + '_' + run.id

        # Set up paths for experiment and logging
        exps_path = f'./exps_hyperparameter/{exp_id}'
        if not os.path.exists(exps_path):
            os.makedirs(exps_path)
        model_last_path = os.path.join(exps_path, 'model_last.pt')
        model_best_path = os.path.join(exps_path, 'model_best.pt')
        log_path = os.path.join(exps_path, 'training_log.txt')

        # Start logging to a file
        with open(log_path, 'w') as log_file:
            log_file.write("Run,Epoch,Train_Loss,Learning_Rate,Val_Loss,Accuracy,Precision,Recall,F1_Score\n")

            # Create the dataset
            training = BallDataSet(path=path, split='train', heatmap_size=heatmap_size, augment=augmentation)
            validation = BallDataSet(path=path, split='validation', heatmap_size=heatmap_size)

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

            # Set up the model
            model = BallTrackNet()
            model = nn.DataParallel(model, device_ids=selected_gpus)
            model = model.to(f'cuda:{selected_gpus[0]}')

            # Set the primary device (first GPU in the list)
            device = torch.device(f'cuda:{selected_gpus[0]}')

            # create optimizer based on config
            if optimizer == "Adam":
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            elif optimizer == "Adelta":
                optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
            else:
                raise ValueError("Invalid optimizer")


            #create criterion based on config
            if loss == "Focal":
                criterion = BinaryFocalLoss()
            elif loss == "WBCE":
                criterion = WeightedBCELoss()
            else:
                raise ValueError("Invalid loss function")
            
            
            # train epochs
            for epoch in range(1, num_epochs + 1):
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

                    # Save the best model
                    if f1 > bestF1:
                        #save f1 and config
                        bestF1 = f1
                        best_config = config
                        #save model
                        torch.save(model.state_dict(), model_best_path)
                else:
                    log_file.write("\n")  # Log training loss only if no validation

                # Save the latest model checkpoint
                torch.save(model.state_dict(), model_last_path)

    # Finish the WandB run
    wandb.finish()


# Main function
if __name__ == '__main__':
    # Set up the sweep
    sweep_id = wandb.sweep(sweep_config, project="ball-tracking")

    # Run the sweep
    wandb.agent(sweep_id, function=hyperparameter_training, count=16)