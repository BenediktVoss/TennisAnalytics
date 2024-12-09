from court_dataset import CourtDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision.transforms import functional as F
import time
import numpy as np
import random
import os
import wandb

##disable warnings
import warnings
warnings.filterwarnings("ignore")

os.environ["WANDB_API_KEY"] = "WANDB_API_KEY"


def create_dataloaders(dataset_path, batch_size):
    train_dataset = CourtDataset(
        path=dataset_path,
        split="train",
        input_height=720,
        input_width=1280,
        model_height=288,
        model_width=512,
        augment=True,
        selected_points=None
    )
    
    val_dataset = CourtDataset(
        path=dataset_path,
        split="validation",
        input_height=720,
        input_width=1280,
        model_height=288,
        model_width=512,
        augment=False,
        selected_points=None
    )

    test_dataset = CourtDataset(
        path=dataset_path,
        split="test",
        input_height=720,
        input_width=1280,
        model_height=288,
        model_width=512,
        augment=False,
        selected_points=None
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


def create_mobilenetv3small(num_coordinates=15):
    # Load pretrained MobileNetV3-Small
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    
    # Extract the number of features output by the last convolutional layer
    num_features = model.classifier[0].in_features  # Fixed to handle changes in feature size dynamically
    
    # Modify the classification head
    num_outputs = num_coordinates * 2  # Each coordinate has (x, y)
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 256),  # Hidden layer
        nn.ReLU(),
        nn.Linear(256, num_outputs)  # Output layer for coordinates
    )
    return model


def create_resnet50(num_coordinates=15):
    # Load pretrained ResNet50
    model = models.resnet50(weights='IMAGENET1K_V2')
    
    # Modify the output layer
    num_outputs = num_coordinates * 2  
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256), 
        nn.ReLU(),
        nn.Linear(256, num_outputs)           
    )
    return model


def train(model, train_loader, optimizer, device, epoch, criterion=torch.nn.MSELoss()):
    start_time = time.time()
    losses = []

    # Wrap train_loader with tqdm for a progress bar
    with tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch") as pbar:
        for batch in train_loader:
            optimizer.zero_grad()
            model.train()

            # Forward pass
            inputs = batch[0].float().to(device)  # Input frames (images)
            keypoints_gt = batch[1].float().to(device)  # Ground truth keypoints
            outputs = model(inputs)  # Predicted keypoints
            
            # Compute loss
            loss = criterion(outputs, keypoints_gt.view(outputs.size()))  # Match shapes for regression
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track losses
            losses.append(loss.item())
            duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

            # Update the tqdm bar
            pbar.set_postfix({
                'loss': round(loss.item(), 6),
                'time': duration
            })
            pbar.update(1)
            
    return np.mean(losses)

def compute_mse(points, positions, img_size):
    #loop over points
    mse = []

    for i in range(len(points)):
        # get the point
        point = points[i]
        # get the position
        position = positions[i]

        # if the point is outside the image, skip
        if point[0] < 0 or point[0] >= img_size[0] or point[1] < 0 or point[1] >= img_size[1]:
            continue

        # if the position is -1,-1 return the maximum distance
        if position[0] == -1 and position[1] == -1:
            continue
        
        # calculate the distance
        distance = np.linalg.norm(np.array(point) - np.array(position))

        # add to mse
        mse.append(distance**2)

    # return the mean
    return np.mean(mse)


def compute_counts(points, positions, img_size, threshold=4):
    # check that len is the same
    assert len(points) == len(positions) 
    
    #loop over points
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(len(points)):
        # get the point
        point = points[i]
        # get the position
        position = positions[i]
        
        # calculate the distance
        distance = np.linalg.norm(np.array(point) - np.array(position))

        # add to tp, fp, fn, tn
        if distance <= threshold:
            # if point is outside the frame
            if point[0] < 0 or point[0] >= img_size[0] or point[1] < 0 or point[1] >= img_size[1]:
                tn += 1
            else:
                tp += 1
        else:
            #if point is outside the frame
            if point[0] < 0 or point[0] >= img_size[0] or point[1] < 0 or point[1] >= img_size[1]:
                fn += 1
            else:
                fp += 1

    # return the metrics
    return tp, fp, tn, fn


def calculate_metrics(tp, fp, tn, fn):
    # Avoid division by zero for precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1-score calculation
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Accuracy calculation
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    

def validate(model, val_loader, device, criterion=torch.nn.MSELoss(), threshold=4, img_size=(1280, 720)):
    model.eval()  # Set the model to evaluation mode
    tp, fp, tn, fn = 0, 0, 0, 0  # Initialize counts
    losses = []
    mse_scores = []

    with tqdm(total=len(val_loader), desc="Validation", unit="batch") as pbar:
        for batch_idx, batch in enumerate(val_loader):
            with torch.no_grad():
                # Prepare data
                inputs = batch[0].float().to(device)  # Input images
                keypoints_gt = batch[1].float().to(device)  # Ground truth keypoints
                outputs = model(inputs)  # Predicted keypoints
                
                # Compute loss
                loss = criterion(outputs, keypoints_gt.view(outputs.size()))  # Match shapes for regression
                losses.append(loss.item())

                # First loop: Iterate over each item in the batch
                for i in range(outputs.shape[0]):  # Loop over batch samples
                    # get predictions
                    positions = outputs[i].view(-1, 2).cpu().numpy()
                    # move to cpu
                    keypoints = keypoints_gt[i].cpu().numpy()
                    
                    #Compute MSE for the batch
                    mse = compute_mse(keypoints, positions, img_size)
                    mse_scores.append(mse)

                    # Compute TP, FP, TN, FN for the batch
                    item_tp, item_fp, item_tn, item_fn = compute_counts(keypoints, positions, img_size, threshold)
                    tp += item_tp
                    fp += item_fp
                    tn += item_tn
                    fn += item_fn

                # Update the tqdm bar
                pbar.set_postfix({
                    'loss': round(np.mean(losses), 6)
                })
                pbar.update(1)

    # Calculate overall metrics
    mean_loss = np.mean(losses)
    mean_mse = np.mean(mse_scores)
    metrics = calculate_metrics(tp, fp, tn, fn)

    return mean_loss, mean_mse, metrics

# Seed for reproducibility
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def setup_paths(exp_id):
    exps_path = f'./exps_hyperparameter/{exp_id}'
    if not os.path.exists(exps_path):
        os.makedirs(exps_path)
    
    paths = {
        "model_last": os.path.join(exps_path, 'model_last.pth'),
        "model_best": os.path.join(exps_path, 'model_best.pth'),
        "log": os.path.join(exps_path, 'training_log.txt')
    }
    return paths


def training(config= None):
    with wandb.init(config=config) as run:
        config = wandb.config
        # get run name
        run_name = run.name
        set_seed(SEED)
        best_val_loss = float("inf")

        #fixed params
        no_improvement_steps = 0  
        val_intervals = 2
        selected_gpus = [0,1]
        patience = 10 
        dataset_path = "../../00_Dataset"
        batch_size = 64
        num_coordinates = 15  # Number of keypoints
        num_epochs = 200

        #tuned params
        learning_rate = config.learning_rate
        model_type = config.model_type  # Options: "mobilenetv3small", "resnet50"
        optimizer_type = config.optimizer_type  # Options: "adam", "adadelta"
        loss_fn = config.loss_fn  # Options: "mse", "l1"
        
        #create name based aon augment, input and output number
        exp_id = f"{run_name}_{model_type}_{optimizer_type}_{loss_fn}"
        
        # Set up paths for experiment and logging
        paths = setup_paths(exp_id)

        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Start logging to a file
        with open(paths["log"], 'w') as log_file:
            log_file.write("Epoch,Train_Loss,Val_Loss\n")

            # Create data loaders
            train_loader, val_loader, test_loader = create_dataloaders(dataset_path, batch_size)

            # select model
            if model_type == "mobilenetv3small":
                model = create_mobilenetv3small(num_coordinates=num_coordinates)
            elif model_type == "resnet50":
                model = create_resnet50(num_coordinates=num_coordinates)
            else:
                raise ValueError(f"Invalid model type: {model_type}")
            
            # Model setup
            model = nn.DataParallel(model, device_ids=selected_gpus)
            model = model.to(f'cuda:{selected_gpus[0]}')
            
            # Loss function and optimizer
            if loss_fn == "mse":
                criterion = nn.MSELoss()
            elif loss_fn == "l1":
                criterion = nn.L1Loss()
            else:
                raise ValueError(f"Invalid loss function: {loss_fn}")

            # Optimizer                                             
            if optimizer_type == "adam":
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_type == "adadelta":
                optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
            else:
                raise ValueError(f"Invalid optimizer type: {optimizer_type}")
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=3
            )

            # Set the primary device (first GPU in the list)
            device = torch.device(f'cuda:{selected_gpus[0]}')
            
            # train epochs
            for epoch in range(1, num_epochs + 1):
                # set epoch seed
                set_seed(SEED + epoch)
                
                # Training step
                train_loss = train(model, train_loader, optimizer, device, epoch, criterion=criterion)
                log_file.write(f"{epoch},{train_loss}")
                # log to wandb
                wandb.log({'train_loss': train_loss, 'epoch': epoch})

                # Validation step
                if epoch > 0 and epoch % val_intervals == 0:
                    val_loss, mean_mse, metrics = validate(model, val_loader, device, criterion=criterion)
                    log_file.write(f",{val_loss},{mean_mse},{metrics['accuracy']},{metrics['f1']}\n")
                    # log to wandb
                    wandb.log({'val_loss': val_loss, 'mean_mse': mean_mse, 'accuracy': metrics['accuracy'], 'f1': metrics['f1'], 'epoch': epoch})

                    # Step the scheduler based on validation loss
                    scheduler.step(val_loss)

                    # Save the best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improvement_steps = 0  # Reset counter
                        #save model
                        torch.save(model.module.state_dict(), paths["model_best"])
                    else:
                        no_improvement_steps += 1  # Increment counter

                    # Early stopping check
                    if no_improvement_steps >= patience:
                        print("Early stopping triggered. No improvement in validation for", patience, "validation steps.")
                        break
                else:
                    log_file.write("\n")  # Log training loss only if no validation

                # Save the latest model checkpoint
                torch.save(model.module.state_dict(), paths["model_last"])

            
    # Finish the WandB run
    wandb.finish()


def run_sweep():
    sweep_config = {
        "method": "bayes", 
        "metric": {"name": "f1", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"distribution": "log_uniform_values", "min": 0.0001, "max": 0.1},  
            "model_type": {"values": ["mobilenetv3small", "resnet50"]},
            "optimizer_type": {"values": ["adam", "adadelta"]},
            "loss_fn": {"values": ["mse", "l1"]}
        }
    }
    
    # Set up the sweep
    sweep_id = wandb.sweep(sweep_config, project="court-detection")

    # Run the sweep
    wandb.agent(sweep_id, function=training, count=16)

if __name__ == "__main__":
    run_sweep()

