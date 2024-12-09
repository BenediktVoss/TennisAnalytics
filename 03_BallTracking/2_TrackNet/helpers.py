import cv2
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from scipy.spatial import distance
from torchvision.ops import sigmoid_focal_loss

class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, predictions, targets):
        # Calculate weights based on residual
        weights = predictions  

        # WBCE components
        term1 = (1 - weights) ** 2 * targets * torch.log(predictions + 1e-8)
        term2 = weights ** 2 * (1 - targets) * torch.log(1 - predictions + 1e-8)

        # Weighted binary cross-entropy
        loss = -torch.sum(term1 + term2)
        return loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="sum"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions, targets):

        # Apply sigmoid if predictions are logits
        predictions = F.sigmoid(predictions)

        # Ensure predictions are probabilities (apply sigmoid if necessary)
        predictions = predictions.clamp(min=1e-8, max=1 - 1e-8)  # Avoid log(0)

        # Compute focal loss components
        term1 = targets * (1 - predictions) ** self.gamma * torch.log(predictions)
        term2 = (1 - targets) * predictions ** self.gamma * torch.log(1 - predictions)

        loss = -self.alpha * term1 - (1 - self.alpha) * term2

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")


def train(model, train_loader, optimizer, device, epoch, max_iters=200, criterion=WeightedBCELoss()):
    start_time = time.time()
    losses = []
    
    
    # Wrap train_loader with tqdm for a progress bar
    with tqdm(total=min(len(train_loader), max_iters), desc=f"Epoch {epoch}", unit="batch") as pbar:
        for iter_id, batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            model.train()

            # Forward pass
            inputs = batch[0].float().to(device)  # Input frames
            heatmap_gt = batch[1].float().to(device)  # Ground truth heatmap
            outputs = model(inputs)

            loss = criterion(outputs, heatmap_gt)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track losses
            losses.append(loss.item())
            end_time = time.time()
            duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

            # Update the tqdm bar
            pbar.set_postfix({
                'loss': round(loss.item(), 6),
                'time': duration
            })
            pbar.update(1)

            if iter_id >= max_iters - 1:
                break

    return np.mean(losses)


def calculate_position(output, threshold):
    # Binarize the heatmap
    binary_map = (output > threshold).cpu().numpy().astype(np.uint8)

    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

    if num_labels > 1:  # Exclude background label 0
        # Find the largest region (excluding background label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Index 1 is the first component
        largest_centroid = centroids[largest_label]
        return int(largest_centroid[0]), int(largest_centroid[1])
    else:
        # No region found
        return -1, -1
    

def calculate_frame_metrics(predicted_point, ground_truth_point, min_dist=4):
    # Check if both predicted and ground truth points are valid
    if predicted_point != (-1, -1) and ground_truth_point != (-1, -1):
        # Calculate distance
        if distance.euclidean(predicted_point, ground_truth_point) < min_dist:
            return ("tp")               # True Positive: Prediction matches ground truth
        else:
            return ("fp")               # False Positive: Prediction exists but doesn't match ground truth
    elif predicted_point != (-1, -1):
        return ("fp")                   # False Positive: Prediction exists but no ground truth
    elif ground_truth_point != (-1, -1):
        return ("fn")                   # False Negative: Ground truth exists but no prediction
    else:
        return ("tn")                   # True Negative: Neither prediction nor ground truth exists


def calculate_metrics(stats):
    tp = stats.get("tp", 0)
    tn = stats.get("tn", 0)
    fp = stats.get("fp", 0)
    fn = stats.get("fn", 0)

    # Avoid division by zero with if 
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn ) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

def tensor_to_point(gt_point):
    # Convert tensor([x], [y]) to (x, y)
    if isinstance(gt_point, list) or isinstance(gt_point, tuple):
        return tuple(p.item() for p in gt_point)
    elif isinstance(gt_point, torch.Tensor):
        return tuple(gt_point.tolist())
    else:
        raise ValueError("Unsupported ground truth format")

def validate(model, val_loader, device, min_dist=4, threshold=0.5, criterion=WeightedBCELoss()):
    losses = []
    # Create a dictionary for tp, tn, fp, fn
    stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    model.eval()

    # Wrap val_loader with tqdm for a progress bar
    with tqdm(total=len(val_loader), desc="Validation", unit="batch") as pbar:
        for iter_id, batch in enumerate(val_loader):
            with torch.no_grad():
                inputs = batch[0].float().to(device)  # Input frames
                heatmap = batch[1].float().to(device)  # Ground truth heatmap
                points = batch[2]  # Ground truth ball positions

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, heatmap)
                losses.append(loss.item())

                for batch_idx in range(outputs.shape[0]):
                    for frame_idx in range(outputs[batch_idx].shape[0]):
                        # Process each batch separately
                        output = outputs[batch_idx][frame_idx]
                        gt_point = points[batch_idx][frame_idx]

                        gt_point_formatted = tensor_to_point(gt_point) 

                        # Calculate predicted position for the frame
                        predicted_position = calculate_position(output, threshold)

                        case = calculate_frame_metrics(predicted_position, gt_point_formatted, min_dist)

                        # Update stats
                        stats[case] += 1

                # Update the tqdm bar
                pbar.set_postfix({'loss': round(np.mean(losses), 6)})
                pbar.update(1)

    # Calculate metrics
    accuracy, precision, recall, f1 = calculate_metrics(stats)

    return np.mean(losses), accuracy, precision, recall, f1