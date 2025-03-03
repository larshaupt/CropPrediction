import torch
import numpy as np
import torchmetrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation dataset.

    Parameters:
        - model (torch.nn.Module): The trained model.
        - dataloader (DataLoader): The DataLoader for the validation dataset.
        - criterion (torch.nn.Module): The loss function.
        - device (torch.device): The device (CPU or CUDA).

    Returns:
        - eval_loss (float): The average loss for the validation set.
        - eval_acc (float): The accuracy on the validation set.
        - confusion_mat (ndarray): The confusion matrix.
        - classification_rep (str): The classification report.
    """
    model.eval()
    eval_loss = 0.0
    all_preds = []
    all_labels = []

    # Iterate over validation data
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["field_ids"].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute average loss
    eval_loss /= len(dataloader)

    # Calculate accuracy
    accuracy = torchmetrics.Accuracy()
    accuracy.update(torch.tensor(all_preds), torch.tensor(all_labels))
    eval_acc = accuracy.compute().item()

    # Confusion matrix
    confusion_mat = confusion_matrix(all_labels, all_preds)

    # Classification report
    classification_rep = classification_report(all_labels, all_preds)

    return eval_loss, eval_acc, confusion_mat, classification_rep

def log_evaluation_metrics(eval_loss, eval_acc, confusion_mat, classification_rep, logger):
    """
    Logs evaluation metrics (loss, accuracy, confusion matrix, classification report).

    Parameters:
        - eval_loss (float): The average loss for the validation set.
        - eval_acc (float): The accuracy on the validation set.
        - confusion_mat (ndarray): The confusion matrix.
        - classification_rep (str): The classification report.
        - logger (logging.Logger): The logger instance.
    """
    logger.info(f"Validation Loss: {eval_loss:.4f}")
    logger.info(f"Validation Accuracy: {eval_acc * 100:.2f}%")
    logger.info("Classification Report:\n" + classification_rep)

def plot_confusion_matrix(confusion_mat, class_names, normalize=False):
    """
    Plots the confusion matrix.

    Parameters:
        - confusion_mat (ndarray): The confusion matrix.
        - class_names (list): List of class names.
        - normalize (bool): Whether to normalize the confusion matrix values.
    """
    if normalize:
        confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
