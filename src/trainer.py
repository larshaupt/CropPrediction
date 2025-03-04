import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import os
import logging
import wandb

class Trainer:
    """
    Trainer class for training and evaluating a model.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, metrics=[], scheduler=None, device=None, config=None, logger=None):
        """
        Initializes the trainer with model, data, and optimization settings.
        
        Parameters:
            - model (nn.Module): The model to train.
            - train_loader (DataLoader): Training DataLoader.
            - val_loader (DataLoader): Validation DataLoader.
            - criterion (nn.Module): Loss function.
            - optimizer (Optimizer): Optimizer for the model.
            - metrics (list, optional): List of metrics to evaluate (default []).
            - scheduler (Scheduler, optional): Learning rate scheduler (default None).
            - device (torch.device, optional): The device to use for training (default CPU).
            - config (dict, optional): Configuration settings (default None).
            - logger (Logger, optional): Logger for logging (default None).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        self.model.to(self.device)

        # Metrics for evaluation
        self.metrics = metrics

        # Set up logging
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("Trainer")
        else:
            self.logger = logger

    def _train_one_epoch(self, epoch):
        """
        Trains the model for one epoch.
        
        Parameters:
            - epoch (int): The current epoch number.
        
        Returns:
            - epoch_loss (float): The average loss for the epoch.
            - epoch_acc (float): The accuracy of the model for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        predictions = []
        targets = []

        # Iterate over the training data
        batch_tqdm = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch_idx, batch in enumerate(batch_tqdm):
            
            
            # Get inputs and labels
            images = batch["image"].to(self.device)
            labels = batch["mask"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)

            # Calculate loss
            loss, target_max = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Update running loss and accuracy
            running_loss += loss.item()
            preds = torch.nn.functional.sigmoid(outputs).detach().cpu()
            target_max = target_max.detach().cpu()
            
            predictions.append(preds)
            targets.append(target_max)
            
            batch_tqdm.set_postfix(loss=loss.item())

        predictions, targets = torch.cat(predictions, axis=0), torch.cat(targets, axis=0)
        epoch_loss = running_loss / len(self.train_loader)
        epoch_metrics = [metric(predictions, targets) for metric in self.metrics]
        return epoch_loss, epoch_metrics

    def _evaluate(self, epoch):
        """
        Evaluates the model on the validation dataset.
        
        Parameters:
            - epoch (int): The current epoch number.
        
        Returns:
            - val_loss (float): The validation loss for the epoch.
            - val_acc (float): The accuracy of the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        predictions = []
        targets = []

        # No gradient calculation needed during evaluation
        with torch.no_grad():
            batch_tqdm = tqdm(self.val_loader, desc=f"Evaluating Epoch {epoch+1}", leave=False)
            for batch_idx, batch in enumerate(batch_tqdm):
                
                images = batch["image"].to(self.device)
                labels = batch["mask"].to(self.device)

                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss, target_max = self.criterion(outputs, labels)

                # Update running loss and accuracy
                running_loss += loss.item()
                preds = torch.nn.functional.sigmoid(outputs).detach().cpu()
                target_max = target_max.detach().cpu()
                
                predictions.append(preds)
                targets.append(target_max)
                
                batch_tqdm.set_postfix(loss=loss.item())

        val_loss = running_loss / len(self.val_loader)
        predictions, targets = torch.cat(predictions, axis=0), torch.cat(targets, axis=0)
        val_metrics = [metric(predictions, targets) for metric in self.metrics]
        return val_loss, val_metrics

    def train(self, num_epochs):
        """
        Runs the training and evaluation for multiple epochs.
        
        Parameters:
            - num_epochs (int): The number of epochs to train.
        """
        best_val_acc = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            
            
            # Train the model for one epoch
            train_loss, train_metric = self._train_one_epoch(epoch)
            
            # Log the results for the current epoch
            wandb.log({"train_loss": train_loss, "train_acc": train_metric[0]})

            # Evaluate the model
            val_loss, val_metrics = self._evaluate(epoch)
            
            # Log the results for the current validation
            wandb.log({"val_loss": val_loss, "val_acc": val_metrics[0]})

            # Log the results for the current epoch
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                             f"Train Loss: {train_loss:.4f}, Train Accuracy: {val_metrics[0]:.4f}, "
                             f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_metrics[0]:.4f}")

            # Save the best model
            if val_metrics[0] > best_val_acc:
                best_val_acc = val_metrics[0]
                best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics[0])

            # Optionally update the learning rate scheduler
            if self.scheduler:
                self.scheduler.step()

        self.logger.info(f"Training completed. Best Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")

    def save_checkpoint(self, epoch, val_acc):
        """
        Saves the model checkpoint.
        
        Parameters:
            - epoch (int): The epoch number at which the checkpoint is saved.
            - val_acc (float): The validation accuracy at the time of saving.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
        }
        checkpoint_path = os.path.join(self.config.get("output_dir", "./outputs/models"), f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint at {checkpoint_path}")

