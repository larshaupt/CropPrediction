import os
import torch
import logging
import yaml
import wandb  # Import wandb
from pathlib import Path

from torch.optim import Adam
from torchmetrics import Accuracy
from src.data_loader import create_dataloader
from src.model import CropClassifierWithGRU
from src.trainer import Trainer
from src.evaluator import evaluate, log_evaluation_metrics
from src.losses import BCEMILLoss

def setup_logger(log_dir):
    """
    Setup the logger for training.
    
    Parameters:
        - log_dir (str): Directory to save logs.
    
    Returns:
        - logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger('Trainer')
    logger.setLevel(logging.INFO)
    
    # Create directory for logs if not exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a file handler for logging
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Set formatter for logging
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path):
    """
    Loads the configuration YAML file.
    
    Parameters:
        - config_path (str): Path to the config YAML file.
    
    Returns:
        - config (dict): Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config["lr"] = float(config["lr"])
    config["num_classes"] = int(config["num_classes"])
    return config

def main(config_path='config/config.yaml'):
    """
    Main function to initialize the model, data, and trainer, and start training and evaluation.
    
    Parameters:
        - config_path (str): Path to the configuration YAML file.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize Wandb
    wandb.init(project="crop-classification", config=config)  # Initialize Wandb
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Setup logger
    log_dir = Path(config['output_dir']) / 'logs'
    logger = setup_logger(log_dir)
    logger.info("Starting training process")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load the dataset and dataloaders
    data_root = config['data_root']
    split_csv = config['data_split']
    
    train_loader = create_dataloader(data_root, split_csv, batch_size=config['batch_size'], is_train=True, num_workers=4)
    val_loader = create_dataloader(data_root, split_csv, batch_size=config['batch_size'], is_train=False, num_workers=4)
    
    # Create the model
    model = CropClassifierWithGRU(
        num_classes=config['num_classes'],
        gru_hidden_size=config['gru_hidden_size'],
        pretrained=True
    ).to(device)
    
    # Define loss function and optimizer
    criterion = BCEMILLoss(n_classes=config['num_classes'])
    optimizer = Adam(model.parameters(), lr=config['lr'])
    metrics = [Accuracy(task="multilabel", num_classes=config["num_classes"], num_labels=config["num_classes"])]
    
    # Initialize the trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        metrics=metrics,
        logger=logger,
        config=config,
        val_loader=val_loader,
        train_loader=train_loader,
    )
    
    # Train the model
    trainer.train(config['num_epochs'])
    
    # Evaluate the model on the validation set
    eval_loss, eval_acc, confusion_mat, classification_rep = evaluate(model, val_loader, criterion, device)
    log_evaluation_metrics(eval_loss, eval_acc, confusion_mat, classification_rep, logger)
    
    # Log evaluation metrics to Wandb
    wandb.log({"eval_loss": eval_loss, "eval_acc": eval_acc})
    
    # Save the model
    model_path = Path(config['output_dir']) / 'models' / 'final_model.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Finish the Wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
