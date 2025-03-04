import os
import torch
import logging
import yaml
import wandb  # Import wandb
from pathlib import Path

from torch.optim import Adam
from torchmetrics import Accuracy, F1Score
from src.data_loader import create_dataloader
from src.model import CropClassifierWithGRU
from src.trainer import Trainer
from src.metrics import F1ScorePerClass
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
    device = config['device']
    logger.info(f"Using device: {device}")
    
    # Load the dataset and dataloaders
    data_root = config['data_root']
    split_csv = config['data_split']
    
    train_loader = create_dataloader(data_root, split_csv, batch_size=config['batch_size'], is_train=True, num_workers=4)
    val_loader = create_dataloader(data_root, split_csv, batch_size=config['batch_size'], is_train=False, num_workers=4)
    
    class_distribution = train_loader.dataset.get_class_distribution(config["num_classes"])
    
    # Create the model
    model = CropClassifierWithGRU(
        num_classes=config['num_classes'],
        gru_hidden_size=config['gru_hidden_size'],
        pretrained=True
    ).to(device)
    
    # Define loss function and optimizer
    criterion = BCEMILLoss(n_classes=config['num_classes'], merge_intercropping=True, class_distribution=class_distribution, device=device)
    optimizer = Adam(model.parameters(), lr=config['lr'])
    metrics = {
        "accuracy": Accuracy(task="multilabel", num_labels=config["num_classes"]),
        "f1_score_macro": F1Score(task="multilabel", average="macro", num_labels=config["num_classes"]),
        "f1_score_micro": F1Score(task="multilabel", average="micro", num_labels=config["num_classes"]),
        "f1_score_weighted": F1Score(task="multilabel", average="weighted", num_labels=config["num_classes"])
        }
    for i_class in range(config["num_classes"]):
        metrics[f"f1_score_{i_class}"] = F1ScorePerClass(class_idx=i_class)
    
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
    
    
    # Save the model
    model_path = Path(config['output_dir']) / 'models' / 'final_model.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Finish the Wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
