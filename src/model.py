# src/model.py

import torch
import torch.nn as nn
import torchgeo.models as geo_models

class CropClassifierWithGRU(nn.Module):
    """
    This model combines a pre-trained ResNet-50 model for feature extraction with a GRU
    layer to account for the temporal resolution of the images.
    """
    def __init__(self, num_classes, gru_hidden_size=256, gru_num_layers=2, dropout=0.5, pretrained=True):
        """
        Initializes the model.

        Parameters:
            - num_classes (int): Number of output classes for classification.
            - gru_hidden_size (int): The hidden size for the GRU layer.
            - gru_num_layers (int): The number of GRU layers.
            - dropout (float): Dropout rate for regularization.
            - pretrained (bool): Whether to use pre-trained weights for the ResNet model.
        """
        super(CropClassifierWithGRU, self).__init__()
        
        if pretrained:
            # Load the pre-trained ResNet-50 model from TorchGeo
            self.resnet = geo_models.resnet50(weights=geo_models.ResNet50_Weights.SENTINEL2_ALL_DECUR)
        else:
            # Initialize a new ResNet-50 model
            self.resnet = geo_models.resnet50(pretrained=False)
        
        # GRU layer to handle temporal resolution of the image data
        self.gru = nn.GRU(
            input_size=1000,  # ResNet-50 output feature size
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,  # (batch_size, sequence_length, features)
            dropout=dropout,
            bidirectional=False  # Unidirectional GRU
        )

        # A dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer for classification
        self.fc = nn.Linear(gru_hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
            - x (Tensor): Input tensor with shape (batch_size, sequence_length, n_bands, height, width)
        
        Returns:
            - output (Tensor): The classification output.
        """
        batch_size, sequence_length, _, _, _ = x.shape
        
        # Extract features from each timepoint using the ResNet model
        resnet_features = []
        for t in range(sequence_length):
            # Reshape to match the input requirements of the ResNet model
            img_t = x[:, t, :, :, :]
            features_t = self.resnet(img_t)
            resnet_features.append(features_t)
        
        # Stack the features over the temporal dimension (sequence_length)
        resnet_features = torch.stack(resnet_features, dim=1)  # Shape: (batch_size, sequence_length, feature_size)

        # Flatten the last 2 dimensions
        resnet_features = resnet_features.view(batch_size, sequence_length, -1)
        
        # Pass the temporal features through the GRU layer
        gru_out, _ = self.gru(resnet_features)  # Output shape: (batch_size, sequence_length, gru_hidden_size)
        
        # Take the last time step's output (or use some aggregation)
        final_gru_out = gru_out[:, -1, :]  # Shape: (batch_size, gru_hidden_size)

        # Apply dropout
        final_gru_out = self.dropout(final_gru_out)

        # Classification head
        output = self.fc(final_gru_out)  # Shape: (batch_size, num_classes)
        
        return output
