import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutional Block 1
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3)
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=3)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        
        # Convolutional Block 4
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm3d(64)
        self.pool4 = nn.MaxPool3d(kernel_size=3)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * feature_size, 128)  # Adjust feature_size based on input size
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(128, 2)
        self.dropout2 = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return F.softmax(x, dim=1)

# Create the model
model = CNNModel()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# Assume feature_size is defined, or compute it based on your input dimensions
