import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    # Input Layer (4 features of iris) --> 
        # Hidden Layer1 (number of neurons) --> 
            # H2 (n) --> 
                # output (3 classes of iris)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__() # Instantiate our nn.Module
        self.fc1 = nn.Linear(in_features, h1) # Input Layer
        self.fc2 = nn.Linear(h1, h2) # Hidden Layer
        self.out = nn.Linear(h2, out_features) # Output Layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) # relu = Rectified linear Unit (if number is under 0, just call it 0)
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x
    
# Set seed
torch.manual_seed(41)

# Create an instance of model
model = Model()
