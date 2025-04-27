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

import matplotlib.pyplot as plt
import pandas as pd

# Load Data
url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df = pd.read_csv(url)

my_df

# Change variety to numeric
my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
my_df['variety'] = my_df['variety'].replace('Versicolor', 1.0)
my_df['variety'] = my_df['variety'].replace('Virginica', 2.0)
my_df

# Set X, y
X = my_df.drop('variety',axis=1)
y = my_df['variety']
# Convert to numpy arrays
X = X.values
y = y.values


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# X features to FloatTensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()
# Choose Adam Optimizer, lr = learning rate (if error doesnt go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Train our model
# Epochs? (one run trhough all the training data in our network)
epochs = 200
losses = [] # To append losses (see progession can be seen)
for i in range(epochs):
    # Go forwards and get a prediction
    y_pred = model.forward(X_train) # Get predicted results
    
    # Measure the loss/error
    loss = criterion(y_pred, y_train) # Predicted value vs the y_train value
    
    # Keep track of losses
    losses.append(loss.detach().numpy())
    
    # Print every 10 Epochs
    if i % 10 == 0:
        print(f'Epoch_: {i} and loss: {loss}')
        
    # Do some back propagation: take the error rate of forward propogation and feed it back
    # through the network to fine tuine the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Graph it out
plt.plot(range(epochs),losses)
plt.ylabel("loss/error")
plt.xlabel("Epoch")
plt.show()
