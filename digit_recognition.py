

#Import necessary libraries
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Figure out how many images are in the train_set and test_set.
print(len(train_set))
print(len(test_set))

# Import necessary PyTorch libraries
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #: Define layers of the neural network
        self.fc1 = nn.Linear(28*28,10) # First fully connected layer
        self.fc1 = nn.Linear(10,10) # First fully connected layer


    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)  # TODO: add an activation function
        return F.log_softmax(x,dim=1)

# Create an instance of the network
model = Net()
print(model)




# Import optimizer
from torch.optim import SGD

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Set the number of epochs
num_epochs =20

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0

    for images, labels in train_loader:
        # Complete Training pass
        optimizer.zero_grad()   # Clear gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

        running_loss += loss.item()
    else:

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    # evaluate on the test_loader
    test_loss = 0.0
    model.eval()  # Set the model to evaluation mode
    for images, labels in test_loader:
        # Evaluation pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
    else:
        print(f" Loss: {test_loss/len(test_loader)}")

print("Training is finished!")

# plot the model complexity graph
# Define different learning rates to represent model complexity
import matplotlib.pyplot as plt

# Define different learning rates to represent model complexity
learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
train_losses = []
val_losses = []

# Training loop for different learning rates
for lr in learning_rates:
    model = Net()  # Reinitialize the model
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Evaluate the model
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    val_losses.append(test_loss / len(test_loader))

# Plot the model complexity graph
plt.plot(learning_rates, train_losses, label='Training Loss')
plt.plot(learning_rates, val_losses, label='Validation Loss')
plt.xlabel('Learning Rate (Model Complexity)')
plt.ylabel('Loss')
plt.title('Model Complexity Graph')
plt.legend()
plt.show()


# Define the neural network class with more complexity
class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        # Define more layers for a complex model
        self.fc1 = nn.Linear(28*28, 128)  # From input layer to hidden layer 1
        self.fc2 = nn.Linear(128, 64)     # From hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(64, 32)      # From hidden layer 2 to hidden layer 3
        self.fc4 = nn.Linear(32, 10)      # From hidden layer 3 to output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 28 * 28)
        # Apply layers with ReLU activation functions
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return F.log_softmax(x, dim=1)

# Create an instance of the complex network
complex_model = ComplexNet()
print(complex_model)

"""## Implementing Early Stopping 🛑"""


# Complete this code to implement Early stopping
patience = 10
min_delta = 0.2
best_loss = None
patience_counter = 0

# Training loop with early stopping
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # Training pass

        running_loss += loss.item()

    # evaluation phase
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            validation_loss +=loss.item()

    # Calculate average losses
    training_loss = running_loss / len(train_loader)
    validation_loss /= len(test_loader)

    print(f"Epoch {epoch+1}, Training Loss: {training_loss}, Validation Loss: {validation_loss}")

    # Early stopping logic
    if best_loss is None or validation_loss < best_loss - min_delta:
        best_loss = validation_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

print("Training is finished!")

# What does min_delta and patience refer to ?
# min_delta: Minimum required improvement in validation loss or metric.
# patience: Number of epochs to wait for improvement before stopping.
# What is different from the first training ?
#first training we have noticed an improvement (decrease) in validation loss but the complex model is more stagnant

"""## Experimenting with Dropout 🌧️"""

class NetWithDropout(nn.Module):
    def __init__(self):
        super(NetWithDropout, self).__init__()
        # Define layers of the neural network
        self.fc1 = nn.Linear(28 * 28, 128)
        self.dropout1 = nn.Dropout()  # Dropout layer with 20% probability
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout()  # Dropout layer with 50% probability
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 28 * 28)
        # Forward pass with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Create an instance of the network with dropout
model_with_dropout = NetWithDropout()
print(model_with_dropout)

class NetWithDropout(nn.Module):
    def __init__(self):
        super(NetWithDropout, self).__init__()
        # Define layers of the neural network
        self.fc1 = nn.Linear(28 * 28, 128)
        self.dropout1 = nn.Dropout()  # Dropout layer with 20% probability
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout()  # Dropout layer with 50% probability
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 28 * 28)
        # Forward pass with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Create an instance of the network with dropout
model_with_dropout = NetWithDropout()
print(model_with_dropout)

# Train the dropout model

# Training loop for different learning rates
for lr in learning_rates:
    model = NetWithDropout()  # Reinitialize the model
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses = []  # List to store training losses
    val_losses = []    # List to store validation losses
    # Train the model
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

    # Evaluate the model
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    val_losses.append(test_loss / len(test_loader))

print(f"Epoch {epoch+1}, Training Loss: {training_loss}, Validation Loss: {validation_loss}")

