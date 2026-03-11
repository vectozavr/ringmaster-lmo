import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from function import NeuralNetworkFunction, NeuralNetworkFunction_Data_Loader

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

# Define a simple two-layer neural network
class TwoLayerNN(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, output_size=10):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoLayerNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Function to extract model parameters as a flattened tensor
def get_model_parameters(model):
    return torch.cat([p.view(-1) for p in model.parameters()]).detach()

# Neural network function
neural_network_function = NeuralNetworkFunction_Data_Loader(train_loader=train_loader, number_of_classes=10, neural_network_name='two_layer_neural_net_relu')

# Store loss for plotting
loss_values = []
loss_2 = []

# Training loop
for epoch in range(1000):
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Store loss values
    loss_values.append(loss.item())
    loss_2.append(neural_network_function.value(get_model_parameters(model).cpu().numpy()))

print("Training completed!")

# Plot loss curves
plt.plot(loss_values, label="PyTorch Loss")
plt.plot(loss_2, label="Function Loss", linestyle="dashed")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.show()