import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 📍 Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_updates = 750_000
batch_size = 23
loss_log_file = "losses_23batch.txt"
constant_lr = 8.5e-5  # Estimated average LR from previous scheduler

# 🔄 CIFAR-100 Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 📦 CIFAR-100 Dataset & DataLoader
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 🧠 Efficient CNN for CIFAR-100
class SmallCIFAR100CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 100)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

# 🎯 Setup
model = SmallCIFAR100CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=constant_lr)

# 🧾 Log file
loss_file = open(loss_log_file, "w")
losses = []
update_counter = 0

print("🚀 Training on CIFAR-100 (batch size 23, constant LR) for 750,000 updates...\n")

# 🔁 Training loop
while update_counter < max_updates:
    for images, labels in train_loader:
        if update_counter >= max_updates:
            break

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        update_counter += 1
        loss_val = loss.item()
        losses.append(loss_val)
        loss_file.write(f"{update_counter},{loss_val:.6f}\n")

        if update_counter % 1000 == 0 or update_counter == 1:
            percent = (update_counter / max_updates) * 100
            print(f"[{update_counter}/{max_updates}] Loss: {loss_val:.4f} ({percent:.2f}%)")

loss_file.close()
print("\n✅ Training complete. Losses saved to 'losses_23batch.txt'.")
