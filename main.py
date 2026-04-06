import ssl

# To bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Data Preparation (Data Preprocessing)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


# 2. Model Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes (digits 0-9)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten operation
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN()

# 3. Optimization and Loss Function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model architecture is ready!")
print(model)

# 4. Training Loop
epochs = 5  # We will train the model for 5 epochs

print("\nTraining starting... (This may take a few minutes depending on your computer's speed)")

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs (images) and labels (actual digits)
        inputs, labels = data

        # Zero the parameter gradients for optimization
        optimizer.zero_grad()

        # 1. Forward Pass: Model makes a prediction
        outputs = model(inputs)

        # 2. Calculate Loss: Difference between prediction and actual
        loss = criterion(outputs, labels)

        # 3. Backward Pass: Propagate errors backward
        loss.backward()

        # 4. Optimization: Update weights
        optimizer.step()

        running_loss += loss.item()

        # Print statistics every 300 batches
        if i % 300 == 299:
            print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] Loss: {running_loss / 300:.3f}')
            running_loss = 0.0

print("Training completed successfully!")

# 5. Testing Phase and Accuracy Measurement
# Download and prepare the test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

correct = 0
total = 0

# Set the model to evaluation mode
model.eval()

print("\nTesting phase starting...")

# No need to calculate gradients during testing, this saves CPU/memory
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)

        # Choose the class with the highest probability
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the 10,000 test images: {accuracy:.2f}%')

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 6. Model Explainability Heatmap (Saliency Map)
print("\nGenerating Heatmap visualization...")

model.eval()

# Test veri setinden bir resim al
dataiter = iter(testloader)
images, labels = next(dataiter)

# İlk resmi seç
img = images[0].unsqueeze(0).clone()
label = labels[0]

# Geri yayılım (backprop) için resmin gradyanlarını açık tut
img.requires_grad_()

# 1. Tahmin (Forward)
output = model(img)
pred_class = output.argmax(dim=1).item()

# 2. Geri Yayılım (Backward)
model.zero_grad()
output[0, pred_class].backward()

# 3. Piksellerin gradyanlarını (etki haritasını) al
saliency = img.grad.data.abs().squeeze().numpy()

# Isı haritasını 0 ile 1 arasına sabitle (Sıfırlanma hatasını kesin çözer)
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

# 4. Grafikleri Çizdir
img_np = img.squeeze().detach().numpy()

plt.figure(figsize=(10, 4))

# Orijinal Resim
plt.subplot(1, 3, 1)
plt.imshow(img_np, cmap='gray')
plt.title(f"Original Image\nActual: {label.item()}, Pred: {pred_class}")
plt.axis('off')

# Isı Haritası (Sıcaklık)
plt.subplot(1, 3, 2)
plt.imshow(saliency, cmap='hot')
plt.title("Saliency Heatmap")
plt.axis('off')

# Üst Üste Binmiş (Overlay)
plt.subplot(1, 3, 3)
plt.imshow(img_np, cmap='gray')
plt.imshow(saliency, cmap='hot', alpha=0.6)
plt.title("Overlay")
plt.axis('off')

plt.tight_layout()
plt.show()

print("Heatmap generated successfully! You can use this plot in your report and video.")
